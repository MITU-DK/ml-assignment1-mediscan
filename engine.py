"""
Training / inference loop shared by train.py and the Part 2 scripts.
Images are cached as uint8 arrays and augmented a whole batch at a time with
affine_grid, so no DataLoader workers are needed. Uses the GPU if there is one
(Kaggle / Colab), otherwise the CPU.
"""
import os
import math
import copy
import time

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F

from common import CACHE, MODELS, set_seed, metrics_at
from models import build, count_params

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", max(1, os.cpu_count() - 2))))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

DEFAULT = dict(model="arogya", res=128, aug="safe", epochs=15, lr=1e-3, bs=32, wd=1e-4,
               dropout=0.2, residual=True, se=True, meta=False, pos_weight=1.0,
               sampler=None, clahe=False, patience=5)

_cache = {}


def apply_clahe(x):
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return np.stack([cl.apply(im) for im in x])


def load_data(res, clahe=False):
    """returns (images uint8 [N,res,res], split dataframe)"""
    key = (res, clahe)
    if key not in _cache:
        df = pd.read_csv(os.path.join(CACHE, "split.csv"))
        src = 96 if res == 96 else 224
        X = np.load(os.path.join(CACHE, f"img{src}.npy"))
        if res not in (96, 224):
            X = np.stack([cv2.resize(im, (res, res), interpolation=cv2.INTER_AREA) for im in X])
        if clahe:
            X = apply_clahe(X)
        _cache[key] = (X, df)
    return _cache[key]


def meta_tensor(df, idx):
    m = df.loc[idx, ["meta_mean", "meta_aspect", "meta_area"]].values.astype(np.float32)
    # rough standardisation with fixed constants (from the audit numbers) so it is not fit on val
    m = (m - np.array([0.48, 1.35, 1.4], dtype=np.float32)) / np.array([0.08, 0.2, 0.9], dtype=np.float32)
    return torch.from_numpy(m)


def to_tensor(X, idx):
    x = torch.from_numpy(X[idx]).float().unsqueeze(1) / 255.0
    return (x - 0.5) / 0.25


def augment(x, policy, gen):
    """x: [B,1,H,W] normalised. policy in none / safe / full / strong"""
    if policy == "none":
        return x
    B = x.shape[0]
    ang = (torch.rand(B, generator=gen) * 2 - 1) * math.radians(10)
    scale = 1.0 / (0.9 + torch.rand(B, generator=gen) * 0.2)      # zoom 0.9-1.1
    cos, sin = torch.cos(ang) * scale, torch.sin(ang) * scale
    flip = torch.ones(B)
    if policy == "full":
        flip = torch.where(torch.rand(B, generator=gen) < 0.5, -1.0, 1.0)
    theta = torch.zeros(B, 2, 3)
    theta[:, 0, 0] = cos * flip
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin * flip
    theta[:, 1, 1] = cos
    theta[:, :, 2] = (torch.rand(B, 2, generator=gen) * 2 - 1) * 0.05   # small shift
    grid = F.affine_grid(theta.to(x.device), x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    if policy == "strong":
        # brightness / contrast jitter, +-25%
        c = 0.75 + torch.rand(B, 1, 1, 1, generator=gen) * 0.5
        b = (torch.rand(B, 1, 1, 1, generator=gen) * 2 - 1) * 0.5
        x = x * c.to(x.device) + b.to(x.device)
    return x


@torch.no_grad()
def predict(model, X, df, idx, cfg, bs=128, return_emb=False):
    model.eval()
    dev = next(model.parameters()).device
    probs, embs = [], []
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        x = to_tensor(X, b).to(dev)
        m = meta_tensor(df, b).to(dev) if cfg.get("meta") else None
        if return_emb:
            e = model.embed(x, m)
            embs.append(e.cpu().numpy())
            logit = model.fc(e).squeeze(1)
        else:
            logit = model(x, m)
        probs.append(torch.sigmoid(logit).cpu().numpy())
    p = np.concatenate(probs)
    return (p, np.concatenate(embs)) if return_emb else p


def train(cfg, train_idx, val_idx, seed, verbose=True):
    """train one model, keep the epoch with best val ROC-AUC. returns dict"""
    cfg = {**DEFAULT, **cfg}
    if os.environ.get("SMOKE"):          # tiny run, only to check the scripts don't crash
        cfg["epochs"] = 1
        train_idx, val_idx = train_idx[::25], val_idx[::5]
    set_seed(seed)
    X, df = load_data(cfg["res"], cfg["clahe"])
    y = df.label.values.astype(np.float32)
    model = build(cfg).to(DEVICE)
    lr = cfg["lr"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["wd"])
    steps = cfg["epochs"] * math.ceil(len(train_idx) / cfg["bs"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    pw = torch.tensor(cfg["pos_weight"], device=DEVICE)
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.RandomState(seed)

    hist = {"train_loss": [], "val_loss": [], "val_auc": [], "val_acc": [], "train_acc": []}
    best, best_auc, bad = None, -1, 0
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        if cfg["sampler"] == "oversample":
            pos = train_idx[y[train_idx] == 1]
            neg = train_idx[y[train_idx] == 0]
            neg = rng.choice(neg, len(pos), replace=True)
            order = rng.permutation(np.concatenate([pos, neg]))
        else:
            order = rng.permutation(train_idx)
        tl, tc, n = 0.0, 0, 0
        for s in range(0, len(order), cfg["bs"]):
            b = order[s:s + cfg["bs"]]
            if len(b) < 2:
                continue
            x = augment(to_tensor(X, b).to(DEVICE), cfg["aug"], gen)
            m = meta_tensor(df, b).to(DEVICE) if cfg["meta"] else None
            t = torch.from_numpy(y[b]).to(DEVICE)
            logit = model(x, m)
            loss = F.binary_cross_entropy_with_logits(logit, t, pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(b)
            tc += ((logit > 0).float() == t).sum().item()
            n += len(b)
        pv = predict(model, X, df, val_idx, cfg)
        yv = y[val_idx]
        vl = F.binary_cross_entropy(torch.tensor(pv).clamp(1e-6, 1 - 1e-6), torch.tensor(yv)).item()
        mv = metrics_at(yv, pv)
        hist["train_loss"].append(tl / n)
        hist["train_acc"].append(tc / n)
        hist["val_loss"].append(vl)
        hist["val_auc"].append(mv["roc_auc"])
        hist["val_acc"].append(mv["accuracy"])
        if verbose:
            print(f"  ep{ep + 1:02d} train_loss {tl / n:.4f} val_loss {vl:.4f} "
                  f"val_auc {mv['roc_auc']:.4f} val_acc {mv['accuracy']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if mv["roc_auc"] > best_auc + 1e-4:
            best_auc, best, bad = mv["roc_auc"], copy.deepcopy(model.state_dict()), 0
            hist["best_epoch"] = ep + 1
        else:
            bad += 1
            if cfg["patience"] and bad >= cfg["patience"]:
                if verbose:
                    print("  early stop")
                break
    model.load_state_dict(best)
    model = model.cpu()
    pv = predict(model, X, df, val_idx, cfg)
    return {"model": model, "cfg": cfg, "hist": hist, "val_prob": pv, "val_idx": val_idx,
            "val_metrics": metrics_at(y[val_idx], pv), "params": count_params(model),
            "train_sec": time.time() - t0}


def split_idx(fold=0):
    df = pd.read_csv(os.path.join(CACHE, "split.csv"))
    pool = df.index[(df.split == "pool") & (df.fold != fold)].values
    val = df.index[(df.split == "pool") & (df.fold == fold)].values
    return pool, val


def test_idx():
    df = pd.read_csv(os.path.join(CACHE, "split.csv"))
    if os.environ.get("SMOKE"):     # smoke tests must never read the locked test set
        return split_idx(0)[1]
    return df.index[df.split == "test"].values


def save_model(res, name):
    torch.save({"state": res["model"].state_dict(), "cfg": res["cfg"]}, os.path.join(MODELS, name))


def load_model(name):
    ck = torch.load(os.path.join(MODELS, name), map_location="cpu", weights_only=False)
    m = build(ck["cfg"])
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck["cfg"]
