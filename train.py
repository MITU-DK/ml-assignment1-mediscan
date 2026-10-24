"""
Part 1, tasks 4-9. Everything is evaluated on fold 0 of the training pool (the
validation set). The locked test set is NOT used here.

usage:
    python train.py --exp shortcut      # task 4  crippled models
    python train.py --exp baselines     # task 5  majority + HOG/LR
    python train.py --exp leak          # Rohit's likely protocol: image-level vs patient-level split
    python train.py --exp transfer      # task 6  ResNet18, 3 seeds
    python train.py --exp arogya        # task 7  ArogyaNet + 2 ablations, 3 seeds each
    python train.py --exp twobranch     # task 8
    python train.py --exp res_aug       # task 9
    python train.py --exp all
"""
import os
import argparse

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

from common import save_json, metrics_at, mean_std, FIG, RES, set_seed
from engine import train, load_data, split_idx, save_model
from models import ArogyaNet, arch_table, count_params

SEEDS = [0, 1, 2]
PRED_DIR = os.path.join(RES, "preds")
os.makedirs(PRED_DIR, exist_ok=True)


def run_seeds(name, cfg, seeds=SEEDS, save=True):
    """train the same config over several seeds, save metrics + val probs + model"""
    tr, va = split_idx(0)
    runs = []
    for s in seeds:
        print(f"== {name} seed {s}")
        r = train(cfg, tr, va, s)
        runs.append({"seed": s, **r["val_metrics"], "best_epoch": r["hist"].get("best_epoch"),
                     "params": r["params"], "train_sec": r["train_sec"], "hist": r["hist"]})
        np.savez(os.path.join(PRED_DIR, f"{name}_s{s}.npz"), idx=va, prob=r["val_prob"])
        if save:
            save_model(r, f"{name}_s{s}.pt")
    out = {"name": name, "cfg": r["cfg"], "runs": runs, "summary": mean_std(runs)}
    save_json(out, f"p1_{name}.json")
    s = out["summary"]
    print(f"   {name}: AUC {s['roc_auc']['mean']:.4f}±{s['roc_auc']['std']:.4f}  "
          f"acc {s['accuracy']['mean']:.4f}±{s['accuracy']['std']:.4f}")
    return out


# ------------------------------------------------------------------ task 4
def exp_shortcut():
    X, df = load_data(224)
    tr, va = split_idx(0)
    y = df.label.values
    out = {}

    def fit_eval(F, name):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
        clf.fit(F[tr], y[tr])
        p = clf.predict_proba(F[va])[:, 1]
        out[name] = metrics_at(y[va], p)
        print(name, {k: round(v, 3) for k, v in out[name].items() if isinstance(v, float)})

    small = np.stack([cv2.resize(im, (16, 16), interpolation=cv2.INTER_AREA) for im in X]).reshape(len(X), -1)
    fit_eval(small.astype(np.float32), "lr_16x16_full")

    # mask the lung field: grey out the central box where the lungs are, keep only the border
    masked = X.copy()
    h0, h1, w0, w1 = int(0.10 * 224), int(0.90 * 224), int(0.12 * 224), int(0.88 * 224)
    masked[:, h0:h1, w0:w1] = 128
    small_m = np.stack([cv2.resize(im, (16, 16), interpolation=cv2.INTER_AREA) for im in masked]).reshape(len(X), -1)
    fit_eval(small_m.astype(np.float32), "lr_16x16_lungs_masked")

    # no pixels at all: only the 3 acquisition numbers
    meta = df[["meta_mean", "meta_aspect", "meta_area"]].values
    fit_eval(meta, "lr_meta_only(mean,aspect,area)")
    fit_eval(df[["meta_aspect", "meta_area"]].values, "lr_size_only(aspect,area)")

    out["chance_majority_acc"] = float(max(y[va].mean(), 1 - y[va].mean()))
    save_json(out, "p1_shortcut.json")

    fig, ax = plt.subplots(1, 4, figsize=(11, 3))
    i0, i1 = va[y[va] == 0][0], va[y[va] == 1][0]
    ax[0].imshow(small[i0].reshape(16, 16), cmap="gray"); ax[0].set_title("NORMAL 16x16")
    ax[1].imshow(small[i1].reshape(16, 16), cmap="gray"); ax[1].set_title("PNEUMONIA 16x16")
    ax[2].imshow(masked[i0], cmap="gray"); ax[2].set_title("lungs masked")
    ax[3].imshow(small_m[i0].reshape(16, 16), cmap="gray"); ax[3].set_title("masked, 16x16")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p1_shortcut_inputs.png"), dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ task 5
def hog_features(X, size=128):
    def one(im):
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
        return hog(im, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    return np.stack([one(im) for im in X]).astype(np.float32)


def exp_baselines():
    X, df = load_data(224)
    tr, va = split_idx(0)
    y = df.label.values
    out = {}
    maj = int(round(y[tr].mean()))
    out["majority"] = metrics_at(y[va], np.full(len(va), float(maj)))
    out["majority"]["pr_auc"] = float(y[va].mean())       # a constant score gives PR-AUC = prevalence
    out["majority"]["roc_auc"] = 0.5
    H = hog_features(X)
    np.save(os.path.join(RES, "preds", "hog_features.npy"), H)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.01))
    clf.fit(H[tr], y[tr])
    p = clf.predict_proba(H[va])[:, 1]
    out["hog_logreg"] = metrics_at(y[va], p)
    out["hog_dim"] = int(H.shape[1])
    np.savez(os.path.join(PRED_DIR, "hog_logreg_s0.npz"), idx=va, prob=p)
    for k in ["majority", "hog_logreg"]:
        print(k, {a: round(b, 3) for a, b in out[k].items() if isinstance(b, float)})
    save_json(out, "p1_baselines.json")


# ------------------------------------------------------------------ Rohit's protocol
def exp_leak():
    """
    How do you get 98%? One easy way: pool the images and split them at random, so
    images of the same child land in train and test. Compare that with a split
    grouped by patient, same model, same sizes. Uses the training pool only.
    """
    X, df = load_data(224)
    pool = df.index[df.split == "pool"].values
    y = df.label.values
    H = hog_features(X[pool])
    groups = df.group.values[pool]
    res = {"image_level": [], "patient_level": []}
    for s in range(5):
        for kind, splitter in [("image_level", ShuffleSplit(1, test_size=0.2, random_state=s)),
                               ("patient_level", GroupShuffleSplit(1, test_size=0.2, random_state=s))]:
            a, b = next(splitter.split(H, y[pool], groups))
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.01))
            clf.fit(H[a], y[pool][a])
            p = clf.predict_proba(H[b])[:, 1]
            m = metrics_at(y[pool][b], p)
            shared = len(set(groups[a]) & set(groups[b]))
            m["test_groups_seen_in_train"] = shared
            res[kind].append(m)
    out = {k: {"summary": mean_std(v), "groups_shared": [r["test_groups_seen_in_train"] for r in v]} for k, v in res.items()}
    print({k: (round(v["summary"]["accuracy"]["mean"], 4), v["groups_shared"]) for k, v in out.items()})

    # same thing with the CNN, one seed each - the CNN can memorise a child far better than HOG
    cfg = {"epochs": 12}
    pool_df = df.loc[pool]
    for kind in ["image_level", "patient_level"]:
        sp = ShuffleSplit(1, test_size=0.2, random_state=0) if kind == "image_level" \
            else GroupShuffleSplit(1, test_size=0.2, random_state=0)
        a, b = next(sp.split(pool, y[pool], pool_df.group.values))
        r = train(cfg, pool[a], pool[b], seed=0, verbose=False)
        out[kind]["arogyanet_seed0"] = r["val_metrics"]
        print(kind, "ArogyaNet acc", round(r["val_metrics"]["accuracy"], 4))
    save_json(out, "p1_leak.json")


# ------------------------------------------------------------------ task 6
def exp_transfer():
    # 224 is what the ImageNet filters were trained at; fewer epochs since it's slow on CPU
    run_seeds("resnet18", {"model": "resnet18", "res": 224, "epochs": 8, "lr": 3e-4, "patience": 3})


# ------------------------------------------------------------------ task 7
def exp_arogya():
    m = ArogyaNet()
    rows = arch_table(m, 128)
    tot = count_params(m)
    lines = ["| layer | output shape | params |", "|---|---|---|"]
    for n, sh, p in rows:
        lines.append(f"| {n} | {' x '.join(map(str, sh))} | {p:,} |")
    lines.append(f"| **total** | | **{tot:,}** |")
    with open(os.path.join(RES, "arogyanet_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    params = {"full": tot, "no_residual": count_params(ArogyaNet(residual=False)),
              "no_se": count_params(ArogyaNet(se=False)), "two_branch": count_params(ArogyaNet(meta=True))}
    save_json(params, "arogyanet_params.json")
    draw_arch(rows)

    run_seeds("arogya_full", {})
    run_seeds("arogya_no_residual", {"residual": False})
    run_seeds("arogya_no_se", {"se": False})


def draw_arch(rows):
    fig, ax = plt.subplots(figsize=(4.2, 9))
    ax.axis("off")
    boxes = [("input 1x128x128", "")] + [(n, " x ".join(map(str, sh))) for n, sh, _ in rows]
    for k, (n, sh) in enumerate(boxes):
        yy = 1 - k / len(boxes)
        col = "#dbe9f6" if "Block" in n else "#f3e3c3"
        ax.add_patch(plt.Rectangle((0.1, yy - 0.06), 0.8, 0.055, fc=col, ec="k", lw=0.8))
        ax.text(0.5, yy - 0.033, f"{n}\n{sh}", ha="center", va="center", fontsize=7)
        if k:
            ax.annotate("", (0.5, yy - 0.005), (0.5, yy + 1 / len(boxes) - 0.06),
                        arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.text(0.5, -0.05, "Block = DSConv(3x3 dw + 1x1 pw) -> DSConv -> SE -> (+ skip) -> ReLU",
            ha="center", fontsize=6.5, wrap=True)
    ax.set_ylim(-0.08, 1)
    fig.savefig(os.path.join(FIG, "p1_arogyanet_diagram.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ task 8
def exp_twobranch():
    run_seeds("arogya_two_branch", {"meta": True})


# ------------------------------------------------------------------ task 9
def exp_res_aug():
    run_seeds("arogya_res96", {"res": 96})
    run_seeds("arogya_res224", {"res": 224})
    run_seeds("arogya_aug_none", {"aug": "none"})
    run_seeds("arogya_aug_flip_rot_zoom", {"aug": "full"})


EXPS = {"shortcut": exp_shortcut, "baselines": exp_baselines, "leak": exp_leak,
        "transfer": exp_transfer, "arogya": exp_arogya, "twobranch": exp_twobranch, "res_aug": exp_res_aug}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", choices=list(EXPS) + ["all"])
    a = ap.parse_args()
    set_seed(0)
    todo = list(EXPS) if a.exp == "all" else [a.exp]
    for e in todo:
        print(f"\n########## {e}")
        EXPS[e]()
