"""
Part 2, task 6: generalisation test on the other hospital. Run ONCE.

External data: COVID-19 Radiography Database, folders Normal/ (label 0) and
Viral Pneumonia/ (label 1) only. Nothing else in the repo reads this folder.

Steps
  1. load + same preprocessing as training (grey -> 224 area resize -> 128)
  2. pixel statistics, compared with the Part 1 audit numbers
  3. overlap check: this database was partly assembled from the same Kaggle pneumonia
     set, so some "external" images may be our own training images. pHash + NCC, same
     rule as audit.py. Scored on all images and on the clean (non-overlapping) part.
  4. final model at the frozen threshold t*
  5. the fix: CLAHE model (trained earlier in final_model.py) at its own val-chosen t*
"""
import os
import glob
from multiprocessing import Pool

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import imagehash
from sklearn.metrics import roc_curve

from common import DATA, CACHE, RES, FIG, save_json, load_json, metrics_at, set_seed
from engine import load_model, predict, apply_clahe
from audit import hamming_matrix, CAND_BITS, NCC_CONFIRM, _small

EXT = os.environ.get("EXTERNAL_DIR", os.path.join(DATA, "external_raw", "COVID-19_Radiography_Dataset"))
CAMP_PREV = 0.10


def load_one(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = g.shape
    g224 = cv2.resize(g, (224, 224), interpolation=cv2.INTER_AREA)
    return (cv2.resize(g224, (128, 128), interpolation=cv2.INTER_AREA),
            (w, h, g.mean(), g.std()), str(imagehash.phash(Image.fromarray(g))))


def score(y, p, t):
    # the clean subset can be empty or one-class if most external images are copies of ours
    if len(y) == 0:
        return {"n": 0}
    return {"n": int(len(y)), **metrics_at(y, p, t)}


def cost_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    cost = 20 * CAMP_PREV * (1 - tpr) + (1 - CAMP_PREV) * fpr
    return float(np.clip(thr[np.argmin(cost)], 0, 1))


def main():
    set_seed(0)
    paths, labels = [], []
    for folder, lab in [("Normal", 0), ("Viral Pneumonia", 1)]:
        fs = sorted(glob.glob(os.path.join(EXT, folder, "images", "*.png")))
        paths += fs
        labels += [lab] * len(fs)
    y = np.array(labels)
    print(f"external: {len(y)} images, {int(y.sum())} viral pneumonia")
    with Pool(min(16, os.cpu_count())) as pool:
        res = pool.map(load_one, paths, chunksize=64)
    X = np.stack([r[0] for r in res])
    st = np.array([r[1] for r in res], dtype=np.float64)
    ph = [r[2] for r in res]
    out = {"n": int(len(y)), "n_normal": int((y == 0).sum()), "n_pneumonia": int(y.sum())}

    # ---- 2. pixel statistics vs the primary data (Part 1 audit table)
    aud = pd.read_csv(os.path.join(CACHE, "audit_table.csv"))
    def summ(w, h, m, s):
        return {"width_median": float(np.median(w)), "height_median": float(np.median(h)),
                "aspect_median": float(np.median(w / h)), "mean_int": float(np.mean(m)), "std_int": float(np.mean(s))}
    out["pixel_stats"] = {
        "primary_NORMAL": summ(*aud[aud.label == 0][["width", "height", "mean_int", "std_int"]].values.T),
        "primary_PNEU": summ(*aud[aud.label == 1][["width", "height", "mean_int", "std_int"]].values.T),
        "external_NORMAL": summ(*st[y == 0].T),
        "external_PNEU": summ(*st[y == 1].T),
    }
    print(out["pixel_stats"])

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for d, lab, ls in [(aud.mean_int, "primary", "-"), (st[:, 2], "external", "--")]:
        ax[0].hist(d, bins=60, density=True, histtype="step", ls=ls, lw=1.5, label=lab)
    for d, lab, ls in [(aud.std_int, "primary", "-"), (st[:, 3], "external", "--")]:
        ax[1].hist(d, bins=60, density=True, histtype="step", ls=ls, lw=1.5, label=lab)
    ax[0].set_title("mean brightness"); ax[1].set_title("contrast (std)")
    for a in ax:
        a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p2_external_shift.png"), dpi=120)
    plt.close(fig)

    # ---- 3. overlap with our own data
    H = hamming_matrix(ph, aud.phash.tolist())
    ii, jj = np.where(H <= CAND_BITS)
    overlap = np.zeros(len(y), bool)
    if len(ii):
        with Pool(min(16, os.cpu_count())) as pool:
            ue, ua = np.unique(ii), np.unique(jj)
            se = dict(zip(ue, pool.map(_small, [paths[i] for i in ue], chunksize=32)))
            sa = dict(zip(ua, pool.map(_small, aud.path.values[ua].tolist(), chunksize=32)))
        ncc = np.array([se[i] @ sa[j] / se[i].size for i, j in zip(ii, jj)])
        overlap[np.unique(ii[ncc >= NCC_CONFIRM])] = True
    out["overlap_with_primary"] = {"n": int(overlap.sum()), "normal": int((overlap & (y == 0)).sum()),
                                   "pneumonia": int((overlap & (y == 1)).sum())}
    print("overlap:", out["overlap_with_primary"])
    clean = ~overlap

    # ---- 4. final model, frozen threshold
    t_star = load_json("p2_threshold.json")["threshold_camp"]
    internal = load_json("p2_test.json")["final_at_t_star"]
    m, cfg = load_model("final.pt")
    df_dummy = None
    p = predict(m, X, df_dummy, np.arange(len(y)), cfg)
    out["final"] = {"threshold": t_star, "all": score(y, p, t_star), "clean": score(y[clean], p[clean], t_star)}
    out["internal_test_reference"] = {k: internal[k] for k in
                                      ["accuracy", "balanced_accuracy", "recall", "specificity", "roc_auc", "pr_auc"]}
    ref = out["final"]["clean"] if "roc_auc" in out["final"]["clean"] else out["final"]["all"]
    out["drop_vs_internal"] = {k: float(ref[k] - internal[k]) for k in out["internal_test_reference"]}

    # ---- 5. fix: CLAHE-trained model, threshold picked on ITS validation probs (not here)
    v = np.load(os.path.join(RES, "preds", "final_clahe_val.npz"))
    split = pd.read_csv(os.path.join(CACHE, "split.csv"))
    t_cl = cost_threshold(split.label.values[v["idx"]], v["prob"])
    mc, cfgc = load_model("final_clahe.pt")
    pc = predict(mc, apply_clahe(X), None, np.arange(len(y)), cfgc)
    out["clahe_fix"] = {"threshold": t_cl, "all": score(y, pc, t_cl), "clean": score(y[clean], pc[clean], t_cl)}

    for nm in ["final", "clahe_fix"]:
        for part in ["all", "clean"]:
            r = out[nm][part]
            if "roc_auc" not in r:
                print(nm, part, "n =", r["n"], "- not enough images to score")
                continue
            print(f"{nm:10s} {part:5s} acc {r['accuracy']:.3f} bal {r['balanced_accuracy']:.3f} "
                  f"sens {r['recall']:.3f} spec {r['specificity']:.3f} auc {r['roc_auc']:.3f}")
    save_json(out, "p2_external.json")


if __name__ == "__main__":
    main()
