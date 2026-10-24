"""
Part 1, tasks 1-3: audit Rohit's data before training anything.

  - inventory + pixel statistics (sizes, aspect, greyscale, brightness per class)
  - exact duplicates (md5) and near duplicates (pHash), within and across splits
  - patient-id overlap between splits (from the file names)
  - confidence interval for "100% on the 16 image val set"

usage: python audit.py
"""
import os
import re
import hashlib
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import imagehash
import cv2
from scipy import stats

from common import all_primary_images, save_json, FIG, CACHE, CLASSES

CAND_BITS = 10       # pHash hamming distance <= 10 (of 64) -> candidate pair
NCC_CONFIRM = 0.97   # ...confirmed as near duplicate if pixel correlation >= 0.97


def patient_id(path):
    # pneumonia files look like person1234_bacteria_5678.jpeg,
    # normal files like IM-0115-0001.jpeg or NORMAL2-IM-1234-0001.jpeg
    fn = os.path.basename(path)
    m = re.match(r"(person\d+)_", fn)
    if m:
        return m.group(1)
    m = re.match(r"((?:NORMAL2-)?IM-\d+)-", fn)
    if m:
        return m.group(1)
    return fn


def inspect(row):
    path = row["path"]
    with open(path, "rb") as f:
        raw = f.read()
    md5 = hashlib.md5(raw).hexdigest()
    im = Image.open(path)
    mode = im.mode
    w, h = im.size
    if mode == "L":
        is_grey = True
        g = np.asarray(im, dtype=np.uint8)
    else:
        rgb = np.asarray(im.convert("RGB"), dtype=np.int16)
        # "true greyscale" = all three channels identical
        is_grey = bool(np.all(rgb[..., 0] == rgb[..., 1]) and np.all(rgb[..., 1] == rgb[..., 2]))
        g = np.asarray(im.convert("L"), dtype=np.uint8)
    ph = imagehash.phash(Image.fromarray(g))
    return {**row, "md5": md5, "mode": mode, "width": w, "height": h,
            "aspect": w / h, "area": w * h, "grey": is_grey,
            "mean_int": float(g.mean()), "std_int": float(g.std()),
            "phash": str(ph),
            # ids are only unique inside one folder (checked below), so prefix the folder
            "patient": row["orig_split"] + ":" + patient_id(path)}


def _small(path):
    x = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (64, 64), interpolation=cv2.INTER_AREA)
    x = x.astype(np.float32).ravel()
    return (x - x.mean()) / (x.std() + 1e-6)


def near_dup_candidates(df, H):
    ii, jj = np.where(np.triu(H <= CAND_BITS, 1))
    with Pool(min(16, os.cpu_count())) as pool:
        small = np.stack(pool.map(_small, df.path.tolist(), chunksize=64))
    ncc = np.einsum("ij,ij->i", small[ii], small[jj]) / small.shape[1]
    return list(zip(ii, jj)), ncc


def hamming_matrix(hashes_a, hashes_b):
    # hashes as uint64 -> popcount of xor, done in numpy for speed
    a = np.array([int(h, 16) for h in hashes_a], dtype=np.uint64)
    b = np.array([int(h, 16) for h in hashes_b], dtype=np.uint64)
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    out = np.empty((len(a), len(b)), dtype=np.int16)
    for s in range(0, len(a), 512):        # chunked, the full xor cube is too big for RAM
        x = a[s:s + 512, None] ^ b[None, :]
        out[s:s + 512] = lut[x.view(np.uint8).reshape(x.shape[0], len(b), 8)].sum(-1)
    return out


def main():
    rows = all_primary_images()
    print(f"found {len(rows)} images")
    with Pool(min(16, os.cpu_count())) as pool:
        recs = pool.map(inspect, rows, chunksize=32)
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(CACHE, "audit_table.csv"), index=False)
    df["cls"] = df.label.map(lambda i: CLASSES[i])

    out = {}

    # ---------------- 1. inventory -----------------
    inv = df.groupby(["orig_split", "cls"]).size().unstack(fill_value=0)
    print(inv)
    out["inventory"] = {s: inv.loc[s].to_dict() for s in inv.index}
    out["total_images"] = len(df)
    out["modes"] = df["mode"].value_counts().to_dict()
    out["true_greyscale"] = int(df.grey.sum())
    out["not_greyscale"] = int((~df.grey).sum())

    per_cls = {}
    for c in CLASSES:
        d = df[df.cls == c]
        per_cls[c] = {k: {"mean": float(d[k].mean()), "median": float(d[k].median()),
                          "min": float(d[k].min()), "max": float(d[k].max())}
                      for k in ["width", "height", "aspect", "mean_int", "std_int"]}
    out["per_class_stats"] = per_cls

    # do the classes differ in size / brightness? Mann-Whitney + simple AUC
    from sklearn.metrics import roc_auc_score
    sep = {}
    for k in ["width", "height", "aspect", "area", "mean_int", "std_int"]:
        a, b = df[df.label == 0][k], df[df.label == 1][k]
        p = stats.mannwhitneyu(a, b).pvalue
        auc = roc_auc_score(df.label, df[k])
        sep[k] = {"mannwhitney_p": float(p), "auc_alone": float(max(auc, 1 - auc))}
    out["class_separation_by_single_stat"] = sep
    print("single-statistic AUCs:", {k: round(v["auc_alone"], 3) for k, v in sep.items()})

    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    for i, (k, lab) in enumerate([("width", "width (px)"), ("height", "height (px)"),
                                  ("aspect", "aspect ratio w/h"), ("mean_int", "mean brightness"),
                                  ("std_int", "brightness std (contrast)"), ("area", "area (Mpx)")]):
        a = ax.flat[i]
        for c, col in zip(CLASSES, ["tab:blue", "tab:red"]):
            v = df[df.cls == c][k]
            if k == "area":
                v = v / 1e6
            a.hist(v, bins=50, alpha=0.55, density=True, color=col, label=c)
        a.set_title(f"{lab}   (AUC alone = {sep[k]['auc_alone']:.2f})", fontsize=10)
        a.legend(fontsize=8)
    fig.suptitle("NORMAL vs PNEUMONIA - size and brightness distributions (all splits)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p1_size_brightness_hist.png"), dpi=120)
    plt.close(fig)

    # ---------------- 2. duplicates -----------------
    exact = df.groupby("md5").filter(lambda g: len(g) > 1)
    out["exact_duplicate_groups"] = int(exact.md5.nunique())
    out["exact_duplicate_files"] = int(len(exact))

    H = hamming_matrix(df.phash.tolist(), df.phash.tolist())
    np.fill_diagonal(H, 99)
    split = df.orig_split.to_numpy(dtype=object)

    # pHash alone is too coarse for chest films - every paediatric AP film looks alike at
    # 8x8 DCT, and two different children can sit 2-4 bits apart. So: pHash gives candidates,
    # pixel correlation on 64x64 confirms them.
    cand, ncc = near_dup_candidates(df, H)
    out["phash_candidates_le_%d_bits" % CAND_BITS] = len(cand)
    out["phash_only_pairs_le_4_bits"] = int((np.triu(H <= 4, 1)).sum())
    out["phash_only_train_test_pairs_le_4_bits"] = int(
        sum(1 for i, j in zip(*np.where(np.triu(H <= 4, 1))) if {split[i], split[j]} == {"train", "test"}))
    within, across = Counter(), Counter()
    for (i, j), c in zip(cand, ncc):
        if c < NCC_CONFIRM:
            continue
        if split[i] == split[j]:
            within[split[i]] += 1
        else:
            across["-".join(sorted([split[i], split[j]]))] += 1
    out["near_dup_rule"] = f"pHash <= {CAND_BITS} bits AND NCC(64x64) >= {NCC_CONFIRM}"
    out["near_dup_pairs_within_split"] = dict(within)
    out["near_dup_pairs_across_split"] = dict(across)
    print("confirmed near dups within:", dict(within), " across:", dict(across))

    # most similar train/test pairs, ranked by pixel correlation (the honest "worst five")
    tt = [(c, i, j) for (i, j), c in zip(cand, ncc) if {split[i], split[j]} == {"train", "test"}]
    tt.sort(reverse=True)
    out["max_ncc_any_train_test_pair"] = float(tt[0][0]) if tt else None
    fig, ax = plt.subplots(5, 2, figsize=(6, 14))
    worst_info = []
    for r, (c, i, j) in enumerate(tt[:5]):
        if split[i] != "test":
            i, j = j, i
        for col, idx in enumerate([i, j]):
            ax[r, col].imshow(Image.open(df.path[idx]).convert("L"), cmap="gray")
            ax[r, col].set_title(f"{df.orig_split[idx]}/{df.cls[idx]}\n{os.path.basename(df.path[idx])}", fontsize=7)
            ax[r, col].axis("off")
        ax[r, 0].text(-0.1, 0.5, f"ncc={c:.3f}\nd={H[i, j]}", transform=ax[r, 0].transAxes, ha="right", fontsize=8)
        worst_info.append({"test": os.path.basename(df.path[i]), "train": os.path.basename(df.path[j]),
                           "hamming": int(H[i, j]), "ncc": float(c)})
    fig.suptitle("5 most similar test/train pairs (pHash candidates, ranked by NCC)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p1_worst_pairs.png"), dpi=110)
    plt.close(fig)
    out["worst_train_test_pairs"] = worst_info

    # a confirmed within-train duplicate pair, for contrast
    dup = [(c, i, j) for (i, j), c in zip(cand, ncc) if c >= NCC_CONFIRM]
    if dup:
        c, i, j = max(dup)
        fig, ax = plt.subplots(1, 2, figsize=(6, 3.4))
        for col, idx in enumerate([i, j]):
            ax[col].imshow(Image.open(df.path[idx]).convert("L"), cmap="gray")
            ax[col].set_title(f"{df.orig_split[idx]}/{os.path.basename(df.path[idx])}", fontsize=7)
            ax[col].axis("off")
        fig.suptitle(f"a confirmed duplicate (ncc={c:.3f}, d={H[i, j]})", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "p1_true_duplicate.png"), dpi=110)
        plt.close(fig)

    # patient ids. The same "personNNN" string shows up in train and test, but is it the
    # same child? If yes, same-id pairs should look more alike than random pairs.
    raw_id = np.array([patient_id(p) for p in df.path], dtype=object)
    pn = np.where(df.label.values == 1)[0]          # only pneumonia files have person ids
    Hp = H[np.ix_(pn, pn)]
    sid = raw_id[pn][:, None] == raw_id[pn][None, :]
    sp = split[pn]
    w = (sp[:, None] == "train") & (sp[None, :] == "train")
    tt = (sp[:, None] == "train") & (sp[None, :] == "test")
    np.fill_diagonal(sid, False)
    shared = set(raw_id[split == "train"]) & set(raw_id[split == "test"])
    out["patient_id_check"] = {
        "id_strings_in_train_and_test": len(shared),
        "median_phash_same_id_within_train": float(np.median(Hp[sid & w])),
        "median_phash_diff_id_within_train": float(np.median(Hp[~sid & w])),
        "median_phash_same_id_train_vs_test": float(np.median(Hp[sid & tt])),
        "median_phash_diff_id_train_vs_test": float(np.median(Hp[~sid & tt])),
    }
    out["patients_per_split"] = df.groupby("orig_split").patient.nunique().to_dict()
    out["patients_with_multiple_images"] = int((df.patient.value_counts() > 1).sum())
    out["images_per_patient_max"] = int(df.patient.value_counts().max())
    print("patient id check:", out["patient_id_check"])

    # ---------------- 3. val set CI -----------------
    n = 16
    lo_cp = (0.05 / 2) ** (1 / n)                    # Clopper-Pearson lower bound when x = n
    z = 1.96
    p = 1.0
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    out["val_ci_16_of_16"] = {"clopper_pearson": [float(lo_cp), 1.0],
                              "wilson": [float(centre - half), float(min(1, centre + half))],
                              "rule_of_three_lower": 1 - 3 / n}
    print("95% CI for 16/16:", out["val_ci_16_of_16"])

    save_json(out, "audit.json")


if __name__ == "__main__":
    main()
