"""
Part 1, task 3: throw away Rohit's train/val/test folders and make a split we can defend.

Plan
  * pool all 5,856 images
  * group them: same patient id  OR  pHash near-duplicate -> same group
    (so one child / one re-saved image can never sit on both sides)
  * 15% of groups -> locked TEST set (stratified). Nothing reads it until the final eval.
  * remaining 85% = training pool, cut into 5 stratified group folds.
    fold 0 is used as the validation set for single runs, all 5 for Part 2 CV.
  * cache every image as greyscale uint8 at 224x224 and 96x96 + the 3 metadata numbers

usage: python split.py      (needs audit.py to have run first)
"""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import StratifiedGroupKFold

from common import CACHE, save_json, set_seed
from audit import hamming_matrix, near_dup_candidates, NCC_CONFIRM

SEED = 42


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a


def load_resized(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = g.shape
    meta = (g.mean() / 255.0, w / h, (w * h) / 1e6)   # mean intensity, aspect, area (Mpx)
    return (cv2.resize(g, (224, 224), interpolation=cv2.INTER_AREA),
            cv2.resize(g, (96, 96), interpolation=cv2.INTER_AREA),
            meta)


def main():
    set_seed(SEED)
    df = pd.read_csv(os.path.join(CACHE, "audit_table.csv"))
    n = len(df)

    uf = UnionFind(n)
    first_of_patient = {}
    for i, p in enumerate(df.patient):
        if p in first_of_patient:
            uf.union(first_of_patient[p], i)
        else:
            first_of_patient[p] = i

    # confirmed near duplicates (same rule as audit.py) also go in one group
    H = hamming_matrix(df.phash.tolist(), df.phash.tolist())
    cand, ncc = near_dup_candidates(df, H)
    ii = [i for (i, j), c in zip(cand, ncc) if c >= NCC_CONFIRM]
    jj = [j for (i, j), c in zip(cand, ncc) if c >= NCC_CONFIRM]
    for i, j in zip(ii, jj):
        uf.union(i, j)
    df["group"] = [uf.find(i) for i in range(n)]
    print(f"{n} images -> {df.group.nunique()} groups "
          f"({len(ii)} near-dup pairs merged into groups)")

    # a group can (rarely) mix labels, stratify on the image label anyway
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)   # 1/7 ~ 14-15% test
    pool_idx, test_idx = next(sgkf.split(df, df.label, df.group))
    df["split"] = "pool"
    df.loc[test_idx, "split"] = "test"

    df["fold"] = -1
    pool = df[df.split == "pool"]
    sgkf5 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for k, (_, va) in enumerate(sgkf5.split(pool, pool.label, pool.group)):
        df.loc[pool.index[va], "fold"] = k

    # sanity: no group crosses test/pool
    g_test = set(df[df.split == "test"].group)
    g_pool = set(df[df.split == "pool"].group)
    assert not (g_test & g_pool), "group leakage!"
    assert not (set(df[df.split == "test"].patient) & set(df[df.split == "pool"].patient))

    print("caching resized images ...")
    with Pool(min(16, os.cpu_count())) as p:
        res = p.map(load_resized, df.path.tolist(), chunksize=32)
    np.save(os.path.join(CACHE, "img224.npy"), np.stack([r[0] for r in res]))
    np.save(os.path.join(CACHE, "img96.npy"), np.stack([r[1] for r in res]))
    meta = np.array([r[2] for r in res], dtype=np.float32)
    df["meta_mean"], df["meta_aspect"], df["meta_area"] = meta[:, 0], meta[:, 1], meta[:, 2]
    df.to_csv(os.path.join(CACHE, "split.csv"), index=False)

    summary = {"seed": SEED, "groups": int(df.group.nunique())}
    for s in ["pool", "test"]:
        d = df[df.split == s]
        summary[s] = {"n": len(d), "normal": int((d.label == 0).sum()), "pneumonia": int((d.label == 1).sum())}
    for k in range(5):
        d = df[df.fold == k]
        summary[f"fold{k}"] = {"n": len(d), "pneumonia_frac": float(d.label.mean())}
    print(summary)
    save_json(summary, "split.json")


if __name__ == "__main__":
    main()
