# shared helpers for the MediScan audit - paths, seeding, metrics, data loading
import os
import json
import random
import time

import numpy as np
import torch
from sklearn import metrics as skm

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "chest_xray")          # train/ val/ test/ as Rohit left them
CACHE = os.path.join(DATA, "cache")
# MEDISCAN_OUT lets a quick smoke test write somewhere else instead of the real results
OUT = os.environ.get("MEDISCAN_OUT", ROOT)
RES = os.path.join(OUT, "results")
FIG = os.path.join(OUT, "figures")
MODELS = os.path.join(OUT, "models")
for d in (CACHE, RES, FIG, MODELS, os.path.join(RES, "preds")):
    os.makedirs(d, exist_ok=True)

CLASSES = ["NORMAL", "PNEUMONIA"]   # label 1 = pneumonia (the positive class)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[seed] python/numpy/torch seed = {seed}")


def save_json(obj, name):
    path = os.path.join(RES, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print("saved", path)


def load_json(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def list_images(split_dir):
    """returns list of (path, label) for one split folder"""
    out = []
    for lab, cls in enumerate(CLASSES):
        folder = os.path.join(split_dir, cls)
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith((".jpeg", ".jpg", ".png")):
                out.append((os.path.join(folder, fn), lab))
    return out


def all_primary_images():
    """every image in Rohit's folders, with the folder it came from"""
    rows = []
    for split in ["train", "val", "test"]:
        for p, y in list_images(os.path.join(RAW, split)):
            rows.append({"path": p, "label": y, "orig_split": split})
    return rows


def metrics_at(y_true, prob, thr=0.5):
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = skm.confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": skm.accuracy_score(y_true, pred),
        "balanced_accuracy": skm.balanced_accuracy_score(y_true, pred),
        "precision": skm.precision_score(y_true, pred, zero_division=0),
        "recall": skm.recall_score(y_true, pred, zero_division=0),
        "specificity": tn / max(tn + fp, 1),
        "f1": skm.f1_score(y_true, pred, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(thr),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = skm.roc_auc_score(y_true, prob)
        out["pr_auc"] = skm.average_precision_score(y_true, prob)
    return out


def mean_std(list_of_dicts, keys=("accuracy", "balanced_accuracy", "precision",
                                  "recall", "specificity", "f1", "roc_auc", "pr_auc")):
    res = {}
    for k in keys:
        vals = [d[k] for d in list_of_dicts if k in d]
        if vals:
            res[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
    return res


class Timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.sec = time.time() - self.t
