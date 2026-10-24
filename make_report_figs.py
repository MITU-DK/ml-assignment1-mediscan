"""
Re-draws the 5 closest train/test pairs from results/audit.json as a wide 2x5 strip
for the report (audit.py draws the same pairs as a tall 5x2 figure).
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from common import RES, FIG, RAW


def find(name):
    for split in ["train", "test", "val"]:
        for cls in ["NORMAL", "PNEUMONIA"]:
            p = os.path.join(RAW, split, cls, name)
            if os.path.exists(p):
                return p, f"{split}/{cls}"
    raise FileNotFoundError(name)


pairs = json.load(open(os.path.join(RES, "audit.json")))["worst_train_test_pairs"]
fig, ax = plt.subplots(2, 5, figsize=(13, 5.6))
for c, pr in enumerate(pairs):
    for r, key in enumerate(["test", "train"]):
        p, where = find(pr[key])
        ax[r, c].imshow(Image.open(p).convert("L"), cmap="gray")
        ax[r, c].set_title(f"{where}\n{pr[key]}", fontsize=7)
        ax[r, c].axis("off")
    ax[1, c].text(0.5, -0.08, f"NCC {pr['ncc']:.3f}, pHash d={pr['hamming']}",
                  transform=ax[1, c].transAxes, ha="center", va="top", fontsize=8)
fig.suptitle("5 most similar test (top) / train (bottom) pairs - all different children", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "p1_worst_pairs_wide.png"), dpi=120)
print("saved figures/p1_worst_pairs_wide.png")
