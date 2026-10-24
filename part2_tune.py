"""
Part 2, task 3: hyperparameter search + regularisation curves.

Random search, budget = 12 configs x 8 epochs each, ArogyaNet at 128px, scored by
validation ROC-AUC on fold 0. Then two long runs (30 epochs, no early stopping)
to show where overfitting starts:
    unregularised : dropout 0, weight decay 0, no augmentation
    regularised   : best searched config + augmentation (rot/zoom, no flip)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import save_json, FIG
from engine import train, split_idx

BUDGET = 12
SEARCH_EPOCHS = 8


def sample(rng):
    return {"lr": float(10 ** rng.uniform(-3.7, -2.3)),
            "bs": int(rng.choice([16, 32, 64])),
            "dropout": float(rng.uniform(0.0, 0.5)),
            "wd": float(10 ** rng.uniform(-6, -2))}


def main():
    tr, va = split_idx(0)
    rng = np.random.RandomState(123)
    trials = []
    for t in range(BUDGET):
        hp = sample(rng)
        r = train({**hp, "epochs": SEARCH_EPOCHS, "patience": 0}, tr, va, seed=0, verbose=False)
        trials.append({**hp, "val_auc": r["val_metrics"]["roc_auc"], "val_acc": r["val_metrics"]["accuracy"],
                       "best_epoch": r["hist"]["best_epoch"]})
        print(t, {k: round(v, 5) if isinstance(v, float) else v for k, v in trials[-1].items()}, flush=True)
    trials.sort(key=lambda d: -d["val_auc"])
    best = {k: trials[0][k] for k in ["lr", "bs", "dropout", "wd"]}
    print("best", best)

    curves = {}
    for name, cfg in [("unregularised", {"dropout": 0.0, "wd": 0.0, "aug": "none"}),
                      ("regularised", {**best, "aug": "safe"})]:
        r = train({**cfg, "epochs": 30, "patience": 0}, tr, va, seed=0)
        h = r["hist"]
        vl = np.array(h["val_loss"])
        curves[name] = {"cfg": cfg, "hist": h, "min_val_loss_epoch": int(vl.argmin() + 1),
                        "best_auc_epoch": h["best_epoch"],
                        "val_auc_at_best": float(max(h["val_auc"])), "val_auc_last": float(h["val_auc"][-1]),
                        "val_loss_min": float(vl.min()), "val_loss_last": float(vl[-1]),
                        "train_loss_last": float(h["train_loss"][-1])}

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, ls in [("unregularised", "-"), ("regularised", "--")]:
        h = curves[name]["hist"]
        ep = np.arange(1, len(h["train_loss"]) + 1)
        ax[0].plot(ep, h["train_loss"], ls, color="tab:blue", label=f"{name} train")
        ax[0].plot(ep, h["val_loss"], ls, color="tab:red", label=f"{name} val")
        ax[1].plot(ep, h["val_auc"], ls, label=name)
        ax[0].axvline(curves[name]["min_val_loss_epoch"], ls=ls, color="grey", lw=0.8)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("BCE loss"); ax[0].legend(fontsize=8)
    ax[0].set_title("loss (grey line = min val loss)")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val ROC-AUC"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p2_reg_curves.png"), dpi=120)
    plt.close(fig)

    save_json({"budget": f"{BUDGET} random configs x {SEARCH_EPOCHS} epochs, seed 0, fold-0 val AUC",
               "trials": trials, "best": best, "curves": curves}, "p2_tune.json")


if __name__ == "__main__":
    main()
