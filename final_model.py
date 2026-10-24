"""
Trains the models Part 2 needs, all ArogyaNet with the hyperparameters picked by part2_tune.py,
trained on folds 1-4, fold 0 = validation (threshold + early stopping). Seed 0.

  models/final.pt              plain BCE           -> the model we ship
  models/final_classweight.pt  class-weighted BCE  -> Part 2.1 comparison
  models/final_clahe.pt        CLAHE preprocessing -> the one fix tried in Part 2.6

Also fits the out-of-distribution guard used by predict.py: Mahalanobis distance of
the penultimate embedding to the training embeddings (Ledoit-Wolf covariance).
"""
import os
import numpy as np
from sklearn.covariance import LedoitWolf

from common import load_json, save_json, MODELS, RES
from engine import train, split_idx, save_model, load_data, predict

SEED = 0


def final_cfg():
    best = load_json("p2_tune.json")["best"]
    return {**best, "bs": int(best["bs"]), "aug": "safe", "epochs": 15, "patience": 5}


def main():
    cfg = final_cfg()
    tr, va = split_idx(0)
    _, df = load_data(cfg.get("res", 128))
    y = df.label.values
    n_pos, n_neg = (y[tr] == 1).sum(), (y[tr] == 0).sum()

    variants = {
        "final": cfg,
        # pos_weight < 1 down-weights the majority class (pneumonia) so both classes count equally
        "final_classweight": {**cfg, "pos_weight": float(n_neg / n_pos)},
        "final_clahe": {**cfg, "clahe": True},
    }
    summary = {"cfg": cfg, "n_train": int(len(tr)), "n_val": int(len(va))}
    for name, c in variants.items():
        print("==", name)
        r = train(c, tr, va, SEED)
        save_model(r, name + ".pt")
        np.savez(os.path.join(RES, "preds", f"{name}_val.npz"), idx=r["val_idx"], prob=r["val_prob"])
        summary[name] = {"val": r["val_metrics"], "hist": r["hist"], "params": r["params"],
                         "pos_weight": c.get("pos_weight", 1.0)}
        if name == "final":
            model, va_used = r["model"], r["val_idx"]

    # OOD guard on the shipped model
    X, df = load_data(cfg.get("res", 128))
    tr_ood = tr[::25] if os.environ.get("SMOKE") else tr
    _, e_tr = predict(model, X, df, tr_ood, cfg, return_emb=True)
    _, e_va = predict(model, X, df, va_used, cfg, return_emb=True)
    lw = LedoitWolf().fit(e_tr)
    prec = lw.precision_

    def maha(e):
        d = e - lw.location_
        return np.sqrt(np.einsum("ij,jk,ik->i", d, prec, d))

    d_va = maha(e_va)
    cut = float(np.percentile(d_va, 99.5))
    np.savez(os.path.join(MODELS, "ood.npz"), mean=lw.location_, prec=prec, cut=cut)
    summary["ood"] = {"maha_val_median": float(np.median(d_va)), "maha_cut_99.5pct": cut}
    print("OOD cut", cut)
    save_json(summary, "p2_final_models.json")


if __name__ == "__main__":
    main()
