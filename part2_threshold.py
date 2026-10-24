"""
Part 2, task 1: class weights vs threshold moving, and a cost-based threshold.

Everything here uses the VALIDATION set (fold 0). The chosen threshold is then
frozen and used unchanged on the test set and the external hospital.

Cost matrix:   missed pneumonia (FN) = 20,  false alarm (FP) = 1,  correct = 0
Camp prevalence: 10% pneumonia (9:1 NORMAL), not the 73% in the dataset.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve, roc_auc_score

from common import save_json, metrics_at, FIG, RES, CACHE
import pandas as pd

C_FN, C_FP = 20.0, 1.0
CAMP_PREV = 0.10


def load_val(name):
    d = np.load(os.path.join(RES, "preds", f"{name}_val.npz"))
    df = pd.read_csv(os.path.join(CACHE, "split.csv"))
    return d["prob"], df.label.values[d["idx"]]


def expected_cost(tpr, fpr, prev):
    return C_FN * prev * (1 - tpr) + C_FP * (1 - prev) * fpr


def main():
    p, y = load_val("final")
    pw, _ = load_val("final_classweight")
    out = {"cost_FN": C_FN, "cost_FP": C_FP, "camp_prevalence": CAMP_PREV,
           "val_prevalence": float(y.mean())}

    # ---- class weights change the MODEL, threshold moving changes only the DECISION
    out["auc_plain"] = roc_auc_score(y, p)
    out["auc_classweight"] = roc_auc_score(y, pw)
    out["spearman_plain_vs_weighted"] = float(spearmanr(p, pw).correlation)
    out["mean_prob_plain"] = float(p.mean())
    out["mean_prob_weighted"] = float(pw.mean())
    out["plain@0.5"] = metrics_at(y, p, 0.5)
    out["weighted@0.5"] = metrics_at(y, pw, 0.5)
    # move the plain model's threshold until its specificity matches the weighted model
    target_spec = out["weighted@0.5"]["specificity"]
    ts = np.unique(p)
    specs = np.array([((p[y == 0] < t).mean()) for t in ts])
    t_match = float(ts[np.argmin(np.abs(specs - target_spec))])
    out["plain@matched_threshold"] = metrics_at(y, p, t_match)

    # ---- cost-optimal threshold at camp prevalence
    fpr, tpr, thr = roc_curve(y, p)
    thr = np.clip(thr, 0, 1)
    cost_camp = expected_cost(tpr, fpr, CAMP_PREV)
    cost_data = expected_cost(tpr, fpr, y.mean())
    k = int(np.argmin(cost_camp))
    t_star = float(thr[k])
    k_d = int(np.argmin(cost_data))
    out["threshold_camp"] = t_star
    out["threshold_if_dataset_prevalence"] = float(thr[k_d])
    out["val_at_t_star"] = metrics_at(y, p, t_star)
    out["expected_cost_per_child"] = {
        "t_star": float(cost_camp[k]),
        "t=0.5": float(expected_cost(*_rates(y, p, 0.5), CAMP_PREV)),
        "always_normal": C_FN * CAMP_PREV,
        "always_refer": C_FP * (1 - CAMP_PREV),
    }

    # ---- "refer to clinician" band: between the 99.5% sensitivity point and the
    # 98% specificity point. Below t_lo almost no pneumonia is left -> confident NORMAL,
    # above t_hi almost no normals -> confident PNEUMONIA. t* sits inside the band.
    t_lo = float(thr[np.argmax(tpr >= 0.995)])
    t_hi = float(thr[np.where(fpr <= 0.02)[0][-1]])
    out["refer_band"] = [min(t_lo, t_hi), max(t_lo, t_hi)]
    band = (p >= out["refer_band"][0]) & (p <= out["refer_band"][1])
    out["refer_band_val"] = {"share_referred": float(band.mean()),
                             "metrics_outside_band_at_t_star": metrics_at(y[~band], p[~band], t_star)}

    # ---- why 99% accuracy is the wrong target: arithmetic at 10% prevalence
    sens, spec = out["val_at_t_star"]["recall"], out["val_at_t_star"]["specificity"]
    per1000 = {"sick": 1000 * CAMP_PREV, "healthy": 1000 * (1 - CAMP_PREV)}
    out["camp_per_1000_children_at_t_star"] = {
        "caught": per1000["sick"] * sens, "missed": per1000["sick"] * (1 - sens),
        "false_alarms": per1000["healthy"] * (1 - spec),
        "accuracy": CAMP_PREV * sens + (1 - CAMP_PREV) * spec,
        "ppv": CAMP_PREV * sens / (CAMP_PREV * sens + (1 - CAMP_PREV) * (1 - spec)),
        "npv": (1 - CAMP_PREV) * spec / ((1 - CAMP_PREV) * spec + CAMP_PREV * (1 - sens)),
    }
    out["camp_always_normal"] = {"accuracy": 1 - CAMP_PREV, "missed_per_1000": per1000["sick"]}
    # a "99% accurate" model can still put all of its 1% error into missed cases
    out["camp_99pct_acc_worst_case"] = {"missed_per_1000": 10, "share_of_sick_missed": 10 / per1000["sick"]}
    print({k: v for k, v in out.items() if not isinstance(v, dict)})
    print("t* =", t_star, out["val_at_t_star"])
    save_json(out, "p2_threshold.json")

    # ---- figures
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    bins = np.linspace(0, 1, 41)
    for arr, lab in [(p, "plain BCE"), (pw, "class-weighted")]:
        ax[0].hist(arr[y == 0], bins, alpha=0.45, label=f"{lab} - NORMAL", density=True)
        ax[0].hist(arr[y == 1], bins, alpha=0.45, label=f"{lab} - PNEU", density=True, histtype="step", lw=1.5)
    ax[0].set_title("val probabilities: class weights move the scores")
    ax[0].legend(fontsize=7)

    fw, tw, _ = roc_curve(y, pw)
    ax[1].plot(fpr, tpr, label=f"plain AUC={out['auc_plain']:.3f}")
    ax[1].plot(fw, tw, "--", label=f"class-weighted AUC={out['auc_classweight']:.3f}")
    ax[1].scatter([fpr[k]], [tpr[k]], c="red", zorder=5, label=f"chosen t*={t_star:.3f}")
    m5 = metrics_at(y, p, 0.5)
    ax[1].scatter([1 - m5["specificity"]], [m5["recall"]], c="k", marker="x", zorder=5, label="t=0.5")
    ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR"); ax[1].set_title("ROC (val)"); ax[1].legend(fontsize=7)
    ax[1].set_xlim(0, 0.5); ax[1].set_ylim(0.5, 1.01)

    with np.errstate(invalid="ignore", divide="ignore"):
        prec_camp = CAMP_PREV * tpr / (CAMP_PREV * tpr + (1 - CAMP_PREV) * fpr)
        prec_data = y.mean() * tpr / (y.mean() * tpr + (1 - y.mean()) * fpr)
    ax[2].plot(tpr, prec_data, label=f"at dataset prevalence ({y.mean():.0%})")
    ax[2].plot(tpr, prec_camp, label="at camp prevalence (10%)")
    ax[2].scatter([tpr[k]], [prec_camp[k]], c="red", zorder=5, label="chosen t*")
    ax[2].set_xlabel("recall"); ax[2].set_ylabel("precision"); ax[2].set_title("PR curve (val)")
    ax[2].legend(fontsize=7); ax[2].set_ylim(0, 1.02)

    ax[3].plot(thr, cost_camp, label="camp (10%)")
    ax[3].plot(thr, cost_data, label=f"dataset ({y.mean():.0%})")
    ax[3].axvline(t_star, color="red", lw=0.8)
    ax[3].set_xlabel("threshold"); ax[3].set_ylabel("expected cost per child")
    ax[3].set_title("cost (FN=20, FP=1)"); ax[3].legend(fontsize=7); ax[3].set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p2_threshold.png"), dpi=120)
    plt.close(fig)


def _rates(y, p, t):
    pred = p >= t
    return pred[y == 1].mean(), pred[y == 0].mean()


if __name__ == "__main__":
    main()
