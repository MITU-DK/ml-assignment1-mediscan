"""
Part 2, task 2 (CV half): 5-fold stratified, patient-grouped cross-validation on the
training pool with the final config. Each fold is scored at the frozen camp threshold
t* from part2_threshold.py and at 0.5. Test set not used.
"""
from common import save_json, load_json, metrics_at, mean_std
from engine import train, split_idx, load_data
from final_model import final_cfg


def main():
    cfg = final_cfg()
    t_star = load_json("p2_threshold.json")["threshold_camp"]
    _, df = load_data(cfg.get("res", 128))
    y = df.label.values
    at_t, at_half = [], []
    for k in range(5):
        tr, va = split_idx(k)
        r = train(cfg, tr, va, seed=k, verbose=False)
        va = r["val_idx"]
        at_t.append(metrics_at(y[va], r["val_prob"], t_star))
        at_half.append(metrics_at(y[va], r["val_prob"], 0.5))
        print(f"fold {k}: AUC {at_t[-1]['roc_auc']:.4f}  recall@t* {at_t[-1]['recall']:.4f}  "
              f"spec@t* {at_t[-1]['specificity']:.4f}", flush=True)
    out = {"threshold": t_star, "folds_at_t_star": at_t, "folds_at_0.5": at_half,
           "summary_at_t_star": mean_std(at_t), "summary_at_0.5": mean_std(at_half)}
    save_json(out, "p2_cv.json")


if __name__ == "__main__":
    main()
