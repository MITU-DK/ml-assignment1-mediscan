"""
Collects results/*.json into reports/tables.md, so every number in the reports
can be traced to a json file and the script that wrote it.
"""
import os
import json
import numpy as np

from common import RES, ROOT

REP = os.path.join(ROOT, "reports")
os.makedirs(REP, exist_ok=True)
L = []


def j(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def ms(s, k):
    if k not in s:
        return "-"
    return f"{s[k]['mean']:.3f} ± {s[k]['std']:.3f}"


def h(t, src):
    L.append(f"\n## {t}\n\n_source: {src}_\n")


METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def row(name, m):
    return f"| {name} | " + " | ".join(f3(m.get(k)) for k in METRICS) + " |"


HEAD = "| model | acc | bal. acc | precision | recall | F1 | ROC-AUC | PR-AUC |\n|---|---|---|---|---|---|---|---|"


def main():
    a = j("audit.json")
    if a:
        h("Inventory (Rohit's folders)", "audit.py -> results/audit.json")
        L.append("| split | NORMAL | PNEUMONIA |\n|---|---|---|")
        for s, d in a["inventory"].items():
            L.append(f"| {s} | {d['NORMAL']} | {d['PNEUMONIA']} |")
        L.append(f"\nFile modes: {a['modes']}; true greyscale (RGB files with 3 equal channels count): "
                 f"{a['true_greyscale']} / {a['total_images']}.\n")
        L.append("| statistic | NORMAL median | PNEUMONIA median | AUC of this number alone |\n|---|---|---|---|")
        for k in ["width", "height", "aspect", "mean_int", "std_int"]:
            L.append(f"| {k} | {a['per_class_stats']['NORMAL'][k]['median']:.2f} | "
                     f"{a['per_class_stats']['PNEUMONIA'][k]['median']:.2f} | "
                     f"{a['class_separation_by_single_stat'][k]['auc_alone']:.3f} |")
        h("Duplicates", "audit.py")
        L.append(f"- exact duplicates (md5): {a['exact_duplicate_groups']} groups, {a['exact_duplicate_files']} files")
        L.append(f"- pHash only, <= 4 bits: {a['phash_only_pairs_le_4_bits']} pairs, of which train-test: "
                 f"{a['phash_only_train_test_pairs_le_4_bits']}")
        L.append(f"- confirmed near duplicates ({a['near_dup_rule']}): within split {a['near_dup_pairs_within_split']}, "
                 f"across splits {a['near_dup_pairs_across_split'] or 0}")
        L.append(f"- highest pixel correlation of any train-test pair: {a['max_ncc_any_train_test_pair']:.3f}")
        L.append(f"- patient-id check: {a['patient_id_check']}")
        c = a["val_ci_16_of_16"]
        L.append(f"\n95% CI for 16/16 correct: Clopper-Pearson [{c['clopper_pearson'][0]:.3f}, 1], "
                 f"Wilson [{c['wilson'][0]:.3f}, 1]")

    s = j("split.json")
    if s:
        h("New split", "split.py -> results/split.json")
        L.append(f"{s['groups']} groups (patient id + confirmed duplicates). Pool {s['pool']}, locked test {s['test']}.")

    sc = j("p1_shortcut.json")
    if sc:
        h("Shortcut hunt (fold-0 validation)", "train.py --exp shortcut")
        L.append(HEAD)
        for k, v in sc.items():
            if isinstance(v, dict):
                L.append(row(k, v))
        L.append(f"\nchance (majority) accuracy = {sc['chance_majority_acc']:.3f}")

    b = j("p1_baselines.json")
    rows = []
    if b:
        rows += [row("majority class", b["majority"]), row("HOG + logistic regression", b["hog_logreg"])]
    for name, label in [("resnet18", "ResNet18 ImageNet, 224px"), ("arogya_full", "ArogyaNet full"),
                        ("arogya_no_residual", "ArogyaNet - no residual"), ("arogya_no_se", "ArogyaNet - no SE"),
                        ("arogya_two_branch", "ArogyaNet two-branch (+meta)"),
                        ("arogya_res96", "ArogyaNet 96px"), ("arogya_res224", "ArogyaNet 224px"),
                        ("arogya_aug_none", "ArogyaNet no augmentation"),
                        ("arogya_aug_flip_rot_zoom", "ArogyaNet flip+rot+zoom")]:
        r = j(f"p1_{name}.json")
        if r:
            sm = r["summary"]
            rows.append(f"| {label} (3 seeds) | " + " | ".join(ms(sm, k) for k in METRICS) + " |")
    if rows:
        h("Part 1 models on fold-0 validation, threshold 0.5 (mean ± sd over seeds)", "train.py, results/p1_*.json")
        L.append(HEAD)
        L.extend(rows)

    full = j("p1_arogya_full.json")
    if full:
        h("Ablations: paired difference vs full ArogyaNet (same seed), ROC-AUC and balanced accuracy",
          "train.py --exp arogya / twobranch / res_aug")
        L.append("| variant | Δ ROC-AUC (mean ± sd) | Δ bal. acc (mean ± sd) | inside noise? |\n|---|---|---|---|")
        base = {r["seed"]: r for r in full["runs"]}
        for name in ["arogya_no_residual", "arogya_no_se", "arogya_two_branch", "arogya_res96",
                     "arogya_res224", "arogya_aug_none", "arogya_aug_flip_rot_zoom"]:
            r = j(f"p1_{name}.json")
            if not r:
                continue
            out = []
            for k in ["roc_auc", "balanced_accuracy"]:
                d = np.array([x[k] - base[x["seed"]][k] for x in r["runs"]])
                out.append((d.mean(), d.std(ddof=1)))
            noise = "yes" if abs(out[0][0]) <= 2 * out[0][1] else "no"
            L.append(f"| {name} | {out[0][0]:+.4f} ± {out[0][1]:.4f} | {out[1][0]:+.4f} ± {out[1][1]:.4f} | {noise} |")
        p = j("arogyanet_params.json")
        if p:
            L.append(f"\nParameter counts: {p}")

    lk = j("p1_leak.json")
    if lk:
        h("Rohit's likely protocol: image-level vs patient-level random split (inside the pool)", "train.py --exp leak")
        L.append("| split type | HOG/LR acc (5 repeats) | HOG/LR ROC-AUC | groups shared train/test | ArogyaNet acc (1 seed) | ArogyaNet AUC |\n|---|---|---|---|---|---|")
        for k, v in lk.items():
            a_ = v.get("arogyanet_seed0", {})
            L.append(f"| {k} | {ms(v['summary'], 'accuracy')} | {ms(v['summary'], 'roc_auc')} | "
                     f"{v['groups_shared']} | {f3(a_.get('accuracy'))} | {f3(a_.get('roc_auc'))} |")

    t = j("p2_tune.json")
    if t:
        h("Hyperparameter search", "part2_tune.py")
        L.append(f"Budget: {t['budget']}\n")
        L.append("| lr | batch | dropout | weight decay | val AUC | best epoch |\n|---|---|---|---|---|---|")
        for r in t["trials"]:
            L.append(f"| {r['lr']:.2e} | {r['bs']} | {r['dropout']:.2f} | {r['wd']:.1e} | {r['val_auc']:.4f} | {r['best_epoch']} |")
        for k, c in t["curves"].items():
            L.append(f"\n- {k}: min val loss at epoch {c['min_val_loss_epoch']} ({c['val_loss_min']:.3f}), "
                     f"val loss at epoch 30 {c['val_loss_last']:.3f}, train loss at 30 {c['train_loss_last']:.3f}, "
                     f"best val AUC {c['val_auc_at_best']:.4f} at epoch {c['best_auc_epoch']}, last {c['val_auc_last']:.4f}")

    th = j("p2_threshold.json")
    if th:
        h("Class weights vs threshold moving, cost-based threshold (validation)", "part2_threshold.py")
        L.append(f"- ROC-AUC plain {th['auc_plain']:.4f}, class-weighted {th['auc_classweight']:.4f}; "
                 f"Spearman rank corr between them {th['spearman_plain_vs_weighted']:.4f}")
        L.append(f"- mean predicted P(pneu): plain {th['mean_prob_plain']:.3f}, weighted {th['mean_prob_weighted']:.3f}")
        L.append("\n| setting | threshold | recall | specificity | precision | bal. acc |\n|---|---|---|---|---|---|")
        for k in ["plain@0.5", "weighted@0.5", "plain@matched_threshold", "val_at_t_star"]:
            m = th[k]
            L.append(f"| {k} | {m['threshold']:.3f} | {m['recall']:.3f} | {m['specificity']:.3f} | "
                     f"{m['precision']:.3f} | {m['balanced_accuracy']:.3f} |")
        L.append(f"\n- t* (camp, 10% prevalence, FN=20, FP=1) = {th['threshold_camp']:.4f}; "
                 f"same rule at dataset prevalence would give {th['threshold_if_dataset_prevalence']:.4f}")
        L.append(f"- expected cost per child: {th['expected_cost_per_child']}")
        L.append(f"- refer band: {th['refer_band']}")
        L.append(f"- per 1000 camp children at t*: {th['camp_per_1000_children_at_t_star']}")

    cv = j("p2_cv.json")
    if cv:
        h("5-fold patient-grouped CV on the pool, at t*", "part2_cv.py")
        L.append("| metric | mean ± sd |\n|---|---|")
        for k in ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"]:
            L.append(f"| {k} | {ms(cv['summary_at_t_star'], k)} |")

    te = j("p2_test.json")
    if te:
        h("Locked test set, final model at t*", "evaluate_test.py")
        m = te["final_at_t_star"]
        L.append(f"n = {te['n_test']}, prevalence {te['test_prevalence']:.3f}, t* = {te['threshold']:.4f}\n")
        L.append(f"TN {m['tn']}  FP {m['fp']}  FN {m['fn']}  TP {m['tp']}\n")
        L.append("| metric | value |\n|---|---|")
        for k in ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc",
                  "camp_ppv", "camp_npv", "camp_accuracy", "camp_expected_cost"]:
            L.append(f"| {k} | {m[k]:.4f} |")
        # band is read from p2_threshold.json (chosen on val) and applied to the saved test predictions
        band = j("p2_threshold.json")["refer_band"]
        d = np.load(os.path.join(RES, "preds", "test_final.npz"))
        import pandas as pd
        from common import CACHE
        yt = pd.read_csv(os.path.join(CACHE, "split.csv")).label.values[d["idx"]]
        inb = (d["prob"] >= band[0]) & (d["prob"] <= band[1])
        from common import metrics_at
        mo = metrics_at(yt[~inb], d["prob"][~inb], te["threshold"])
        L.append(f"\nRefer band [{band[0]:.3f}, {band[1]:.3f}] (chosen on val): {inb.mean():.1%} of test images fall in it "
                 f"and go to a clinician. On the rest: recall {mo['recall']:.4f}, specificity {mo['specificity']:.4f}, "
                 f"FN {mo['fn']}, FP {mo['fp']} (n={(~inb).sum()}).")
        h("Ensemble / classical-on-embeddings table (locked test, threshold 0.5)", "evaluate_test.py")
        L.append("| model | ROC-AUC | PR-AUC | acc | bal. acc | F1 |\n|---|---|---|---|---|---|")
        for k, v in te["ensemble_table_at_0.5"].items():
            L.append(f"| {k} | " + " | ".join(f3(v.get(x)) for x in ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "f1"]) + " |")
        h("Feature space", "evaluate_test.py")
        L.append(f"{te['feature_space']}\n")
        L.append("| group | n | width | height | area | mean int | std int | mean prob |\n|---|---|---|---|---|---|---|---|")
        for k, v in te["error_profile"].items():
            L.append(f"| {k} | {v['n']} | {v['width']} | {v['height']} | {v['area']:.0f} | {v['mean_int']} | {v['std_int']} | {v['mean_prob']:.3f} |")

    ex = j("p2_external.json")
    if ex:
        h("External hospital (run once)", "external_eval.py")
        L.append(f"n = {ex['n']} ({ex['n_normal']} normal, {ex['n_pneumonia']} viral pneumonia); "
                 f"overlap with our own data: {ex['overlap_with_primary']}\n")
        L.append("| model | subset | threshold | acc | bal. acc | recall | specificity | ROC-AUC | PR-AUC |\n|---|---|---|---|---|---|---|---|---|")
        for nm in ["final", "clahe_fix"]:
            for part in ["all", "clean"]:
                m = ex[nm][part]
                if "roc_auc" not in m:     # clean subset is all NORMAL: only specificity exists
                    L.append(f"| {nm} | {part} (all NORMAL, n={m.get('n')}) | {ex[nm]['threshold']:.3f} | "
                             f"{f3(m['accuracy'])} | - | - | {f3(m['specificity'])} | - | - |")
                    continue
                L.append(f"| {nm} | {part} | {ex[nm]['threshold']:.3f} | " + " | ".join(
                    f3(m[k]) for k in ["accuracy", "balanced_accuracy", "recall", "specificity", "roc_auc", "pr_auc"]) + " |")
        i = ex["internal_test_reference"]
        L.append("| final | internal locked test | | " + " | ".join(
            f3(i[k]) for k in ["accuracy", "balanced_accuracy", "recall", "specificity", "roc_auc", "pr_auc"]) + " |")
        L.append(f"\nPixel statistics: {json.dumps(ex['pixel_stats'])}")

    sh = j("p2_ship.json")
    if sh:
        h("Shipping check", "ship_check.py")
        L.append(f"- model file sizes (MB): {sh['size_MB']}")
        L.append(f"- latency 1 thread: {sh['latency_ms_1_thread']}, 2 threads: {sh['latency_ms_2_thread']}, "
                 f"model load {sh['model_load_ms']:.0f} ms")
        L.append("\n| input | verdict | P(pneumonia) |\n|---|---|---|")
        for k, v in sh["sample_predictions"].items():
            L.append(f"| {k} | {v['verdict']} | {v.get('probability')} |")

    lp = j("p2_ship_laptop.json")
    if lp:
        h("Latency on the i7-12700H laptop (CPU only)", "SHIP_OUT=p2_ship_laptop.json python ship_check.py")
        L.append(f"- 1 thread: {lp['latency_ms_1_thread']}, 2 threads: {lp['latency_ms_2_thread']}, "
                 f"model load {lp['model_load_ms']:.0f} ms")

    with open(os.path.join(REP, "tables.md"), "w") as f:
        f.write("# All result tables (generated by make_tables.py)\n" + "\n".join(L) + "\n")
    print("wrote reports/tables.md")


if __name__ == "__main__":
    main()
