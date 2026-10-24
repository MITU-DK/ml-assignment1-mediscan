"""
The ONLY script that reads the locked test set (837 images, patient-disjoint from the pool).
Run it once, after every decision (config, threshold, band) has been frozen on validation.

  Part 2.2  final model at t*: confusion matrix + all metrics
  Part 2.4  ensemble table: Part 1 models, soft vote, RF / AdaBoost on CNN embeddings
  Part 2.5  PCA / t-SNE of embeddings, k-means(k=2) vs labels, what the errors share
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, ConfusionMatrixDisplay

from common import save_json, load_json, metrics_at, FIG, RES, MODELS, CACHE, set_seed
from engine import load_model, load_data, predict, split_idx, test_idx

CAMP_PREV = 0.10


def probs_for(name, idx):
    m, cfg = load_model(name)
    X, df = load_data(cfg.get("res", 128), cfg.get("clahe", False))
    return predict(m, X, df, idx, cfg)


def main():
    set_seed(0)
    te = test_idx()
    tr, va = split_idx(0)
    df = pd.read_csv(os.path.join(CACHE, "split.csv"))
    y = df.label.values
    yt = y[te]
    thr = load_json("p2_threshold.json")
    t_star = thr["threshold_camp"]
    out = {"n_test": int(len(te)), "test_prevalence": float(yt.mean()), "threshold": t_star}

    # ---------------- 2.2 final model
    m, cfg = load_model("final.pt")
    X, _ = load_data(cfg.get("res", 128))
    p_final, emb_te = predict(m, X, df, te, cfg, return_emb=True)
    _, emb_tr = predict(m, X, df, tr, cfg, return_emb=True)
    fm = metrics_at(yt, p_final, t_star)
    sens, spec = fm["recall"], fm["specificity"]
    fm["camp_ppv"] = CAMP_PREV * sens / (CAMP_PREV * sens + (1 - CAMP_PREV) * (1 - spec))
    fm["camp_npv"] = (1 - CAMP_PREV) * spec / ((1 - CAMP_PREV) * spec + CAMP_PREV * (1 - sens))
    fm["camp_accuracy"] = CAMP_PREV * sens + (1 - CAMP_PREV) * spec
    fm["camp_expected_cost"] = 20 * CAMP_PREV * (1 - sens) + (1 - CAMP_PREV) * (1 - spec)
    out["final_at_t_star"] = fm
    out["final_at_0.5"] = metrics_at(yt, p_final, 0.5)
    lo, hi = thr["refer_band"]
    band = (p_final >= lo) & (p_final <= hi)
    out["refer_band"] = {"band": [lo, hi], "share_referred": float(band.mean()),
                         "metrics_outside_band": metrics_at(yt[~band], p_final[~band], t_star)}
    print("final @t*:", {k: round(v, 4) for k, v in fm.items() if isinstance(v, float)})

    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ConfusionMatrixDisplay.from_predictions(yt, (p_final >= t_star).astype(int),
                                            display_labels=["NORMAL", "PNEU"], ax=ax, colorbar=False)
    ax.set_title(f"locked test, t*={t_star:.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p2_test_confusion.png"), dpi=130)
    plt.close(fig)

    # ---------------- 2.4 ensembles
    table = {}
    member_probs = {}
    for fam in ["arogya_full", "arogya_no_residual", "arogya_no_se", "arogya_two_branch", "resnet18"]:
        files = sorted(glob.glob(os.path.join(MODELS, f"{fam}_s*.pt")))
        ps = [probs_for(os.path.basename(f), te) for f in files]
        if not ps:
            continue
        member_probs[fam] = ps
        rows = [metrics_at(yt, q, 0.5) for q in ps]
        table[fam + " (mean of seeds)"] = {k: float(np.mean([r[k] for r in rows]))
                                           for k in ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "f1"]}
        table[fam + " (mean of seeds)"]["n_models"] = len(ps)

    # HOG + LR refit on the same training folds
    H = np.load(os.path.join(RES, "preds", "hog_features.npy"))
    hog = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.01)).fit(H[tr], y[tr])
    p_hog = hog.predict_proba(H[te])[:, 1]
    table["HOG + logistic regression"] = metrics_at(yt, p_hog, 0.5)

    # soft voting: image-only CNNs from Part 1 (two-branch left out on purpose, it reads the size shortcut)
    vote_members = member_probs.get("arogya_full", []) + member_probs.get("resnet18", [])
    if vote_members:
        p_vote = np.mean(vote_members, axis=0)
        table["soft vote: ArogyaNet x3 + ResNet18 x3"] = metrics_at(yt, p_vote, 0.5)
        p_vote2 = np.mean(vote_members + [p_hog], axis=0)
        table["soft vote: + HOG/LR"] = metrics_at(yt, p_vote2, 0.5)

    rf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1).fit(emb_tr, y[tr])
    ada = AdaBoostClassifier(n_estimators=300, random_state=0).fit(emb_tr, y[tr])
    table["RandomForest on final-model embeddings"] = metrics_at(yt, rf.predict_proba(emb_te)[:, 1], 0.5)
    table["AdaBoost on final-model embeddings"] = metrics_at(yt, ada.predict_proba(emb_te)[:, 1], 0.5)
    table["final ArogyaNet (shipped)"] = metrics_at(yt, p_final, 0.5)
    keep = ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "f1"]
    out["ensemble_table_at_0.5"] = {k: {kk: float(v[kk]) for kk in keep if kk in v} for k, v in table.items()}
    for k, v in out["ensemble_table_at_0.5"].items():
        print(f"{k:45s}", {kk: round(vv, 4) for kk, vv in v.items()})

    # ---------------- 2.5 feature space
    pred = (p_final >= t_star).astype(int)
    wrong = pred != yt
    Z = StandardScaler().fit_transform(emb_te)
    pca = PCA(2, random_state=0).fit(Z)
    zp = pca.transform(Z)
    zt = TSNE(2, random_state=0, init="pca", perplexity=30).fit_transform(Z)
    km = KMeans(2, n_init=10, random_state=0).fit(Z)
    lab = km.labels_
    purity = sum(np.bincount(yt[lab == c]).max() for c in [0, 1]) / len(yt)
    out["feature_space"] = {"pca_explained_var_2d": pca.explained_variance_ratio_.tolist(),
                            "kmeans_ARI": float(adjusted_rand_score(yt, lab)), "kmeans_purity": float(purity),
                            "n_wrong": int(wrong.sum()),
                            "n_false_pos": int((wrong & (yt == 0)).sum()), "n_false_neg": int((wrong & (yt == 1)).sum())}
    # what do the errors have in common? compare acquisition stats
    audit = pd.read_csv(os.path.join(CACHE, "audit_table.csv")).set_index("path")
    st = audit.loc[df.path.values[te], ["width", "height", "area", "mean_int", "std_int"]].values
    cmp = {}
    for nm, mask in [("correct", ~wrong), ("false_pos (NORMAL called PNEU)", wrong & (yt == 0)),
                     ("false_neg (PNEU missed)", wrong & (yt == 1)), ("all NORMAL", yt == 0), ("all PNEU", yt == 1)]:
        if mask.sum():
            cmp[nm] = dict(zip(["width", "height", "area", "mean_int", "std_int"], st[mask].mean(0).round(1).tolist()))
            cmp[nm]["mean_prob"] = float(p_final[mask].mean())
            cmp[nm]["n"] = int(mask.sum())
    out["error_profile"] = cmp
    print(cmp)

    fig, ax = plt.subplots(2, 2, figsize=(10, 9))
    for row, (Zd, nm) in enumerate([(zp, "PCA"), (zt, "t-SNE")]):
        for c, col, cl in [(0, "tab:blue", "NORMAL"), (1, "tab:red", "PNEUMONIA")]:
            ax[row, 0].scatter(*Zd[yt == c].T, s=5, c=col, label=cl, alpha=0.6)
        ax[row, 1].scatter(*Zd[~wrong].T, s=5, c="lightgrey", label="correct")
        ax[row, 1].scatter(*Zd[wrong & (yt == 0)].T, s=18, c="tab:blue", marker="x", label="false positive")
        ax[row, 1].scatter(*Zd[wrong & (yt == 1)].T, s=18, c="tab:red", marker="x", label="false negative")
        ax[row, 0].set_title(f"{nm} - true class")
        ax[row, 1].set_title(f"{nm} - correct / wrong at t*")
        for a in ax[row]:
            a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "p2_embeddings.png"), dpi=120)
    plt.close(fig)

    idx_wrong = te[wrong][:10]
    if len(idx_wrong):
        fig, ax = plt.subplots(2, 5, figsize=(12, 5))
        for a in ax.flat:
            a.axis("off")
        for a, i, pr in zip(ax.flat, idx_wrong, p_final[wrong][:10]):
            a.imshow(X[i], cmap="gray")
            a.set_title(f"true {['N', 'P'][y[i]]}  p={pr:.2f}", fontsize=8)
        fig.suptitle("misclassified test images (first 10)")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "p2_errors.png"), dpi=110)
        plt.close(fig)

    np.savez(os.path.join(RES, "preds", "test_final.npz"), idx=te, prob=p_final)
    save_json(out, "p2_test.json")


if __name__ == "__main__":
    main()
