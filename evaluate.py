import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from open_set import MODES, NAMES, mahalanobis, softmax

N_BOOT = 1000
SEED = 42
BOOT_METRICS = ["Known accuracy", "Mahalanobis AUROC", "Softmax AUROC"]


def load_mode(mode):
    d = np.load(os.path.join("outputs", mode, "embeddings.npz"), allow_pickle=True)
    mh = np.load(os.path.join("outputs", mode, "mahalanobis.npz"), allow_pickle=True)
    kt_prob, ut_prob = softmax(d["known_test_logits"]), softmax(d["unknown_test_logits"])
    return {"ids": np.r_[d["known_test_ids"], d["unknown_test_ids"]].astype(str),
            "y_true": d["known_test_labels"], "y_pred": kt_prob.argmax(1),
            "kt_dist": mahalanobis(d["known_test_emb"], mh["means"], mh["precision"])[0],
            "ut_dist": mahalanobis(d["unknown_test_emb"], mh["means"], mh["precision"])[0],
            "kt_msp": 1 - kt_prob.max(1), "ut_msp": 1 - ut_prob.max(1)}


def auroc(known, unknown):
    return roc_auc_score(np.r_[np.zeros(len(known)), np.ones(len(unknown))], np.r_[known, unknown])


def stats(r, ik, iu):
    return {"Known accuracy": np.mean(r["y_true"][ik] == r["y_pred"][ik]),
            "Mahalanobis AUROC": auroc(r["kt_dist"][ik], r["ut_dist"][iu]),
            "Softmax AUROC": auroc(r["kt_msp"][ik], r["ut_msp"][iu])}


def main():
    runs = {m: load_mode(m) for m in MODES}
    for m in MODES:
        assert np.array_equal(runs[m]["ids"], runs["image"]["ids"]), "test sets differ between models!"
    n_k, n_u = len(runs["image"]["y_true"]), len(runs["image"]["ut_dist"])
    print(f"All three models evaluated on the same {n_k} known + {n_u} unknown test artifacts.")

    rows = []
    for m in MODES:
        km = json.load(open(os.path.join("outputs", m, "metrics_known.json")))
        om = json.load(open(os.path.join("outputs", m, "open_set_metrics.json")))
        point = stats(runs[m], np.arange(n_k), np.arange(n_u))
        assert abs(point["Known accuracy"] - km["accuracy"]) < 1e-9 and abs(point["Mahalanobis AUROC"] - om["auroc"]) < 1e-9
        rows.append({"Model": NAMES[m], "Accuracy": km["accuracy"], "Known macro-F1": km["macro_f1"],
                     "AUROC": om["auroc"], "F1": om["f1"], "Recall": om["recall"], "FPR": om["fpr"],
                     "Precision": om["precision"], "Softmax AUROC": om["softmax_baseline_auroc"]})
    final = pd.DataFrame(rows).round(4)
    final.to_csv(os.path.join("outputs", "final_results.csv"), index=False)
    print("Consistency check OK: recomputed values match the saved metrics files.\n")
    print("FINAL RESULTS")
    print("  Accuracy, Known macro-F1 = known-class classification on the known test set")
    print("  AUROC, F1, Recall, FPR, Precision = Known vs Unknown detection (Mahalanobis, positive = Unknown)")
    print(final.to_string(index=False))

    rng = np.random.default_rng(SEED)
    boot = {m: {k: [] for k in BOOT_METRICS} for m in MODES}
    for _ in range(N_BOOT):
        ik, iu = rng.integers(0, n_k, n_k), rng.integers(0, n_u, n_u)
        for m in MODES:
            for k, v in stats(runs[m], ik, iu).items():
                boot[m][k].append(v)
    boot = {m: {k: np.array(v) for k, v in b.items()} for m, b in boot.items()}

    ci_rows = []
    for m in MODES:
        point = stats(runs[m], np.arange(n_k), np.arange(n_u))
        for k in BOOT_METRICS:
            lo, hi = np.percentile(boot[m][k], [2.5, 97.5])
            ci_rows.append({"Model": NAMES[m], "Metric": k, "Value": point[k], "95% CI low": lo, "95% CI high": hi})
    ci = pd.DataFrame(ci_rows).round(4)
    ci.to_csv(os.path.join("outputs", "final_confidence_intervals.csv"), index=False)
    print(f"\n95% BOOTSTRAP CONFIDENCE INTERVALS ({N_BOOT} resamples of the test set, seed {SEED})")
    print(ci.to_string(index=False))

    comparisons = [("multimodal", "image", "Known accuracy"), ("multimodal", "image", "Mahalanobis AUROC"),
                   ("multimodal", "metadata", "Mahalanobis AUROC"), ("image", "metadata", "Mahalanobis AUROC"),
                   ("image", "image", "Softmax vs Mahalanobis"), ("multimodal", "multimodal", "Softmax vs Mahalanobis")]
    diff_rows = []
    for a, b, metric in comparisons:
        if metric == "Softmax vs Mahalanobis":
            diffs = boot[a]["Softmax AUROC"] - boot[a]["Mahalanobis AUROC"]
            label = f"{NAMES[a]}: Softmax AUROC - Mahalanobis AUROC"
        else:
            diffs = boot[a][metric] - boot[b][metric]
            label = f"{metric}: {NAMES[a]} - {NAMES[b]}"
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        diff_rows.append({"Comparison": label, "Mean difference": diffs.mean(), "95% CI low": lo,
                          "95% CI high": hi, "CI excludes 0": bool(lo > 0 or hi < 0)})
    diff = pd.DataFrame(diff_rows).round(4)
    diff.to_csv(os.path.join("outputs", "final_differences.csv"), index=False)
    print("\nPAIRED DIFFERENCES (same bootstrap samples for both models)")
    print("  If the CI excludes 0, the difference is unlikely to be just luck in which test images we have.")
    pd.set_option("display.width", 200)
    print(diff.to_string(index=False))

    metrics = ["Accuracy", "AUROC", "F1", "Recall", "FPR"]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, m in enumerate(MODES):
        vals = final.loc[i, metrics].values.astype(float)
        bars = ax.bar(x + (i - 1) * 0.27, vals, 0.27, label=NAMES[m])
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
        for col, key in [("Accuracy", "Known accuracy"), ("AUROC", "Mahalanobis AUROC")]:
            idx = metrics.index(col)
            lo, hi = np.percentile(boot[m][key], [2.5, 97.5])
            ax.errorbar(x[idx] + (i - 1) * 0.27, vals[idx], yerr=[[vals[idx] - lo], [hi - vals[idx]]],
                        color="black", capsize=3)
    ax.set_xticks(x, ["Known\nAccuracy", "Open-set\nAUROC", "Open-set\nF1", "Open-set\nRecall", "Open-set\nFPR"])
    ax.set_ylim(0, 1.1)
    ax.set_title("Image only vs Metadata only vs Image + Metadata (error bars = 95% bootstrap CI)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join("outputs", "metric_comparison.png"), dpi=120)
    plt.close(fig)

    d = np.load(os.path.join("outputs", "multimodal", "embeddings.npz"), allow_pickle=True)
    pca = PCA(n_components=2, random_state=SEED).fit(d["train_emb"])
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for split, marker, size in [("known_test", "o", 14), ("unknown_test", "x", 22)]:
        pts, cats = pca.transform(d[f"{split}_emb"]), d[f"{split}_category"].astype(str)
        for cat in sorted(set(cats)):
            label = f"{cat} (unseen)" if split == "unknown_test" else cat
            ax.scatter(*pts[cats == cat].T, s=size, marker=marker, alpha=0.7, label=label)
    ax.set_title("PCA of Image + Metadata embeddings (fitted on train)\ncircles = known test, crosses = unseen")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join("outputs", "embedding_pca.png"), dpi=120)
    plt.close(fig)
    print("\nSaved outputs/final_results.csv, final_confidence_intervals.csv, final_differences.csv, "
          "metric_comparison.png, embedding_pca.png")


if __name__ == "__main__":
    main()
