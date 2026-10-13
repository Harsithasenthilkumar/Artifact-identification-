import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

MODES = ["image", "metadata", "multimodal"]
NAMES = {"image": "Image only", "metadata": "Metadata only", "multimodal": "Image + Metadata"}
REG = 0.01
PERCENTILE = 95


def fit_mahalanobis(emb, labels, n_classes, reg=REG):
    means = np.stack([emb[labels == c].mean(axis=0) for c in range(n_classes)])
    centered = emb - means[labels]
    cov = centered.T @ centered / len(emb)
    lam = reg * np.trace(cov) / cov.shape[0]
    precision = np.linalg.inv(cov + lam * np.eye(cov.shape[0]))
    return means, precision, lam


def mahalanobis(emb, means, precision):
    diffs = emb[:, None, :] - means[None, :, :]
    d2 = np.einsum("ncd,de,nce->nc", diffs, precision, diffs)
    d = np.sqrt(np.maximum(d2, 0))
    return d.min(axis=1), d.argmin(axis=1)


def softmax(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def detection_metrics(is_unknown, pred_unknown):
    tp = int(np.sum(pred_unknown & is_unknown))
    tn = int(np.sum(~pred_unknown & ~is_unknown))
    fp = int(np.sum(pred_unknown & ~is_unknown))
    fn = int(np.sum(~pred_unknown & is_unknown))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "recall": recall,
            "fpr": fp / (fp + tn), "precision": precision, "f1": f1}


def evaluate_mode(mode):
    d = np.load(os.path.join("outputs", mode, "embeddings.npz"), allow_pickle=True)
    classes = [str(c) for c in d["classes"]]
    means, precision, lam = fit_mahalanobis(d["train_emb"], d["train_labels"], len(classes))

    val_dist, _ = mahalanobis(d["val_emb"], means, precision)
    threshold = float(np.percentile(val_dist, PERCENTILE))

    kt_dist, kt_near = mahalanobis(d["known_test_emb"], means, precision)
    ut_dist, ut_near = mahalanobis(d["unknown_test_emb"], means, precision)
    scores = np.concatenate([kt_dist, ut_dist])
    is_unknown = np.concatenate([np.zeros(len(kt_dist), bool), np.ones(len(ut_dist), bool)])
    pred_unknown = scores > threshold

    kt_prob, ut_prob = softmax(d["known_test_logits"]), softmax(d["unknown_test_logits"])
    msp_score = 1 - np.concatenate([kt_prob.max(1), ut_prob.max(1)])

    m = {"mode": mode, "regularization_lambda": float(lam), "threshold": threshold,
         "threshold_rule": f"{PERCENTILE}th percentile of validation (known-only) distances",
         "val_fpr": float(np.mean(val_dist > threshold)),
         "auroc": roc_auc_score(is_unknown, scores),
         "softmax_baseline_auroc": roc_auc_score(is_unknown, msp_score),
         **detection_metrics(is_unknown, pred_unknown)}

    unseen = d["unknown_test_category"]
    m["per_unseen_category"] = {}
    for cat in sorted(set(unseen)):
        mask = unseen == cat
        y = np.concatenate([np.zeros(len(kt_dist)), np.ones(mask.sum())])
        m["per_unseen_category"][cat] = {
            "auroc": roc_auc_score(y, np.concatenate([kt_dist, ut_dist[mask]])),
            "detected_as_unknown": float(np.mean(ut_dist[mask] > threshold)),
            "nearest_known_class_counts": pd.Series([classes[i] for i in ut_near[mask]]).value_counts().to_dict()}

    kt_pred = kt_prob.argmax(1)
    accepted = kt_dist <= threshold
    m["known_test_accepted"] = int(accepted.sum())
    m["accuracy_on_accepted_known"] = float(np.mean(kt_pred[accepted] == d["known_test_labels"][accepted]))

    m["threshold_sensitivity"] = {}
    for pct in [90, 95, 99]:
        t = np.percentile(val_dist, pct)
        s = detection_metrics(is_unknown, scores > t)
        m["threshold_sensitivity"][f"p{pct}"] = {k: round(s[k], 4) for k in ["recall", "fpr", "f1"]}

    ut_prob_pred = ut_prob.argmax(1)
    rows = pd.DataFrame({
        "object_id": np.concatenate([d["known_test_ids"], d["unknown_test_ids"]]),
        "split": ["known_test"] * len(kt_dist) + ["unknown_test"] * len(ut_dist),
        "category": np.concatenate([d["known_test_category"], unseen]),
        "is_unknown": is_unknown,
        "mahalanobis_distance": scores.round(3),
        "predicted_unknown": pred_unknown,
        "nearest_known_class": [classes[i] for i in np.concatenate([kt_near, ut_near])],
        "classifier_prediction": [classes[i] for i in np.concatenate([kt_pred, ut_prob_pred])],
        "classifier_confidence": np.concatenate([kt_prob.max(1), ut_prob.max(1)]).round(4)})
    rows.to_csv(os.path.join("outputs", mode, "open_set_scores.csv"), index=False)
    np.savez(os.path.join("outputs", mode, "mahalanobis.npz"), means=means, precision=precision,
             threshold=threshold, classes=np.array(classes))
    with open(os.path.join("outputs", mode, "open_set_metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    return m, is_unknown, scores, kt_dist, ut_dist


def main():
    results, curves = [], {}
    for mode in MODES:
        m, is_unknown, scores, kt_dist, ut_dist = evaluate_mode(mode)
        results.append(m)
        curves[mode] = (is_unknown, scores, kt_dist, ut_dist, m["threshold"])

        print(f"\n=== {NAMES[mode]} ===")
        print(f"Mahalanobis fitted on train embeddings only; regularization lambda = {m['regularization_lambda']:.4g}")
        print(f"Threshold = {m['threshold']:.3f} ({m['threshold_rule']}); validation FPR = {m['val_fpr']:.3f}")
        print(f"Open-set test: {m['TN'] + m['FP']} known + {m['TP'] + m['FN']} unknown")
        print(f"  TP {m['TP']}  FN {m['FN']}  FP {m['FP']}  TN {m['TN']}")
        print(f"  AUROC {m['auroc']:.4f}   F1 {m['f1']:.4f}   Recall {m['recall']:.4f}   "
              f"FPR {m['fpr']:.4f}   Precision {m['precision']:.4f}")
        print(f"  Softmax-confidence baseline AUROC: {m['softmax_baseline_auroc']:.4f}")
        print(f"  Classifier accuracy on known-test samples accepted as known: "
              f"{m['accuracy_on_accepted_known']:.4f} ({m['known_test_accepted']} accepted)")
        for cat, v in m["per_unseen_category"].items():
            print(f"  unseen '{cat}': AUROC {v['auroc']:.4f}, detected {v['detected_as_unknown']:.3f}, "
                  f"nearest known class {v['nearest_known_class_counts']}")
        print("  Threshold sensitivity (reported only, not used to choose):", m["threshold_sensitivity"])

    summary = pd.DataFrame([{"Model": NAMES[m["mode"]], "AUROC": m["auroc"], "F1": m["f1"],
                             "Recall": m["recall"], "FPR": m["fpr"], "Precision": m["precision"],
                             "Softmax AUROC": m["softmax_baseline_auroc"]} for m in results]).round(4)
    summary.to_csv(os.path.join("outputs", "open_set_summary.csv"), index=False)
    print("\nOPEN-SET SUMMARY (positive class = Unknown)")
    print(summary.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 5))
    for mode, (is_unknown, scores, *_rest) in curves.items():
        fpr, tpr, _ = roc_curve(is_unknown, scores)
        ax.plot(fpr, tpr, label=f"{NAMES[mode]} (AUROC {roc_auc_score(is_unknown, scores):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random")
    ax.set_xlabel("False positive rate (known flagged as unknown)")
    ax.set_ylabel("True positive rate (unknown detected)")
    ax.set_title("Known vs Unknown ROC (Mahalanobis)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join("outputs", "open_set_roc.png"), dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (mode, (_, _, kt_dist, ut_dist, thr)) in zip(axes, curves.items()):
        bins = np.linspace(0, np.percentile(np.concatenate([kt_dist, ut_dist]), 99), 40)
        ax.hist(kt_dist, bins=bins, alpha=0.6, label="known test")
        ax.hist(ut_dist, bins=bins, alpha=0.6, label="unknown test")
        ax.axvline(thr, color="k", linestyle="--", label="threshold (from validation)")
        ax.set_title(NAMES[mode])
        ax.set_xlabel("min Mahalanobis distance")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join("outputs", "open_set_distances.png"), dpi=120)
    plt.close(fig)
    print("\nSaved outputs/open_set_summary.csv, open_set_roc.png, open_set_distances.png "
          "and per-model files in outputs/<mode>/")


if __name__ == "__main__":
    main()
