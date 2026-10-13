import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from open_set import MODES, NAMES, PERCENTILE, detection_metrics, fit_mahalanobis, mahalanobis, softmax
from train import SPLITS, load_inputs

OUT = os.path.join("outputs", "exploratory")


def per_category_auroc(known_scores, unknown_scores, unseen_cats):
    result = {}
    for cat in sorted(set(unseen_cats)):
        mask = unseen_cats == cat
        y = np.r_[np.zeros(len(known_scores)), np.ones(mask.sum())]
        result[f"AUROC {cat}"] = roc_auc_score(y, np.r_[known_scores, unknown_scores[mask]])
    return result


def mahalanobis_row(name, x, train_labels, n_classes, unseen_cats):
    means, precision, _ = fit_mahalanobis(x["train"], train_labels, n_classes)
    dist = {s: mahalanobis(x[s], means, precision)[0] for s in SPLITS}
    threshold = np.percentile(dist["val"], PERCENTILE)
    scores = np.r_[dist["known_test"], dist["unknown_test"]]
    is_unknown = np.r_[np.zeros(len(dist["known_test"]), bool), np.ones(len(dist["unknown_test"]), bool)]
    m = detection_metrics(is_unknown, scores > threshold)
    row = {"representation": name, "AUROC": roc_auc_score(is_unknown, scores),
           "Recall": m["recall"], "FPR": m["fpr"], "F1": m["f1"],
           **per_category_auroc(dist["known_test"], dist["unknown_test"], unseen_cats)}
    medians = {"representation": name, **{f"median {s}": float(np.median(dist[s])) for s in SPLITS}}
    medians["val / train ratio"] = medians["median val"] / medians["median train"]
    return row, medians


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join("data", "dataset_config.json")) as f:
        classes = json.load(f)["known"]
    print("POST-HOC EXPLORATORY ANALYSIS")
    print("Computed after seeing the main open-set results. It helps explain them; it does NOT replace them.")
    print("Same rules as the main experiment: fit on train, threshold = 95th percentile of validation.\n")

    frames, inputs = load_inputs("multimodal", "data", "features")
    train_labels = frames["train"]["category"].map({c: i for i, c in enumerate(classes)}).values
    unseen_cats = frames["unknown_test"]["category"].values
    raw_image = {s: inputs[s][0].numpy() for s in SPLITS}
    raw_meta = {s: inputs[s][1].numpy() for s in SPLITS}
    raw_both = {s: np.hstack([raw_image[s], raw_meta[s]]) for s in SPLITS}

    maha_rows, median_rows = [], []
    for name, x in [("Raw ViT features (768, no trained head)", raw_image),
                    ("Raw metadata vector (125, no trained head)", raw_meta),
                    ("Raw ViT + metadata (893, no trained head)", raw_both)]:
        row, med = mahalanobis_row(name, x, train_labels, len(classes), unseen_cats)
        maha_rows.append(row)
        median_rows.append(med)

    softmax_rows = []
    for mode in MODES:
        d = np.load(os.path.join("outputs", mode, "embeddings.npz"), allow_pickle=True)
        x = {s: d[f"{s}_emb"] for s in SPLITS}
        row, med = mahalanobis_row(f"Trained head: {NAMES[mode]} (main result)", x, d["train_labels"],
                                   len(classes), d["unknown_test_category"])
        maha_rows.append(row)
        median_rows.append(med)

        kt = 1 - softmax(d["known_test_logits"]).max(1)
        ut = 1 - softmax(d["unknown_test_logits"]).max(1)
        y = np.r_[np.zeros(len(kt)), np.ones(len(ut))]
        softmax_rows.append({"model": NAMES[mode], "softmax AUROC": roc_auc_score(y, np.r_[kt, ut]),
                             **per_category_auroc(kt, ut, d["unknown_test_category"])})

    maha = pd.DataFrame(maha_rows).round(4)
    soft = pd.DataFrame(softmax_rows).round(4)
    medians = pd.DataFrame(median_rows).round(3)
    maha.to_csv(os.path.join(OUT, "mahalanobis_representations.csv"), index=False)
    soft.to_csv(os.path.join(OUT, "softmax_per_category.csv"), index=False)
    medians.to_csv(os.path.join(OUT, "distance_medians.csv"), index=False)

    pd.set_option("display.width", 250)
    print("A. Mahalanobis on different representations (positive class = Unknown)")
    print(maha.to_string(index=False))
    print("\nB. Softmax-confidence baseline, per unseen category")
    print(soft.to_string(index=False))
    print("\nC. Median min-Mahalanobis distance per split (large val/train ratio = train embeddings "
          "are much tighter than new known data, i.e. overfitting)")
    print(medians.to_string(index=False))
    print(f"\nSaved CSV tables to {OUT}/")


if __name__ == "__main__":
    main()
