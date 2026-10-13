import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

try:
    import faiss
except (ImportError, OSError) as e:
    faiss = None
    FAISS_ERROR = str(e)

SEED = 42


def l2_normalize(x):
    return (x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)).astype(np.float32)


def build_index(reference):
    reference = l2_normalize(reference)
    if faiss is None:
        return reference
    index = faiss.IndexFlatIP(reference.shape[1])
    index.add(reference)
    return index


def search(index, queries, k):
    queries = l2_normalize(queries)
    if faiss is None:
        sims = queries @ index.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        return np.take_along_axis(sims, idx, axis=1), idx
    return index.search(queries, k)


def plot_examples(examples, ref_ids, ref_cats, data_dir, path):
    k = len(examples[0]["neighbours"])
    fig, axes = plt.subplots(len(examples), k + 1, figsize=(2.6 * (k + 1), 2.9 * len(examples)))
    axes = np.atleast_2d(axes)
    for r, ex in enumerate(examples):
        decision = "predicted UNKNOWN" if ex["flagged"] else "MISSED: accepted as known"
        cells = [(ex["query_id"], f"QUERY: {decision}\n(true: {ex['query_category']}, unseen)")]
        cells += [(ref_ids[j], f"#{n + 1} {ref_cats[j]}\nsimilarity {s:.3f}")
                  for n, (j, s) in enumerate(zip(ex["neighbours"], ex["similarities"]))]
        for c, (oid, title) in enumerate(cells):
            ax = axes[r, c]
            ax.imshow(Image.open(os.path.join(data_dir, "images", f"{oid}.jpg")).convert("RGB"))
            ax.set_title(title, fontsize=9, color="red" if c == 0 else "black")
            ax.axis("off")
    fig.suptitle("Top-K most similar KNOWN artifacts (similar, not the true identity)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="multimodal", choices=["image", "metadata", "multimodal"])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--data", default="data")
    args = p.parse_args()
    out_dir = os.path.join("outputs", "retrieval", args.mode)
    os.makedirs(out_dir, exist_ok=True)

    if faiss is None:
        print(f"WARNING: faiss could not be loaded ({FAISS_ERROR}).")
        print("Falling back to exact NumPy search, which gives the same neighbours as faiss.IndexFlatIP.")
    else:
        print(f"Using FAISS {faiss.__version__}: IndexFlatIP (exact search, cosine similarity)")

    d = np.load(os.path.join("outputs", args.mode, "embeddings.npz"), allow_pickle=True)
    ref_ids, ref_cats = d["train_ids"], np.array([str(c) for c in d["train_category"]])
    index = build_index(d["train_emb"])
    if faiss is not None:
        faiss.write_index(index, os.path.join(out_dir, "faiss_index.bin"))
    pd.DataFrame({"row": range(len(ref_ids)), "object_id": ref_ids, "category": ref_cats}).to_csv(
        os.path.join(out_dir, "index_reference.csv"), index=False)
    print(f"Reference set: {len(ref_ids)} known training artifacts, embedding size {d['train_emb'].shape[1]}")

    kt_sims, kt_idx = search(index, d["known_test_emb"], args.k)
    kt_true = np.array([str(c) for c in d["known_test_category"]])
    same = ref_cats[kt_idx] == kt_true[:, None]
    sanity = {"precision_at_1": float(same[:, 0].mean()), f"precision_at_{args.k}": float(same.mean())}
    print(f"\nSanity check on KNOWN test queries (do neighbours share the query's class?)")
    print(f"  Precision@1 = {sanity['precision_at_1']:.4f}   Precision@{args.k} = {sanity[f'precision_at_{args.k}']:.4f}")

    ut_sims, ut_idx = search(index, d["unknown_test_emb"], args.k)
    ut_ids = d["unknown_test_ids"]
    ut_cats = np.array([str(c) for c in d["unknown_test_category"]])
    scores = pd.read_csv(os.path.join("outputs", args.mode, "open_set_scores.csv"), dtype={"object_id": str})
    scores = scores[scores["split"] == "unknown_test"].set_index("object_id")
    flagged = scores.loc[[str(o) for o in ut_ids], "predicted_unknown"].values.astype(bool)

    rows = []
    for q in range(len(ut_ids)):
        for r in range(args.k):
            j = ut_idx[q, r]
            rows.append({"query_id": ut_ids[q], "query_true_category": ut_cats[q],
                         "predicted_unknown": flagged[q], "rank": r + 1, "neighbour_id": ref_ids[j],
                         "neighbour_category": ref_cats[j], "cosine_similarity": round(float(ut_sims[q, r]), 4)})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "unknown_topk.csv"), index=False)

    summary = {"mode": args.mode, "k": args.k, "reference_size": int(len(ref_ids)),
               "search": "faiss.IndexFlatIP" if faiss is not None else "numpy exact (faiss unavailable)",
               "known_test_sanity": sanity,
               "mean_top1_similarity_known_test": float(kt_sims[:, 0].mean()),
               "mean_top1_similarity_unknown_test": float(ut_sims[:, 0].mean()), "per_unseen_category": {}}
    print(f"\nMean top-1 similarity: known test {summary['mean_top1_similarity_known_test']:.4f}   "
          f"unknown test {summary['mean_top1_similarity_unknown_test']:.4f}")
    print(f"\nWhat the unseen artifacts retrieve (all top-{args.k} neighbours counted):")
    for cat in sorted(set(ut_cats)):
        mask = ut_cats == cat
        counts = pd.Series(ref_cats[ut_idx[mask]].ravel()).value_counts()
        summary["per_unseen_category"][cat] = {
            "queries": int(mask.sum()), "flagged_unknown": int(flagged[mask].sum()),
            "mean_top1_similarity": float(ut_sims[mask, 0].mean()),
            "retrieved_category_share": (counts / counts.sum()).round(3).to_dict()}
        print(f"  {cat:17s} ({mask.sum()} queries, {flagged[mask].sum()} flagged unknown): "
              f"{(counts / counts.sum()).round(2).to_dict()}")
    with open(os.path.join(out_dir, "retrieval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    rng = np.random.default_rng(SEED)
    examples = []
    for cat in sorted(set(ut_cats)):
        candidates = np.where((ut_cats == cat) & flagged)[0]
        if len(candidates) == 0:
            candidates = np.where(ut_cats == cat)[0]
        q = rng.choice(candidates)
        examples.append({"query_id": ut_ids[q], "query_category": cat, "flagged": bool(flagged[q]),
                         "neighbours": ut_idx[q], "similarities": ut_sims[q]})
        status = "flagged UNKNOWN" if flagged[q] else "NOT flagged - no flagged example in this category"
        print(f"\nExample: unseen artifact {ut_ids[q]} (true category '{cat}', hidden from the model) - {status}")
        for n, (j, s) in enumerate(zip(ut_idx[q], ut_sims[q]), 1):
            print(f"  {n}. {ref_cats[j]:14s} object {ref_ids[j]}   similarity {s:.4f}")
    plot_examples(examples, ref_ids, ref_cats, args.data, os.path.join(out_dir, "retrieval_examples.png"))
    print(f"\nNote: retrieved categories are the most SIMILAR known categories, not the true identity.")
    print(f"Saved index, top-{args.k} table, summary and example figure to {out_dir}/")


if __name__ == "__main__":
    main()
