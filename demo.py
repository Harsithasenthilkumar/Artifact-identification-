import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.models import ViT_B_16_Weights, vit_b_16

from metadata_features import encode
from open_set import mahalanobis, softmax
from retrieval import build_index, search
from train import build_model

try:
    import faiss
except (ImportError, OSError):
    faiss = None


def image_feature(path):
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    vit = vit_b_16(weights=weights)
    vit.heads = torch.nn.Identity()
    vit.eval()
    x = weights.transforms()(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        return vit(x).numpy().astype(np.float32)


def pick_input(args, dataset):
    if args.sample:
        row = dataset[dataset["split"] == args.sample].sample(1, random_state=args.seed).iloc[0]
        return os.path.join(args.data, "images", f"{row['object_id']}.jpg"), row
    if args.object_id:
        row = dataset[dataset["object_id"] == args.object_id].iloc[0]
        return os.path.join(args.data, "images", f"{args.object_id}.jpg"), row
    oid = os.path.splitext(os.path.basename(args.image))[0]
    match = dataset[dataset["object_id"] == oid]
    return args.image, (match.iloc[0] if len(match) else None)


def main():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="path to any image file")
    group.add_argument("--object_id", help="a MET object id from our dataset")
    group.add_argument("--sample", choices=["known_test", "unknown_test"], help="pick a random test artifact")
    p.add_argument("--mode", default="auto", choices=["auto", "image", "multimodal"])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--data", default="data")
    args = p.parse_args()

    dataset = pd.read_csv(os.path.join(args.data, "dataset_all.csv"), dtype=str)
    path, row = pick_input(args, dataset)
    mode = args.mode
    if mode == "auto":
        mode = "multimodal" if row is not None else "image"
    if mode == "multimodal" and row is None:
        raise SystemExit("No MET metadata for this image, so the Image + Metadata model cannot be used. Use --mode image.")

    print(f"Input image: {path}")
    if row is not None:
        show = {c: (row[c] if isinstance(row[c], str) else "-") for c in
                ["Department", "Culture", "Period", "Medium", "Object Begin Date", "Object End Date"]}
        print(f"MET object {row['object_id']} | metadata: {show['Department']} | {show['Culture']} | "
              f"{show['Period']} | {show['Medium']} | {show['Object Begin Date']} to {show['Object End Date']}")
        note = " (TRAINING artifact - not a fair test!)" if row["split"] == "train" else ""
        print(f"[for checking only, not given to the model] true category: {row['category']}, split: {row['split']}{note}")
    else:
        print("No MET metadata found for this image.")
    print(f"Model: {'Image + Metadata' if mode == 'multimodal' else 'Image only'}\n")

    inputs = [torch.tensor(image_feature(path))]
    if mode == "multimodal":
        with open(os.path.join("features", "metadata_encoder.json")) as f:
            encoder = json.load(f)
        inputs.append(torch.tensor(encode(pd.DataFrame([row]), encoder)))

    emb_file = np.load(os.path.join("outputs", mode, "embeddings.npz"), allow_pickle=True)
    classes = [str(c) for c in emb_file["classes"]]
    model = build_model(mode, [x.shape[1] for x in inputs], len(classes))
    model.load_state_dict(torch.load(os.path.join("outputs", mode, "model.pt")))
    model.eval()
    with torch.no_grad():
        logits, emb = model(*inputs)
    probs = softmax(logits.numpy())[0]
    emb = emb.numpy()

    mh = np.load(os.path.join("outputs", mode, "mahalanobis.npz"), allow_pickle=True)
    dist, nearest = mahalanobis(emb, mh["means"], mh["precision"])
    dist, threshold = float(dist[0]), float(mh["threshold"])
    pred, conf = classes[int(probs.argmax())], float(probs.max())

    neighbours = []
    if dist <= threshold:
        verdict = f"KNOWN\nCategory: {pred}\nConfidence: {conf:.2f}"
        print("=" * 45)
        print("Prediction: KNOWN")
        print(f"Category:   {pred}")
        print(f"Confidence: {conf:.4f}  (softmax probability)")
        print(f"Unknown score (Mahalanobis distance): {dist:.2f}  <=  threshold {threshold:.2f}")
        print("=" * 45)
    else:
        index_path = os.path.join("outputs", "retrieval", mode, "faiss_index.bin")
        if faiss is not None and os.path.exists(index_path):
            index = faiss.read_index(index_path)
        else:
            index = build_index(emb_file["train_emb"])
        sims, idx = search(index, emb, args.k)
        ref_ids, ref_cats = emb_file["train_ids"], [str(c) for c in emb_file["train_category"]]
        neighbours = [(ref_ids[j], ref_cats[j], float(s)) for j, s in zip(idx[0], sims[0])]
        verdict = f"UNKNOWN\nUnknown score: {dist:.1f} (> {threshold:.1f})"
        print("=" * 45)
        print("Prediction: UNKNOWN")
        print(f"Unknown score (Mahalanobis distance): {dist:.2f}  >  threshold {threshold:.2f}")
        print(f"(Without open-set detection, the classifier would have forced: {pred}, confidence {conf:.2f})")
        print(f"\nTop-{args.k} similar KNOWN artifacts (FAISS, cosine similarity):")
        for n, (oid, cat, s) in enumerate(neighbours, 1):
            print(f"  {n}. {cat:14s} object {oid}   similarity {s:.4f}")
        print("Note: these are similar known artifacts, not the true identity of the object.")
        print("=" * 45)

    os.makedirs(os.path.join("outputs", "demo"), exist_ok=True)
    name = row["object_id"] if row is not None else os.path.splitext(os.path.basename(path))[0]
    fig, axes = plt.subplots(1, 1 + len(neighbours), figsize=(3 * (1 + len(neighbours)), 3.6))
    axes = np.atleast_1d(axes)
    axes[0].imshow(Image.open(path).convert("RGB"))
    axes[0].set_title(verdict, fontsize=10, color="red" if neighbours else "green")
    for ax, (oid, cat, s) in zip(axes[1:], neighbours):
        ax.imshow(Image.open(os.path.join(args.data, "images", f"{oid}.jpg")).convert("RGB"))
        ax.set_title(f"{cat}\nsimilarity {s:.3f}", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out = os.path.join("outputs", "demo", f"demo_{name}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"\nSaved result figure to {out}")


if __name__ == "__main__":
    main()
