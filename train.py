import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, recall_score)
from torch.utils.data import DataLoader, TensorDataset

from models import ImageModel, MetadataModel, MultimodalModel

SEED = 42
SPLITS = ["train", "val", "known_test", "unknown_test"]
CONFIG = {"embed_dim": 256, "meta_hidden": 128, "meta_embed_dim": 64, "fusion_dim": 256, "dropout": 0.3,
          "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 64, "epochs": 30}
FEATURE_FILES = {"image": ["image_features.npz"], "metadata": ["metadata_features.npz"],
                 "multimodal": ["image_features.npz", "metadata_features.npz"]}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_inputs(mode, data_dir, feat_dir):
    frames = {s: pd.read_csv(os.path.join(data_dir, f"{s}.csv"), dtype={"object_id": str}) for s in SPLITS}
    inputs = {s: [] for s in SPLITS}
    for fname in FEATURE_FILES[mode]:
        npz = np.load(os.path.join(feat_dir, fname), allow_pickle=True)
        row = {str(oid): i for i, oid in enumerate(npz["object_ids"])}
        for s in SPLITS:
            idx = [row[o] for o in frames[s]["object_id"]]
            inputs[s].append(torch.tensor(npz["features"][idx], dtype=torch.float32))
    return frames, inputs


def build_model(mode, in_dims, n_classes):
    if mode == "image":
        return ImageModel(in_dims[0], CONFIG["embed_dim"], n_classes, CONFIG["dropout"])
    if mode == "metadata":
        return MetadataModel(in_dims[0], CONFIG["meta_hidden"], CONFIG["meta_embed_dim"], n_classes, CONFIG["dropout"])
    return MultimodalModel(in_dims[0], in_dims[1], CONFIG["embed_dim"], CONFIG["meta_hidden"],
                           CONFIG["meta_embed_dim"], CONFIG["fusion_dim"], n_classes, CONFIG["dropout"])


def run_epoch(model, loader, loss_fn, optimizer=None):
    model.train(optimizer is not None)
    total_loss, correct, n = 0.0, 0, 0
    for *xs, y in loader:
        with torch.set_grad_enabled(optimizer is not None):
            logits, _ = model(*xs)
            loss = loss_fn(logits, y)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


def plot_curves(history, out_dir, mode):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, ax, title in [("loss", axes[0], "Loss"), ("acc", axes[1], "Accuracy")]:
        ax.plot(history["epoch"], history[f"train_{key}"], label="train")
        ax.plot(history["epoch"], history[f"val_{key}"], label="validation")
        ax.set_xlabel("epoch")
        ax.set_title(f"{title} ({mode})")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["image", "metadata", "multimodal"], required=True)
    p.add_argument("--data", default="data")
    p.add_argument("--features", default="features")
    args = p.parse_args()

    set_seed(SEED)
    out_dir = os.path.join("outputs", args.mode)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(args.data, "dataset_config.json")) as f:
        classes = json.load(f)["known"]

    frames, inputs = load_inputs(args.mode, args.data, args.features)
    labels = {s: torch.tensor(frames[s]["category"].map({c: i for i, c in enumerate(classes)})
                              .fillna(-1).astype(int).values) for s in SPLITS}

    in_dims = [x.shape[1] for x in inputs["train"]]
    model = build_model(args.mode, in_dims, len(classes))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(TensorDataset(*inputs["train"], labels["train"]),
                              batch_size=CONFIG["batch_size"], shuffle=True, generator=g)
    val_loader = DataLoader(TensorDataset(*inputs["val"], labels["val"]), batch_size=256)

    print(f"Mode: {args.mode}   classes: {classes}")
    print(f"Input dims: {in_dims}   trainable parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"train {len(labels['train'])}   val {len(labels['val'])}")
    history = {k: [] for k in ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]}
    best_val_loss, best_epoch = float("inf"), 0
    for epoch in range(1, CONFIG["epochs"] + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, loss_fn, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, loss_fn)
        for k, v in zip(history, [epoch, tr_loss, tr_acc, va_loss, va_acc]):
            history[k].append(v)
        if va_loss < best_val_loss:
            best_val_loss, best_epoch = va_loss, epoch
            torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
        print(f"epoch {epoch:2d}  train loss {tr_loss:.3f} acc {tr_acc:.3f}   val loss {va_loss:.3f} acc {va_acc:.3f}")
    print(f"Best epoch (lowest validation loss): {best_epoch}")

    model.load_state_dict(torch.load(os.path.join(out_dir, "model.pt")))
    model.eval()
    saved = {}
    with torch.no_grad():
        for s in SPLITS:
            logits, emb = model(*inputs[s])
            saved[f"{s}_emb"] = emb.numpy()
            saved[f"{s}_logits"] = logits.numpy()
            saved[f"{s}_labels"] = labels[s].numpy()
            saved[f"{s}_ids"] = frames[s]["object_id"].values
            saved[f"{s}_category"] = frames[s]["category"].values
    np.savez(os.path.join(out_dir, "embeddings.npz"), classes=np.array(classes), **saved)

    probs = torch.softmax(torch.tensor(saved["known_test_logits"]), dim=1).numpy()
    y_true = saved["known_test_labels"]
    y_pred = probs.argmax(1)
    pred_df = pd.DataFrame({"object_id": saved["known_test_ids"],
                            "true": [classes[i] for i in y_true],
                            "pred": [classes[i] for i in y_pred],
                            "confidence": probs.max(1).round(4)})
    for i, c in enumerate(classes):
        pred_df[f"prob_{c}"] = probs[:, i].round(4)
    pred_df.to_csv(os.path.join(out_dir, "predictions_known_test.csv"), index=False)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    metrics = {"mode": args.mode, "n_known_test": int(len(y_true)), "best_epoch": best_epoch,
               "accuracy": accuracy_score(y_true, y_pred),
               "macro_f1": f1_score(y_true, y_pred, average="macro"),
               "macro_recall": recall_score(y_true, y_pred, average="macro"),
               "per_class_recall": dict(zip(classes, recall_score(y_true, y_pred, average=None).round(3).tolist())),
               "mean_confidence_correct": float(probs.max(1)[y_true == y_pred].mean()),
               "mean_confidence_wrong": float(probs.max(1)[y_true != y_pred].mean()) if (y_true != y_pred).any() else None,
               "confusion_matrix": cm.tolist()}
    with open(os.path.join(out_dir, "metrics_known.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({**CONFIG, "mode": args.mode, "seed": SEED, "classes": classes}, f, indent=2)
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)

    plot_curves(history, out_dir, args.mode)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Known-test confusion matrix ({args.mode})")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=120)
    plt.close(fig)

    print(f"\nKNOWN-TEST RESULTS ({args.mode}, {len(y_true)} samples)")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Macro F1:     {metrics['macro_f1']:.4f}")
    print(f"  Macro Recall: {metrics['macro_recall']:.4f}")
    print("  Per-class recall:", metrics["per_class_recall"])
    print("  Confusion matrix (rows = true, columns = predicted):")
    print(pd.DataFrame(cm, index=classes, columns=classes).to_string())
    print(f"\nSaved model, metrics, predictions, embeddings and plots to {out_dir}/")


if __name__ == "__main__":
    main()
