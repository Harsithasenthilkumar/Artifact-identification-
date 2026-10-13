import argparse
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

CATEGORICAL = ["Department", "Culture", "Period"]
NUMERIC = ["Object Begin Date", "Object End Date"]
TEXT = "Medium"
MIN_COUNT = 5
MAX_VALUES = 30
MAX_WORDS = 50
STOPWORDS = {"and", "with", "in", "on", "of", "the", "or", "a", "to", "from", "over", "under"}


def words(text):
    if not isinstance(text, str):
        return set()
    return set(re.findall(r"[a-z]+", text.lower())) - STOPWORDS


def fit_encoder(train, banned_words):
    enc = {"categorical": {}, "numeric": {}}
    for col in CATEGORICAL:
        counts = train[col].value_counts()
        enc["categorical"][col] = counts[counts >= MIN_COUNT].index[:MAX_VALUES].tolist()
    for col in NUMERIC:
        vals = pd.to_numeric(train[col], errors="coerce")
        enc["numeric"][col] = {"median": float(vals.median()), "mean": float(vals.mean()),
                               "std": float(vals.std()) or 1.0}
    counts = Counter(w for m in train[TEXT] for w in words(m) if w not in banned_words)
    enc["medium_words"] = [w for w, c in counts.most_common(MAX_WORDS) if c >= MIN_COUNT]
    return enc


def feature_names(enc):
    names = []
    for col in CATEGORICAL:
        names += [f"{col}={v}" for v in enc["categorical"][col]] + [f"{col}=other", f"{col}=missing"]
    names += NUMERIC + [f"medium:{w}" for w in enc["medium_words"]]
    return names


def encode(df, enc):
    parts = []
    for col in CATEGORICAL:
        vocab = enc["categorical"][col]
        block = np.zeros((len(df), len(vocab) + 2), dtype=np.float32)
        for i, v in enumerate(df[col]):
            if pd.isna(v):
                block[i, -1] = 1
            elif v in vocab:
                block[i, vocab.index(v)] = 1
            else:
                block[i, -2] = 1
        parts.append(block)
    for col in NUMERIC:
        s = enc["numeric"][col]
        vals = pd.to_numeric(df[col], errors="coerce").fillna(s["median"])
        parts.append(((vals - s["mean"]) / s["std"]).values.astype(np.float32)[:, None])
    vocab = enc["medium_words"]
    block = np.zeros((len(df), len(vocab)), dtype=np.float32)
    for i, m in enumerate(df[TEXT]):
        for w in words(m):
            if w in vocab:
                block[i, vocab.index(w)] = 1
    parts.append(block)
    return np.concatenate(parts, axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="features")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(os.path.join(args.data, "dataset_all.csv"), dtype=str)
    with open(os.path.join(args.data, "dataset_config.json")) as f:
        known = json.load(f)["known"]
    banned = {w for name in known for w in words(name)}

    train = df[df["split"] == "train"]
    enc = fit_encoder(train, banned)
    X = encode(df, enc)
    names = feature_names(enc)

    np.savez(os.path.join(args.out, "metadata_features.npz"), object_ids=df["object_id"].values, features=X)
    with open(os.path.join(args.out, "metadata_encoder.json"), "w") as f:
        json.dump({**enc, "feature_names": names, "fitted_on": "train split only"}, f, indent=2)

    print("Fields used:", CATEGORICAL + NUMERIC + [TEXT])
    print("Not used (label leakage): Title, Object Name, Classification")
    print(f"Encoder fitted on {len(train)} training rows only")
    print(f"Feature vector length: {X.shape[1]}")
    for col in CATEGORICAL:
        print(f"  {col}: {len(enc['categorical'][col])} values + other + missing")
    print(f"  dates: {len(NUMERIC)} standardised numbers")
    print(f"  medium words ({len(enc['medium_words'])}): {enc['medium_words']}")
    print(f"  known-class words blocked from medium vocabulary: {sorted(banned)}")

    print("\nShare of rows whose value is 'other' (not seen in training) or 'missing':")
    rows = []
    for split in ["train", "val", "known_test", "unknown_test"]:
        part = X[(df["split"] == split).values]
        row = {"split": split}
        for col in CATEGORICAL:
            row[f"{col} other"] = round(part[:, names.index(f"{col}=other")].mean(), 2)
            row[f"{col} missing"] = round(part[:, names.index(f"{col}=missing")].mean(), 2)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved {X.shape} to {args.out}/metadata_features.npz")


if __name__ == "__main__":
    main()
