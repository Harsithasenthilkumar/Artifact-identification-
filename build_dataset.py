import argparse
import hashlib
import json
import os
import time

import pandas as pd
import requests
from PIL import Image

KNOWN = ["bowl", "vase", "figure", "coin", "lamp", "cylinder seal"]
UNSEEN = ["jug", "cuneiform tablet", "ring"]

SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
API = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{}"
SIGNATURE_COLS = ["Department", "category", "Title", "Culture", "Period", "Medium", "Object Begin Date"]
OUT_COLS = ["object_id", "object_number", "category", "known", "split", "image_path", "image_md5",
            "Department", "Culture", "Period", "Medium", "Object Begin Date", "Object End Date",
            "Country", "Region"]


def get_image_url(session, object_id):
    for attempt in range(4):
        try:
            r = session.get(API.format(object_id), timeout=20)
            if r.status_code == 200:
                return r.json().get("primaryImageSmall", "")
            if r.status_code == 404:
                return ""
        except requests.RequestException:
            pass
        time.sleep(3 * (attempt + 1))
    return ""


def download_image(session, url, path):
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        Image.open(path).verify()
        return True
    except Exception:
        if os.path.exists(path):
            os.remove(path)
        return False


def file_md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def collect(session, rows, n_needed, img_dir, seen_hashes, delay):
    kept, no_image, duplicate = [], 0, 0
    for _, row in rows.iterrows():
        if len(kept) >= n_needed:
            break
        path = os.path.join(img_dir, f"{row['Object ID']}.jpg")
        if not os.path.exists(path):
            url = get_image_url(session, row["Object ID"])
            time.sleep(delay)
            if not url or not download_image(session, url, path):
                no_image += 1
                continue
        h = file_md5(path)
        if h in seen_hashes:
            duplicate += 1
            continue
        seen_hashes.add(h)
        kept.append({**row.to_dict(), "image_path": path, "image_md5": h})
    return kept, no_image, duplicate


def check(name, ok):
    print(f"  [{'OK' if ok else 'FAILED'}] {name}")
    if not ok:
        raise SystemExit("Dataset check failed - stop and debug before continuing.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default="outputs/phase1/candidate_pool.csv")
    p.add_argument("--data", default="data")
    p.add_argument("--n_known", type=int, default=400)
    p.add_argument("--n_unseen", type=int, default=150)
    p.add_argument("--max_per_signature", type=int, default=5)
    p.add_argument("--delay", type=float, default=0.1)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if os.path.exists(os.path.join(args.data, "train.csv")) and not args.force:
        raise SystemExit("data/train.csv already exists. The split is fixed - use --force only to rebuild it.")

    img_dir = os.path.join(args.data, "images")
    os.makedirs(img_dir, exist_ok=True)

    pool = pd.read_csv(args.pool, dtype=str)
    pool = pool[pool["category"].isin(KNOWN + UNSEEN)]
    print(f"Records in chosen categories: {len(pool)}")
    pool = pool.drop_duplicates(subset="Object Number")
    print(f"After removing duplicate Object Numbers: {len(pool)}")
    pool = pool.sample(frac=1, random_state=SEED)
    pool = pool.groupby(SIGNATURE_COLS, dropna=False).head(args.max_per_signature)
    print(f"After keeping at most {args.max_per_signature} records per identical-metadata group: {len(pool)}")

    session = requests.Session()
    session.headers["User-Agent"] = "student-project-met-dataset/1.0"
    seen_hashes, all_rows = set(), []
    print("\nDownloading images (re-running resumes from files already in data/images):")
    for cat in KNOWN + UNSEEN:
        n_needed = args.n_known if cat in KNOWN else args.n_unseen
        rows = pool[pool["category"] == cat]
        kept, no_img, dup = collect(session, rows, n_needed, img_dir, seen_hashes, args.delay)
        all_rows += kept
        warn = "  <-- fewer than target" if len(kept) < n_needed else ""
        print(f"  {cat:18s} candidates {len(rows):5d}  kept {len(kept):4d}  "
              f"no_image {no_img:3d}  duplicate_image {dup:3d}{warn}")

    df = pd.DataFrame(all_rows).rename(columns={"Object ID": "object_id", "Object Number": "object_number"})
    df["known"] = df["category"].isin(KNOWN)
    df["split"] = "unknown_test"
    for cat in KNOWN:
        idx = df.index[df["category"] == cat].tolist()
        n_train = int(TRAIN_FRAC * len(idx))
        n_val = int(VAL_FRAC * len(idx))
        df.loc[idx[:n_train], "split"] = "train"
        df.loc[idx[n_train:n_train + n_val], "split"] = "val"
        df.loc[idx[n_train + n_val:], "split"] = "known_test"
    df = df[OUT_COLS]

    df.to_csv(os.path.join(args.data, "dataset_all.csv"), index=False)
    for split in ["train", "val", "known_test", "unknown_test"]:
        df[df["split"] == split].to_csv(os.path.join(args.data, f"{split}.csv"), index=False)
    config = {"known": KNOWN, "unseen": UNSEEN, "seed": SEED, "train_frac": TRAIN_FRAC,
              "val_frac": VAL_FRAC, "test_frac": round(1 - TRAIN_FRAC - VAL_FRAC, 2),
              "n_known_per_class": args.n_known, "n_unseen_per_class": args.n_unseen,
              "max_per_signature": args.max_per_signature}
    with open(os.path.join(args.data, "dataset_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("\nSamples per category and split:")
    table = pd.crosstab(df["category"], df["split"], margins=True, margins_name="total")
    print(table[["train", "val", "known_test", "unknown_test", "total"]].to_string())

    print("\nChecks:")
    splits = ["train", "val", "known_test", "unknown_test"]
    ids = {s: set(df.loc[df["split"] == s, "object_id"]) for s in splits}
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            check(f"no object overlap between {a} and {b}", len(ids[a] & ids[b]) == 0)
    known_part = df[df["split"].isin(["train", "val", "known_test"])]
    check("train/val/known_test contain only known categories", known_part["category"].isin(KNOWN).all())
    check("unknown_test contains only unseen categories",
          df.loc[df["split"] == "unknown_test", "category"].isin(UNSEEN).all())
    for s in ["train", "val", "known_test"]:
        present = set(df.loc[df["split"] == s, "category"])
        check(f"every known category present in {s}", set(KNOWN) <= present)
    check("no duplicate images anywhere (md5)", df["image_md5"].is_unique)
    check("no duplicate object numbers anywhere", df["object_number"].is_unique)
    check("every image file exists", df["image_path"].map(os.path.exists).all())
    print(f"\nSaved {len(df)} samples to {args.data}/")


if __name__ == "__main__":
    main()
