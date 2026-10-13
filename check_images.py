import argparse
import os
import re
import time

import pandas as pd
import requests

API = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{}"


def fetch(session, object_id):
    for attempt in range(3):
        try:
            r = session.get(API.format(object_id), timeout=20)
            if r.status_code == 200:
                return "ok", r.json()
            if r.status_code == 404:
                return "not_found", None
            status = f"http_{r.status_code}"
        except requests.RequestException as e:
            status = f"error_{type(e).__name__}"
        time.sleep(1 + attempt)
    return status, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default="outputs/phase1/candidate_pool.csv")
    p.add_argument("--out", default="outputs/phase1")
    p.add_argument("--per_class", type=int, default=40)
    p.add_argument("--save_thumbs", type=int, default=6)
    p.add_argument("--delay", type=float, default=0.1)
    args = p.parse_args()

    pool = pd.read_csv(args.pool, dtype=str)
    session = requests.Session()
    session.headers["User-Agent"] = "student-project-met-inspection/1.0"

    raw = []
    for name, g in pool.groupby("category"):
        sample = g.sample(n=min(args.per_class, len(g)), random_state=42)
        thumb_dir = os.path.join(args.out, "thumbs", re.sub(r"[^a-z0-9]+", "_", name))
        saved = 0
        for oid in sample["Object ID"]:
            status, data = fetch(session, oid)
            img = data.get("primaryImageSmall", "") if data else ""
            raw.append({"category": name, "Object ID": oid, "status": status,
                        "has_image": bool(img),
                        "n_additional_images": len(data.get("additionalImages", [])) if data else 0})
            if img and saved < args.save_thumbs:
                try:
                    os.makedirs(thumb_dir, exist_ok=True)
                    content = session.get(img, timeout=30).content
                    with open(os.path.join(thumb_dir, f"{oid}.jpg"), "wb") as f:
                        f.write(content)
                    saved += 1
                except requests.RequestException:
                    pass
            time.sleep(args.delay)
        ok = sum(r["status"] == "ok" for r in raw if r["category"] == name)
        print(f"{name:25s} checked {len(sample):3d}  api_ok {ok:3d}")

    raw = pd.DataFrame(raw)
    raw.to_csv(f"{args.out}/image_check_raw.csv", index=False)

    if (raw["status"] != "ok").all():
        print("\nEvery API request failed. Status codes seen:", raw["status"].value_counts().to_dict())
        print("Check your internet connection or try again later.")
        return

    counts = pool["category"].value_counts()
    ok = raw[raw["status"] == "ok"]
    summary = ok.groupby("category").agg(api_ok=("has_image", "size"), with_image=("has_image", "sum"))
    summary["image_rate"] = (summary["with_image"] / summary["api_ok"]).round(2)
    summary["pool_count"] = counts.reindex(summary.index)
    summary["est_usable_images"] = (summary["pool_count"] * summary["image_rate"]).round().astype(int)
    summary = summary.sort_values("est_usable_images", ascending=False)
    summary.to_csv(f"{args.out}/image_availability.csv")

    pd.set_option("display.width", 200)
    print("\nImage availability (estimated from a random sample per category):")
    print(summary.to_string())
    print("\nNon-ok API statuses:", raw.loc[raw["status"] != "ok", "status"].value_counts().to_dict())
    print(f"Thumbnails saved under {args.out}/thumbs/ - open them and check image quality.")


if __name__ == "__main__":
    main()
