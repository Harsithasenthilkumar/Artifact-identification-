import argparse
import os
import sys

import pandas as pd
import requests

# Official MET Open Access metadata CSV (repo: https://github.com/metmuseum/openaccess)
MET_CSV_URL = "https://media.githubusercontent.com/media/metmuseum/openaccess/master/MetObjects.csv"

KEY_COLS = ["Object ID", "Object Number", "Is Public Domain", "Department", "Object Name",
            "Title", "Classification", "Culture", "Period", "Dynasty", "Medium",
            "Object Begin Date", "Object End Date", "Country", "Region", "Geography Type",
            "Dimensions", "Excavation", "AccessionYear"]

# Departments that mostly hold archaeological / ancient material.
# Matched as lowercase substrings so small naming differences still work. Edit if needed.
ARCH_KEYWORDS = ["egyptian", "greek and roman", "near eastern", "west asian",
                 "africa, oceania", "asian art", "islamic", "medieval", "cloisters"]

META_COLS = ["Culture", "Period", "Medium", "Object Begin Date", "Country", "Region", "Dimensions"]
SIGNATURE_COLS = ["Department", "category", "Title", "Culture", "Period", "Medium", "Object Begin Date"]


def download_csv(csv_path):
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 1_000_000:
        print(f"Found {csv_path} ({os.path.getsize(csv_path) / 1e6:.0f} MB), skipping download.")
        return
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    print(f"Downloading MET Open Access metadata (~318 MB) from:\n{MET_CSV_URL}")
    tmp = csv_path + ".part"
    try:
        with requests.get(MET_CSV_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            done = 0
            with open(tmp, "wb") as f:
                for block in r.iter_content(chunk_size=1 << 20):
                    f.write(block)
                    done += len(block)
                    print(f"\r  {done / 1e6:.0f} MB", end="", flush=True)
        print()
    except requests.RequestException as e:
        sys.exit(f"\nDownload failed: {e}\n"
                 f"Download it manually from {MET_CSV_URL} and save it as {csv_path}")
    os.replace(tmp, csv_path)
    if os.path.getsize(csv_path) < 1_000_000:
        sys.exit(f"{csv_path} is too small - it is probably a Git LFS pointer, not the real data. "
                 f"Delete it and download again from {MET_CSV_URL}")


def load(csv_path):
    header = pd.read_csv(csv_path, nrows=0, encoding="utf-8-sig").columns.tolist()
    use = [c for c in KEY_COLS if c in header]
    missing = [c for c in KEY_COLS if c not in header]

    fill_counts = pd.Series(0, index=header)
    total = 0
    parts = []
    for chunk in pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig", chunksize=100_000):
        chunk = chunk.apply(lambda s: s.str.strip())
        chunk = chunk.replace("", pd.NA)
        fill_counts += chunk.notna().sum()
        total += len(chunk)
        parts.append(chunk[use])
    df = pd.concat(parts, ignore_index=True)
    fill = (fill_counts / total).rename("fill_rate").sort_values(ascending=False)
    return df, header, missing, fill


def category_stats(pool):
    rows = []
    for name, g in pool.groupby("category"):
        dept_counts = g["Department"].value_counts()
        title_has_name = g["Title"].fillna("").str.lower().str.contains(name, regex=False).mean()
        dup_rows = g.duplicated(subset=SIGNATURE_COLS, keep=False).sum()
        row = {"category": name, "count": len(g),
               "n_departments": g["Department"].nunique(),
               "top_department": dept_counts.index[0],
               "top_dept_share": round(dept_counts.iloc[0] / len(g), 2),
               "title_contains_label": round(title_has_name, 2),
               "metadata_duplicate_rows": int(dup_rows)}
        for c in META_COLS:
            if c in g.columns:
                row[f"fill_{c}"] = round(g[c].notna().mean(), 2)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/MetObjects.csv")
    p.add_argument("--out", default="outputs/phase1")
    p.add_argument("--min_count", type=int, default=300)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    download_csv(args.csv)

    df, header, missing, fill = load(args.csv)
    print(f"Total records inspected: {len(df)}")
    print(f"Columns in file: {len(header)}")
    print("Columns:", header)
    if missing:
        print("WARNING - expected columns not found:", missing)
    fill.to_csv(f"{args.out}/column_fill_rates.csv")
    print("\nFill rate of every column (fraction non-empty):")
    print(fill.round(3).to_string())

    print("\nDuplicate Object IDs:", df["Object ID"].duplicated().sum())
    print("Duplicate Object Numbers:", df["Object Number"].duplicated().sum())

    df["public_domain"] = df["Is Public Domain"].str.lower().eq("true")
    print(f"\nPublic-domain records: {df['public_domain'].sum()} of {len(df)}")

    dept = df.groupby("Department")["public_domain"].agg(total="size", public_domain="sum")
    dept = dept.sort_values("public_domain", ascending=False)
    dept.to_csv(f"{args.out}/department_counts.csv")
    print("\nRecords per department:")
    print(dept.to_string())

    dep_lower = df["Department"].fillna("").str.lower()
    is_arch = dep_lower.apply(lambda d: any(k in d for k in ARCH_KEYWORDS))
    print("\nDepartments treated as archaeological:", sorted(df.loc[is_arch, "Department"].unique()))

    arch = df[is_arch & df["public_domain"] & df["Object Name"].notna()].copy()
    arch["category"] = arch["Object Name"].str.lower().str.strip()
    print(f"Public-domain archaeological records with an Object Name: {len(arch)}")
    print(f"Distinct object names: {arch['category'].nunique()}")

    names = arch["category"].value_counts()
    names.to_csv(f"{args.out}/object_name_counts.csv")
    print("\nTop 40 object names (archaeological, public domain):")
    print(names.head(40).to_string())

    if "Classification" in arch.columns:
        print("\nTop 20 Classification values (for reference only):")
        print(arch["Classification"].value_counts().head(20).to_string())

    candidates = names[names >= args.min_count].index
    pool = arch[arch["category"].isin(candidates)]
    stats = category_stats(pool)
    stats.to_csv(f"{args.out}/candidate_categories.csv", index=False)
    pool.to_csv(f"{args.out}/candidate_pool.csv", index=False)

    pd.set_option("display.width", 250)
    print(f"\nCandidate categories (>= {args.min_count} public-domain records): {len(stats)}")
    print(stats.to_string(index=False))
    print(f"\nRows in candidate pool: {len(pool)}")
    print(f"Saved reports to {args.out}/")


if __name__ == "__main__":
    main()