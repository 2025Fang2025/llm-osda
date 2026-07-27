"""Step 02: Download Amazon Product Metadata (parquet) and build ads_corpus.jsonl.

Goal: 100 real ads spanning 50 distinct sub-categories (2 per sub-category).
Downloads first parquet shard from multiple Amazon major categories.
Uses each product's own `main_category` field as the fine-grained category.

Usage:
    cd data_pipeline
    python3 scripts/02_make_ads_corpus.py
"""

import json
import random
import argparse
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path

from utils_io import write_jsonl

HF_PARQUET_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023"
    "/resolve/main/raw_meta_{cat}/full-00000-of-{shards}.parquet"
)

# All available categories with their shard counts, ordered by file size (small first)
CATEGORIES_INFO = [
    ("All_Beauty",                 "00001"),   #  60 MB
    ("Musical_Instruments",        "00002"),   # 124 MB
    ("Handmade_Products",          "00001"),   # 132 MB
    ("Cell_Phones_and_Accessories","00007"),   # 205 MB
    ("Toys_and_Games",             "00005"),   # 218 MB
    ("Arts_Crafts_and_Sewing",     "00004"),   # 220 MB
    ("Electronics",                "00010"),   # 220 MB
    ("Industrial_and_Scientific",  "00002"),   # 228 MB
    # Gift_Cards skipped (0 MB, no useful product data)
]


def download_parquet_and_extract(cat, shards_str, max_rows=8000):
    """Download first parquet shard, read up to max_rows, return parsed items."""
    import pyarrow.parquet as pq

    url = HF_PARQUET_URL.format(cat=cat, shards=shards_str)
    print("  GET {} ...".format(url))

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()
    except Exception as e:
        print("  [skip] Failed: {}".format(e))
        return []

    print("  Downloaded {:.1f} MB".format(len(data) / 1e6))

    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    tmp.write(data)
    tmp.close()

    try:
        table = pq.read_table(tmp.name)
        df = table.to_pandas()
    except Exception as e:
        print("  [skip] Parquet parse failed: {}".format(e))
        return []
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    print("  Total rows in shard: {}".format(len(df)))
    if len(df) > max_rows:
        df = df.head(max_rows)

    # Print available columns for debugging (first category only)
    items = []
    for _, row in df.iterrows():
        title = str(row.get("title") or "").strip()

        # Handle description (may be numpy array, list, or string)
        desc_raw = row.get("description")
        if desc_raw is None:
            desc = ""
        elif hasattr(desc_raw, '__len__') and not isinstance(desc_raw, str):
            desc = " ".join([str(d) for d in desc_raw if d])
        else:
            desc = str(desc_raw)
        desc = desc.strip()[:200]

        # Handle price
        price_val = row.get("price")
        price = 0.0
        if price_val is not None:
            try:
                price = float(str(price_val).replace("$", "").replace(",", "").strip())
            except (ValueError, TypeError):
                price = 0.0

        # Use main_category as the fine-grained sub-category
        main_cat_val = row.get("main_category")
        if main_cat_val is None or str(main_cat_val).strip() in ("", "nan", "None"):
            sub_cat = cat.replace("_", " ")
        else:
            sub_cat = str(main_cat_val).strip()

        if not title or price <= 0 or len(desc) < 10:
            continue

        items.append({
            "major_category": cat,
            "sub_category": sub_cat,
            "title": title,
            "short_description": desc,
            "price": round(price, 2),
            "asin": str(row.get("parent_asin") or ""),
        })

    unique_sub = len(set(it["sub_category"] for it in items))
    print("  {}: valid={}, unique sub-categories={}".format(cat, len(items), unique_sub))
    return items


def select_diverse_ads(all_items, n_categories=50, per_category=2, seed=42):
    """Group by sub_category, pick per_category items from top n_categories."""
    random.seed(seed)

    by_cat = defaultdict(list)
    for item in all_items:
        by_cat[item["sub_category"]].append(item)

    # Filter out generic/empty categories
    skip = {"", "N/A", "Unknown", "All Departments", "nan", "None"}
    valid_cats = {k: v for k, v in by_cat.items() if k not in skip and len(v) >= 1}

    # Sort by count descending (prefer categories with more products = more representative)
    sorted_cats = sorted(valid_cats.keys(), key=lambda c: len(valid_cats[c]), reverse=True)

    selected = []
    used_cats = []

    for cat_name in sorted_cats:
        if len(used_cats) >= n_categories:
            break
        pool = valid_cats[cat_name]
        picks = random.sample(pool, min(per_category, len(pool)))
        selected.extend(picks)
        used_cats.append(cat_name)

    print("\nSelected {} ads from {} distinct sub-categories".format(
        len(selected), len(used_cats)))
    return selected, used_cats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-categories", type=int, default=50,
                        help="Target number of distinct categories")
    parser.add_argument("--per-category", type=int, default=2,
                        help="Ads per category")
    args = parser.parse_args()

    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "pipeline.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    seed = cfg["project"]["seed"]

    target_ads = args.n_categories * args.per_category
    print("=" * 60)
    print("Target: {} ads across {} categories".format(target_ads, args.n_categories))
    print("Downloading from {} Amazon major categories".format(len(CATEGORIES_INFO)))
    print("=" * 60)

    all_items = []
    for cat, shards in CATEGORIES_INFO:
        print("\n--- {} (shards={}) ---".format(cat, shards))
        items = download_parquet_and_extract(cat, shards)
        all_items.extend(items)

        unique_cats = len(set(it["sub_category"] for it in all_items))
        print("  Running total: {} items, {} unique sub-categories".format(
            len(all_items), unique_cats))

        # Stop early if we have enough diversity
        if unique_cats >= args.n_categories * 2:
            print("  >>> Enough diversity! Stopping download early.")
            break

    selected, used_cats = select_diverse_ads(
        all_items, args.n_categories, args.per_category, seed)

    # Assign ad_id and clean up fields
    for idx, item in enumerate(selected):
        item["ad_id"] = "A_{:04d}".format(idx)
        item["category"] = item.pop("sub_category")
        item.pop("major_category", None)

    out_path = Path(__file__).resolve().parents[1] / "data" / "ads_corpus.jsonl"
    write_jsonl(out_path, selected)
    print("\nWrote {} ads -> {}".format(len(selected), out_path))

    # Save category list
    cat_doc_path = Path(__file__).resolve().parents[1] / "data" / "ads_categories.json"
    cat_doc_path.write_text(
        json.dumps(used_cats, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Category list ({}) -> {}".format(len(used_cats), cat_doc_path))

    print("\nSample ads:")
    for item in selected[:5]:
        print("  [{}] {} | {} | ${}".format(
            item["ad_id"], item["category"][:30], item["title"][:40], item["price"]))

    # Summary stats
    cats_count = defaultdict(int)
    for item in selected:
        cats_count[item["category"]] += 1
    print("\nCategory distribution:")
    for c, n in sorted(cats_count.items()):
        print("  {} -> {} ads".format(c, n))


if __name__ == "__main__":
    main()
