"""
download_data.py
================
Downloads the MovieLens 100K dataset from GroupLens Research.
Dataset: https://grouplens.org/datasets/movielens/100k/

Run: python download_data.py
"""

import urllib.request
import zipfile
import os
import shutil

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR    = "data"
ZIP_PATH    = os.path.join(DATA_DIR, "ml-100k.zip")

def download_movielens_100k():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 55)
    print("  Downloading MovieLens 100K Dataset")
    print("  Source: GroupLens Research (grouplens.org)")
    print("=" * 55)

    if os.path.exists(os.path.join(DATA_DIR, "ml-100k", "u.data")):
        print("\n[✓] Dataset already exists at data/ml-100k/")
        return

    print(f"\n[1] Downloading from {DATASET_URL} ...")
    urllib.request.urlretrieve(DATASET_URL, ZIP_PATH,
        reporthook=lambda b, bs, total:
            print(f"\r    Progress: {min(100, int(b*bs*100/total))}%", end="", flush=True))
    print("\n[✓] Download complete.")

    print("\n[2] Extracting archive ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)
    os.remove(ZIP_PATH)
    print("[✓] Extracted to data/ml-100k/")

    print("\n[3] Verifying files ...")
    required = ["u.data", "u.item", "u.user", "u.genre", "u.occupation"]
    base = os.path.join(DATA_DIR, "ml-100k")
    for f in required:
        size = os.path.getsize(os.path.join(base, f))
        print(f"    {f:<20} {size:>10,} bytes  ✓")

    print("\n" + "=" * 55)
    print("  Dataset Ready!")
    print("  Ratings  : 100,000")
    print("  Users    : 943")
    print("  Movies   : 1,682")
    print("  Run next : python recommender.py")
    print("=" * 55)

if __name__ == "__main__":
    download_movielens_100k()
