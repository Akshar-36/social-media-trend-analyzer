import os
import sys
import json
import hashlib
from collections import Counter
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

FILES = {
    "calibration": os.path.join(DATA_DIR, "relevance_calibration_set.json"),
    "test": os.path.join(DATA_DIR, "relevance_held_out_test_set.json"),
    "robustness": os.path.join(DATA_DIR, "relevance_robustness_set.json"),
    "metadata": os.path.join(DATA_DIR, "relevance_curated_metadata.json")
}

print("=== STEP 1: VERIFYING DATASET INTEGRITY ===")

loaded_data = {}
file_hashes = {}

for name, path in FILES.items():
    if not os.path.exists(path):
        print(f"ERROR: Missing file {path}")
        sys.exit(1)
        
    with open(path, 'rb') as f:
        content = f.read()
        sha256 = hashlib.sha256(content).hexdigest()
        file_hashes[name] = sha256
        
    with open(path, 'r', encoding='utf-8') as f:
        loaded_data[name] = json.load(f)
        
    print(f"File: {os.path.basename(path)}")
    print(f"  Size: {len(content):,} bytes")
    print(f"  SHA-256: {sha256}")
    if isinstance(loaded_data[name], list):
        print(f"  Records: {len(loaded_data[name])}")
    else:
        print(f"  Keys: {list(loaded_data[name].keys())}")
    print()

# Check record counts
calib = loaded_data['calibration']
test = loaded_data['test']
robust = loaded_data['robustness']

assert len(calib) == 720, f"Expected 720 calib records, got {len(calib)}"
assert len(test) == 1080, f"Expected 1080 test records, got {len(test)}"
assert len(robust) == 1260, f"Expected 1260 robustness records, got {len(robust)}"
print("Record counts verified: Calibration=720, Test=1080, Robustness=1260.")

# Check Category Balance
calib_cats = Counter(r['expected_category'] for r in calib)
test_cats = Counter(r['expected_category'] for r in test)

print("\n--- Category Balance Verification ---")
print("Calibration set category counts (Target = 80 per category):")
for cat, count in sorted(calib_cats.items()):
    print(f"  {cat:15s}: {count}")
    assert count == 80, f"Calibration category {cat} count {count} != 80"

print("\nHeld-out test set category counts (Target = 120 per category):")
for cat, count in sorted(test_cats.items()):
    print(f"  {cat:15s}: {count}")
    assert count == 120, f"Test category {cat} count {count} != 120"
print("Category balance strictly verified (perfect uniform balance across all categories).")

# Check Account Isolation (Zero Leakage)
calib_users = set(r['username_hash'] for r in calib)
test_users = set(r['username_hash'] for r in test)
user_overlap = calib_users.intersection(test_users)

print("\n--- Account Isolation Verification ---")
print(f"Unique accounts in Calibration: {len(calib_users)}")
print(f"Unique accounts in Test:        {len(test_users)}")
print(f"Account overlap:                {len(user_overlap)}")
assert len(user_overlap) == 0, f"CRITICAL LEAKAGE: {len(user_overlap)} accounts overlap!"
print("Account separation strictly verified: 0 overlapping accounts.")

# Check Post ID Isolation
calib_posts = set(r['post_id'] for r in calib)
test_posts = set(r['post_id'] for r in test)
post_overlap = calib_posts.intersection(test_posts)
assert len(post_overlap) == 0, f"CRITICAL LEAKAGE: {len(post_overlap)} posts overlap!"
print("Post ID separation strictly verified: 0 overlapping posts.")

# Check Query Variants
for split_name, dataset in [("Calibration", calib), ("Test", test)]:
    missing_natural = sum(1 for r in dataset if not r.get('query_natural'))
    missing_masked = sum(1 for r in dataset if not r.get('query_masked'))
    missing_short = sum(1 for r in dataset if not r.get('query_short'))
    assert missing_natural == 0 and missing_masked == 0 and missing_short == 0, f"Missing query variants in {split_name}"
print("Query variants verified: query_natural, query_masked, query_short 100% complete.")

# Check Category Name Leakage
calib_leakage = sum(1 for r in calib if r['category_name_leakage']) / len(calib) * 100
test_leakage = sum(1 for r in test if r['category_name_leakage']) / len(test) * 100
print(f"\n--- Category Name Leakage Statistics ---")
print(f"Calibration set: {calib_leakage:.2f}% of queries contain literal category name")
print(f"Held-out test set: {test_leakage:.2f}% of queries contain literal category name")

# Robustness set breakdown
robust_types = Counter(r['robustness_type'] for r in robust)
print(f"\n--- Robustness Set Breakdown ---")
for r_type, count in robust_types.most_common():
    print(f"  {r_type:25s}: {count}")

print("\nALL INTEGRITY CHECKS PASSED SUCCESSFULLY.")
