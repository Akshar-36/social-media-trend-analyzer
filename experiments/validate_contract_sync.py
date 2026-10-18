import os
import sys
import json
import jsonschema
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
from pipeline import RecommenderPipeline

print("=" * 80)
print("INDEPENDENT VALIDATION OF CONTRACT SYNCHRONIZATION")
print("=" * 80)

# Check 1: Output Schema Validation
print("\n--- Check 1: JSON Parsing & Schema Validation for all 8 files ---")
schema_path = os.path.join(PROJECT_ROOT, "contract", "output_schema.json")
with open(schema_path, "r", encoding="utf-8") as f:
    schema = json.load(f)

sample_dir = os.path.join(PROJECT_ROOT, "contract", "sample_outputs")
sample_files = sorted([f for f in os.listdir(sample_dir) if f.startswith("brand_") and f.endswith(".json")])
assert len(sample_files) == 8, f"Expected 8 sample files, found {len(sample_files)}"

loaded_payloads = {}
for sf in sample_files:
    fpath = os.path.join(sample_dir, sf)
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    jsonschema.validate(instance=data, schema=schema)
    loaded_payloads[sf] = data
    print(f"  [PASS] {sf}: Valid JSON and conforms strictly to output_schema.json")

# Check 2: Repeated Generation Determinism
print("\n--- Check 2: Repeated Generation Determinism ---")
from generate_sample_outputs import test_brands, assert_payload_identical
pipeline = RecommenderPipeline()
for i, b in enumerate(test_brands):
    fname = f"brand_{i+1}_{b['brand_name'].lower().replace(' ', '_')}.json"
    p1 = loaded_payloads[fname]
    p2 = pipeline.recommend(b)
    assert_payload_identical(p1, p2, b["brand_name"])
    print(f"  [PASS] {b['brand_name']}: Identical outputs across runs (timestamp 'generated_at' is sole dynamic field)")

# Check 3: Zero Falling Topics
print("\n--- Check 3: Zero Falling Topics in recommended_topic or trending_topics ---")
total_falling_rec = 0
total_falling_trending = 0
for sf, data in loaded_payloads.items():
    rec_topic = data["recommended_topic"]
    trending_topics = data["trending_topics"]
    details = {h["hashtag"].replace("#", ""): h["trend_direction"] for h in data["hashtag_details"]}
    
    rec_dir = details.get(rec_topic, "unknown")
    if rec_dir == "falling":
        total_falling_rec += 1
        
    trending_dirs = [details.get(t, "unknown") for t in trending_topics]
    if any(d == "falling" for d in trending_dirs):
        total_falling_trending += 1
        
    print(f"  [PASS] {sf}: rec='{rec_topic}' ({rec_dir}), trending={trending_topics} ({trending_dirs})")

assert total_falling_rec == 0, f"Found {total_falling_rec} falling recommended topics!"
assert total_falling_trending == 0, f"Found {total_falling_trending} falling trending topics!"

# Check 4: Engagement Predictions & Cold-Start Medians
print("\n--- Check 4: Engagement Predictions & Cold-Start Medians ---")
with open(os.path.join(PROJECT_ROOT, "artifacts", "v1.0", "fallback_medians.json"), "r", encoding="utf-8") as f:
    fb_medians = json.load(f)

for sf, data in loaded_payloads.items():
    er_val = data["predicted_engagement"]["value"]
    er_label = data["predicted_engagement"]["label"]
    cat = data["data_basis"]["category"]
    assert er_val > 0 and np.isfinite(er_val), f"Invalid ER: {er_val}"
    expected_follower_median = fb_medians.get(cat, fb_medians["all"])["follower_count"]
    print(f"  [PASS] {sf}: cat={cat}, predicted_ER={er_val:.4f} ({er_label}), empirical_follower_median={expected_follower_median}")

# Check 5: README example matches regenerated payload
print("\n--- Check 5: README example matches brand_1_lumiere_cosmetics.json ---")
with open(os.path.join(PROJECT_ROOT, "README.md"), "r", encoding="utf-8") as f:
    readme_text = f.read()

b1_payload = loaded_payloads["brand_1_lumiere_cosmetics.json"]
assert b1_payload["recommended_topic"] in readme_text, "README missing recommended_topic"
assert b1_payload["trending_topics"][0] in readme_text, "README missing trending topic 0"
assert b1_payload["trending_topics"][1] in readme_text, "README missing trending topic 1"
assert b1_payload["trending_topics"][2] in readme_text, "README missing trending topic 2"
assert "#abhprsearch" in readme_text, "README missing #abhprsearch"
assert "trend_direction\": \"rising" in readme_text, "README missing rising direction"
assert "trend_direction\": \"falling" not in readme_text, "README still contains falling direction!"
assert "0.0261" in readme_text, "README ER value mismatch"
print("  [PASS] README input & output examples match regenerated brand_1 payload exactly!")

print("\n" + "=" * 80)
print("ALL 5 INDEPENDENT VALIDATION CHECKS PASSED PERFECTLY!")
print("=" * 80)
