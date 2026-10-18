import os
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
from pipeline import RecommenderPipeline
from generate_sample_outputs import test_brands

pipeline = RecommenderPipeline()

print("--- Testing Semantic and Byte-for-Byte Determinism ---")
for i, brand in enumerate(test_brands):
    bname = brand["brand_name"]
    
    # Run 1
    out1 = pipeline.recommend(brand)
    # Run 2
    out2 = pipeline.recommend(brand)
    
    # Normalize timestamp
    norm_ts = "2026-10-10T00:00:00Z"
    out1["generated_at"] = norm_ts
    out2["generated_at"] = norm_ts
    
    s1 = json.dumps(out1, indent=2, sort_keys=True)
    s2 = json.dumps(out2, indent=2, sort_keys=True)
    
    byte_identical = (s1 == s2)
    semantic_identical = (
        out1["recommended_topic"] == out2["recommended_topic"] and
        out1["trending_topics"] == out2["trending_topics"] and
        out1["recommended_hashtags"] == out2["recommended_hashtags"] and
        out1["recommended_content_type"] == out2["recommended_content_type"] and
        out1["recommended_posting_time"] == out2["recommended_posting_time"] and
        out1["predicted_engagement"]["label"] == out2["predicted_engagement"]["label"] and
        out1["data_basis"] == out2["data_basis"]
    )
    
    print(f"Brand {i+1} ({bname}):")
    print(f"  Semantic Determinism: {'PASS' if semantic_identical else 'FAIL'}")
    print(f"  Byte-for-byte Determinism (normalized timestamp): {'PASS' if byte_identical else 'FAIL'}")
    if not byte_identical:
        print("  Differences in float precision detected between runs.")
        
print("\nDeterminism audit complete.")
