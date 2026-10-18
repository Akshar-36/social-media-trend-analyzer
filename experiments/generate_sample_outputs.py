import os
import sys
import json
import jsonschema
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
from pipeline import RecommenderPipeline

# Load schemas and configurations
with open(os.path.join(PROJECT_ROOT, "contract", "output_schema.json"), "r", encoding="utf-8") as f:
    schema = json.load(f)

with open(os.path.join(PROJECT_ROOT, "artifacts", "v1.0", "fallback_medians.json"), "r", encoding="utf-8") as f:
    fallback_medians = json.load(f)

test_brands = [
    {
        "brand_name": "Lumiere Cosmetics",
        "business_description": "Cruelty-free, vegan makeup and skincare.",
        "industry": "beauty",
        "target_audience": "young adults interested in clean beauty",
        "timezone": "America/New_York"
    },
    {
        "brand_name": "Happy Tails",
        "business_description": "Organic dog treats and eco-friendly toys.",
        "industry": "pet",
        "target_audience": "dog owners who care about sustainability"
    },
    {
        "brand_name": "Tech Innovate",
        "business_description": "B2B SaaS software for HR management.",
        "industry": "software",
        "target_audience": "HR professionals"
    },
    {
        "brand_name": "Nomad Eats",
        "business_description": "Gourmet food truck serving fusion tacos.",
        "industry": "food",
        "target_audience": "foodies and festival goers"
    },
    {
        "brand_name": "FitLife Gear",
        "business_description": "High performance activewear and yoga mats.",
        "industry": "fitness",
        "target_audience": "yoga enthusiasts and gym goers"
    },
    {
        "brand_name": "Wanderlust Travels",
        "business_description": "Curated adventure travel packages.",
        "industry": "travel",
        "target_audience": "solo travelers and adventurers"
    },
    {
        "brand_name": "Cozy Home",
        "business_description": "Minimalist furniture and home decor.",
        "industry": "interior",
        "target_audience": "new homeowners"
    },
    {
        "brand_name": "Urban Style",
        "business_description": "Streetwear and limited edition sneakers.",
        "industry": "fashion",
        "target_audience": "hypebeasts and fashion enthusiasts"
    }
]

def assert_payload_identical(p1, p2, brand_name):
    """Verify that p1 and p2 are identical except for generated_at."""
    c1 = json.loads(json.dumps(p1))
    c2 = json.loads(json.dumps(p2))
    
    t1 = c1.pop("generated_at")
    t2 = c2.pop("generated_at")
    assert t1 and t2, f"{brand_name}: missing generated_at"
    
    # Check predicted engagement value with approx tolerance
    v1 = c1["predicted_engagement"].pop("value")
    v2 = c2["predicted_engagement"].pop("value")
    assert abs(v1 - v2) < 1e-6, f"{brand_name}: non-deterministic predicted ER {v1} vs {v2}"
    
    # Check all hashtag details float values
    for h1, h2 in zip(c1["hashtag_details"], c2["hashtag_details"]):
        assert h1["hashtag"] == h2["hashtag"]
        assert h1["trend_direction"] == h2["trend_direction"]
        assert abs(h1["relevance"] - h2["relevance"]) < 1e-6
        assert abs(h1["trend_score"] - h2["trend_score"]) < 1e-6
        assert abs(h1["score"] - h2["score"]) < 1e-6
        assert abs(h1["historical_engagement"] - h2["historical_engagement"]) < 1e-6
    
    # Pop float-sensitive lists to compare rest directly
    c1.pop("hashtag_details")
    c2.pop("hashtag_details")
    
    assert c1 == c2, f"{brand_name}: mismatch in static payload fields: {c1} != {c2}"

def run(write=False):
    pipeline = RecommenderPipeline()
    output_dir = os.path.join(PROJECT_ROOT, "contract", "sample_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Running {'WRITE & SYNC' if write else 'VALIDATION ONLY'} ---")
    
    results = {}
    validation_report = []
    
    for i, brand in enumerate(test_brands):
        bname = brand["brand_name"]
        print(f"Processing {i+1}/8: {bname}...")
        
        # Pass 1
        out1 = pipeline.recommend(brand)
        jsonschema.validate(instance=out1, schema=schema)
        
        # Pass 2 (Determinism Check)
        out2 = pipeline.recommend(brand)
        jsonschema.validate(instance=out2, schema=schema)
        assert_payload_identical(out1, out2, bname)
        
        # Validate direction constraints
        rec_topic = out1["recommended_topic"]
        trend_topics = out1["trending_topics"]
        details = {h["hashtag"].replace("#", ""): h["trend_direction"] for h in out1["hashtag_details"]}
        
        rec_dir = details.get(rec_topic, "unknown")
        assert rec_dir != "falling", f"Brand {bname}: recommended_topic {rec_topic} is falling!"
        
        for t in trend_topics:
            t_dir = details.get(t, "unknown")
            assert t_dir != "falling", f"Brand {bname}: trending_topic {t} is falling!"
            
        # Validate engagement prediction
        er = out1["predicted_engagement"]["value"]
        assert er > 0 and np.isfinite(er), f"Brand {bname}: invalid ER {er}"
        
        cat = out1["data_basis"]["category"]
        match_type = out1["data_basis"]["category_match"]
        fname = f"brand_{i+1}_{bname.lower().replace(' ', '_')}.json"
        
        if write:
            fpath = os.path.join(output_dir, fname)
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(out1, f, indent=2)
            print(f"  -> WROTE {fpath}")
            
            # Re-read from disk and validate
            with open(fpath, "r", encoding="utf-8") as f:
                disk_payload = json.load(f)
            jsonschema.validate(instance=disk_payload, schema=schema)
            
        results[fname] = out1
        validation_report.append({
            "brand": bname,
            "category": cat,
            "match": match_type,
            "recommended_topic": f"{rec_topic} ({rec_dir})",
            "trending_topics": trend_topics,
            "predicted_er": round(er, 4),
            "er_label": out1["predicted_engagement"]["label"],
            "content_type": out1["recommended_content_type"]["value"],
            "posting_time": f"{out1['recommended_posting_time']['day']} {out1['recommended_posting_time']['start']}-{out1['recommended_posting_time']['end']}"
        })
        
    print("\n--- Summary Validation Table ---")
    header = "| Brand | Category | Match | Rec Topic | Trending Topics | ER | Type | Post Time |"
    sep = "|---|---|---|---|---|---|---|---|"
    print(header)
    print(sep)
    for r in validation_report:
        tt_str = ", ".join(r["trending_topics"][:2])
        print(f"| {r['brand']} | {r['category']} | {r['match']} | {r['recommended_topic']} | {tt_str} | {r['predicted_er']} ({r['er_label']}) | {r['content_type']} | {r['posting_time']} |")
        
    print("\nAll 8 reference brands passed schema, determinism, and direction checks!")
    return results

if __name__ == "__main__":
    write_flag = "--write" in sys.argv
    run(write=write_flag)
