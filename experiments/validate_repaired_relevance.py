import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import joblib

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from pipeline import RecommenderPipeline
from relevance import predict_category, get_brand_text

ACTIVE_CATEGORIES = ['beauty', 'family', 'fashion', 'fitness', 'food', 'interior', 'pet', 'travel']
ALL_CLASSES = ACTIVE_CATEGORIES + ['all']

print("==================================================")
print("REPAIRED RELEVANCE PRODUCTION VALIDATION")
print("==================================================")

# 1. Artifact Integrity Check
print("\n--- 1. Artifact Integrity Check ---")
artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts", "v1.0")
artifact_files = sorted(os.listdir(artifacts_dir))
for af in artifact_files:
    fpath = os.path.join(artifacts_dir, af)
    sz = os.path.getsize(fpath)
    with open(fpath, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()[:16]
    print(f"  {af:35s} Size: {sz:10d} bytes | SHA256[:16]: {sha}")

# 2. Diagnostic Brands End-to-End Pipeline Verification
print("\n--- 2. Diagnostic Brands End-to-End Pipeline Trace ---")
pipeline = RecommenderPipeline()

diag_cases = [
    {
        "brand_name": "CampusWear",
        "business_description": "We sell trendy college apparel, hoodies, and sweatpants for students.",
        "industry": "fashion",
        "target_audience": "university students and young adults",
        "expected": "fashion"
    },
    {
        "brand_name": "FitForge",
        "business_description": "High-intensity functional training gear, barbells, and resistance bands.",
        "industry": "fitness equipment",
        "target_audience": "crossfit athletes and gym lifters",
        "expected": "fitness"
    },
    {
        "brand_name": "GlowTheory",
        "business_description": "Clean, vegan skincare essentials for radiant and healthy skin.",
        "industry": "beauty & skincare",
        "target_audience": "skincare enthusiasts seeking a natural glow",
        "expected": "beauty"
    },
    {
        "brand_name": "SpiceRoute",
        "business_description": "Artisanal organic spice blends and small-batch curry pastes.",
        "industry": "gourmet food & cooking",
        "target_audience": "home cooks and foodies",
        "expected": "food"
    },
    {
        "brand_name": "UrbanNest",
        "business_description": "Mid-century modern home decor, handcrafted ceramics, and modular storage.",
        "industry": "interior design & furniture",
        "target_audience": "apartment renters and interior designers",
        "expected": "interior"
    },
    {
        "brand_name": "PawPal",
        "business_description": "Hypoallergenic dog treats, durable chew toys, and ergonomic leashes.",
        "industry": "pet supplies",
        "target_audience": "dog owners and pet lovers",
        "expected": "pet"
    },
    {
        "brand_name": "WanderWeekends",
        "business_description": "Curated weekend road trip itineraries, glamping guides, and hiking gear rentals.",
        "industry": "travel & outdoor adventure",
        "target_audience": "weekend travelers and hikers",
        "expected": "travel"
    },
    {
        "brand_name": "CreatorHub",
        "business_description": "All-in-one link-in-bio, analytics, and invoicing platform for digital content creators.",
        "industry": "B2B SaaS / digital marketing",
        "target_audience": "social media creators and agencies",
        "expected": "all"
    }
]

diag_results = []
for c in diag_cases:
    inp = {
        "brand_name": c["brand_name"],
        "business_description": c["business_description"],
        "industry": c["industry"],
        "target_audience": c["target_audience"]
    }
    out = pipeline.recommend(inp)
    assert "error" not in out, f"Error in pipeline for {c['brand_name']}: {out}"
    cat_res = out["data_basis"]["category"]
    match_res = out["data_basis"]["category_match"]
    passed = (cat_res == c["expected"])
    diag_results.append({
        "Brand": c["brand_name"],
        "Expected": c["expected"],
        "Resolved Category": cat_res,
        "Match Type": match_res,
        "Status": "PASS" if passed else "FAIL",
        "Recommended Topic": out.get("recommended_topic"),
        "Hashtags Count": len(out.get("recommended_hashtags", []))
    })

df_diag = pd.DataFrame(diag_results)
print(df_diag.to_string(index=False))

# 3. Held-Out Relevance Benchmark Comparison (Baseline vs Repaired)
print("\n--- 3. Held-Out Relevance Benchmark (N = 1,080) ---")
test_path = os.path.join(PROJECT_ROOT, "experiments", "data", "relevance_held_out_test_set.json")
with open(test_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# Extract embeddings and ground truth
test_queries = [r['query_natural'] for r in test_data]
y_true = [r['expected_category'] if r['expected_category'] != 'out_of_domain' else 'all' for r in test_data]
in_mask = np.array([r['expected_category'] != 'out_of_domain' for r in test_data])
ood_mask = ~in_mask

model = pipeline.model
centroids = pipeline.centroids
single_centroid_matrix = np.stack([centroids[c] for c in ACTIVE_CATEGORIES])

test_embs = model.encode(test_queries, batch_size=64, show_progress_bar=False)
sim_matrix = cosine_similarity(test_embs, single_centroid_matrix)

top1_idx = np.argmax(sim_matrix, axis=1)
top1_sims = np.max(sim_matrix, axis=1)

masked_sim = sim_matrix.copy()
for i in range(len(sim_matrix)):
    masked_sim[i, top1_idx[i]] = -1.0
top2_sims = np.max(masked_sim, axis=1)
margins = top1_sims - top2_sims
top1_cats = [ACTIVE_CATEGORIES[i] for i in top1_idx]

# A. Previous Baseline Decision Rule: top1 >= tau (0.34308)
preds_baseline = []
for i in range(len(sim_matrix)):
    if top1_sims[i] >= 0.34308:
        preds_baseline.append(top1_cats[i])
    else:
        preds_baseline.append('all')
preds_baseline = np.array(preds_baseline)

# B. Repaired Production Rule: margin >= 0.010 AND top1 >= 0.250
preds_repaired = []
for i in range(len(sim_matrix)):
    if margins[i] >= 0.010 and top1_sims[i] >= 0.250:
        preds_repaired.append(top1_cats[i])
    else:
        preds_repaired.append('all')
preds_repaired = np.array(preds_repaired)

y_true_arr = np.array(y_true)

def compute_metrics(preds):
    in_true = y_true_arr[in_mask]
    in_pred = preds[in_mask]
    ood_true = y_true_arr[ood_mask]
    ood_pred = preds[ood_mask]
    
    in_correct = (in_pred == in_true).sum()
    in_fb = (in_pred == 'all').sum()
    n_in = len(in_true)
    
    overall_acc = (preds == y_true_arr).sum() / len(y_true_arr)
    in_acc = in_correct / n_in
    in_cov = (n_in - in_fb) / n_in
    in_prec = in_correct / (n_in - in_fb) if (n_in - in_fb) > 0 else 0.0
    false_fb = in_fb / n_in
    ood_rej = (ood_pred == 'all').sum() / len(ood_true)
    macro_f1 = f1_score(y_true_arr, preds, labels=ALL_CLASSES, average='macro', zero_division=0)
    
    # Fashion recall
    f_mask = (y_true_arr == 'fashion')
    f_recall = (preds[f_mask] == 'fashion').sum() / f_mask.sum()
    
    return {
        "Overall Acc": overall_acc,
        "In-Domain Acc": in_acc,
        "In-Domain Prec": in_prec,
        "Coverage": in_cov,
        "False Fallback": false_fb,
        "OOD Rejection": ood_rej,
        "Macro-F1": macro_f1,
        "Fashion Recall": f_recall
    }

m_base = compute_metrics(preds_baseline)
m_repair = compute_metrics(preds_repaired)

comparison = [
    {"Metric": "Overall Accuracy", "Baseline (tau=0.343)": f"{m_base['Overall Acc']*100:.2f}%", "Repaired (margin>=0.01, floor>=0.25)": f"{m_repair['Overall Acc']*100:.2f}%", "Delta": f"{(m_repair['Overall Acc'] - m_base['Overall Acc'])*100:+.2f}%"},
    {"Metric": "In-Domain Accuracy", "Baseline (tau=0.343)": f"{m_base['In-Domain Acc']*100:.2f}%", "Repaired (margin>=0.01, floor>=0.25)": f"{m_repair['In-Domain Acc']*100:.2f}%", "Delta": f"{(m_repair['In-Domain Acc'] - m_base['In-Domain Acc'])*100:+.2f}%"},
    {"Metric": "In-Domain Precision", "Baseline (tau=0.343)": f"{m_base['In-Domain Prec']*100:.2f}%", "Repaired (margin>=0.01, floor>=0.25)": f"{m_repair['In-Domain Prec']*100:.2f}%", "Delta": f"{(m_repair['In-Domain Prec'] - m_base['In-Domain Prec'])*100:+.2f}%"},
    {"Metric": "Coverage", "Baseline (tau=0.343)": f"{m_base['Coverage']*100:.2f}%", "Repaired (margin>=0.01, floor>=0.25)": f"{m_repair['Coverage']*100:.2f}%", "Delta": f"{(m_repair['Coverage'] - m_base['Coverage'])*100:+.2f}%"},
    {"Metric": "False Fallback", "Baseline (tau=0.343)": f"{m_base['False Fallback']*100:.2f}%", "Repaired (margin>=0.01, floor>=0.25)": f"{m_repair['False Fallback']*100:.2f}%", "Delta": f"{(m_repair['False Fallback'] - m_base['False Fallback'])*100:+.2f}%"},
    {"Metric": "OOD Rejection", "Baseline (tau=0.343)": f"{m_base['OOD Rejection']*100:.2f}%", "Repaired (margin>=0.01, floor>=0.25)": f"{m_repair['OOD Rejection']*100:.2f}%", "Delta": f"{(m_repair['OOD Rejection'] - m_base['OOD Rejection'])*100:+.2f}%"},
    {"Metric": "Macro-F1", "Baseline (tau=0.343)": f"{m_base['Macro-F1']:.4f}", "Repaired (margin>=0.01, floor>=0.25)": f"{m_repair['Macro-F1']:.4f}", "Delta": f"{(m_repair['Macro-F1'] - m_base['Macro-F1']):+.4f}"},
    {"Metric": "Fashion Recall", "Baseline (tau=0.343)": f"{m_base['Fashion Recall']*100:.2f}%", "Repaired (margin>=0.01, floor>=0.25)": f"{m_repair['Fashion Recall']*100:.2f}%", "Delta": f"{(m_repair['Fashion Recall'] - m_base['Fashion Recall'])*100:+.2f}%"},
]

df_comp = pd.DataFrame(comparison)
print(df_comp.to_string(index=False))

# 4. Per-Category Breakdown (Baseline vs Repaired)
print("\n--- 4. Per-Category Precision, Recall, F1 Breakdown ---")
rep_base = classification_report(y_true_arr, preds_baseline, labels=ALL_CLASSES, output_dict=True, zero_division=0)
rep_repair = classification_report(y_true_arr, preds_repaired, labels=ALL_CLASSES, output_dict=True, zero_division=0)

cat_rows = []
for c in ALL_CLASSES:
    cat_rows.append({
        "Category": c,
        "Base_Prec": round(rep_base[c]['precision'], 3),
        "Base_Rec": round(rep_base[c]['recall'], 3),
        "Base_F1": round(rep_base[c]['f1-score'], 3),
        "Repair_Prec": round(rep_repair[c]['precision'], 3),
        "Repair_Rec": round(rep_repair[c]['recall'], 3),
        "Repair_F1": round(rep_repair[c]['f1-score'], 3),
        "Rec_Delta": f"{(rep_repair[c]['recall'] - rep_base[c]['recall'])*100:+.1f}%"
    })
df_cat = pd.DataFrame(cat_rows)
print(df_cat.to_string(index=False))
