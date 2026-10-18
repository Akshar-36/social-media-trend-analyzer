import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "experiments", "data")
TEST_PATH = os.path.join(DATA_DIR, "relevance_held_out_test_set.json")
ROBUST_PATH = os.path.join(DATA_DIR, "relevance_robustness_set.json")
FROZEN_PATH = os.path.join(DATA_DIR, "frozen_relevance_parameters.json")
CENTROIDS_PATH = os.path.join(PROJECT_ROOT, "artifacts", "v1.0", "category_centroids.pkl")
TRENDS_PATH = os.path.join(PROJECT_ROOT, "artifacts", "v1.0", "trends.parquet")
H_EMB_PATH = os.path.join(PROJECT_ROOT, "artifacts", "v1.0", "hashtag_embeddings.parquet")

ACTIVE_CATEGORIES = ['beauty', 'family', 'fashion', 'fitness', 'food', 'interior', 'pet', 'travel']
ALL_CLASSES = ACTIVE_CATEGORIES + ['all']

# Load frozen parameters
with open(FROZEN_PATH, 'r', encoding='utf-8') as f:
    frozen_cfg = json.load(f)

FROZEN_MARGIN = frozen_cfg['margin']
FROZEN_FLOOR = frozen_cfg['floor']
BASELINE_TAU = 0.34308

print("=== LOADED FROZEN PARAMETERS ===")
print(f"Frozen Margin: {FROZEN_MARGIN}")
print(f"Frozen Floor:  {FROZEN_FLOOR}")
print(f"Baseline Tau:  {BASELINE_TAU}")

# Load test and robustness datasets
with open(TEST_PATH, 'r', encoding='utf-8') as f:
    test_data = json.load(f)
with open(ROBUST_PATH, 'r', encoding='utf-8') as f:
    robust_data = json.load(f)

# Load centroids
with open(CENTROIDS_PATH, 'rb') as f:
    centroids = pickle.load(f)
centroid_matrix = np.stack([centroids[c] for c in ACTIVE_CATEGORIES])

# Load trends for TF-IDF baseline
trends_df = pd.read_parquet(TRENDS_PATH)
h_emb_df = pd.read_parquet(H_EMB_PATH)

# Build TF-IDF representations
import re
def split_hashtag(ht):
    s = ht.lstrip('#')
    words = re.findall(r'[A-Za-z]+|\d+', s)
    return " ".join([w.lower() for w in words])

cat_pool_texts = {}
for cat in ACTIVE_CATEGORIES:
    pool_hts = trends_df[trends_df['pool'] == cat]['hashtag'].tolist()
    tokens = [split_hashtag(h) for h in pool_hts]
    cat_pool_texts[cat] = " ".join(tokens)

tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1, sublinear_tf=True)
tfidf_centroid_matrix = tfidf_vec.fit_transform([cat_pool_texts[c] for c in ACTIVE_CATEGORIES])

# Load MiniLM
print("Loading MiniLM...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Helper: evaluate prediction array
def compute_metrics(y_true, y_pred, in_mask, ood_mask):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    in_correct = (y_pred[in_mask] == y_true[in_mask]).sum()
    in_fb = (y_pred[in_mask] == 'all').sum()
    n_in = in_mask.sum()
    
    in_acc = in_correct / n_in
    in_cov = (n_in - in_fb) / n_in
    in_prec = in_correct / (n_in - in_fb) if (n_in - in_fb) > 0 else 0.0
    false_fb_rate = in_fb / n_in
    
    ood_correct = (y_pred[ood_mask] == 'all').sum()
    n_ood = ood_mask.sum()
    ood_rej = ood_correct / n_ood if n_ood > 0 else 0.0
    
    overall_acc = (in_correct + ood_correct) / len(y_true)
    macro_f1 = f1_score(y_true, y_pred, labels=ALL_CLASSES, average='macro', zero_division=0)
    
    return {
        "overall_acc": float(overall_acc),
        "in_domain_acc": float(in_acc),
        "in_domain_cov": float(in_cov),
        "in_domain_prec": float(in_prec),
        "false_fallback_rate": float(false_fb_rate),
        "ood_rejection_rate": float(ood_rej),
        "macro_f1": float(macro_f1),
        "in_correct": int(in_correct),
        "in_fallback": int(in_fb),
        "ood_correct": int(ood_correct)
    }

# ==============================================================
# PART 1: HELD-OUT TEST EVALUATION (N = 1,080)
# ==============================================================
print("\n=== [PART 1] EVALUATION ON HELD-OUT TEST SET (N = 1,080) ===")
test_queries = [r['query_natural'] for r in test_data]
test_embs = model.encode(test_queries, batch_size=64, show_progress_bar=False)
test_sims = cosine_similarity(test_embs, centroid_matrix)

# MiniLM rankings
test_top1_idx = np.argmax(test_sims, axis=1)
test_top1_sims = np.max(test_sims, axis=1)

masked_sims = test_sims.copy()
for i in range(len(test_queries)):
    masked_sims[i, test_top1_idx[i]] = -1.0
test_top2_sims = np.max(masked_sims, axis=1)
test_margins = test_top1_sims - test_top2_sims
test_top1_cats = [ACTIVE_CATEGORIES[idx] for idx in test_top1_idx]

# Ground truth
y_true_test = [r['expected_category'] if r['expected_category'] != 'out_of_domain' else 'all' for r in test_data]
in_mask_test = np.array([r['expected_category'] != 'out_of_domain' for r in test_data])
ood_mask_test = ~in_mask_test

# 1. Baseline A (tau = 0.34308)
preds_baseline = ['all' if sim < BASELINE_TAU else cat for sim, cat in zip(test_top1_sims, test_top1_cats)]
res_baseline = compute_metrics(y_true_test, preds_baseline, in_mask_test, ood_mask_test)

# 2. Argmax B (no fallback)
preds_argmax = test_top1_cats
res_argmax = compute_metrics(y_true_test, preds_argmax, in_mask_test, ood_mask_test)

# 3. Calibrated C (frozen: m=0.010, floor=0.250)
preds_calibrated = []
for i in range(len(test_queries)):
    if test_margins[i] >= FROZEN_MARGIN and test_top1_sims[i] >= FROZEN_FLOOR:
        preds_calibrated.append(test_top1_cats[i])
    else:
        preds_calibrated.append('all')
res_calibrated = compute_metrics(y_true_test, preds_calibrated, in_mask_test, ood_mask_test)

# 4. TF-IDF Baseline D (sim > 0.0)
test_tfidf_vecs = tfidf_vec.transform(test_queries)
test_tfidf_sims = cosine_similarity(test_tfidf_vecs, tfidf_centroid_matrix)
tfidf_top1_idx = np.argmax(test_tfidf_sims, axis=1)
tfidf_top1_sims = np.max(test_tfidf_sims, axis=1)
preds_tfidf = ['all' if s == 0.0 else ACTIVE_CATEGORIES[idx] for s, idx in zip(tfidf_top1_sims, tfidf_top1_idx)]
res_tfidf = compute_metrics(y_true_test, preds_tfidf, in_mask_test, ood_mask_test)

print("\n--- HELD-OUT TEST SUMMARY COMPARISON ---")
comp_df = pd.DataFrame([
    {"method": "A. Production Baseline (tau=0.3431)", **res_baseline},
    {"method": "B. MiniLM Argmax (no fallback)", **res_argmax},
    {"method": f"C. Calibrated MiniLM (m={FROZEN_MARGIN:.3f}, fl={FROZEN_FLOOR:.2f})", **res_calibrated},
    {"method": "D. TF-IDF Baseline (sim > 0.0)", **res_tfidf}
])
cols = ['method', 'overall_acc', 'in_domain_acc', 'in_domain_cov', 'in_domain_prec', 'false_fallback_rate', 'ood_rejection_rate', 'macro_f1']
print(comp_df[cols].to_string(index=False))

# Per-category report on Held-Out Test for Calibrated Model
print("\n--- PER-CATEGORY CLASSIFICATION REPORT (CALIBRATED MINILM ON HELD-OUT TEST) ---")
rep = classification_report(y_true_test, preds_calibrated, labels=ALL_CLASSES, zero_division=0, output_dict=True)
rep_df = pd.DataFrame(rep).transpose()
print(rep_df[['precision', 'recall', 'f1-score', 'support']].to_string())

print("\n--- CONFUSION MATRIX (CALIBRATED MINILM ON HELD-OUT TEST) ---")
cm = confusion_matrix(y_true_test, preds_calibrated, labels=ALL_CLASSES)
cm_df = pd.DataFrame(cm, index=ALL_CLASSES, columns=ALL_CLASSES)
print(cm_df.to_string())

# Performance by Query Length
print("\n--- PERFORMANCE BY QUERY LENGTH (HELD-OUT TEST) ---")
lengths = [r['word_count'] for r in test_data]
len_bins = [
    ("Short (< 25 words)", [i for i, l in enumerate(lengths) if l < 25]),
    ("Medium (25 - 50 words)", [i for i, l in enumerate(lengths) if 25 <= l <= 50]),
    ("Long (> 50 words)", [i for i, l in enumerate(lengths) if l > 50])
]

len_results = []
for lbl, idxs in len_bins:
    sub_true = np.array(y_true_test)[idxs]
    sub_pred = np.array(preds_calibrated)[idxs]
    sub_in = in_mask_test[idxs]
    sub_ood = ood_mask_test[idxs]
    metrics = compute_metrics(sub_true, sub_pred, sub_in, sub_ood)
    len_results.append({"Length Bucket": lbl, "Count": len(idxs), **metrics})
print(pd.DataFrame(len_results)[['Length Bucket', 'Count', 'overall_acc', 'in_domain_acc', 'in_domain_cov', 'false_fallback_rate', 'ood_rejection_rate', 'macro_f1']].to_string(index=False))

# Performance for Category-Name-Present vs Category-Name-Masked
print("\n--- PERFORMANCE BY CATEGORY NAME LEAKAGE ---")
leak_indices = [i for i, r in enumerate(test_data) if r['category_name_leakage']]
no_leak_indices = [i for i, r in enumerate(test_data) if not r['category_name_leakage']]

leak_metrics = compute_metrics(
    np.array(y_true_test)[leak_indices], np.array(preds_calibrated)[leak_indices],
    in_mask_test[leak_indices], ood_mask_test[leak_indices]
)
no_leak_metrics = compute_metrics(
    np.array(y_true_test)[no_leak_indices], np.array(preds_calibrated)[no_leak_indices],
    in_mask_test[no_leak_indices], ood_mask_test[no_leak_indices]
)
leak_df = pd.DataFrame([
    {"Group": "Category Name Present (Leakage)", "Count": len(leak_indices), **leak_metrics},
    {"Group": "Category Name Absent (Non-leaked)", "Count": len(no_leak_indices), **no_leak_metrics}
])
print(leak_df[['Group', 'Count', 'overall_acc', 'in_domain_acc', 'in_domain_cov', 'false_fallback_rate', 'ood_rejection_rate', 'macro_f1']].to_string(index=False))

# ==============================================================
# PART 2: ROBUSTNESS / HARD SET EVALUATION (N = 1,260)
# ==============================================================
print("\n=== [PART 2] EVALUATION ON ROBUSTNESS SET (N = 1,260) ===")

# Break down robustness set by type
rob_types = set(r['robustness_type'] for r in robust_data)
rob_results = []

for r_type in sorted(rob_types):
    sub_items = [r for r in robust_data if r['robustness_type'] == r_type]
    sub_queries = [r['query'] for r in sub_items]
    sub_embs = model.encode(sub_queries, batch_size=64, show_progress_bar=False)
    sub_sims = cosine_similarity(sub_embs, centroid_matrix)
    
    top1_i = np.argmax(sub_sims, axis=1)
    top1_s = np.max(sub_sims, axis=1)
    
    m_sims = sub_sims.copy()
    for i in range(len(sub_queries)):
        m_sims[i, top1_i[i]] = -1.0
    top2_s = np.max(m_sims, axis=1)
    sub_margins = top1_s - top2_s
    sub_top1_cats = [ACTIVE_CATEGORIES[idx] for idx in top1_i]
    
    sub_preds = []
    for i in range(len(sub_queries)):
        if sub_margins[i] >= FROZEN_MARGIN and top1_s[i] >= FROZEN_FLOOR:
            sub_preds.append(sub_top1_cats[i])
        else:
            sub_preds.append('all')
            
    sub_true = [r['expected_category'] if r['expected_category'] != 'out_of_domain' else 'all' for r in sub_items]
    sub_in = np.array([r['expected_category'] != 'out_of_domain' for r in sub_items])
    sub_ood = ~sub_in
    
    m = compute_metrics(sub_true, sub_preds, sub_in, sub_ood)
    rob_results.append({
        "Robustness Stress Type": r_type,
        "Count": len(sub_items),
        **m
    })

print(pd.DataFrame(rob_results)[['Robustness Stress Type', 'Count', 'overall_acc', 'in_domain_acc', 'in_domain_cov', 'false_fallback_rate', 'ood_rejection_rate', 'macro_f1']].to_string(index=False))

# Trace the 8 Diagnostic Brands from Curated Benchmark
print("\n--- TRACE OF 8 CORE DIAGNOSTIC BRANDS UNDER CALIBRATED RULE ---")
diag_items = [r for r in robust_data if r.get('brand_name') in ['CampusWear', 'GlowTheory', 'FitForge', 'SpiceRoute', 'UrbanNest', 'PawPal', 'WanderWeekends', 'CreatorHub']]

diag_records = []
for d in diag_items:
    b_text = d['query']
    b_emb = model.encode([b_text])[0]
    sims = cosine_similarity([b_emb], centroid_matrix)[0]
    
    top1_idx = np.argmax(sims)
    top1_sim = float(sims[top1_idx])
    
    masked = sims.copy()
    masked[top1_idx] = -1.0
    top2_idx = np.argmax(masked)
    top2_sim = float(sims[top2_idx])
    
    margin = top1_sim - top2_sim
    top1_cat = ACTIVE_CATEGORIES[top1_idx]
    
    # Decisions under each baseline
    pred_baseline = 'all' if top1_sim < BASELINE_TAU else top1_cat
    pred_argmax = top1_cat
    pred_calibrated = top1_cat if (margin >= FROZEN_MARGIN and top1_sim >= FROZEN_FLOOR) else 'all'
    
    # TF-IDF
    tfidf_sim = cosine_similarity(tfidf_vec.transform([b_text]), tfidf_centroid_matrix)[0]
    pred_tfidf = 'all' if np.max(tfidf_sim) == 0.0 else ACTIVE_CATEGORIES[np.argmax(tfidf_sim)]
    
    diag_records.append({
        "Brand": d['brand_name'],
        "Expected": d['expected_category'],
        "Top1_Cat": top1_cat,
        "Top1_Sim": top1_sim,
        "Top2_Sim": top2_sim,
        "Margin": margin,
        "Pred_Baseline": pred_baseline,
        "Pred_Argmax": pred_argmax,
        "Pred_Calibrated": pred_calibrated,
        "Pred_TFIDF": pred_tfidf
    })

print(pd.DataFrame(diag_records).to_string(index=False))

# ==============================================================
# PART 3: CATEGORY HINT SEPARATE ANALYSIS
# ==============================================================
print("\n=== [PART 3] CATEGORY HINT SEPARATE ANALYSIS ===")
curated_bench = [r for r in robust_data if r['robustness_type'] == 'curated_benchmark_case']
hint_results = []

for c in curated_bench:
    b_text = c['query']
    hint = c.get('category_hint')
    b_emb = model.encode([b_text])[0]
    sims = cosine_similarity([b_emb], centroid_matrix)[0]
    top1_idx = np.argmax(sims)
    top1_cat = ACTIVE_CATEGORIES[top1_idx]
    top1_sim = float(sims[top1_idx])
    
    masked = sims.copy()
    masked[top1_idx] = -1.0
    top2_sim = float(np.max(masked))
    margin = top1_sim - top2_sim
    
    # 1. No hint (calibrated rule)
    p_no_hint = top1_cat if (margin >= FROZEN_MARGIN and top1_sim >= FROZEN_FLOOR) else 'all'
    
    # 2. Hard hint
    p_hard_hint = hint if (hint and hint in ACTIVE_CATEGORIES) else p_no_hint
    
    # 3. Hint as validation/prior (only trust hint if similarity to hinted category >= 0.20 floor)
    if hint and hint in ACTIVE_CATEGORIES:
        hint_cat_idx = ACTIVE_CATEGORIES.index(hint)
        p_prior_hint = hint if sims[hint_cat_idx] >= 0.20 else 'all'
    else:
        p_prior_hint = p_no_hint
        
    hint_results.append({
        "expected": c['expected_category'],
        "hint": hint,
        "p_no_hint": p_no_hint,
        "p_hard_hint": p_hard_hint,
        "p_prior_hint": p_prior_hint
    })

df_h = pd.DataFrame(hint_results)
h_true = [r['expected'] if r['expected'] != 'out_of_domain' else 'all' for r in hint_results]
h_in = np.array([r['expected'] != 'out_of_domain' for r in hint_results])
h_ood = ~h_in

m_no_hint = compute_metrics(h_true, df_h['p_no_hint'], h_in, h_ood)
m_hard_hint = compute_metrics(h_true, df_h['p_hard_hint'], h_in, h_ood)
m_prior_hint = compute_metrics(h_true, df_h['p_prior_hint'], h_in, h_ood)

hint_comp_df = pd.DataFrame([
    {"Condition": "A. No Hint (Autonomous Calibrated Rule)", **m_no_hint},
    {"Condition": "B. Hard Category Hint (Blind Caller Trust)", **m_hard_hint},
    {"Condition": "C. Hint as Validation Prior (Floor >= 0.20)", **m_prior_hint}
])
print(hint_comp_df[['Condition', 'overall_acc', 'in_domain_acc', 'in_domain_cov', 'false_fallback_rate', 'ood_rejection_rate', 'macro_f1']].to_string(index=False))

# ==============================================================
# PART 4: GENERALIZATION GAP SUMMARY
# ==============================================================
print("\n=== [PART 4] GENERALIZATION GAP ANALYSIS ===")
gen_df = pd.DataFrame([
    {
        "Dataset Split": "Calibration Set (N = 720)",
        "Overall Acc": frozen_cfg['calibration_metrics']['overall_accuracy'],
        "In-Domain Acc": frozen_cfg['calibration_metrics']['in_domain_accuracy'],
        "Coverage": frozen_cfg['calibration_metrics']['in_domain_coverage'],
        "False Fallback Rate": frozen_cfg['calibration_metrics']['false_fallback_rate'],
        "OOD Rejection": frozen_cfg['calibration_metrics']['ood_rejection_rate'],
        "Macro-F1": frozen_cfg['calibration_metrics']['macro_f1']
    },
    {
        "Dataset Split": "Held-Out Test Set (N = 1,080)",
        "Overall Acc": res_calibrated['overall_acc'],
        "In-Domain Acc": res_calibrated['in_domain_acc'],
        "Coverage": res_calibrated['in_domain_cov'],
        "False Fallback Rate": res_calibrated['false_fallback_rate'],
        "OOD Rejection": res_calibrated['ood_rejection_rate'],
        "Macro-F1": res_calibrated['macro_f1']
    },
    {
        "Dataset Split": "Generalization Gap (Test - Calib)",
        "Overall Acc": res_calibrated['overall_acc'] - frozen_cfg['calibration_metrics']['overall_accuracy'],
        "In-Domain Acc": res_calibrated['in_domain_acc'] - frozen_cfg['calibration_metrics']['in_domain_accuracy'],
        "Coverage": res_calibrated['in_domain_cov'] - frozen_cfg['calibration_metrics']['in_domain_coverage'],
        "False Fallback Rate": res_calibrated['false_fallback_rate'] - frozen_cfg['calibration_metrics']['false_fallback_rate'],
        "OOD Rejection": res_calibrated['ood_rejection_rate'] - frozen_cfg['calibration_metrics']['ood_rejection_rate'],
        "Macro-F1": res_calibrated['macro_f1'] - frozen_cfg['calibration_metrics']['macro_f1']
    }
])
print(gen_df.to_string(index=False))

# Save all results to benchmark_evaluation_results.json
full_results = {
    "frozen_parameters": frozen_cfg,
    "held_out_comparison": comp_df.to_dict(orient='records'),
    "per_category_report": rep,
    "query_length_results": len_results,
    "category_leakage_results": leak_df.to_dict(orient='records'),
    "robustness_results": rob_results,
    "diagnostic_trace": diag_records,
    "hint_analysis": hint_comp_df.to_dict(orient='records'),
    "generalization_gap": gen_df.to_dict(orient='records')
}

out_full_path = os.path.join(DATA_DIR, "benchmark_evaluation_results.json")
with open(out_full_path, 'w', encoding='utf-8') as f:
    json.dump(full_results, f, indent=2)

print(f"\nAll benchmark evaluation metrics saved to: {out_full_path}")
