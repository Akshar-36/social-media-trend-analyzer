import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "experiments", "data")
CALIB_PATH = os.path.join(DATA_DIR, "relevance_calibration_set.json")
TEST_PATH = os.path.join(DATA_DIR, "relevance_held_out_test_set.json")
ROBUST_PATH = os.path.join(DATA_DIR, "relevance_robustness_set.json")
FROZEN_PATH = os.path.join(DATA_DIR, "frozen_relevance_parameters.json")
CENTROIDS_PATH = os.path.join(PROJECT_ROOT, "artifacts", "v1.0", "category_centroids.pkl")

ACTIVE_CATEGORIES = ['beauty', 'family', 'fashion', 'fitness', 'food', 'interior', 'pet', 'travel']
ALL_CLASSES = ACTIVE_CATEGORIES + ['all']

# Frozen decision rule parameters
with open(FROZEN_PATH, 'r', encoding='utf-8') as f:
    frozen_cfg = json.load(f)
FROZEN_MARGIN = frozen_cfg['margin']   # 0.010
FROZEN_FLOOR = frozen_cfg['floor']     # 0.250

print(f"=== RELEVANCE REPRESENTATION EXPERIMENT ===")
print(f"Frozen Decision Gate: Margin >= {FROZEN_MARGIN:.3f}, Floor >= {FROZEN_FLOOR:.3f}")

# 1. Load Data
with open(CALIB_PATH, 'r', encoding='utf-8') as f:
    calib_data = json.load(f)
with open(TEST_PATH, 'r', encoding='utf-8') as f:
    test_data = json.load(f)
with open(ROBUST_PATH, 'r', encoding='utf-8') as f:
    robust_data = json.load(f)
with open(CENTROIDS_PATH, 'rb') as f:
    prod_centroids = pickle.load(f)

# Production single-centroid matrix (8, 384)
single_centroid_matrix = np.stack([prod_centroids[c] for c in ACTIVE_CATEGORIES])

print("Loading SentenceTransformer all-MiniLM-L6-v2...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Encode Calibration Data for Training Prototypes & Exemplars
print("\n[Step 1/5] Encoding Calibration Set for Representation Learning...")
calib_embs_by_cat = {c: [] for c in ACTIVE_CATEGORIES}
for r in calib_data:
    cat = r['raw_source_category']
    if cat in calib_embs_by_cat:
        calib_embs_by_cat[cat].append(r['query_natural'])

for c in ACTIVE_CATEGORIES:
    embs = model.encode(calib_embs_by_cat[c], batch_size=32, show_progress_bar=False)
    calib_embs_by_cat[c] = embs

# 3. Learn Prototypes & Medoid Exemplars
print("\n[Step 2/5] Learning Prototypes and Exemplars from Calibration Data...")
prototypes_dict = {}  # k -> {cat: matrix (k, 384)}
medoids_dict = {}     # k -> {cat: matrix (k, 384)}

for k in [3, 5, 10]:
    prototypes_dict[k] = {}
    medoids_dict[k] = {}
    for c in ACTIVE_CATEGORIES:
        X = calib_embs_by_cat[c]
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        prototypes_dict[k][c] = kmeans.cluster_centers_
        
        # Medoids: actual data points closest to each cluster center
        medoid_indices = []
        for center in kmeans.cluster_centers_:
            dists = np.linalg.norm(X - center, axis=1)
            medoid_indices.append(np.argmin(dists))
        medoids_dict[k][c] = X[medoid_indices]

# Full calibration exemplar pool (80 per category)
exemplar_pool = {c: calib_embs_by_cat[c] for c in ACTIVE_CATEGORIES}

# 4. Define Scoring Functions for Categories
def score_single_centroid(q_embs):
    # q_embs: (N, 384)
    # returns: (N, 8) cosine similarities
    return cosine_similarity(q_embs, single_centroid_matrix)

def score_prototypes(q_embs, k, agg='max'):
    # For each category c, compute similarity to k prototypes
    # Return (N, 8)
    sims = np.zeros((len(q_embs), len(ACTIVE_CATEGORIES)))
    for c_idx, c in enumerate(ACTIVE_CATEGORIES):
        p_mat = prototypes_dict[k][c] # (k, 384)
        c_sims = cosine_similarity(q_embs, p_mat) # (N, k)
        if agg == 'max':
            sims[:, c_idx] = np.max(c_sims, axis=1)
        elif agg == 'mean':
            sims[:, c_idx] = np.mean(c_sims, axis=1)
        elif agg == 'top2_mean':
            sorted_s = np.sort(c_sims, axis=1)[:, ::-1]
            sims[:, c_idx] = np.mean(sorted_s[:, :min(2, k)], axis=1)
    return sims

def score_exemplars(q_embs, k_val, agg='topk_mean'):
    # Using all 80 calibration exemplars per category
    sims = np.zeros((len(q_embs), len(ACTIVE_CATEGORIES)))
    for c_idx, c in enumerate(ACTIVE_CATEGORIES):
        e_mat = exemplar_pool[c] # (80, 384)
        e_sims = cosine_similarity(q_embs, e_mat) # (N, 80)
        sorted_s = np.sort(e_sims, axis=1)[:, ::-1]
        if agg == 'max':
            sims[:, c_idx] = sorted_s[:, 0]
        elif agg == 'topk_mean':
            sims[:, c_idx] = np.mean(sorted_s[:, :k_val], axis=1)
    return sims

def score_hybrid(q_embs, w=0.5, k=5):
    s_cent = score_single_centroid(q_embs)
    s_proto = score_prototypes(q_embs, k=k, agg='max')
    return w * s_cent + (1 - w) * s_proto

# List of representation methods to benchmark
REPRESENTATION_METHODS = [
    ("Baseline: Single Centroid", lambda q: score_single_centroid(q)),
    ("Proto (k=3, max)", lambda q: score_prototypes(q, k=3, agg='max')),
    ("Proto (k=5, max)", lambda q: score_prototypes(q, k=5, agg='max')),
    ("Proto (k=10, max)", lambda q: score_prototypes(q, k=10, agg='max')),
    ("Proto (k=5, mean)", lambda q: score_prototypes(q, k=5, agg='mean')),
    ("Proto (k=5, top2_mean)", lambda q: score_prototypes(q, k=5, agg='top2_mean')),
    ("Exemplar (k=1, max)", lambda q: score_exemplars(q, k_val=1, agg='max')),
    ("Exemplar (k=3, mean)", lambda q: score_exemplars(q, k_val=3, agg='topk_mean')),
    ("Exemplar (k=5, mean)", lambda q: score_exemplars(q, k_val=5, agg='topk_mean')),
    ("Exemplar (k=10, mean)", lambda q: score_exemplars(q, k_val=10, agg='topk_mean')),
    ("Hybrid (w=0.3 cent, 0.7 proto-k5)", lambda q: score_hybrid(q, w=0.3, k=5)),
    ("Hybrid (w=0.5 cent, 0.5 proto-k5)", lambda q: score_hybrid(q, w=0.5, k=5)),
    ("Hybrid (w=0.7 cent, 0.3 proto-k5)", lambda q: score_hybrid(q, w=0.7, k=5)),
]

# Helper: evaluate similarity matrix under decision gate
def evaluate_similarity_matrix(sim_matrix, y_true_labels, in_domain_mask, margin=FROZEN_MARGIN, floor=FROZEN_FLOOR):
    top1_idx = np.argmax(sim_matrix, axis=1)
    top1_sims = np.max(sim_matrix, axis=1)
    
    masked = sim_matrix.copy()
    for i in range(len(sim_matrix)):
        masked[i, top1_idx[i]] = -1.0
    top2_sims = np.max(masked, axis=1)
    margins = top1_sims - top2_sims
    top1_cats = [ACTIVE_CATEGORIES[i] for i in top1_idx]
    
    # Calibrated decision rule
    preds = []
    for i in range(len(sim_matrix)):
        if margins[i] >= margin and top1_sims[i] >= floor:
            preds.append(top1_cats[i])
        else:
            preds.append('all')
            
    # Argmax decision rule (no fallback)
    preds_argmax = top1_cats
    
    y_true = np.array(y_true_labels)
    preds = np.array(preds)
    preds_argmax = np.array(preds_argmax)
    
    # In-domain metrics
    in_true = y_true[in_domain_mask]
    in_preds = preds[in_domain_mask]
    in_argmax = preds_argmax[in_domain_mask]
    
    in_correct = (in_preds == in_true).sum()
    in_fb = (in_preds == 'all').sum()
    n_in = len(in_true)
    
    in_acc = in_correct / n_in
    in_cov = (n_in - in_fb) / n_in
    in_prec = in_correct / (n_in - in_fb) if (n_in - in_fb) > 0 else 0.0
    false_fb = in_fb / n_in
    in_argmax_acc = (in_argmax == in_true).sum() / n_in
    
    # OOD metrics
    ood_mask = ~in_domain_mask
    ood_true = y_true[ood_mask]
    ood_preds = preds[ood_mask]
    ood_rej = (ood_preds == 'all').sum() / len(ood_true) if len(ood_true) > 0 else 0.0
    
    # Overall & Macro-F1
    overall_acc = (in_correct + (ood_preds == 'all').sum()) / len(y_true)
    macro_f1 = f1_score(y_true, preds, labels=ALL_CLASSES, average='macro', zero_division=0)
    
    # Fashion Recall specifically
    fashion_mask = (y_true == 'fashion')
    fashion_recall = (preds[fashion_mask] == 'fashion').sum() / fashion_mask.sum() if fashion_mask.sum() > 0 else 0.0
    
    return {
        "overall_acc": float(overall_acc),
        "in_domain_acc": float(in_acc),
        "in_domain_cov": float(in_cov),
        "in_domain_prec": float(in_prec),
        "false_fallback": float(false_fb),
        "ood_rejection": float(ood_rej),
        "macro_f1": float(macro_f1),
        "fashion_recall": float(fashion_recall),
        "in_argmax_acc": float(in_argmax_acc),
        "score_dist": {
            "top1_mean": float(np.mean(top1_sims)),
            "top1_std": float(np.std(top1_sims)),
            "margin_mean": float(np.mean(margins)),
            "margin_std": float(np.std(margins))
        },
        "preds": preds.tolist()
    }

# 5. Execute Held-Out Test Evaluation
print("\n[Step 3/5] Executing Benchmark on Held-Out Test Set (N = 1,080)...")
test_queries = [r['query_natural'] for r in test_data]
test_embs = model.encode(test_queries, batch_size=64, show_progress_bar=False)

y_true_test = [r['expected_category'] if r['expected_category'] != 'out_of_domain' else 'all' for r in test_data]
in_mask_test = np.array([r['expected_category'] != 'out_of_domain' for r in test_data])

held_out_results = []
for name, fn in REPRESENTATION_METHODS:
    sim_mat = fn(test_embs)
    m = evaluate_similarity_matrix(sim_mat, y_true_test, in_mask_test)
    held_out_results.append({
        "Method": name,
        "Overall Acc": m['overall_acc'],
        "In-Domain Acc": m['in_domain_acc'],
        "Coverage": m['in_domain_cov'],
        "In-Domain Prec": m['in_domain_prec'],
        "False Fallback": m['false_fallback'],
        "OOD Rejection": m['ood_rejection'],
        "Macro-F1": m['macro_f1'],
        "Fashion Recall": m['fashion_recall'],
        "Argmax In-Acc": m['in_argmax_acc'],
        "Top1 Mean Sim": m['score_dist']['top1_mean'],
        "Margin Mean": m['score_dist']['margin_mean']
    })

df_held_out = pd.DataFrame(held_out_results)
print("\n=== HELD-OUT TEST BENCHMARK RESULTS (N = 1,080) ===")
cols_show = ['Method', 'Overall Acc', 'In-Domain Acc', 'In-Domain Prec', 'Coverage', 'False Fallback', 'OOD Rejection', 'Macro-F1', 'Fashion Recall', 'Top1 Mean Sim']
print(df_held_out[cols_show].to_string(index=False))

# 6. Execute Robustness Set Evaluation
print("\n[Step 4/5] Executing Benchmark on Robustness / Hard Set (N = 1,260)...")
robust_queries = [r['query'] for r in robust_data]
robust_embs = model.encode(robust_queries, batch_size=64, show_progress_bar=False)
y_true_robust = [r['expected_category'] if r['expected_category'] != 'out_of_domain' else 'all' for r in robust_data]
in_mask_robust = np.array([r['expected_category'] != 'out_of_domain' for r in robust_data])

# Evaluate key representations on each robustness slice
rob_types = sorted(list(set(r['robustness_type'] for r in robust_data)))
rob_summary = []

key_methods = [
    ("Baseline: Centroid", lambda q: score_single_centroid(q)),
    ("Proto (k=5, max)", lambda q: score_prototypes(q, k=5, agg='max')),
    ("Exemplar (k=5, mean)", lambda q: score_exemplars(q, k_val=5, agg='topk_mean')),
    ("Hybrid (w=0.5)", lambda q: score_hybrid(q, w=0.5, k=5))
]

for r_type in rob_types:
    sub_indices = [i for i, r in enumerate(robust_data) if r['robustness_type'] == r_type]
    sub_embs = robust_embs[sub_indices]
    sub_true = np.array(y_true_robust)[sub_indices]
    sub_in = in_mask_robust[sub_indices]
    
    row = {"Stress Type": r_type, "Count": len(sub_indices)}
    for m_name, fn in key_methods:
        sim_mat = fn(sub_embs)
        m = evaluate_similarity_matrix(sim_mat, sub_true, sub_in)
        # Use In-Domain Acc if in-domain exists, else OOD rejection
        score_val = m['ood_rejection'] if r_type == 'out_of_domain_rejection' else m['in_domain_acc']
        row[f"{m_name}"] = score_val
    rob_summary.append(row)

df_rob = pd.DataFrame(rob_summary)
print("\n=== ROBUSTNESS STRESS BENCHMARK RESULTS (N = 1,260) ===")
print(df_rob.to_string(index=False))

# 7. CampusWear and 8 Core Diagnostic Brands Trace
print("\n[Step 5/5] Tracing CampusWear and Core Diagnostic Brands...")
diag_indices = [i for i, r in enumerate(robust_data) if r.get('brand_name') in ['CampusWear', 'GlowTheory', 'FitForge', 'SpiceRoute', 'UrbanNest', 'PawPal', 'WanderWeekends', 'CreatorHub']]

diag_records = []
for idx in diag_indices:
    b_item = robust_data[idx]
    b_name = b_item['brand_name']
    b_exp = b_item['expected_category']
    b_emb = robust_embs[idx:idx+1]
    
    row = {"Brand": b_name, "Expected": b_exp}
    for m_name, fn in key_methods:
        sims = fn(b_emb)[0]
        top1_idx = np.argmax(sims)
        top1_sim = float(sims[top1_idx])
        masked = sims.copy()
        masked[top1_idx] = -1.0
        top2_sim = float(np.max(masked))
        margin = top1_sim - top2_sim
        pred = ACTIVE_CATEGORIES[top1_idx] if (margin >= FROZEN_MARGIN and top1_sim >= FROZEN_FLOOR) else 'all'
        row[f"{m_name}_Pred"] = pred
        row[f"{m_name}_Sim"] = round(top1_sim, 3)
        row[f"{m_name}_Margin"] = round(margin, 3)
    diag_records.append(row)

df_diag = pd.DataFrame(diag_records)
print("\n=== DIAGNOSTIC BRANDS TRACE ACROSS REPRESENTATIONS ===")
print(df_diag[['Brand', 'Expected', 'Baseline: Centroid_Pred', 'Proto (k=5, max)_Pred', 'Exemplar (k=5, mean)_Pred', 'Hybrid (w=0.5)_Pred']].to_string(index=False))

# Detailed CampusWear Trace
cw_row = next(r for r in diag_records if r['Brand'] == 'CampusWear')
print("\nDetailed CampusWear Comparison:")
for k, v in cw_row.items():
    print(f"  {k:30s}: {v}")

# 8. Computational Cost & Latency Benchmark
print("\n=== COMPUTATIONAL COST BENCHMARK ===")
timing_queries = test_embs[:100]  # 100 queries
costs = []

for m_name, fn in key_methods:
    # Memory size estimation
    if "Centroid" in m_name:
        param_bytes = single_centroid_matrix.nbytes
        vector_count = len(ACTIVE_CATEGORIES)
    elif "Proto" in m_name:
        param_bytes = sum(v.nbytes for v in prototypes_dict[5].values())
        vector_count = len(ACTIVE_CATEGORIES) * 5
    elif "Exemplar" in m_name:
        param_bytes = sum(v.nbytes for v in exemplar_pool.values())
        vector_count = len(ACTIVE_CATEGORIES) * 80
    elif "Hybrid" in m_name:
        param_bytes = single_centroid_matrix.nbytes + sum(v.nbytes for v in prototypes_dict[5].values())
        vector_count = len(ACTIVE_CATEGORIES) * 6
        
    # Latency: run 20 times over 100 queries
    start_t = time.perf_counter()
    for _ in range(20):
        _ = fn(timing_queries)
    elapsed = (time.perf_counter() - start_t) / (20 * 100) * 1000  # ms per query
    
    costs.append({
        "Representation": m_name,
        "Total Vectors": vector_count,
        "Parameter Size (KB)": round(param_bytes / 1024, 2),
        "Inference Latency (ms/query)": round(elapsed, 4),
        "Latency vs Baseline": f"{elapsed / costs[0]['Inference Latency (ms/query)']:.2f}x" if costs else "1.00x"
    })

df_cost = pd.DataFrame(costs)
print(df_cost.to_string(index=False))

# 9. Save all results to representation_experiment_results.json
out_path = os.path.join(DATA_DIR, "representation_experiment_results.json")
save_payload = {
    "held_out_benchmark": df_held_out.to_dict(orient='records'),
    "robustness_benchmark": df_rob.to_dict(orient='records'),
    "diagnostic_trace": diag_records,
    "computational_cost": costs
}
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(save_payload, f, indent=2)

print(f"\nAll representation experiment results serialized to: {out_path}")
