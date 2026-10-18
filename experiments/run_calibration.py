import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "experiments", "data")
CALIB_PATH = os.path.join(DATA_DIR, "relevance_calibration_set.json")
CENTROIDS_PATH = os.path.join(PROJECT_ROOT, "artifacts", "v1.0", "category_centroids.pkl")

ACTIVE_CATEGORIES = ['beauty', 'family', 'fashion', 'fitness', 'food', 'interior', 'pet', 'travel']
ALL_CLASSES = ACTIVE_CATEGORIES + ['all']

print(f"Loading calibration dataset: {CALIB_PATH}")
with open(CALIB_PATH, 'r', encoding='utf-8') as f:
    calib_data = json.load(f)

print(f"Loading category centroids: {CENTROIDS_PATH}")
with open(CENTROIDS_PATH, 'rb') as f:
    centroids = pickle.load(f)

centroid_matrix = np.stack([centroids[c] for c in ACTIVE_CATEGORIES])

print("Loading SentenceTransformer all-MiniLM-L6-v2...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract and encode natural query texts
queries = [r['query_natural'] for r in calib_data]
print(f"Encoding {len(queries)} calibration queries...")
q_embs = model.encode(queries, batch_size=64, show_progress_bar=False)

# Compute cosine similarities against the 8 centroids
# Shape: (720, 8)
sim_matrix = cosine_similarity(q_embs, centroid_matrix)

# Precompute top-1, top-2, margins
top1_indices = np.argmax(sim_matrix, axis=1)
top1_sims = np.max(sim_matrix, axis=1)

# To find top-2, mask out top-1
masked_sims = sim_matrix.copy()
for i in range(len(queries)):
    masked_sims[i, top1_indices[i]] = -1.0
top2_sims = np.max(masked_sims, axis=1)
margins = top1_sims - top2_sims
top1_cats = [ACTIVE_CATEGORIES[idx] for idx in top1_indices]

# Ground truth mapping: in-domain -> expected_category, out_of_domain -> 'all'
y_true = [r['expected_category'] if r['expected_category'] != 'out_of_domain' else 'all' for r in calib_data]
in_domain_mask = np.array([r['expected_category'] != 'out_of_domain' for r in calib_data])
ood_mask = ~in_domain_mask

# Grid parameters
MARGIN_GRID = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050, 0.060]
FLOOR_GRID = [0.00, 0.10, 0.15, 0.20, 0.25, 0.30]

print(f"\n--- Running Grid Search over {len(MARGIN_GRID)} margins x {len(FLOOR_GRID)} floors (60 configurations) ---")

grid_results = []

for m in MARGIN_GRID:
    for fl in FLOOR_GRID:
        # Decision rule
        preds = []
        for i in range(len(queries)):
            if margins[i] >= m and top1_sims[i] >= fl:
                preds.append(top1_cats[i])
            else:
                preds.append('all')
                
        preds = np.array(preds)
        
        # In-domain metrics
        in_true = np.array(y_true)[in_domain_mask]
        in_preds = preds[in_domain_mask]
        
        in_correct = (in_preds == in_true).sum()
        in_fallback = (in_preds == 'all').sum()
        n_in = len(in_true)
        
        in_acc = in_correct / n_in
        in_cov = (n_in - in_fallback) / n_in
        in_prec = in_correct / (n_in - in_fallback) if (n_in - in_fallback) > 0 else 0.0
        false_fb_rate = in_fallback / n_in
        
        # OOD metrics
        ood_true = np.array(y_true)[ood_mask]
        ood_preds = preds[ood_mask]
        ood_rejected = (ood_preds == 'all').sum()
        n_ood = len(ood_true)
        ood_rej_rate = ood_rejected / n_ood
        
        # Overall metrics
        total_correct = in_correct + ood_rejected
        overall_acc = total_correct / len(y_true)
        macro_f1 = f1_score(y_true, preds, labels=ALL_CLASSES, average='macro', zero_division=0)
        
        grid_results.append({
            'margin': m,
            'floor': fl,
            'overall_acc': overall_acc,
            'macro_f1': macro_f1,
            'in_domain_acc': in_acc,
            'in_domain_cov': in_cov,
            'in_domain_prec': in_prec,
            'false_fallback_rate': false_fb_rate,
            'ood_rejection_rate': ood_rej_rate
        })

df_grid = pd.DataFrame(grid_results)

# Sort by Macro-F1 descending, then Overall Acc descending
df_sorted = df_grid.sort_values(by=['macro_f1', 'overall_acc'], ascending=[False, False]).reset_index(drop=True)

print("\n=== TOP 15 CONFIGURATIONS ON CALIBRATION SET ===")
print(df_sorted.head(15).to_string(index=False))

# Also inspect baseline on calibration set
baseline_tau = 0.34308
preds_base = ['all' if sim < baseline_tau else cat for sim, cat in zip(top1_sims, top1_cats)]
base_in_correct = (np.array(preds_base)[in_domain_mask] == np.array(y_true)[in_domain_mask]).sum()
base_ood_correct = (np.array(preds_base)[ood_mask] == 'all').sum()
base_f1 = f1_score(y_true, preds_base, labels=ALL_CLASSES, average='macro', zero_division=0)
base_overall_acc = (base_in_correct + base_ood_correct) / len(y_true)
base_in_acc = base_in_correct / in_domain_mask.sum()
base_false_fb = (np.array(preds_base)[in_domain_mask] == 'all').sum() / in_domain_mask.sum()
base_ood_rej = base_ood_correct / ood_mask.sum()

print("\n=== PRODUCTION BASELINE (tau = 0.34308) ON CALIBRATION SET ===")
print(f"Overall Acc:         {base_overall_acc:.4f}")
print(f"Macro-F1:            {base_f1:.4f}")
print(f"In-domain Acc:       {base_in_acc:.4f}")
print(f"False Fallback Rate: {base_false_fb:.4f}")
print(f"OOD Rejection Rate:  {base_ood_rej:.4f}")

# Select the top configuration
best_cfg = df_sorted.iloc[0]
print("\n=== SELECTED CALIBRATION PARAMETER PAIR ===")
print(f"Selected Margin (m):        {best_cfg['margin']}")
print(f"Selected Similarity Floor: {best_cfg['floor']}")
print(f"Selection Criterion:       Maximum Macro-F1 across all 9 classes (balances in-domain precision, recall, and OOD rejection).")
print(f"Calibration Overall Acc:   {best_cfg['overall_acc']:.4f}")
print(f"Calibration Macro-F1:      {best_cfg['macro_f1']:.4f}")
print(f"Calibration In-domain Acc: {best_cfg['in_domain_acc']:.4f}")
print(f"Calibration Coverage:      {best_cfg['in_domain_cov']:.4f}")
print(f"Calibration False Fallback:{best_cfg['false_fallback_rate']:.4f}")
print(f"Calibration OOD Rejection: {best_cfg['ood_rejection_rate']:.4f}")

# Save calibration results and frozen parameters
frozen_params = {
    "frozen": True,
    "margin": float(best_cfg['margin']),
    "floor": float(best_cfg['floor']),
    "selection_criterion": "Argmax Macro-F1 on disjoint 720-example calibration set",
    "calibration_metrics": {
        "overall_accuracy": float(best_cfg['overall_acc']),
        "macro_f1": float(best_cfg['macro_f1']),
        "in_domain_accuracy": float(best_cfg['in_domain_acc']),
        "in_domain_coverage": float(best_cfg['in_domain_cov']),
        "in_domain_precision": float(best_cfg['in_domain_prec']),
        "false_fallback_rate": float(best_cfg['false_fallback_rate']),
        "ood_rejection_rate": float(best_cfg['ood_rejection_rate'])
    },
    "baseline_calibration_metrics": {
        "overall_accuracy": float(base_overall_acc),
        "macro_f1": float(base_f1),
        "in_domain_accuracy": float(base_in_acc),
        "false_fallback_rate": float(base_false_fb),
        "ood_rejection_rate": float(base_ood_rej)
    },
    "grid_results_summary": df_sorted.head(15).to_dict(orient='records')
}

out_param_path = os.path.join(DATA_DIR, "frozen_relevance_parameters.json")
with open(out_param_path, 'w', encoding='utf-8') as f:
    json.dump(frozen_params, f, indent=2)

print(f"\nFrozen parameters saved to: {out_param_path}")
