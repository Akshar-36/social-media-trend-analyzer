import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RECORDS_PATH = os.path.join(os.path.dirname(__file__), "data", "brand_similarity_records.json")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "data", "relevance_test_set.json")

with open(RECORDS_PATH, "r", encoding="utf-8") as f:
    records = json.load(f)

with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
    test_set = json.load(f)

# Load hashtag artifacts for P@10 downstream evaluation
h_emb_df = pd.read_parquet(os.path.join(PROJECT_ROOT, "artifacts/v1.0/hashtag_embeddings.parquet"))
trends_df = pd.read_parquet(os.path.join(PROJECT_ROOT, "artifacts/v1.0/trends.parquet"))

ACTIVE_CATEGORIES = ['beauty', 'family', 'fashion', 'fitness', 'food', 'interior', 'pet', 'travel']

# Prepare hashtag embedding matrix
ht_list = h_emb_df['hashtag'].tolist()
ht_mat = np.stack(h_emb_df['embedding'].values)

# Hashtag pool mappings in trends.parquet
cat_to_pool_ht = {}
for cat in ACTIVE_CATEGORIES:
    cat_to_pool_ht[cat] = set(trends_df[trends_df['pool'] == cat]['hashtag'].unique())
all_pool_ht = set(trends_df[trends_df['pool'] == 'all']['hashtag'].unique())

def is_relevant_lexical(brand_cat, hashtag):
    h = hashtag.lower()
    if brand_cat == 'beauty' and any(k in h for k in ['skin', 'beauty', 'makeup', 'glow', 'face', 'cosmetic']): return 1
    if brand_cat == 'family' and any(k in h for k in ['fam', 'mom', 'dad', 'parent', 'kid', 'child', 'baby']): return 1
    if brand_cat == 'fashion' and any(k in h for k in ['fashion', 'style', 'outfit', 'wear', 'dress', 'clothes', 'ootd']): return 1
    if brand_cat == 'fitness' and any(k in h for k in ['fit', 'gym', 'workout', 'train', 'muscle', 'health']): return 1
    if brand_cat == 'food' and any(k in h for k in ['food', 'eat', 'cook', 'recipe', 'meal', 'yum', 'delicious']): return 1
    if brand_cat == 'interior' and any(k in h for k in ['interior', 'home', 'decor', 'room', 'house', 'furniture']): return 1
    if brand_cat == 'pet' and any(k in h for k in ['dog', 'cat', 'pet', 'puppy', 'animal']): return 1
    if brand_cat == 'travel' and any(k in h for k in ['travel', 'trip', 'explore', 'wander', 'vacation']): return 1
    return 0

def is_relevant_semantic(brand_cat, hashtag):
    if brand_cat in cat_to_pool_ht:
        return 1 if hashtag in cat_to_pool_ht[brand_cat] else 0
    return 0

# Function to evaluate a set of predictions
def evaluate_method(predictions, method_name):
    # predictions: list of predicted categories ('all' or active category)
    n_total = len(records)
    
    in_domain_indices = [i for i, r in enumerate(records) if r['expected'] != 'out_of_domain']
    ood_indices = [i for i, r in enumerate(records) if r['expected'] == 'out_of_domain']
    adv_indices = [i for i, r in enumerate(records) if r['type'] == 'adversarial']
    
    # In-domain metrics
    in_correct = 0
    in_fallback = 0
    in_misclass = 0
    
    for i in in_domain_indices:
        pred = predictions[i]
        exp = records[i]['expected']
        if pred == exp:
            in_correct += 1
        elif pred == 'all':
            in_fallback += 1
        else:
            in_misclass += 1
            
    n_in = len(in_domain_indices)
    in_acc = in_correct / n_in
    in_coverage = (n_in - in_fallback) / n_in
    in_precision = in_correct / (n_in - in_fallback) if (n_in - in_fallback) > 0 else 0.0
    false_fallback_rate = in_fallback / n_in
    
    # OOD metrics
    ood_correct_rejection = sum(1 for i in ood_indices if predictions[i] == 'all')
    n_ood = len(ood_indices)
    ood_rejection_rate = ood_correct_rejection / n_ood
    
    # Adversarial rejection
    adv_rejection = sum(1 for i in adv_indices if predictions[i] == 'all')
    n_adv = len(adv_indices)
    adv_rejection_rate = adv_rejection / n_adv
    
    # Overall Accuracy (where correct for OOD is 'all', for in-domain is expected)
    total_correct = in_correct + ood_correct_rejection
    overall_acc = total_correct / n_total
    
    # Macro-F1 across classes
    y_true = [r['expected'] if r['expected'] != 'out_of_domain' else 'all' for r in records]
    y_pred = predictions
    labels = ACTIVE_CATEGORIES + ['all']
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    
    return {
        "method": method_name,
        "overall_acc": overall_acc,
        "in_acc": in_acc,
        "in_coverage": in_coverage,
        "in_precision": in_precision,
        "false_fallback_rate": false_fallback_rate,
        "ood_rejection_rate": ood_rejection_rate,
        "adv_rejection_rate": adv_rejection_rate,
        "macro_f1": macro_f1,
        "in_correct": in_correct,
        "in_fallback": in_fallback,
        "in_misclass": in_misclass,
        "ood_rejected": ood_correct_rejection,
        "total_correct": total_correct,
        "predictions": predictions
    }

# 1. Condition A: Baseline MiniLM (tau = 0.34308)
preds_a = []
tau = 0.34308
for r in records:
    if r['minilm_top1_sim'] >= tau:
        preds_a.append(r['minilm_top1_cat'])
    else:
        preds_a.append('all')
res_a = evaluate_method(preds_a, "A. MiniLM Baseline (tau=0.3431)")

# 2. Condition B: Argmax Diagnostic (no fallback)
preds_b = [r['minilm_top1_cat'] for r in records]
res_b = evaluate_method(preds_b, "B. MiniLM Argmax (no fallback)")

# 3. Condition C: Relative-Margin Diagnostic sweeps
margin_sweep_results = []
for m in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]:
    preds_c = []
    for r in records:
        if r['minilm_margin'] >= m:
            preds_c.append(r['minilm_top1_cat'])
        else:
            preds_c.append('all')
    res_c = evaluate_method(preds_c, f"C. Margin m={m:.2f}")
    margin_sweep_results.append(res_c)

# Also Hybrid: Margin + Minimum Absolute Floor (e.g. margin >= 0.03 and top1 >= 0.25)
hybrid_sweep_results = []
for m in [0.02, 0.03, 0.04, 0.05]:
    for floor in [0.20, 0.25, 0.28]:
        preds_h = []
        for r in records:
            if r['minilm_margin'] >= m and r['minilm_top1_sim'] >= floor:
                preds_h.append(r['minilm_top1_cat'])
            else:
                preds_h.append('all')
        res_h = evaluate_method(preds_h, f"C-Hybrid (m={m:.2f}, floor={floor:.2f})")
        hybrid_sweep_results.append(res_h)

# 4. Condition D: Category-Hint Condition
# D1: Gated by Hint if present; if hint is in active categories, choose hint if sim >= 0.20
preds_d1 = []
for r in records:
    hint = r.get('category_hint')
    if hint and hint in ACTIVE_CATEGORIES:
        # Check if hint has reasonable similarity
        hint_sim = r['minilm_all_sims'][hint]
        if hint_sim >= 0.20:
            preds_d1.append(hint)
        else:
            preds_d1.append('all')
    else:
        # Unhinted: use margin rule m=0.03
        if r['minilm_margin'] >= 0.03 and r['minilm_top1_sim'] >= 0.25:
            preds_d1.append(r['minilm_top1_cat'])
        else:
            preds_d1.append('all')
res_d1 = evaluate_method(preds_d1, "D. Hint Prior + Margin Fallback")

# D2: Pure Hint if present, otherwise Argmax
preds_d2 = []
for r in records:
    hint = r.get('category_hint')
    if hint and hint in ACTIVE_CATEGORIES:
        preds_d2.append(hint)
    else:
        preds_d2.append(r['minilm_top1_cat'])
res_d2 = evaluate_method(preds_d2, "D-Direct (Hint if present, else Argmax)")

# 5. Condition E: TF-IDF Baselines
# E1: TF-IDF Argmax (no fallback)
preds_e1 = [r['tfidf_top1_cat'] for r in records]
res_e1 = evaluate_method(preds_e1, "E1. TF-IDF Argmax (no fallback)")

# E2: TF-IDF Zero-overlap Fallback (sim == 0.0 -> 'all')
preds_e2 = []
for r in records:
    if r['tfidf_top1_sim'] > 0.0:
        preds_e2.append(r['tfidf_top1_cat'])
    else:
        preds_e2.append('all')
res_e2 = evaluate_method(preds_e2, "E2. TF-IDF (sim > 0.0)")

# E3: TF-IDF Threshold sweep
tfidf_sweep = []
for tau_tf in [0.01, 0.02, 0.03, 0.05, 0.08]:
    preds_e3 = []
    for r in records:
        if r['tfidf_top1_sim'] >= tau_tf:
            preds_e3.append(r['tfidf_top1_cat'])
        else:
            preds_e3.append('all')
    tfidf_sweep.append(evaluate_method(preds_e3, f"E3. TF-IDF (tau={tau_tf:.2f})"))

# Compile main comparative results table
main_results = [
    res_a,
    res_b,
    next(r for r in margin_sweep_results if "m=0.03" in r['method']),
    next(r for r in margin_sweep_results if "m=0.05" in r['method']),
    next(r for r in hybrid_sweep_results if "m=0.03, floor=0.25" in r['method']),
    res_d1,
    res_e1,
    res_e2,
    next(r for r in tfidf_sweep if "tau=0.02" in r['method'])
]

df_main = pd.DataFrame(main_results)
cols_to_show = ['method', 'overall_acc', 'in_acc', 'in_coverage', 'in_precision', 'false_fallback_rate', 'ood_rejection_rate', 'adv_rejection_rate', 'macro_f1']

print("=== MAIN EXPERIMENTAL RESULTS ===")
print(df_main[cols_to_show].to_string(index=False))

# Print margin sweep table
df_margin = pd.DataFrame(margin_sweep_results)
print("\n=== MARGIN SENSITIVITY SWEEP ===")
print(df_margin[cols_to_show].to_string(index=False))

# Print hybrid sweep table
df_hybrid = pd.DataFrame(hybrid_sweep_results)
print("\n=== HYBRID SWEEP (Margin + Sim Floor) ===")
print(df_hybrid[cols_to_show].to_string(index=False))

# Save summary results to JSON
summary_out = {
    "main_results": [{k: v for k, v in r.items() if k != 'predictions'} for r in main_results],
    "margin_sweep": [{k: v for k, v in r.items() if k != 'predictions'} for r in margin_sweep_results],
    "hybrid_sweep": [{k: v for k, v in r.items() if k != 'predictions'} for r in hybrid_sweep_results]
}
with open(os.path.join(os.path.dirname(__file__), "data", "experiment_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary_out, f, indent=2)

print("\nSummary saved to paper/experiments/data/experiment_summary.json")
