import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RECORDS_PATH = os.path.join(os.path.dirname(__file__), "data", "brand_similarity_records.json")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "data", "relevance_test_set.json")

with open(RECORDS_PATH, "r", encoding="utf-8") as f:
    records = json.load(f)

with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
    test_set = json.load(f)

h_emb_df = pd.read_parquet(os.path.join(PROJECT_ROOT, "artifacts/v1.0/hashtag_embeddings.parquet"))
trends_df = pd.read_parquet(os.path.join(PROJECT_ROOT, "artifacts/v1.0/trends.parquet"))

ACTIVE_CATEGORIES = ['beauty', 'family', 'fashion', 'fitness', 'food', 'interior', 'pet', 'travel']
ALL_LABELS = ACTIVE_CATEGORIES + ['all']

# Map ground truth
y_true = [r['expected'] if r['expected'] != 'out_of_domain' else 'all' for r in records]

# Baseline preds (tau = 0.34308)
preds_a = [r['minilm_top1_cat'] if r['minilm_top1_sim'] >= 0.34308 else 'all' for r in records]

# Argmax preds
preds_b = [r['minilm_top1_cat'] for r in records]

# Margin m=0.02
preds_c02 = [r['minilm_top1_cat'] if r['minilm_margin'] >= 0.02 else 'all' for r in records]

# Margin m=0.03
preds_c03 = [r['minilm_top1_cat'] if r['minilm_margin'] >= 0.03 else 'all' for r in records]

# Hint preds
preds_d = []
for r in records:
    hint = r.get('category_hint')
    if hint and hint in ACTIVE_CATEGORIES and r['minilm_all_sims'][hint] >= 0.20:
        preds_d.append(hint)
    else:
        preds_d.append(r['minilm_top1_cat'] if r['minilm_margin'] >= 0.03 and r['minilm_top1_sim'] >= 0.25 else 'all')

# TF-IDF (sim > 0.0)
preds_e = [r['tfidf_top1_cat'] if r['tfidf_top1_sim'] > 0.0 else 'all' for r in records]

methods = {
    "A. Baseline (tau=0.3431)": preds_a,
    "B. MiniLM Argmax": preds_b,
    "C. MiniLM Margin (m=0.02)": preds_c02,
    "C. MiniLM Margin (m=0.03)": preds_c03,
    "D. Category Hint": preds_d,
    "E. TF-IDF (sim>0)": preds_e
}

# DOWNSTREAM P@10 CALCULATION
minilm_model = SentenceTransformer('all-MiniLM-L6-v2')
ht_mat = np.stack(h_emb_df['embedding'].values)
ht_names = h_emb_df['hashtag'].tolist()

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

cat_pool_hts = {c: set(trends_df[trends_df['pool'] == c]['hashtag'].unique()) for c in ACTIVE_CATEGORIES}

print("================ DOWNSTREAM P@10 EVALUATION ================\n")

# Evaluate diagnostic and historical brands for P@10
p10_summary = {}

for m_name, preds in methods.items():
    lex_p10_list = []
    sem_p10_list = []
    
    for i, r in enumerate(records):
        if r['expected'] == 'out_of_domain':
            continue
        exp = r['expected']
        pred_cat = preds[i]
        
        # Candidate pool
        if pred_cat == 'all':
            eligible = set(trends_df[trends_df['pool'] == 'all']['hashtag'].unique())
        else:
            eligible = cat_pool_hts.get(pred_cat, set(trends_df[trends_df['pool'] == 'all']['hashtag'].unique()))
            
        pool_indices = [idx for idx, h in enumerate(ht_names) if h in eligible]
        if not pool_indices:
            lex_p10_list.append(0.0)
            sem_p10_list.append(0.0)
            continue
            
        sub_ht_mat = ht_mat[pool_indices]
        sub_ht_names = [ht_names[idx] for idx in pool_indices]
        
        # Encode actual brand text
        b_text = f"{test_set[i]['business_description']} {test_set[i]['industry']} {test_set[i]['target_audience']}".strip()
        b_emb = minilm_model.encode([b_text])[0]
        
        sims = cosine_similarity([b_emb], sub_ht_mat)[0]
        top_10_local_idx = np.argsort(sims)[::-1][:10]
        top_10_hts = [sub_ht_names[idx] for idx in top_10_local_idx]
        
        lex_p = sum(is_relevant_lexical(exp, h) for h in top_10_hts) / 10.0
        sem_p = sum(1 for h in top_10_hts if h in cat_pool_hts[exp]) / 10.0
        
        lex_p10_list.append(lex_p)
        sem_p10_list.append(sem_p)
        
    p10_summary[m_name] = {
        "mean_lexical_p10": float(np.mean(lex_p10_list)),
        "mean_semantic_p10": float(np.mean(sem_p10_list))
    }

print(pd.DataFrame(p10_summary).transpose().to_string())

print("\n================ CAMPUSWEAR DEEP DIVE ================")
cw_idx = [r['brand_name'] for r in records].index('CampusWear')
cw = records[cw_idx]
cw_b = test_set[cw_idx]

print(f"Brand Name: {cw['brand_name']}")
print(f"Description: {cw_b['business_description']}")
print(f"Industry: {cw_b['industry']}, Audience: {cw_b['target_audience']}")
print(f"Expected Category: {cw['expected']}")
print(f"Category Hint: {cw_b.get('category_hint')}")

print("\n--- MiniLM Similarities against Category Centroids ---")
for c in ACTIVE_CATEGORIES:
    mark = " <-- TOP 1" if c == cw['minilm_top1_cat'] else (" <-- TOP 2" if c == cw['minilm_top2_cat'] else "")
    print(f"  {c:10s}: {cw['minilm_all_sims'][c]:.5f}{mark}")
print(f"MiniLM Margin (Top1 - Top2): {cw['minilm_margin']:.5f}")

print("\n--- TF-IDF Similarities against Category Pools ---")
for c in ACTIVE_CATEGORIES:
    mark = " <-- TOP 1" if c == cw['tfidf_top1_cat'] else (" <-- TOP 2" if c == cw['tfidf_top2_cat'] else "")
    print(f"  {c:10s}: {cw['tfidf_all_sims'][c]:.5f}{mark}")
print(f"TF-IDF Margin (Top1 - Top2): {cw['tfidf_margin']:.5f}")

print("\n--- Method-by-Method Decisions for CampusWear ---")
cw_text = f"{cw_b['business_description']} {cw_b['industry']} {cw_b['target_audience']}".strip()
cw_emb = minilm_model.encode([cw_text])[0]

for m_name, p_list in methods.items():
    pred_cat = p_list[cw_idx]
    
    # Trace candidate pool
    if pred_cat == 'all':
        pool_hts = set(trends_df[trends_df['pool'] == 'all']['hashtag'].unique())
        fallback_flag = True
    else:
        pool_hts = cat_pool_hts.get(pred_cat, set(trends_df[trends_df['pool'] == 'all']['hashtag'].unique()))
        fallback_flag = False
        
    pool_indices = [idx for idx, h in enumerate(ht_names) if h in pool_hts]
    sub_mat = ht_mat[pool_indices]
    sub_names = [ht_names[idx] for idx in pool_indices]
    
    sims = cosine_similarity([cw_emb], sub_mat)[0]
    top_5_idx = np.argsort(sims)[::-1][:5]
    top_5_hts = [sub_names[idx] for idx in top_5_idx]
    
    lex_p = sum(is_relevant_lexical('fashion', h) for h in top_5_hts[:5]) / 5.0
    sem_p = sum(1 for h in top_5_hts[:5] if h in cat_pool_hts['fashion']) / 5.0
    
    print(f"\nMethod: {m_name}")
    print(f"  Predicted Category : {pred_cat}")
    print(f"  Fallback Occurred  : {fallback_flag}")
    print(f"  Candidate Pool Size: {len(pool_hts)}")
    print(f"  Top 5 Hashtags     : {top_5_hts}")
    print(f"  Top-5 Semantic Acc : {sem_p:.2f} | Lexical Acc: {lex_p:.2f}")

# Save detailed results to json
with open(os.path.join(PROJECT_ROOT, "experiments/data/deep_dive_summary.json"), "w", encoding="utf-8") as f:
    json.dump({
        "p10_summary": p10_summary,
        "campuswear": {
            "minilm_sims": cw['minilm_all_sims'],
            "tfidf_sims": cw['tfidf_all_sims'],
            "decisions": {m_name: p_list[cw_idx] for m_name, p_list in methods.items()}
        }
    }, f, indent=2)

print("\nDeep dive metrics saved successfully.")
