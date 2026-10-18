import os
import sys
import json
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sentence_transformers import SentenceTransformer

# Ensure reproducible paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "relevance_test_set.json")

print(f"Loading test set from: {DATA_PATH}")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    test_set = json.load(f)

print(f"Loaded {len(test_set)} test brands.")

# Load production artifacts (READ ONLY)
with open(os.path.join(PROJECT_ROOT, "artifacts/v1.0/category_centroids.pkl"), "rb") as f:
    centroids = pickle.load(f)

h_emb_df = pd.read_parquet(os.path.join(PROJECT_ROOT, "artifacts/v1.0/hashtag_embeddings.parquet"))
trends_df = pd.read_parquet(os.path.join(PROJECT_ROOT, "artifacts/v1.0/trends.parquet"))

ACTIVE_CATEGORIES = ['beauty', 'family', 'fashion', 'fitness', 'food', 'interior', 'pet', 'travel']
print("Active categories:", ACTIVE_CATEGORIES)

# Centroids matrix
centroid_matrix = np.stack([centroids[c] for c in ACTIVE_CATEGORIES])

# MiniLM model
print("Loading MiniLM model...")
minilm_model = SentenceTransformer('all-MiniLM-L6-v2')

# Helper: format brand text exactly as in src/relevance.py
def get_brand_text(desc, ind, aud):
    return f"{desc} {ind} {aud}".strip()

# Helper: split hashtags into tokens for TF-IDF
def split_hashtag(ht):
    s = ht.lstrip('#')
    words = re.findall(r'[A-Za-z]+|\d+', s)
    return " ".join([w.lower() for w in words])

# Build TF-IDF representations from category hashtag pools
print("Building TF-IDF category representations...")
cat_pool_texts = {}
for cat in ACTIVE_CATEGORIES:
    pool_hts = trends_df[trends_df['pool'] == cat]['hashtag'].tolist()
    tokens = [split_hashtag(h) for h in pool_hts]
    cat_pool_texts[cat] = " ".join(tokens)

tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1, sublinear_tf=True)
tfidf_centroid_matrix = tfidf_vec.fit_transform([cat_pool_texts[c] for c in ACTIVE_CATEGORIES])

# Historical notebook 05 lexical relevance function
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

# Semantic ground truth relevance: hashtag is in the category's pool in trends.parquet
cat_to_hashtags = {}
for cat in ACTIVE_CATEGORIES:
    cat_to_hashtags[cat] = set(trends_df[trends_df['pool'] == cat]['hashtag'])

def is_relevant_semantic(brand_cat, hashtag):
    if brand_cat in cat_to_hashtags:
        return 1 if hashtag in cat_to_hashtags[brand_cat] else 0
    return 0

# Extract MiniLM embeddings for test brands
print("Encoding test brand texts...")
brand_texts = [get_brand_text(b['business_description'], b['industry'], b['target_audience']) for b in test_set]
brand_embs = minilm_model.encode(brand_texts)

# Precalculate MiniLM similarities against category centroids
# Shape: (N_brands, 8)
minilm_sims = cosine_similarity(brand_embs, centroid_matrix)

# Precalculate TF-IDF similarities
brand_tfidf_vecs = tfidf_vec.transform(brand_texts)
tfidf_sims = cosine_similarity(brand_tfidf_vecs, tfidf_centroid_matrix)

print(f"Computed MiniLM and TF-IDF similarity matrices: {minilm_sims.shape}, {tfidf_sims.shape}")

# Save detailed records
eval_records = []

for idx, b in enumerate(test_set):
    sims = minilm_sims[idx]
    sorted_cat_indices = np.argsort(sims)[::-1]
    
    top1_idx = sorted_cat_indices[0]
    top2_idx = sorted_cat_indices[1]
    
    top1_cat = ACTIVE_CATEGORIES[top1_idx]
    top2_cat = ACTIVE_CATEGORIES[top2_idx]
    
    top1_sim = float(sims[top1_idx])
    top2_sim = float(sims[top2_idx])
    margin = top1_sim - top2_sim
    
    # TF-IDF top predictions
    t_sims = tfidf_sims[idx]
    t_sorted_indices = np.argsort(t_sims)[::-1]
    t_top1_idx = t_sorted_indices[0]
    t_top2_idx = t_sorted_indices[1]
    t_top1_cat = ACTIVE_CATEGORIES[t_top1_idx]
    t_top2_cat = ACTIVE_CATEGORIES[t_top2_idx]
    t_top1_sim = float(t_sims[t_top1_idx])
    t_top2_sim = float(t_sims[t_top2_idx])
    t_margin = t_top1_sim - t_top2_sim
    
    eval_records.append({
        "id": b["id"],
        "brand_name": b["brand_name"],
        "type": b["type"],
        "expected": b["expected_category"],
        "category_hint": b.get("category_hint"),
        "minilm_top1_cat": top1_cat,
        "minilm_top1_sim": top1_sim,
        "minilm_top2_cat": top2_cat,
        "minilm_top2_sim": top2_sim,
        "minilm_margin": margin,
        "minilm_all_sims": {cat: float(s) for cat, s in zip(ACTIVE_CATEGORIES, sims)},
        "tfidf_top1_cat": t_top1_cat,
        "tfidf_top1_sim": t_top1_sim,
        "tfidf_top2_cat": t_top2_cat,
        "tfidf_top2_sim": t_top2_sim,
        "tfidf_margin": t_margin,
        "tfidf_all_sims": {cat: float(s) for cat, s in zip(ACTIVE_CATEGORIES, t_sims)}
    })

eval_df = pd.DataFrame(eval_records)
out_records_path = os.path.join(os.path.dirname(__file__), "data", "brand_similarity_records.json")
with open(out_records_path, "w", encoding="utf-8") as f:
    json.dump(eval_records, f, indent=2)

print(f"Saved evaluation records to {out_records_path}")
