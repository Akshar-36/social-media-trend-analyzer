import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "experiments", "data")
CALIB_PATH = os.path.join(DATA_DIR, "relevance_calibration_set.json")

with open(CALIB_PATH, 'r', encoding='utf-8') as f:
    calib_data = json.load(f)

ACTIVE_CATEGORIES = ['beauty', 'family', 'fashion', 'fitness', 'food', 'interior', 'pet', 'travel']

print("Loading MiniLM...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Group calibration queries by category
cat_queries = {c: [] for c in ACTIVE_CATEGORIES}
for r in calib_data:
    cat = r['raw_source_category']
    if cat in cat_queries:
        cat_queries[cat].append(r['query_natural'])

print("Calibration query counts per category:")
for c, q_list in cat_queries.items():
    print(f"  {c:10s}: {len(q_list)} queries")

# Encode each category
cat_embs = {}
for c in ACTIVE_CATEGORIES:
    embs = model.encode(cat_queries[c], batch_size=32, show_progress_bar=False)
    cat_embs[c] = embs
    print(f"Encoded {c}: shape {embs.shape}")

# Test KMeans k=3, 5, 10
for k in [3, 5, 10]:
    prototypes = {}
    for c in ACTIVE_CATEGORIES:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(cat_embs[c])
        prototypes[c] = kmeans.cluster_centers_
    print(f"Fitted K-Means prototypes for k={k}. Each category has {k} prototype vectors of shape (384,).")
