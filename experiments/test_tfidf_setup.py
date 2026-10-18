import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

trends = pd.read_parquet('paper/artifacts/v1.0/trends.parquet')
pools = ['beauty', 'family', 'fashion', 'fitness', 'food', 'interior', 'pet', 'travel']

def split_hashtag(ht):
    # Strip # and split camelCase or snake_case or numbers
    s = ht.lstrip('#')
    words = re.findall(r'[A-Za-z]+|\d+', s)
    return " ".join([w.lower() for w in words])

# Build category texts from the pools in trends
cat_texts = {}
for p in pools:
    hts = trends[trends['pool'] == p]['hashtag'].tolist()
    words_list = [split_hashtag(h) for h in hts]
    cat_texts[p] = " ".join(words_list)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
tfidf_mat = vectorizer.fit_transform([cat_texts[p] for p in pools])

test_queries = [
    ("CampusWear", "We sell trendy college apparel, hoodies, and sweatpants for students. fashion university students"),
    ("GlowTheory", "Clean, vegan skincare essentials for radiant and healthy skin. beauty skincare enthusiasts"),
    ("FitForge", "High-intensity functional training gear, barbells, and resistance bands. fitness crossfit athletes"),
    ("B3_from_nb", "Trendy streetwear and oversized hoodies fashion apparel teens and young adults")
]

for name, q in test_queries:
    q_vec = vectorizer.transform([q])
    sims = cosine_similarity(q_vec, tfidf_mat)[0]
    best_idx = np.argmax(sims)
    print(f"\n{name}:")
    for p, s in zip(pools, sims):
        print(f"  {p}: {s:.4f}")
    print(f"  -> Top TF-IDF: {pools[best_idx]} ({sims[best_idx]:.4f})")
