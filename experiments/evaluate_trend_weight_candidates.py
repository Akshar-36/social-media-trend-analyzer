import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import joblib

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from relevance import predict_category, get_brand_text
from hashtags import get_candidates_for_brand, calculate_historical_engagement, rank_hashtags

print("=================================================================")
print("OFFLINE TREND ENGINE CANDIDATE WEIGHT BENCHMARK")
print("=================================================================")

# Load Artifacts
trends_path = os.path.join(PROJECT_ROOT, "artifacts", "v1.0", "trends.parquet")
h_emb_path = os.path.join(PROJECT_ROOT, "artifacts", "v1.0", "hashtag_embeddings.parquet")
hist_path = os.path.join(PROJECT_ROOT, "artifacts", "v1.0", "historical_engagement.parquet")
centroids_path = os.path.join(PROJECT_ROOT, "artifacts", "v1.0", "category_centroids.pkl")
config_path = os.path.join(PROJECT_ROOT, "config.yaml")

import yaml
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

trends_df = pd.read_parquet(trends_path)
h_emb_df = pd.read_parquet(h_emb_path)
history_df = pd.read_parquet(hist_path)
with open(centroids_path, 'rb') as f:
    centroids = joblib.load(f)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Candidate Weight Configurations
CANDIDATES = {
    "1. Baseline [0, 0, 0, 1]": [0.0, 0.0, 0.0, 1.0],
    "2. Equal [0.25, 0.25, 0.25, 0.25]": [0.25, 0.25, 0.25, 0.25],
    "3. Momentum [0.3, 0.5, 0, 0.2]": [0.3, 0.5, 0.0, 0.2],
    "4. Balanced [0.3, 0.4, 0.1, 0.2]": [0.3, 0.4, 0.1, 0.2],
    "5. Growth-dominant [0.2, 0.6, 0, 0.2]": [0.2, 0.6, 0.0, 0.2]
}

POOLS = ['all', 'fashion', 'food', 'travel', 'family', 'interior', 'beauty', 'fitness', 'pet']

# -------------------------------------------------------------
# Part 1: Overall and Per-Category Ranking Metrics
# -------------------------------------------------------------
print("\n--- Part 1: Overall Pool Ranking & Direction Metrics ---")

overall_metrics = []

for c_name, w in CANDIDATES.items():
    # Compute candidate score
    c_score = w[0]*trends_df['frequency'] + w[1]*trends_df['growth'] + w[2]*trends_df['engagement'] + w[3]*trends_df['recency']
    df_eval = trends_df.copy()
    df_eval['c_score'] = c_score
    
    # Global correlations
    corr_growth, _ = spearmanr(df_eval['c_score'], df_eval['growth'])
    corr_freq, _ = spearmanr(df_eval['c_score'], df_eval['frequency'])
    
    # High-score falling count (score > 0.8)
    high_score_falling = ((df_eval['c_score'] > 0.8) & (df_eval['trend_direction'] == 'falling')).sum()
    
    # Top 10 aggregated across pools (mean of top-10 per pool)
    pool_top10_falling = []
    pool_top10_rising = []
    pool_top10_stable = []
    
    for p in POOLS:
        pdf = df_eval[df_eval['pool'] == p]
        top10 = pdf.sort_values('c_score', ascending=False).head(10)
        n = len(top10)
        if n > 0:
            pool_top10_falling.append((top10['trend_direction'] == 'falling').sum() / n * 100)
            pool_top10_rising.append((top10['trend_direction'] == 'rising').sum() / n * 100)
            pool_top10_stable.append((top10['trend_direction'] == 'stable').sum() / n * 100)
            
    mean_falling_10 = np.mean(pool_top10_falling)
    mean_rising_10 = np.mean(pool_top10_rising)
    mean_stable_10 = np.mean(pool_top10_stable)
    not_falling_10 = 100.0 - mean_falling_10
    
    overall_metrics.append({
        "Candidate": c_name,
        "Falling@10 (%)": round(mean_falling_10, 1),
        "Rising@10 (%)": round(mean_rising_10, 1),
        "Stable@10 (%)": round(mean_stable_10, 1),
        "Not-Falling@10 (%)": round(not_falling_10, 1),
        "High-Score Falling (N)": high_score_falling,
        "Corr w/ Growth": round(corr_growth, 3),
        "Corr w/ Freq": round(corr_freq, 3)
    })

df_overall = pd.DataFrame(overall_metrics)
print(df_overall.to_string(index=False))

# -------------------------------------------------------------
# Part 2: Per-Category Breakdown
# -------------------------------------------------------------
print("\n--- Part 2: Per-Category Robustness ---")

per_category_rows = []
for p in POOLS:
    pdf_base = trends_df[trends_df['pool'] == p]
    for c_name, w in CANDIDATES.items():
        c_score = w[0]*pdf_base['frequency'] + w[1]*pdf_base['growth'] + w[2]*pdf_base['engagement'] + w[3]*pdf_base['recency']
        pdf = pdf_base.copy()
        pdf['c_score'] = c_score
        
        top10 = pdf.sort_values('c_score', ascending=False).head(10)
        n = len(top10)
        f_count = (top10['trend_direction'] == 'falling').sum()
        r_count = (top10['trend_direction'] == 'rising').sum()
        s_count = (top10['trend_direction'] == 'stable').sum()
        
        c_growth, _ = spearmanr(pdf['c_score'], pdf['growth'])
        c_freq, _ = spearmanr(pdf['c_score'], pdf['frequency'])
        
        per_category_rows.append({
            "Pool": p,
            "Pool Size": len(pdf),
            "Candidate": c_name.split(". ")[1],
            "Falling@10": f"{f_count}/{n}",
            "Rising@10": f"{r_count}/{n}",
            "Stable@10": f"{s_count}/{n}",
            "Corr Growth": round(c_growth, 3),
            "Corr Freq": round(c_freq, 3)
        })

df_cat = pd.DataFrame(per_category_rows)
print(df_cat.to_string(index=False))

# -------------------------------------------------------------
# Part 3: Diagnostic Brand Simulation (Score-Only vs Direction-Filter)
# -------------------------------------------------------------
print("\n--- Part 3: Diagnostic Brand Simulation ---")

diag_brands = [
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

brand_sim_results = []

for b in diag_brands:
    b_name = b["brand_name"]
    b_text = get_brand_text(b["business_description"], b["industry"], b["target_audience"])
    
    # 1. Resolve Category
    best_cat, _, _ = predict_category(
        b_text, model, centroids,
        margin=config.get('relevance_margin', 0.010),
        floor=config.get('relevance_floor', 0.250)
    )
    
    # 2. Get candidate hashtags using base pool
    pool_trends = trends_df[trends_df['pool'] == best_cat].copy()
    eligible_hashtags = pool_trends['hashtag'].tolist()
    pool_emb_df = h_emb_df[h_emb_df['hashtag'].isin(eligible_hashtags)].copy()
    
    from sklearn.metrics.pairwise import cosine_similarity
    b_emb = model.encode([b_text])[0]
    emb_matrix = np.stack(pool_emb_df['embedding'].values)
    sims = cosine_similarity([b_emb], emb_matrix)[0]
    pool_emb_df['relevance'] = sims
    
    # Candidate pool
    cand_base = pool_emb_df.sort_values('relevance', ascending=False).head(config['candidate_pool_size'])
    cand_base = cand_base.merge(pool_trends[['hashtag', 'frequency', 'growth', 'engagement', 'recency', 'trend_direction']], on='hashtag', how='inner')
    
    # Historical engagement
    hist_eng = calculate_historical_engagement(best_cat, cand_base['hashtag'].tolist(), history_df)
    cand_base = cand_base.merge(hist_eng, on='hashtag', how='left')
    cand_base['historical_engagement'] = cand_base['historical_engagement'].fillna(0.0)
    
    for c_name, w in CANDIDATES.items():
        cand = cand_base.copy()
        cand['trend_score'] = (
            w[0]*cand['frequency'] + 
            w[1]*cand['growth'] + 
            w[2]*cand['engagement'] + 
            w[3]*cand['recency']
        )
        
        # Rank hashtags using candidate trend_score
        ranked = rank_hashtags(cand, config)
        top_n = ranked.head(config['n_hashtags']) # 8 hashtags
        
        # Simulation A: Score-Only Selection (top 3 by candidate trend_score)
        trending_score_only = top_n.sort_values('trend_score', ascending=False).head(config['n_topics'])
        topics_so = [h.replace('#', '') for h in trending_score_only['hashtag'].tolist()]
        dirs_so = trending_score_only['trend_direction'].tolist()
        scores_so = [round(float(s), 3) for s in trending_score_only['trend_score'].tolist()]
        falling_count_so = dirs_so.count('falling')
        
        # Simulation B: Score + Direction Filter Selection
        # Exclude 'falling', then sort by trend_score. If all falling, fallback to score-only.
        non_falling_pool = top_n[top_n['trend_direction'] != 'falling']
        if not non_falling_pool.empty:
            trending_filtered = non_falling_pool.sort_values('trend_score', ascending=False).head(config['n_topics'])
        else:
            trending_filtered = top_n.sort_values('trend_score', ascending=False).head(config['n_topics'])
            
        topics_df = [h.replace('#', '') for h in trending_filtered['hashtag'].tolist()]
        dirs_df = trending_filtered['trend_direction'].tolist()
        scores_df = [round(float(s), 3) for s in trending_filtered['trend_score'].tolist()]
        falling_count_df = dirs_df.count('falling')
        
        brand_sim_results.append({
            "Brand": b_name,
            "Category": best_cat,
            "Candidate": c_name.split(". ")[1],
            "ScoreOnly Topics": topics_so,
            "ScoreOnly Dirs": dirs_so,
            "ScoreOnly Scores": scores_so,
            "ScoreOnly Falling Count": falling_count_so,
            "Filter Topics": topics_df,
            "Filter Dirs": dirs_df,
            "Filter Falling Count": falling_count_df
        })

df_brand_sim = pd.DataFrame(brand_sim_results)

# Print diagnostic brand simulation summary
print("\n=== DIAGNOSTIC BRAND SIMULATION SUMMARY ===")
diag_summary = []
for c_name in CANDIDATES.keys():
    sub = df_brand_sim[df_brand_sim['Candidate'] == c_name.split(". ")[1]]
    total_falling_so = sub['ScoreOnly Falling Count'].sum()
    total_falling_df = sub['Filter Falling Count'].sum()
    diag_summary.append({
        "Candidate": c_name,
        "Total Surfaced Falling (Score-Only)": f"{total_falling_so} / 24",
        "Total Surfaced Falling (With Filter)": f"{total_falling_df} / 24",
        "Falling Reduction (%)": f"{((total_falling_so - total_falling_df)/total_falling_so)*100:.1f}%" if total_falling_so > 0 else "0.0%"
    })
print(pd.DataFrame(diag_summary).to_string(index=False))

# Print detailed brand table for Baseline vs Balanced vs Growth-Dominant
print("\nDetailed Brand-by-Brand Comparison (Score-Only):")
for b_name in [b['brand_name'] for b in diag_brands]:
    print(f"\nBrand: {b_name}")
    sub_b = df_brand_sim[df_brand_sim['Brand'] == b_name]
    for _, r in sub_b.iterrows():
        items_str = ", ".join([f"{t} ({d}, {s})" for t, d, s in zip(r['ScoreOnly Topics'], r['ScoreOnly Dirs'], r['ScoreOnly Scores'])])
        print(f"  {r['Candidate']:25s} | Falling: {r['ScoreOnly Falling Count']} | Topics: {items_str}")

# Save full results payload to json
out_path = os.path.join(PROJECT_ROOT, "experiments", "data", "trend_weight_benchmark_results.json")
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump({
        "overall_metrics": overall_metrics,
        "per_category_metrics": per_category_rows,
        "brand_simulation": brand_sim_results,
        "diagnostic_summary": diag_summary
    }, f, indent=2, default=str)

print(f"\nAll benchmark results serialized to: {out_path}")
