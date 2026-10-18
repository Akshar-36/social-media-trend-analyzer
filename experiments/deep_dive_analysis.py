import os
import sys
import yaml
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import ndcg_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from walk_forward_extractor import extract_origin_data, TUNE_ORIGINS, CONFIRM_ORIGINS, ALL_ORIGINS

with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

origin_dfs = {}
for orig in ALL_ORIGINS:
    origin_dfs[orig] = extract_origin_data(orig, config, pool='all', min_posts=10, min_users=3)

# -------------------------------------------------------------
# 1. EVALUATION FUNCTION: FILTERED VS UNFILTERED
# -------------------------------------------------------------
def evaluate_candidate(df, weights, apply_direction_filter=False):
    df_eval = df.copy()
    score = (
        weights[0] * df_eval['frequency'] +
        weights[1] * df_eval['growth'] +
        weights[2] * df_eval['engagement'] +
        weights[3] * df_eval['recency']
    )
    df_eval['c_score'] = score
    
    # Global ranking correlation with target_Y4
    corr_y4, p_val = spearmanr(df_eval['c_score'], df_eval['target_Y4'], nan_policy='omit')
    
    if apply_direction_filter:
        eligible = df_eval[df_eval['trend_direction'] != 'falling']
        if len(eligible) < 10:
            eligible = df_eval  # fallback
    else:
        eligible = df_eval
        
    eligible_sorted = eligible.sort_values('c_score', ascending=False).reset_index(drop=True)
    top10 = eligible_sorted.head(10)
    
    falling_pct = (top10['trend_direction'] == 'falling').mean() * 100.0
    rising_pct = (top10['trend_direction'] == 'rising').mean() * 100.0
    stable_pct = (top10['trend_direction'] == 'stable').mean() * 100.0
    
    # NDCG@10 of the surfaced top 10 against target_Y4
    true_rel = np.asarray([df_eval['target_Y4'].values])
    if apply_direction_filter:
        df_eval_filtered = df_eval.copy()
        df_eval_filtered.loc[df_eval_filtered['trend_direction'] == 'falling', 'c_score'] = -999.0
        pred_scores = np.asarray([df_eval_filtered['c_score'].values])
    else:
        pred_scores = np.asarray([df_eval['c_score'].values])
        
    ndcg = ndcg_score(true_rel, pred_scores, k=10)
    
    return {
        'ndcg10': ndcg,
        'spearman_y4': corr_y4,
        'falling10': falling_pct,
        'rising10': rising_pct,
        'stable10': stable_pct
    }

CANDIDATES = {
    "Baseline [0, 0, 0, 1]": [0.0, 0.0, 0.0, 1.0],
    "Frequency Only [1, 0, 0, 0]": [1.0, 0.0, 0.0, 0.0],
    "Growth Only [0, 1, 0, 0]": [0.0, 1.0, 0.0, 0.0],
    "Equal [0.25, 0.25, 0.25, 0.25]": [0.25, 0.25, 0.25, 0.25],
    "Momentum [0.3, 0.5, 0, 0.2]": [0.3, 0.5, 0.0, 0.2],
    "Balanced (Phase 1) [0.3, 0.4, 0.1, 0.2]": [0.3, 0.4, 0.1, 0.2],
    "Growth Dominant [0.2, 0.6, 0, 0.2]": [0.2, 0.6, 0.0, 0.2],
    "Frequency Dominant [0.7, 0.1, 0.0, 0.2]": [0.7, 0.1, 0.0, 0.2],
    "Frequency-Recency [0.8, 0.0, 0.0, 0.2]": [0.8, 0.0, 0.0, 0.2]
}

print("=" * 80)
print("PART 1: TUNING SET EVALUATION — UNFILTERED VS FILTERED")
print("=" * 80)

rows_tune = []
for name, w in CANDIDATES.items():
    unfilt_ndcg = [evaluate_candidate(origin_dfs[o], w, False)['ndcg10'] for o in TUNE_ORIGINS]
    unfilt_corr = [evaluate_candidate(origin_dfs[o], w, False)['spearman_y4'] for o in TUNE_ORIGINS]
    unfilt_fall = [evaluate_candidate(origin_dfs[o], w, False)['falling10'] for o in TUNE_ORIGINS]
    
    filt_ndcg = [evaluate_candidate(origin_dfs[o], w, True)['ndcg10'] for o in TUNE_ORIGINS]
    filt_fall = [evaluate_candidate(origin_dfs[o], w, True)['falling10'] for o in TUNE_ORIGINS]
    
    rows_tune.append({
        'Candidate': name,
        'Raw NDCG@10': f"{np.mean(unfilt_ndcg):.4f} (+/- {np.std(unfilt_ndcg):.4f})",
        'Raw Corr': f"{np.mean(unfilt_corr):.4f} (+/- {np.std(unfilt_corr):.4f})",
        'Raw Falling%': f"{np.mean(unfilt_fall):.1f}%",
        'Filtered NDCG@10': f"{np.mean(filt_ndcg):.4f} (+/- {np.std(filt_ndcg):.4f})",
        'Filtered Falling%': f"{np.mean(filt_fall):.1f}%"
    })

print(pd.DataFrame(rows_tune).to_string(index=False))

print("\n" + "=" * 80)
print("PART 2: GRID SEARCH OVER SIMPLEX ON TUNING SET (WITH DIRECTION FILTER)")
print("=" * 80)

grid_combos = []
for w1 in np.arange(0.0, 1.05, 0.1):
    for w2 in np.arange(0.0, 1.05 - w1, 0.1):
        for w3 in np.arange(0.0, 1.05 - w1 - w2, 0.1):
            w4 = round(1.0 - w1 - w2 - w3, 2)
            if w4 >= -1e-5:
                w_tuple = (round(w1, 1), round(w2, 1), round(w3, 1), max(0.0, w4))
                if abs(sum(w_tuple) - 1.0) < 1e-4:
                    grid_combos.append(w_tuple)

grid_results = []
for w in grid_combos:
    filt_ndcgs = [evaluate_candidate(origin_dfs[o], w, True)['ndcg10'] for o in TUNE_ORIGINS]
    corrs = [evaluate_candidate(origin_dfs[o], w, False)['spearman_y4'] for o in TUNE_ORIGINS]
    grid_results.append({
        'weights': w,
        'mean_filtered_ndcg': np.mean(filt_ndcgs),
        'std_filtered_ndcg': np.std(filt_ndcgs),
        'mean_corr': np.mean(corrs),
        'score': np.mean(filt_ndcgs) + 0.5 * np.mean(corrs)
    })

grid_df = pd.DataFrame(grid_results).sort_values('mean_filtered_ndcg', ascending=False)
print("Top 5 Grid Candidates by Filtered NDCG@10 on Tuning Set:")
for _, r in grid_df.head(5).iterrows():
    print(f"  Weights={r['weights']} | Filtered NDCG@10={r['mean_filtered_ndcg']:.4f} (+/- {r['std_filtered_ndcg']:.4f}) | Global Corr={r['mean_corr']:.4f}")

locked_candidate_weights = list(grid_df.iloc[0]['weights'])
locked_name = f"Locked Optimal {locked_candidate_weights}"
print(f"\nFORMALLY LOCKED CANDIDATE FROM TUNING SET: {locked_candidate_weights}")

print("\n" + "=" * 80)
print("PART 3: UNTOUCHED CONFIRMATION SET EVALUATION (2019-03-20, 2019-04-17)")
print("=" * 80)

confirm_candidates = {
    "Baseline [0, 0, 0, 1]": [0.0, 0.0, 0.0, 1.0],
    "Phase 1 Balanced [0.3, 0.4, 0.1, 0.2]": [0.3, 0.4, 0.1, 0.2],
    locked_name: locked_candidate_weights
}

rows_confirm = []
for name, w in confirm_candidates.items():
    c1_res_raw = evaluate_candidate(origin_dfs[CONFIRM_ORIGINS[0]], w, False)
    c2_res_raw = evaluate_candidate(origin_dfs[CONFIRM_ORIGINS[1]], w, False)
    c1_res_filt = evaluate_candidate(origin_dfs[CONFIRM_ORIGINS[0]], w, True)
    c2_res_filt = evaluate_candidate(origin_dfs[CONFIRM_ORIGINS[1]], w, True)
    
    mean_raw_ndcg = (c1_res_raw['ndcg10'] + c2_res_raw['ndcg10']) / 2.0
    mean_raw_corr = (c1_res_raw['spearman_y4'] + c2_res_raw['spearman_y4']) / 2.0
    mean_raw_fall = (c1_res_raw['falling10'] + c2_res_raw['falling10']) / 2.0
    
    mean_filt_ndcg = (c1_res_filt['ndcg10'] + c2_res_filt['ndcg10']) / 2.0
    mean_filt_fall = (c1_res_filt['falling10'] + c2_res_filt['falling10']) / 2.0
    
    rows_confirm.append({
        'Candidate': name,
        'C1 Raw NDCG': f"{c1_res_raw['ndcg10']:.4f}",
        'C2 Raw NDCG': f"{c2_res_raw['ndcg10']:.4f}",
        'Mean Confirm Raw NDCG': f"{mean_raw_ndcg:.4f}",
        'Mean Confirm Corr': f"{mean_raw_corr:.4f}",
        'Mean Confirm Raw Falling%': f"{mean_raw_fall:.1f}%",
        'Mean Confirm Filtered NDCG': f"{mean_filt_ndcg:.4f}",
        'Mean Confirm Filtered Falling%': f"{mean_filt_fall:.1f}%"
    })

print(pd.DataFrame(rows_confirm).to_string(index=False))

print("\n" + "=" * 80)
print("PART 4: MEAN-REVERSION & PART-WHOLE CORRELATION AUDIT")
print("=" * 80)

all_df = pd.concat([origin_dfs[o] for o in ALL_ORIGINS], ignore_index=True)

corr_cur_delta, _ = spearmanr(all_df['share_cur'], all_df['target_share_delta'])
corr_cur_future, _ = spearmanr(all_df['share_cur'], all_df['target_future_share'])
corr_growth_delta, _ = spearmanr(all_df['growth'], all_df['target_share_delta'])
corr_growth_future, _ = spearmanr(all_df['growth'], all_df['target_future_share'])

print(f"Spearman Corr(current_share, target_share_delta): {corr_cur_delta:.4f}  (Part-whole negative coupling)")
print(f"Spearman Corr(current_share, target_future_share): {corr_cur_future:.4f}  (Volume persistence)")
print(f"Spearman Corr(historical_growth, target_share_delta): {corr_growth_delta:.4f}  (Mean reversion of growth spikes)")
print(f"Spearman Corr(historical_growth, target_future_share): {corr_growth_future:.4f}  (Zero persistence of growth spikes)")

# Inspect top quartile of historical growth
q75_growth = all_df['growth'].quantile(0.75)
high_growth_df = all_df[all_df['growth'] >= q75_growth]
low_growth_df = all_df[all_df['growth'] < q75_growth]

print(f"\nHigh historical growth (top 25%): Mean share delta = {high_growth_df['target_share_delta'].mean():.6f}, % with delta < 0 = {(high_growth_df['target_share_delta'] < 0).mean()*100:.1f}%")
print(f"Low/Normal historical growth (< 75%): Mean share delta = {low_growth_df['target_share_delta'].mean():.6f}, % with delta < 0 = {(low_growth_df['target_share_delta'] < 0).mean()*100:.1f}%")

print("\n" + "=" * 80)
print("PART 5: SPARSE-HASHTAG SENSITIVITY AUDIT (min_posts=5 vs 10 vs 15)")
print("=" * 80)

for mp in [5, 10, 15, 20]:
    counts = []
    corrs = []
    for o in TUNE_ORIGINS:
        df_mp = extract_origin_data(o, config, pool='all', min_posts=mp, min_users=2)
        counts.append(len(df_mp))
        r, _ = spearmanr(df_mp['growth'], df_mp['target_share_delta'])
        corrs.append(r)
    print(f"min_posts={mp:2d}: Mean eligible hashtags={np.mean(counts):.1f} | Mean Corr(growth, delta_share)={np.mean(corrs):.4f}")
