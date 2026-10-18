import os
import sys
import yaml
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
sys.path.append(os.path.join(PROJECT_ROOT, 'experiments'))

from walk_forward_extractor import extract_origin_data, TUNE_ORIGINS, CONFIRM_ORIGINS, ALL_ORIGINS

with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

print("=" * 70)
print("TREND ENGINE PHASE 2: WALK-FORWARD TEMPORAL BENCHMARK")
print("=" * 70)

# 1. Precompute datasets for all 5 origins
origin_dfs = {}
for orig in ALL_ORIGINS:
    df = extract_origin_data(orig, config, pool='all', min_posts=10, min_users=3)
    if df is not None:
        origin_dfs[orig] = df
        print(f"Origin {orig}: N = {len(df)} eligible hashtags.")
    else:
        print(f"Origin {orig}: FAILED to extract data.")

# -------------------------------------------------------------
# Analysis A: Target Diagnosis & Correlation Matrix
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("ANALYSIS A: CANDIDATE TARGET AUDIT & CORRELATIONS (Tuning Set)")
print("=" * 70)

tune_combined = pd.concat([origin_dfs[o] for o in TUNE_ORIGINS], ignore_index=True)
features = ['frequency', 'growth', 'engagement', 'recency']
targets = ['target_future_share', 'target_share_delta', 'target_rank_delta', 'target_Y4', 'target_legacy']

corr_table = []
for f in features:
    row = {'Feature': f}
    for t in targets:
        r, _ = spearmanr(tune_combined[f], tune_combined[t], nan_policy='omit')
        row[t] = round(r, 4)
    corr_table.append(row)

corr_df = pd.DataFrame(corr_table)
print(corr_df.to_string(index=False))

# Target-Target Correlations
print("\nTarget-to-Target Spearman Correlations (Tuning Set):")
target_corr = tune_combined[targets].corr(method='spearman').round(4)
print(target_corr.to_string())

# Mean reversion analysis on legacy vs Y4
print("\nDenominator / Mean-Reversion Check:")
low_cur_share = tune_combined[tune_combined['share_cur'] < tune_combined['share_cur'].median()]
high_cur_share = tune_combined[tune_combined['share_cur'] >= tune_combined['share_cur'].median()]
print(f"Low current share: Mean legacy target = {low_cur_share['target_legacy'].mean():.4f}, Mean delta share = {low_cur_share['target_share_delta'].mean():.6f}")
print(f"High current share: Mean legacy target = {high_cur_share['target_legacy'].mean():.4f}, Mean delta share = {high_cur_share['target_share_delta'].mean():.6f}")

# -------------------------------------------------------------
# Analysis B: Weight Formulations on Tuning Set
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("ANALYSIS B: EVALUATING WEIGHT CANDIDATES (Tuning Set)")
print("=" * 70)

CANDIDATES = {
    "1. Recency Baseline [0, 0, 0, 1]": [0.0, 0.0, 0.0, 1.0],
    "2. Frequency Only [1, 0, 0, 0]": [1.0, 0.0, 0.0, 0.0],
    "3. Growth Only [0, 1, 0, 0]": [0.0, 1.0, 0.0, 0.0],
    "4. Equal [0.25, 0.25, 0.25, 0.25]": [0.25, 0.25, 0.25, 0.25],
    "5. Momentum [0.3, 0.5, 0, 0.2]": [0.3, 0.5, 0.0, 0.2],
    "6. Balanced (Phase 1) [0.3, 0.4, 0.1, 0.2]": [0.3, 0.4, 0.1, 0.2],
    "7. Growth Dominant [0.2, 0.6, 0, 0.2]": [0.2, 0.6, 0.0, 0.2],
    "8. High Frequency Balanced [0.5, 0.3, 0.1, 0.1]": [0.5, 0.3, 0.1, 0.1]
}

def evaluate_candidate_on_origin(df, weights):
    # Candidate score
    score = (
        weights[0] * df['frequency'] +
        weights[1] * df['growth'] +
        weights[2] * df['engagement'] +
        weights[3] * df['recency']
    )
    df_eval = df.copy()
    df_eval['c_score'] = score
    
    # Rank by candidate score
    df_eval = df_eval.sort_values('c_score', ascending=False).reset_index(drop=True)
    top10 = df_eval.head(10)
    
    # Metrics
    falling_pct = (top10['trend_direction'] == 'falling').mean() * 100.0
    rising_pct = (top10['trend_direction'] == 'rising').mean() * 100.0
    stable_pct = (top10['trend_direction'] == 'stable').mean() * 100.0
    
    # Spearman corr with target_Y4
    corr_y4, _ = spearmanr(df_eval['c_score'], df_eval['target_Y4'], nan_policy='omit')
    
    # NDCG@10 with target_Y4
    true_rel = np.asarray([df_eval['target_Y4'].values])
    pred_scores = np.asarray([df_eval['c_score'].values])
    ndcg = ndcg_score(true_rel, pred_scores, k=10)
    
    return {
        'ndcg10': ndcg,
        'spearman_y4': corr_y4,
        'falling10': falling_pct,
        'rising10': rising_pct,
        'stable10': stable_pct
    }

tuning_results = []
for c_name, w in CANDIDATES.items():
    orig_ndcg = []
    orig_corr = []
    orig_falling = []
    orig_rising = []
    
    for orig in TUNE_ORIGINS:
        res = evaluate_candidate_on_origin(origin_dfs[orig], w)
        orig_ndcg.append(res['ndcg10'])
        orig_corr.append(res['spearman_y4'])
        orig_falling.append(res['falling10'])
        orig_rising.append(res['rising10'])
        
    tuning_results.append({
        'Candidate': c_name,
        'Mean NDCG@10': round(np.mean(orig_ndcg), 4),
        'Std NDCG@10': round(np.std(orig_ndcg), 4),
        'Mean Corr(Y4)': round(np.mean(orig_corr), 4),
        'Std Corr(Y4)': round(np.std(orig_corr), 4),
        'Mean Falling@10 (%)': round(np.mean(orig_falling), 1),
        'Mean Rising@10 (%)': round(np.mean(orig_rising), 1),
    })

tuning_res_df = pd.DataFrame(tuning_results)
print("\nTuning Results Across Origins (2018-12-26, 2019-01-23, 2019-02-20):")
print(tuning_res_df.to_string(index=False))

# -------------------------------------------------------------
# Analysis C: Grid Search on Tuning Set
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("ANALYSIS C: SYSTEMATIC GRID SEARCH ON TUNING SET (Step = 0.1)")
print("=" * 70)

grid_combos = []
for w1 in np.arange(0.0, 1.05, 0.1):
    for w2 in np.arange(0.0, 1.05 - w1, 0.1):
        for w3 in np.arange(0.0, 1.05 - w1 - w2, 0.1):
            w4 = round(1.0 - w1 - w2 - w3, 2)
            if w4 >= -1e-5:
                w_tuple = (round(w1, 1), round(w2, 1), round(w3, 1), max(0.0, w4))
                if abs(sum(w_tuple) - 1.0) < 1e-4:
                    grid_combos.append(w_tuple)

grid_evals = []
for w in grid_combos:
    ndcgs = []
    corrs = []
    fallings = []
    for orig in TUNE_ORIGINS:
        res = evaluate_candidate_on_origin(origin_dfs[orig], w)
        ndcgs.append(res['ndcg10'])
        corrs.append(res['spearman_y4'])
        fallings.append(res['falling10'])
    grid_evals.append({
        'w': w,
        'mean_ndcg': np.mean(ndcgs),
        'mean_corr': np.mean(corrs),
        'mean_falling': np.mean(fallings),
        'utility': np.mean(ndcgs) + np.mean(corrs) - (np.mean(fallings) / 100.0)
    })

grid_df = pd.DataFrame(grid_evals).sort_values('utility', ascending=False)
print("Top 5 Grid Candidates by Composite Objective (NDCG + Corr - Falling Penalty):")
for idx, r in grid_df.head(5).iterrows():
    print(f"  w={r['w']}: Mean NDCG@10={r['mean_ndcg']:.4f}, Mean Corr={r['mean_corr']:.4f}, Falling@10={r['mean_falling']:.1f}%")

best_grid_w = list(grid_df.iloc[0]['w'])
print(f"\nOptimal Grid Candidate on Tuning: {best_grid_w}")

# -------------------------------------------------------------
# Analysis D: Lock Candidate & Evaluate ONCE on Confirmation Set
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("ANALYSIS D: UNTOUCHED CONFIRMATION SET EVALUATION (2019-03-20, 2019-04-17)")
print("=" * 70)

# Compare: Baseline, Phase 1 Balanced, and Best Grid on Confirmation Set
eval_candidates = {
    "Baseline [0, 0, 0, 1]": [0.0, 0.0, 0.0, 1.0],
    "Phase 1 Balanced [0.3, 0.4, 0.1, 0.2]": [0.3, 0.4, 0.1, 0.2],
    f"Tuned Optimal {best_grid_w}": best_grid_w
}

confirm_results = []
for c_name, w in eval_candidates.items():
    res_c1 = evaluate_candidate_on_origin(origin_dfs[CONFIRM_ORIGINS[0]], w)
    res_c2 = evaluate_candidate_on_origin(origin_dfs[CONFIRM_ORIGINS[1]], w)
    
    mean_ndcg = (res_c1['ndcg10'] + res_c2['ndcg10']) / 2.0
    mean_corr = (res_c1['spearman_y4'] + res_c2['spearman_y4']) / 2.0
    mean_falling = (res_c1['falling10'] + res_c2['falling10']) / 2.0
    mean_rising = (res_c1['rising10'] + res_c2['rising10']) / 2.0
    
    confirm_results.append({
        'Candidate': c_name,
        'C1 (2019-03-20) NDCG': round(res_c1['ndcg10'], 4),
        'C2 (2019-04-17) NDCG': round(res_c2['ndcg10'], 4),
        'Mean Confirm NDCG@10': round(mean_ndcg, 4),
        'Mean Confirm Corr(Y4)': round(mean_corr, 4),
        'Mean Confirm Falling@10 (%)': round(mean_falling, 1),
        'Mean Confirm Rising@10 (%)': round(mean_rising, 1),
    })

confirm_df = pd.DataFrame(confirm_results)
print(confirm_df.to_string(index=False))

# -------------------------------------------------------------
# Analysis E: Signal Marginal Value & Ablation Check
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("ANALYSIS E: MARGINAL SIGNAL VALUE & ABLATION (All 5 Origins Combined)")
print("=" * 70)

all_combined = pd.concat([origin_dfs[o] for o in ALL_ORIGINS], ignore_index=True)
for sig in ['frequency', 'growth', 'engagement', 'recency']:
    r_y4, p_val = spearmanr(all_combined[sig], all_combined['target_Y4'])
    r_delta, _ = spearmanr(all_combined[sig], all_combined['target_share_delta'])
    r_future, _ = spearmanr(all_combined[sig], all_combined['target_future_share'])
    print(f"{sig.capitalize():12s}: Corr(Y4)={r_y4:+.4f} (p={p_val:.2e}) | Corr(DeltaShare)={r_delta:+.4f} | Corr(FutureShare)={r_future:+.4f}")

print("\n" + "=" * 70)
print("WALK-FORWARD TEMPORAL BENCHMARK COMPLETE")
print("=" * 70)
