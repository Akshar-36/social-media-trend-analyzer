import os
import sys
import yaml
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from walk_forward_extractor import extract_origin_data, TUNE_ORIGINS, CONFIRM_ORIGINS, ALL_ORIGINS

with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

origin_dfs = {}
for orig in ALL_ORIGINS:
    origin_dfs[orig] = extract_origin_data(orig, config, pool='all', min_posts=10, min_users=3)

CANDIDATES = {
    "Phase 1 Balanced": [0.3, 0.4, 0.1, 0.2],
    "Frequency Dominant": [0.9, 0.0, 0.1, 0.0],
    "Hybrid Stabilized": [0.6, 0.1, 0.1, 0.2],
    "Freq-Recency-Eng": [0.8, 0.0, 0.1, 0.1]
}

def get_candidate_top10(df, weights, filter_falling=True):
    df_eval = df.copy()
    score = (
        weights[0] * df_eval['frequency'] +
        weights[1] * df_eval['growth'] +
        weights[2] * df_eval['engagement'] +
        weights[3] * df_eval['recency']
    )
    df_eval['c_score'] = score
    if filter_falling:
        eligible = df_eval[df_eval['trend_direction'] != 'falling']
        if len(eligible) < 10:
            eligible = df_eval
    else:
        eligible = df_eval
    return eligible.sort_values('c_score', ascending=False).head(10)

def evaluate_metrics(df, weights):
    # compute NDCG, Corr, etc.
    df_eval = df.copy()
    score = (
        weights[0] * df_eval['frequency'] +
        weights[1] * df_eval['growth'] +
        weights[2] * df_eval['engagement'] +
        weights[3] * df_eval['recency']
    )
    df_eval['c_score'] = score
    corr, _ = spearmanr(df_eval['c_score'], df_eval['target_Y4'])
    
    # Filtered
    df_filt = df_eval.copy()
    df_filt.loc[df_filt['trend_direction'] == 'falling', 'c_score'] = -999.0
    true_rel = np.asarray([df_eval['target_Y4'].values])
    pred_scores = np.asarray([df_filt['c_score'].values])
    ndcg = ndcg_score(true_rel, pred_scores, k=10)
    
    top10 = get_candidate_top10(df, weights, filter_falling=True)
    pure_freq_top10 = set(df.sort_values('frequency', ascending=False).head(10)['hashtag'])
    cand_top10 = set(top10['hashtag'])
    jaccard_with_freq = len(pure_freq_top10.intersection(cand_top10)) / 10.0
    
    rising_count = (top10['trend_direction'] == 'rising').sum()
    stable_count = (top10['trend_direction'] == 'stable').sum()
    falling_count = (top10['trend_direction'] == 'falling').sum()
    mean_freq_pct = top10['frequency'].mean()
    mean_growth_pct = top10['growth'].mean()
    
    return {
        'ndcg': ndcg,
        'corr': corr,
        'jaccard_freq': jaccard_with_freq,
        'rising': rising_count,
        'stable': stable_count,
        'falling': falling_count,
        'mean_freq_pct': mean_freq_pct,
        'mean_growth_pct': mean_growth_pct
    }

print("=" * 80)
print("METRICS SUMMARY ACROSS ALL 5 ORIGINS (MEAN +/- STD)")
print("=" * 80)

summary_rows = []
for name, w in CANDIDATES.items():
    metrics_list = [evaluate_metrics(origin_dfs[o], w) for o in ALL_ORIGINS]
    summary_rows.append({
        'Candidate': name,
        'Weights': str(w),
        'NDCG@10': f"{np.mean([m['ndcg'] for m in metrics_list]):.4f} +/- {np.std([m['ndcg'] for m in metrics_list]):.4f}",
        'Corr(Y4)': f"{np.mean([m['corr'] for m in metrics_list]):.4f}",
        'Freq Top-10 Overlap': f"{np.mean([m['jaccard_freq'] for m in metrics_list])*100:.1f}%",
        'Rising / 10': f"{np.mean([m['rising'] for m in metrics_list]):.1f}",
        'Stable / 10': f"{np.mean([m['stable'] for m in metrics_list]):.1f}",
        'Mean Freq Pct': f"{np.mean([m['mean_freq_pct'] for m in metrics_list]):.3f}",
        'Mean Growth Pct': f"{np.mean([m['mean_growth_pct'] for m in metrics_list]):.3f}",
    })

print(pd.DataFrame(summary_rows).to_string(index=False))

print("\n" + "=" * 80)
print("REPRESENTATIVE EXAMPLES: TOP 10 RECOMMENDATIONS AT CONFIRMATION ORIGIN 2019-03-20")
print("=" * 80)

df_c1 = origin_dfs['2019-03-20']
for name, w in CANDIDATES.items():
    top10 = get_candidate_top10(df_c1, w, filter_falling=True)
    print(f"\n--- {name} {w} ---")
    cols = ['hashtag', 'posts_cur', 'posts_next', 'frequency', 'growth', 'trend_direction', 'target_share_delta', 'target_Y4']
    display_df = top10[cols].copy()
    display_df['freq_pct'] = display_df['frequency'].round(2)
    display_df['growth_pct'] = display_df['growth'].round(2)
    display_df['d_share'] = display_df['target_share_delta'].apply(lambda x: f"{x:+.5f}")
    display_df['Y4'] = display_df['target_Y4'].round(3)
    out = display_df[['hashtag', 'posts_cur', 'posts_next', 'freq_pct', 'growth_pct', 'trend_direction', 'd_share', 'Y4']]
    print(out.to_string(index=False))
