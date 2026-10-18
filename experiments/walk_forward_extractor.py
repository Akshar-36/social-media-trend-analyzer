import os
import sys
import yaml
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

DATA_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'data')
posts_path = os.path.join(DATA_DIR, 'temporal_posts.parquet')
hashtags_path = os.path.join(DATA_DIR, 'temporal_hashtags.parquet')

print("Loading cached temporal data...")
posts = pd.read_parquet(posts_path)
hashtags = pd.read_parquet(hashtags_path)

with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

# Origins
TUNE_ORIGINS = ['2018-12-26', '2019-01-23', '2019-02-20']
CONFIRM_ORIGINS = ['2019-03-20', '2019-04-17']
ALL_ORIGINS = TUNE_ORIGINS + CONFIRM_ORIGINS

def extract_origin_data(origin, cfg, pool='all', min_posts=10, min_users=3):
    """
    Extracts features up to origin (strict separation) and future targets in (origin, origin + 28d].
    """
    period_days = cfg['period_days']
    origin_dt = pd.to_datetime(origin, utc=True)
    prev_start = origin_dt - pd.Timedelta(days=2 * period_days)
    curr_start = origin_dt - pd.Timedelta(days=period_days)
    next_end = origin_dt + pd.Timedelta(days=period_days)

    # 1. Feature context: (origin - 56d, origin]
    pool_posts = posts[(posts['posted_at'] > prev_start) & (posts['posted_at'] <= origin_dt)].copy()
    if pool != 'all':
        pool_posts = pool_posts[pool_posts['category'] == pool]

    if pool_posts.empty:
        return None

    pool_posts['is_current'] = pool_posts['posted_at'] > curr_start
    curr_pool_posts = pool_posts[pool_posts['is_current']].copy()

    # Winsorize ER in current window
    winsor_pct = cfg['er_winsor_pct']
    if not curr_pool_posts['engagement_rate'].isna().all():
        er_cap = curr_pool_posts['engagement_rate'].quantile(winsor_pct)
        curr_pool_posts['engagement_rate_win'] = curr_pool_posts['engagement_rate'].clip(upper=er_cap)
    else:
        curr_pool_posts['engagement_rate_win'] = np.nan

    # Hashtag aggregation
    ph = pool_posts[['post_id', 'username', 'posted_at', 'is_current']].merge(hashtags, on='post_id')
    stats = ph.groupby('hashtag').agg(
        posts_cur=('is_current', 'sum'),
        posts_prev=('is_current', lambda x: (~x).sum()),
        users_total=('username', 'nunique')
    ).reset_index()
    stats['posts_total'] = stats['posts_cur'] + stats['posts_prev']

    eligible = stats[
        (stats['posts_total'] >= min_posts) &
        (stats['users_total'] >= min_users)
    ].copy()

    if eligible.empty:
        return None

    pool_cur_posts = pool_posts['is_current'].sum()
    pool_prev_posts = (~pool_posts['is_current']).sum()
    if pool_cur_posts == 0 or pool_prev_posts == 0:
        return None

    eligible['share_cur'] = eligible['posts_cur'] / pool_cur_posts
    eligible['share_prev'] = eligible['posts_prev'] / pool_prev_posts

    # Features
    eligible['frequency_raw'] = eligible['share_cur'] * 10000
    a = cfg.get('growth_smoothing', 0.001)
    eligible['growth_raw'] = np.log((eligible['share_cur'] + a) / (eligible['share_prev'] + a))

    curr_ph = curr_pool_posts[['post_id', 'posted_at', 'engagement_rate_win']].merge(hashtags, on='post_id')
    half_life = period_days / 4
    age_days = (origin_dt - curr_ph['posted_at']).dt.total_seconds() / 86400.0
    curr_ph['recency_raw'] = np.exp(-np.log(2) * age_days / half_life)

    eng_rec = curr_ph.groupby('hashtag').agg(
        engagement_raw=('engagement_rate_win', 'mean'),
        recency_raw=('recency_raw', 'mean')
    ).reset_index()

    eligible = eligible.merge(eng_rec, on='hashtag', how='left')
    eligible['engagement_raw'] = eligible['engagement_raw'].fillna(0)
    eligible['recency_raw'] = eligible['recency_raw'].fillna(0)

    for col in ['frequency', 'growth', 'engagement', 'recency']:
        eligible[col] = eligible[f'{col}_raw'].rank(pct=True)

    # Direction classification
    theta = cfg.get('theta', 0.103116)
    eligible['trend_direction'] = 'stable'
    eligible.loc[eligible['growth_raw'] > theta, 'trend_direction'] = 'rising'
    eligible.loc[eligible['growth_raw'] < -theta, 'trend_direction'] = 'falling'

    # 2. Future context: strictly (origin, origin + 28d]
    next_posts = posts[(posts['posted_at'] > origin_dt) & (posts['posted_at'] <= next_end)].copy()
    if pool != 'all':
        next_posts = next_posts[next_posts['category'] == pool]

    pool_next_posts = len(next_posts)
    if pool_next_posts == 0:
        return None

    next_ph = next_posts[['post_id']].merge(hashtags, on='post_id')
    next_counts = next_ph.groupby('hashtag').size().reset_index(name='posts_next')

    df = eligible.merge(next_counts, on='hashtag', how='left')
    df['posts_next'] = df['posts_next'].fillna(0)
    df['share_next'] = df['posts_next'] / pool_next_posts

    # Candidate Targets
    # Target 1: future share level
    df['target_future_share'] = df['share_next']

    # Target 2: absolute share change
    df['target_share_delta'] = df['share_next'] - df['share_cur']

    # Target 3: future rank delta
    df['rank_cur'] = df['share_cur'].rank(pct=True)
    df['rank_next'] = df['share_next'].rank(pct=True)
    df['target_rank_delta'] = df['rank_next'] - df['rank_cur']

    # Target 4: composite utility Y4 = 0.5 * rank(future share) + 0.5 * rank(delta share)
    rank_future_share = df['share_next'].rank(pct=True)
    rank_delta_share = df['target_share_delta'].rank(pct=True)
    df['target_Y4'] = 0.5 * rank_future_share + 0.5 * rank_delta_share

    # Legacy Target (for diagnosis)
    df['target_legacy'] = np.log((df['share_next'] + 1e-5) / (df['share_cur'] + 1e-5))

    return df

print("Extractor function built successfully.")
