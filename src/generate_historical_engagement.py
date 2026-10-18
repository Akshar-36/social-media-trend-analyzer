"""
Deterministic producer for artifacts/v1.0/historical_engagement.parquet.

Input Data:
- data/processed/posts_clean.parquet
- data/processed/hashtags_clean.parquet
- artifacts/v1.0/trends.parquet
- config.yaml

Output:
- artifacts/v1.0/historical_engagement.parquet
  Schema: [hashtag: str, historical_engagement: float64, pool: str]
"""

import os
import yaml
import pandas as pd
from dotenv import find_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def load_config():
    with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as f:
        return yaml.safe_load(f)

def generate_historical_engagement(origin_date='2019-05-15'):
    config = load_config()
    period_days = config.get('period_days', 28)
    origin_dt = pd.to_datetime(origin_date, utc=True)
    cutoff_date = origin_dt - pd.Timedelta(days=period_days)
    
    posts_path = os.path.join(PROJECT_ROOT, 'data/processed/posts_clean.parquet')
    hashtags_path = os.path.join(PROJECT_ROOT, 'data/processed/hashtags_clean.parquet')
    trends_path = os.path.join(PROJECT_ROOT, 'artifacts/v1.0/trends.parquet')
    output_path = os.path.join(PROJECT_ROOT, 'artifacts/v1.0/historical_engagement.parquet')
    
    if not os.path.exists(posts_path) or not os.path.exists(hashtags_path):
        raise FileNotFoundError(
            f"Source datasets missing. Ensure {posts_path} and {hashtags_path} exist to rebuild."
        )
        
    print(f"Loading posts and hashtags for historical engagement calculation (cutoff: {cutoff_date})...")
    posts = pd.read_parquet(posts_path)
    hashtags = pd.read_parquet(hashtags_path)
    trends = pd.read_parquet(trends_path)
    
    posts['posted_at'] = pd.to_datetime(posts['posted_at'], utc=True)
    pools = ['all', 'beauty', 'family', 'fashion', 'fitness', 'food', 'interior', 'pet', 'travel']
    
    results = []
    for pool in pools:
        pool_trends = trends[trends['pool'] == pool]
        if pool_trends.empty:
            continue
        eligible_hashtags = set(pool_trends['hashtag'].unique())
        
        if pool == 'all':
            pool_posts = posts[posts['has_followers'] == True]
        else:
            pool_posts = posts[(posts['category'] == pool) & (posts['has_followers'] == True)]
            
        hist_posts = pool_posts[pool_posts['posted_at'] <= cutoff_date]
        if hist_posts.empty:
            continue
            
        hist_tags = hist_posts[['post_id', 'engagement_rate']].merge(hashtags, on='post_id', how='inner')
        hist_tags = hist_tags[hist_tags['hashtag'].isin(eligible_hashtags)]
        
        if hist_tags.empty:
            continue
            
        pool_agg = hist_tags.groupby('hashtag')['engagement_rate'].mean().reset_index()
        pool_agg.rename(columns={'engagement_rate': 'historical_engagement'}, inplace=True)
        pool_agg['pool'] = pool
        results.append(pool_agg)
        
    final_df = pd.concat(results, ignore_index=True)
    # Ensure correct schema types
    final_df['hashtag'] = final_df['hashtag'].astype(str)
    final_df['historical_engagement'] = final_df['historical_engagement'].astype(float)
    final_df['pool'] = final_df['pool'].astype(str)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_parquet(output_path, index=False)
    print(f"Generated {len(final_df)} historical engagement rows at {output_path}.")
    return final_df

if __name__ == '__main__':
    generate_historical_engagement()
