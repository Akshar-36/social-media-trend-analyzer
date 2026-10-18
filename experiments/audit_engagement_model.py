import os
import sys
import yaml
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Load temporal posts
posts_df = pd.read_parquet('experiments/data/temporal_posts.parquet')
print(f"Total posts loaded: {len(posts_df)}")

# Filter valid posts: followers > 0, ER not null, ER <= 1.0, followers >= 5000 (config threshold)
valid = posts_df[
    (posts_df['followers'] >= 5000) & 
    (posts_df['engagement_rate'].notnull()) & 
    (posts_df['engagement_rate'] > 0) & 
    (posts_df['engagement_rate'] <= 1.0)
].copy()
print(f"Valid posts (followers >= 5000, 0 < ER <= 1.0): {len(valid)}")

valid['y'] = np.log1p(100 * valid['engagement_rate'])
valid['log_followers'] = np.log1p(valid['followers'])
valid['year'] = valid['posted_at'].dt.year

# -------------------------------------------------------------
# 1. TARGET VALIDITY AUDIT: TEMPORAL DRIFT OF DENOMINATOR
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("1. TARGET VALIDITY: RECORDED ER BY POST YEAR")
print("=" * 70)
year_stats = valid.groupby('year').agg(
    n_posts=('post_id', 'count'),
    mean_likes=('likes', 'mean'),
    median_likes=('likes', 'median'),
    mean_er=('engagement_rate', 'mean'),
    median_er=('engagement_rate', 'median'),
    mean_y=('y', 'mean')
).reset_index()
print(year_stats.to_string(index=False))

# -------------------------------------------------------------
# 2. MATHEMATICAL COUPLING AUDIT: LOG_FOLLOWERS VS TARGET
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("2. MATHEMATICAL COUPLING: LOG_FOLLOWERS VS TARGET")
print("=" * 70)
r_f_y, _ = pearsonr(valid['log_followers'], valid['y'])
rho_f_y, _ = spearmanr(valid['log_followers'], valid['y'])
r_f_likes, _ = pearsonr(valid['log_followers'], np.log1p(valid['likes']))
print(f"Pearson Corr(log_followers, y): {r_f_y:.4f}")
print(f"Spearman Corr(log_followers, y): {rho_f_y:.4f}")
print(f"Pearson Corr(log_followers, log_likes): {r_f_likes:.4f}")

# -------------------------------------------------------------
# 3. BASELINES & ABLATION EXPERIMENT ON TEMPORAL POSTS
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("3. BASELINE EVALUATION (CHRONOLOGICAL SPLIT)")
print("=" * 70)

# Chronological split: Train on posts before 2019-01-01, Test on 2019 posts
train_mask = valid['posted_at'] < pd.to_datetime('2019-01-01', utc=True)
test_mask = valid['posted_at'] >= pd.to_datetime('2019-01-01', utc=True)

train_df = valid[train_mask].copy()
test_df = valid[test_mask].copy()
print(f"Train set (pre-2019): {len(train_df)} rows")
print(f"Test set (2019): {len(test_df)} rows")

def eval_preds(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

y_test = test_df['y'].values

# Baseline 1: Global Mean
b1_pred = np.full_like(y_test, train_df['y'].mean())
m_b1 = eval_preds(y_test, b1_pred)

# Baseline 2: Category Mean
cat_means = train_df.groupby('category')['y'].mean().to_dict()
b2_pred = test_df['category'].map(cat_means).fillna(train_df['y'].mean()).values
m_b2 = eval_preds(y_test, b2_pred)

# Baseline 3: Follower-only OLS (y ~ log_followers)
lr_f = LinearRegression().fit(train_df[['log_followers']], train_df['y'])
b3_pred = lr_f.predict(test_df[['log_followers']])
m_b3 = eval_preds(y_test, b3_pred)

# Baseline 4: Category + Follower OLS
train_cats = pd.get_dummies(train_df['category'], drop_first=True)
test_cats = pd.get_dummies(test_df['category'], drop_first=True)
# align cols
for col in train_cats.columns:
    if col not in test_cats.columns:
        test_cats[col] = 0
test_cats = test_cats[train_cats.columns]

X_train_cf = pd.concat([train_df[['log_followers']].reset_index(drop=True), train_cats.reset_index(drop=True)], axis=1)
X_test_cf = pd.concat([test_df[['log_followers']].reset_index(drop=True), test_cats.reset_index(drop=True)], axis=1)

lr_cf = LinearRegression().fit(X_train_cf, train_df['y'])
b4_pred = lr_cf.predict(X_test_cf)
m_b4 = eval_preds(y_test, b4_pred)

# Model 5: Random Forest with all available metadata (hour, day, followers, category)
train_df['hour'] = train_df['posted_at'].dt.hour
train_df['day'] = train_df['posted_at'].dt.dayofweek
test_df['hour'] = test_df['posted_at'].dt.hour
test_df['day'] = test_df['posted_at'].dt.dayofweek

X_train_full = pd.concat([
    train_df[['log_followers', 'hour', 'day']].reset_index(drop=True), 
    train_cats.reset_index(drop=True)
], axis=1)
X_test_full = pd.concat([
    test_df[['log_followers', 'hour', 'day']].reset_index(drop=True), 
    test_cats.reset_index(drop=True)
], axis=1)

rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train_full, train_df['y'])
b5_pred = rf.predict(X_test_full)
m_b5 = eval_preds(y_test, b5_pred)

res_table = [
    {'Model': '1. Global Mean Baseline', **m_b1},
    {'Model': '2. Category Mean Baseline', **m_b2},
    {'Model': '3. Follower-Only OLS (log_followers)', **m_b3},
    {'Model': '4. Category + Follower OLS', **m_b4},
    {'Model': '5. Random Forest (Followers+Category+Hour+Day)', **m_b5}
]
print(pd.DataFrame(res_table).to_string(index=False))

# -------------------------------------------------------------
# 4. ABLATION: WHAT DOES METADATA ADD WITHOUT LOG_FOLLOWERS?
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("4. ABLATION: RF WITHOUT LOG_FOLLOWERS (Category + Timing Only)")
print("=" * 70)
X_train_no_f = pd.concat([train_df[['hour', 'day']].reset_index(drop=True), train_cats.reset_index(drop=True)], axis=1)
X_test_no_f = pd.concat([test_df[['hour', 'day']].reset_index(drop=True), test_cats.reset_index(drop=True)], axis=1)

rf_no_f = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_no_f.fit(X_train_no_f, train_df['y'])
pred_no_f = rf_no_f.predict(X_test_no_f)
m_no_f = eval_preds(y_test, pred_no_f)
print(f"RF (NO log_followers) -> RMSE: {m_no_f['RMSE']:.4f}, MAE: {m_no_f['MAE']:.4f}, R2: {m_no_f['R2']:.4f}")
