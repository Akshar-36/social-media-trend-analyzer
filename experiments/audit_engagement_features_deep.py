import os
import sys
import json
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'data')
INFLUENCERS_PATH = r'C:\Users\singh\Downloads\influencers.txt'
INFO_DIR = r'C:\Users\singh\Downloads\metadata\info'

print("Loading data and extracting complete 19-feature cold matrix...")

# Load influencers map
inf_map = {}
with open(INFLUENCERS_PATH, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f.readlines()[2:]:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            u, cat = parts[0].strip(), parts[1].strip()
            if cat in ['fasion', 'fashion 0.5']: cat = 'fashion'
            fol = int(parts[2].strip()) if parts[2].strip().isdigit() else None
            inf_map[u] = (cat, fol)

files = [f for f in os.listdir(INFO_DIR) if f.endswith('.info')]

rows = []
for fn in files:
    fp = os.path.join(INFO_DIR, fn)
    with open(fp, 'r', encoding='utf-8', errors='replace') as f:
        try:
            d = json.load(f)
        except Exception:
            continue
    pid = d.get('id')
    uname = d.get('owner', {}).get('username')
    if not uname:
        parts = fn[:-5].rsplit('-', 1)
        if len(parts) == 2: uname = parts[0]
    ts = d.get('taken_at_timestamp')
    if not ts or not pid: continue
    
    cat, fol = inf_map.get(uname, ('other', None))
    if not fol or fol < 5000:
        continue
        
    likes = d.get('edge_media_preview_like', {}).get('count', 0) or 0
    comments = d.get('edge_media_to_parent_comment', {}).get('count', 0) or 0
    er = (likes + comments) / fol
    if er <= 0 or er > 1.0:
        continue
        
    cap_edges = d.get('edge_media_to_caption', {}).get('edges', [])
    cap = cap_edges[0]['node']['text'] if cap_edges else ''
    caption_length = len(cap)
    
    usertag_count = len(d.get('edge_media_to_tagged_user', {}).get('edges', []))
    
    typename = d.get('__typename', 'GraphImage')
    media_type = 'carousel' if typename == 'GraphSidecar' else 'image'
    n_media = len(d.get('edge_sidecar_to_children', {}).get('edges', [])) if media_type == 'carousel' else 1
    if n_media == 0: n_media = 1
    
    ht_count = len(set(re.findall(r'#\w+', cap.lower())))
    
    posted_at = pd.to_datetime(ts, unit='s', utc=True)
    
    rows.append({
        'post_id': pid,
        'username': uname,
        'posted_at': posted_at,
        'category': cat,
        'followers': fol,
        'likes': likes,
        'comments': comments,
        'engagement_rate': er,
        'y': np.log1p(100 * er),
        'log_followers': np.log1p(fol),
        'hour_of_day': posted_at.hour,
        'day_of_week': posted_at.dayofweek,
        'is_weekend': 1 if posted_at.dayofweek in [5, 6] else 0,
        'hashtag_count': ht_count,
        'caption_length': caption_length,
        'usertag_count': usertag_count,
        'n_media': n_media,
        'media_type': media_type
    })

df = pd.DataFrame(rows)
print(f"Extracted {len(df)} valid posts with complete metadata.")

# Add one-hot category and media_type dummies
categories = ['beauty', 'family', 'fashion', 'fitness', 'food', 'interior', 'other', 'pet', 'travel']
for c in categories:
    df[f'category_{c}'] = (df['category'] == c).astype(int)

df['media_type_carousel'] = (df['media_type'] == 'carousel').astype(int)
df['media_type_image'] = (df['media_type'] == 'image').astype(int)

# Chronological split: pre-2019 vs 2019
train_mask = df['posted_at'] < pd.to_datetime('2019-01-01', utc=True)
test_mask = df['posted_at'] >= pd.to_datetime('2019-01-01', utc=True)

train_df = df[train_mask].copy()
test_df = df[test_mask].copy()
print(f"Train (pre-2019): {len(train_df)} rows | Test (2019 holdout): {len(test_df)} rows")

def eval_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': round(rmse, 4), 'MAE': round(mae, 4), 'R2': round(r2, 4)}

y_train = train_df['y'].values
y_test = test_df['y'].values

# Load production model
prod_artifact = joblib.load('artifacts/v1.0/model.joblib')
prod_rf = prod_artifact['model']
allowlist = prod_artifact['allowlist']

print("\n" + "=" * 80)
print("1. CLEAN BASELINE COMPARISON ON CHRONOLOGICAL 2019 HOLDOUT")
print("=" * 80)

# Baseline 1: Global Mean
b1_pred = np.full_like(y_test, y_train.mean())
m_b1 = eval_metrics(y_test, b1_pred)

# Baseline 2: Category Mean
cat_means = train_df.groupby('category')['y'].mean().to_dict()
b2_pred = test_df['category'].map(cat_means).fillna(y_train.mean()).values
m_b2 = eval_metrics(y_test, b2_pred)

# Baseline 3: Follower-only OLS
lr_f = LinearRegression().fit(train_df[['log_followers']], y_train)
b3_pred = lr_f.predict(test_df[['log_followers']])
m_b3 = eval_metrics(y_test, b3_pred)

# Baseline 4: Category + Follower OLS
cat_cols = [f'category_{c}' for c in categories if c != 'other']
lr_cf = LinearRegression().fit(train_df[['log_followers'] + cat_cols], y_train)
b4_pred = lr_cf.predict(test_df[['log_followers'] + cat_cols])
m_b4 = eval_metrics(y_test, b4_pred)

# Model 5: Re-trained Full Cold RF
rf_retrained = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf_retrained.fit(train_df[allowlist], y_train)
m_rf_retrained = eval_metrics(y_test, rf_retrained.predict(test_df[allowlist]))

# Model 6: Loaded Production RF (artifacts/v1.0/model.joblib)
m_prod_rf = eval_metrics(y_test, prod_rf.predict(test_df[allowlist]))

baselines_df = pd.DataFrame([
    {'Model': '1. Global Mean Baseline', **m_b1},
    {'Model': '2. Category Mean Baseline', **m_b2},
    {'Model': '3. Follower-Only OLS (log_followers)', **m_b3},
    {'Model': '4. Category + Follower OLS', **m_b4},
    {'Model': '5. Re-trained Full RF (Cold Features)', **m_rf_retrained},
    {'Model': '6. Production RF (artifacts/v1.0/model.joblib)', **m_prod_rf}
])
print(baselines_df.to_string(index=False))

print("\n" + "=" * 80)
print("2. INCREMENTAL FEATURE-GROUP ABLATION (Out-of-Sample 2019 Holdout)")
print("=" * 80)

feature_groups = {
    "A. Follower Only": ['log_followers'],
    "B. Category Only": [f'category_{c}' for c in categories],
    "C. Follower + Category": ['log_followers'] + [f'category_{c}' for c in categories],
    "D. Follower + Category + Timing": ['log_followers'] + [f'category_{c}' for c in categories] + ['hour_of_day', 'day_of_week', 'is_weekend'],
    "E. Follower + Category + Timing + HashtagCount": ['log_followers'] + [f'category_{c}' for c in categories] + ['hour_of_day', 'day_of_week', 'is_weekend', 'hashtag_count'],
    "F. Full Cold Features (All 19 Features)": allowlist,
    "G. Full Cold Features MINUS Followers (Category+Timing+Meta)": [f for f in allowlist if f != 'log_followers'],
}

ablation_rows = []
for g_name, feats in feature_groups.items():
    rf_g = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_g.fit(train_df[feats], y_train)
    m_g = eval_metrics(y_test, rf_g.predict(test_df[feats]))
    ablation_rows.append({'Feature Group': g_name, 'N_Features': len(feats), **m_g})

print(pd.DataFrame(ablation_rows).to_string(index=False))

print("\n" + "=" * 80)
print("3. CONSISTENCY CHECK: TRAINING VS INFERENCE DEFAULTS")
print("=" * 80)
# In production, caption_length and usertag_count are not provided by brand and fall back to constants.
# Let's test what happens to the Production RF if test data uses production fallbacks for caption & usertags:
test_df_prod_sim = test_df[allowlist].copy()
with open('artifacts/v1.0/fallback_medians.json') as f:
    fb = json.load(f)

# Apply category fallbacks for caption_length and usertag_count
for c in categories:
    mask_c = test_df[f'category_{c}'] == 1
    test_df_prod_sim.loc[mask_c, 'caption_length'] = fb[c]['caption_length']
    test_df_prod_sim.loc[mask_c, 'usertag_count'] = fb[c]['usertag_count']

pred_prod_sim = prod_rf.predict(test_df_prod_sim)
m_prod_sim = eval_metrics(y_test, pred_prod_sim)
print(f"Production RF with REAL post captions/usertags -> RMSE: {m_prod_rf['RMSE']}, R2: {m_prod_rf['R2']}")
print(f"Production RF with FALLBACK caption/usertags   -> RMSE: {m_prod_sim['RMSE']}, R2: {m_prod_sim['R2']}")
print(f"Performance delta when caption/usertag unavailable: Delta R2 = {m_prod_sim['R2'] - m_prod_rf['R2']:+.4f}")
