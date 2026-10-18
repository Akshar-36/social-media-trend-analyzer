import os, json, re, time
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

INFLUENCERS_PATH = r'C:\Users\singh\Downloads\influencers.txt'
INFO_DIR = r'C:\Users\singh\Downloads\metadata\info'

posts_out = os.path.join(DATA_DIR, 'temporal_posts.parquet')
hashtags_out = os.path.join(DATA_DIR, 'temporal_hashtags.parquet')

if os.path.exists(posts_out) and os.path.exists(hashtags_out):
    print("Temporal data already cached at:")
    print(" ", posts_out)
    print(" ", hashtags_out)
    exit(0)

t0 = time.time()
inf_map = {}
with open(INFLUENCERS_PATH, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f.readlines()[2:]:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            u, cat = parts[0].strip(), parts[1].strip()
            if cat == 'fasion' or cat == 'fashion 0.5': cat = 'fashion'
            fol = int(parts[2].strip()) if parts[2].strip().isdigit() else None
            inf_map[u] = (cat, fol)

files = [f for f in os.listdir(INFO_DIR) if f.endswith('.info')]
print(f'Loading {len(files)} info files...')

posts_data = []
hashtags_data = []

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
    
    likes = d.get('edge_media_preview_like', {}).get('count', 0) or 0
    comments = d.get('edge_media_to_parent_comment', {}).get('count', 0) or 0
    cat, fol = inf_map.get(uname, ('other', None))
    
    er = (likes + comments) / fol if fol and fol > 0 else np.nan
    posts_data.append((pid, uname, ts, cat, fol, likes, comments, er))
    
    cap_edges = d.get('edge_media_to_caption', {}).get('edges', [])
    cap = cap_edges[0]['node']['text'] if cap_edges else ''
    for h in set(re.findall(r'#\w+', cap.lower())):
        hashtags_data.append((pid, h))

posts_df = pd.DataFrame(posts_data, columns=['post_id', 'username', 'ts', 'category', 'followers', 'likes', 'comments', 'engagement_rate'])
posts_df['posted_at'] = pd.to_datetime(posts_df['ts'], unit='s', utc=True)
hashtags_df = pd.DataFrame(hashtags_data, columns=['post_id', 'hashtag'])

posts_df.to_parquet(posts_out)
hashtags_df.to_parquet(hashtags_out)
t1 = time.time()
print(f'Cached {len(posts_df)} posts and {len(hashtags_df)} hashtags in {t1-t0:.2f}s.')
