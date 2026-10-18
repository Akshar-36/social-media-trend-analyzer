import os
import sys
import re
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from collections import Counter

# Set UTF-8 encoding for stdout
sys.stdout.reconfigure(encoding='utf-8')

INFLUENCERS_PATH = r"C:\Users\singh\Downloads\influencers.txt"
INFO_DIR = r"C:\Users\singh\Downloads\metadata\info"

print("--- Step 1: Parsing Influencer Metadata ---")
inf_map = {}
with open(INFLUENCERS_PATH, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

for line in lines[2:]:
    parts = line.strip().split('\t')
    if len(parts) >= 2:
        uname = parts[0].strip()
        cat = parts[1].strip()
        if cat == 'fasion': cat = 'fashion'
        if cat == 'fashion 0.5': cat = 'fashion'
        followers = int(parts[2].strip()) if len(parts) > 2 and parts[2].strip().isdigit() else None
        posts_cnt = int(parts[4].strip()) if len(parts) > 4 and parts[4].strip().isdigit() else None
        inf_map[uname] = {
            'category': cat,
            'followers': followers,
            'posts_cnt': posts_cnt
        }

print(f"Total mapped influencers: {len(inf_map)}")

print("\n--- Step 2: Scanning Raw Info Files ---")
info_files = [f for f in os.listdir(INFO_DIR) if f.endswith('.info')]
print(f"Total .info files detected: {len(info_files)}")

posts_records = []
parse_errors = 0
unmapped_users = 0

for i, fn in enumerate(info_files):
    filepath = os.path.join(INFO_DIR, fn)
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            d = json.load(f)
            
        pid = d.get('id')
        owner = d.get('owner', {})
        uname = owner.get('username')
        
        # If username not in owner, try filename
        if not uname:
            parts = fn[:-5].rsplit('-', 1)
            if len(parts) == 2:
                uname = parts[0]
                
        inf_info = inf_map.get(uname, {})
        cat = inf_info.get('category')
        if not cat:
            unmapped_users += 1
            
        ts = d.get('taken_at_timestamp')
        
        # Caption extraction
        caption = None
        edges = d.get('edge_media_to_caption', {}).get('edges', [])
        if edges:
            caption = edges[0].get('node', {}).get('text')
            
        likes = d.get('edge_media_preview_like', {}).get('count')
        comments = d.get('edge_media_preview_comment', {}).get('count')
        is_video = d.get('is_video', False)
        typename = d.get('__typename')
        
        # Extract hashtags from caption
        ht_list = re.findall(r'#(\w+)', caption) if caption else []
        
        posts_records.append({
            'filename': fn,
            'post_id': str(pid) if pid else None,
            'username': uname,
            'category': cat,
            'timestamp': ts,
            'has_caption': bool(caption and caption.strip()),
            'caption_len': len(caption) if caption else 0,
            'caption_words': len(caption.split()) if caption else 0,
            'hashtags_count': len(ht_list),
            'likes': likes,
            'comments': comments,
            'is_video': is_video,
            'typename': typename
        })
    except Exception as e:
        parse_errors += 1

print(f"Successfully processed {len(posts_records)} posts. Parse errors: {parse_errors}")
print(f"Unmapped users count: {unmapped_users}")

df = pd.DataFrame(posts_records)

print("\n--- Step 3: Dataset Summary Statistics ---")
print(f"Total rows: {len(df)}")
print(f"Unique post_ids: {df['post_id'].nunique()}")
print(f"Duplicate post_ids: {df['post_id'].duplicated().sum()}")
print(f"Unique usernames: {df['username'].nunique()}")

print("\n--- Category Breakdown of Posts ---")
cat_post_counts = df['category'].value_counts(dropna=False)
print(cat_post_counts)

print("\n--- Category Breakdown of Unique Accounts in Posts ---")
user_cat_df = df[['username', 'category']].drop_duplicates()
user_cat_counts = user_cat_df['category'].value_counts(dropna=False)
print(user_cat_counts)

print("\n--- Caption Availability ---")
non_empty_caps = df['has_caption'].sum()
empty_caps = len(df) - non_empty_caps
print(f"Posts with non-empty caption: {non_empty_caps} ({non_empty_caps/len(df)*100:.2f}%)")
print(f"Posts with empty/missing caption: {empty_caps} ({empty_caps/len(df)*100:.2f}%)")

print("\n--- Caption Lengths (where non-empty) ---")
cap_lens = df[df['has_caption']]['caption_len']
cap_words = df[df['has_caption']]['caption_words']
print(f"Characters: mean={cap_lens.mean():.1f}, median={cap_lens.median():.1f}, min={cap_lens.min()}, max={cap_lens.max()}")
print(f"Words: mean={cap_words.mean():.1f}, median={cap_words.median():.1f}, min={cap_words.min()}, max={cap_words.max()}")

print("\n--- Hashtag Counts in Caption ---")
ht_counts = df['hashtags_count']
print(f"Posts with at least 1 hashtag: {(ht_counts > 0).sum()} ({(ht_counts > 0).sum()/len(df)*100:.2f}%)")
print(f"Hashtags per post: mean={ht_counts.mean():.1f}, median={ht_counts.median():.1f}, max={ht_counts.max()}")

print("\n--- Timestamp Analysis ---")
valid_ts = df['timestamp'].dropna()
print(f"Posts with valid timestamp: {len(valid_ts)}")
if len(valid_ts) > 0:
    min_date = datetime.fromtimestamp(valid_ts.min(), tz=timezone.utc)
    max_date = datetime.fromtimestamp(valid_ts.max(), tz=timezone.utc)
    print(f"Earliest post date (UTC): {min_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Latest post date (UTC):   {max_date.strftime('%Y-%m-%d %H:%M:%S')}")

print("\n--- Posts per Account Distribution ---")
posts_per_user = df['username'].value_counts()
print(f"Min posts per user: {posts_per_user.min()}")
print(f"Median posts per user: {posts_per_user.median()}")
print(f"Mean posts per user: {posts_per_user.mean():.2f}")
print(f"Max posts per user: {posts_per_user.max()}")
print(f"Users with >= 2 posts: {(posts_per_user >= 2).sum()}")
print(f"Users with >= 3 posts: {(posts_per_user >= 3).sum()}")

# Save scan summary
summary = {
    "total_files": len(info_files),
    "parsed_posts": len(df),
    "unique_post_ids": int(df['post_id'].nunique()),
    "unique_accounts": int(df['username'].nunique()),
    "category_counts_posts": cat_post_counts.to_dict(),
    "category_counts_accounts": user_cat_counts.to_dict(),
    "non_empty_captions": int(non_empty_caps),
    "empty_captions": int(empty_caps),
    "min_date_utc": min_date.strftime('%Y-%m-%d') if len(valid_ts) > 0 else None,
    "max_date_utc": max_date.strftime('%Y-%m-%d') if len(valid_ts) > 0 else None
}
os.makedirs(os.path.join(PROJECT_ROOT, "experiments/data"), exist_ok=True)
with open(os.path.join(PROJECT_ROOT, "experiments/data/raw_data_scan_summary.json"), 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print("\nScan complete. Summary written to paper/experiments/data/raw_data_scan_summary.json")
