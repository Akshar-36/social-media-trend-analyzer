import os
import glob
import json
import pandas as pd
from collections import Counter

INFLUENCERS_PATH = r"C:\Users\singh\Downloads\influencers.txt"
INFO_DIR = r"C:\Users\singh\Downloads\metadata\info"

print(f"Checking influencers file: {INFLUENCERS_PATH}")
if os.path.exists(INFLUENCERS_PATH):
    # Parse influencers.txt
    # Format: Username \t Category \t #Followers \t #Followees \t #Posts
    # Skip header lines
    records = []
    with open(INFLUENCERS_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    print(f"Total lines in influencers.txt: {len(lines)}")
    for line in lines[2:]: # skip header and separator
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            uname = parts[0].strip()
            cat = parts[1].strip()
            followers = parts[2].strip() if len(parts) > 2 else None
            posts_cnt = parts[4].strip() if len(parts) > 4 else None
            records.append({
                'username': uname,
                'category': cat,
                'followers': followers,
                'posts_count': posts_cnt
            })
            
    inf_df = pd.DataFrame(records)
    print(f"Parsed {len(inf_df)} influencers.")
    print("Raw category counts in influencers.txt:")
    print(inf_df['category'].value_counts())
    
    # Check normalized categories
    inf_df['norm_category'] = inf_df['category'].replace({'fasion': 'fashion'})
    print("\nNormalized category counts:")
    print(inf_df['norm_category'].value_counts())
else:
    print(f"ERROR: {INFLUENCERS_PATH} does not exist!")

print(f"\nChecking info directory: {INFO_DIR}")
if os.path.exists(INFO_DIR):
    info_files = os.listdir(INFO_DIR)
    print(f"Total files in info directory: {len(info_files)}")
    
    # Check filenames pattern: <username>-<post_id>.info
    usernames_in_files = []
    for fn in info_files:
        if fn.endswith('.info'):
            # Split from right on '-'
            parts = fn[:-5].rsplit('-', 1)
            if len(parts) == 2:
                usernames_in_files.append(parts[0])
                
    user_counts = Counter(usernames_in_files)
    print(f"Distinct usernames in info files: {len(user_counts)}")
    print(f"Sample top users by post count in info: {user_counts.most_common(5)}")
    
    # Check overlap with influencers.txt
    inf_users = set(inf_df['username']) if 'inf_df' in locals() else set()
    overlap_users = set(user_counts.keys()).intersection(inf_users)
    print(f"Overlap between info users and influencers.txt: {len(overlap_users)} accounts")
    
    # Sample 5 files and inspect JSON structure
    print("\nInspecting sample post JSON structures:")
    sample_files = [os.path.join(INFO_DIR, fn) for fn in info_files[:5]]
    for sf in sample_files:
        try:
            with open(sf, 'r', encoding='utf-8') as f:
                d = json.load(f)
            pid = d.get('id')
            uname = d.get('owner', {}).get('username')
            ts = d.get('taken_at_timestamp')
            
            # Caption extraction
            edges = d.get('edge_media_to_caption', {}).get('edges', [])
            caption = edges[0].get('node', {}).get('text') if edges else None
            
            likes = d.get('edge_media_preview_like', {}).get('count')
            comments = d.get('edge_media_preview_comment', {}).get('count')
            
            print(f"File: {os.path.basename(sf)}")
            print(f"  username: {uname}, post_id: {pid}, ts: {ts}, likes: {likes}, comments: {comments}")
            print(f"  caption: {repr(caption)[:80] if caption else 'NONE'}")
        except Exception as e:
            print(f"  Error reading {sf}: {e}")
