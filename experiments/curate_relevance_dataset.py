import os
import sys
import re
import json
import random
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# Source paths
INFLUENCERS_PATH = r"C:\Users\singh\Downloads\influencers.txt"
INFO_DIR = r"C:\Users\singh\Downloads\metadata\info"

# Output directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DATA_DIR = os.path.join(PROJECT_ROOT, "experiments", "data")
os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

SEED = 42
TARGET_ACCOUNTS_PER_CAT = 200
CALIBRATION_RATIO = 0.40  # 80 calibration, 120 test per category

ACTIVE_CATEGORIES = ['beauty', 'family', 'fashion', 'fitness', 'food', 'interior', 'pet', 'travel']
ALL_CATEGORIES = ACTIVE_CATEGORIES + ['other']

CATEGORY_KEYWORDS = {
    'beauty': ['skin', 'skincare', 'beauty', 'makeup', 'glow', 'cosmetics', 'lipstick', 'serum', 'eyeshadow', 'facial'],
    'family': ['family', 'mom', 'dad', 'parent', 'kid', 'child', 'baby', 'toddler', 'mother', 'father', 'parenthood'],
    'fashion': ['fashion', 'style', 'outfit', 'wear', 'dress', 'clothes', 'ootd', 'streetwear', 'apparel', 'shoes'],
    'fitness': ['fitness', 'fit', 'gym', 'workout', 'training', 'muscle', 'health', 'exercise', 'crossfit', 'bodybuilding'],
    'food': ['food', 'eat', 'cook', 'recipe', 'meal', 'yum', 'delicious', 'baking', 'dinner', 'culinary', 'restaurant'],
    'interior': ['interior', 'home', 'decor', 'room', 'house', 'furniture', 'design', 'livingroom', 'architecture', 'homedecor'],
    'pet': ['dog', 'cat', 'pet', 'puppy', 'animal', 'kitten', 'dogs', 'cats', 'pets', 'woof'],
    'travel': ['travel', 'trip', 'explore', 'wander', 'vacation', 'wanderlust', 'adventure', 'travelphotography', 'journey', 'tourist']
}

def clean_caption_text(caption):
    if not caption:
        return ""
    # Strip URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', caption)
    # Strip @mentions
    text = re.sub(r'@\w+', '', text)
    # Convert hashtags to spaced plain words
    text = re.sub(r'#(\w+)', r'\1', text)
    # Normalize extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_predominantly_latin(text, min_ratio=0.60):
    if not text:
        return False
    latin_chars = len(re.findall(r'[A-Za-z0-9\s.,!?:;\-\'\"\/]', text))
    return (latin_chars / len(text)) >= min_ratio

def mask_category_names(text, target_cat):
    # Mask the exact category name and close variants
    words_to_mask = [target_cat]
    if target_cat == 'pet': words_to_mask.extend(['pets'])
    if target_cat == 'family': words_to_mask.extend(['families'])
    pattern = r'\b(' + '|'.join(words_to_mask) + r')\b'
    return re.sub(pattern, '[MASK]', text, flags=re.IGNORECASE)

def create_short_query(text, min_words=4, max_words=8):
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:random.randint(min_words, max_words)])

def get_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]

def run_curation():
    print(f"=== Starting Curated Relevance Dataset Generation ===")
    print(f"Seed: {SEED}")
    print(f"Target per category: {TARGET_ACCOUNTS_PER_CAT} accounts")
    
    random.seed(SEED)
    np.random.seed(SEED)
    
    # 1. Load Influencer Category Map
    print(f"\n[1/6] Loading influencer metadata from: {INFLUENCERS_PATH}")
    inf_map = {}
    with open(INFLUENCERS_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        for idx, line in enumerate(f):
            if idx < 2: continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                uname = parts[0].strip()
                cat = parts[1].strip()
                if cat in ['fasion', 'fashion 0.5']: cat = 'fashion'
                followers = int(parts[2].strip()) if len(parts) > 2 and parts[2].strip().isdigit() else None
                inf_map[uname] = {'category': cat, 'followers': followers}
                
    print(f"Loaded {len(inf_map)} influencer account mappings.")
    
    # 2. Scan Raw Posts and Filter
    print(f"\n[2/6] Reading and filtering raw post metadata from: {INFO_DIR}")
    info_files = [f for f in os.listdir(INFO_DIR) if f.endswith('.info')]
    print(f"Total info files found: {len(info_files)}")
    
    exclusion_reasons = Counter()
    user_to_candidate_posts = defaultdict(list)
    
    for fn in info_files:
        filepath = os.path.join(INFO_DIR, fn)
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                d = json.load(f)
        except Exception:
            exclusion_reasons['corrupted_json'] += 1
            continue
            
        pid = str(d.get('id', ''))
        owner = d.get('owner', {})
        uname = owner.get('username')
        if not uname:
            parts = fn[:-5].rsplit('-', 1)
            uname = parts[0] if len(parts) == 2 else None
            
        if not uname or uname not in inf_map:
            exclusion_reasons['unmapped_user'] += 1
            continue
            
        cat = inf_map[uname]['category']
        if cat not in ALL_CATEGORIES:
            exclusion_reasons['unknown_category'] += 1
            continue
            
        edges = d.get('edge_media_to_caption', {}).get('edges', [])
        caption = edges[0].get('node', {}).get('text') if edges else None
        
        if not caption or not caption.strip():
            exclusion_reasons['empty_caption'] += 1
            continue
            
        cleaned = clean_caption_text(caption)
        words = cleaned.split()
        
        if len(words) < 6:
            exclusion_reasons['caption_too_short'] += 1
            continue
            
        if len(words) > 150:
            exclusion_reasons['caption_too_long'] += 1
            continue
            
        if not is_predominantly_latin(cleaned, min_ratio=0.60):
            exclusion_reasons['non_latin_script'] += 1
            continue
            
        ts = d.get('taken_at_timestamp')
        likes = d.get('edge_media_preview_like', {}).get('count', 0)
        comments = d.get('edge_media_preview_comment', {}).get('count', 0)
        
        user_to_candidate_posts[uname].append({
            'filename': fn,
            'post_id': pid,
            'username': uname,
            'category': cat,
            'followers': inf_map[uname].get('followers'),
            'timestamp': ts,
            'likes': likes,
            'comments': comments,
            'raw_caption': caption,
            'cleaned_caption': cleaned,
            'word_count': len(words),
            'char_count': len(cleaned)
        })
        
    print(f"Exclusion statistics:")
    for reason, count in exclusion_reasons.most_common():
        print(f"  {reason:25s}: {count}")
        
    print(f"\nUnique accounts with usable post candidates: {len(user_to_candidate_posts)}")
    cat_to_users = defaultdict(list)
    for u, posts in user_to_candidate_posts.items():
        cat = posts[0]['category']
        cat_to_users[cat].append(u)
        
    for c in ALL_CATEGORIES:
        print(f"  {c:10s}: {len(cat_to_users[c])} eligible accounts")
        
    # 3. Stratified Account Sampling
    print(f"\n[3/6] Sampling {TARGET_ACCOUNTS_PER_CAT} accounts per category...")
    selected_records = []
    
    for cat in ALL_CATEGORIES:
        available_users = cat_to_users[cat]
        random.shuffle(available_users)
        
        sample_n = min(len(available_users), TARGET_ACCOUNTS_PER_CAT)
        sampled_users = available_users[:sample_n]
        
        for u in sampled_users:
            posts = user_to_candidate_posts[u]
            # Select post closest to median length (ideal brand blurb representation)
            posts.sort(key=lambda p: abs(p['word_count'] - 35))
            chosen = posts[0]
            selected_records.append(chosen)
            
    print(f"Total sampled posts/records: {len(selected_records)}")
    
    # 4. Account-Level Split: Calibration (40%) vs Held-Out Test (60%)
    print(f"\n[4/6] Partitioning by account: Calibration vs Held-out Test...")
    calib_records = []
    test_records = []
    
    # Group by category to stratify
    cat_to_records = defaultdict(list)
    for r in selected_records:
        cat_to_records[r['category']].append(r)
        
    for cat in ALL_CATEGORIES:
        recs = cat_to_records[cat]
        random.shuffle(recs)
        n_calib = int(len(recs) * CALIBRATION_RATIO)
        calib_recs = recs[:n_calib]
        test_recs = recs[n_calib:]
        
        calib_records.extend(calib_recs)
        test_records.extend(test_recs)
        
    print(f"Calibration Set records: {len(calib_records)} ({len(calib_records)//len(ALL_CATEGORIES)} per category)")
    print(f"Held-Out Test Set records: {len(test_records)} ({len(test_records)//len(ALL_CATEGORIES)} per category)")
    
    # Verify zero account leakage
    calib_users = set(r['username'] for r in calib_records)
    test_users = set(r['username'] for r in test_records)
    overlap = calib_users.intersection(test_users)
    assert len(overlap) == 0, f"LEAKAGE DETECTED: {len(overlap)} accounts in both splits!"
    print("Zero-account leakage verified: Calibration and Test accounts are completely disjoint.")
    
    # 5. Format Records and Generate Query Variants
    def format_evaluation_payload(rec_list, split_name):
        formatted = []
        for idx, r in enumerate(rec_list):
            cat = r['category']
            clean_text = r['cleaned_caption']
            masked_text = mask_category_names(clean_text, cat)
            short_text = create_short_query(clean_text)
            
            # Category name leakage check
            cat_name_present = bool(re.search(r'\b' + cat + r'\b', clean_text, re.IGNORECASE))
            
            # Obvious category keywords count
            kw_matches = [kw for kw in CATEGORY_KEYWORDS.get(cat, []) if re.search(r'\b' + kw + r'\b', clean_text, re.IGNORECASE)]
            
            # Format record
            rec_id = f"{split_name[:3].upper()}-{idx+1:04d}"
            formatted.append({
                "id": rec_id,
                "split": split_name,
                "source": "instagram_raw_metadata",
                "username_hash": get_hash(r['username']),
                "post_id": r['post_id'],
                "timestamp_utc": datetime.fromtimestamp(r['timestamp'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S') if r['timestamp'] else None,
                "expected_category": cat if cat != 'other' else 'out_of_domain',
                "raw_source_category": cat,
                "followers": r['followers'],
                "likes": r['likes'],
                "comments": r['comments'],
                "query_natural": clean_text,
                "query_masked": masked_text,
                "query_short": short_text,
                "category_name_leakage": cat_name_present,
                "keyword_matches": kw_matches,
                "keyword_count": len(kw_matches),
                "word_count": r['word_count'],
                "char_count": r['char_count']
            })
        return formatted
        
    formatted_calib = format_evaluation_payload(calib_records, "calibration")
    formatted_test = format_evaluation_payload(test_records, "held_out_test")
    
    # 6. Build Robustness / Hard Set
    print(f"\n[5/6] Building Robustness / Hard Dataset...")
    robust_records = []
    
    # Load 54 curated benchmark cases
    curated_path = os.path.join(OUTPUT_DATA_DIR, "relevance_test_set.json")
    if os.path.exists(curated_path):
        with open(curated_path, 'r', encoding='utf-8') as f:
            curated_54 = json.load(f)
        for c in curated_54:
            robust_records.append({
                "id": f"ROB-BENCH-{c['id']}",
                "robustness_type": "curated_benchmark_case",
                "source": "manual_curated_54",
                "brand_name": c.get('brand_name'),
                "expected_category": c['expected_category'],
                "query": f"{c['business_description']} {c['industry']} {c['target_audience']}".strip(),
                "category_hint": c.get('category_hint'),
                "rationale": c.get('rationale')
            })
        print(f"  Added {len(curated_54)} cases from curated 54-brand benchmark.")
        
    # Add Short-query variants from held-out test
    for r in formatted_test:
        if r['expected_category'] != 'out_of_domain':
            robust_records.append({
                "id": f"ROB-SHORT-{r['id']}",
                "robustness_type": "short_query_stress",
                "source": "instagram_raw_metadata",
                "expected_category": r['expected_category'],
                "query": r['query_short'],
                "parent_record_id": r['id'],
                "word_count": len(r['query_short'].split())
            })
            
    # Add Category-name-masked variants from held-out test
    for r in formatted_test:
        if r['category_name_leakage'] and r['expected_category'] != 'out_of_domain':
            robust_records.append({
                "id": f"ROB-MASKED-{r['id']}",
                "robustness_type": "category_name_masked",
                "source": "instagram_raw_metadata",
                "expected_category": r['expected_category'],
                "query": r['query_masked'],
                "parent_record_id": r['id']
            })
            
    # Add Out-of-Domain cases from test set
    for r in formatted_test:
        if r['expected_category'] == 'out_of_domain':
            robust_records.append({
                "id": f"ROB-OOD-{r['id']}",
                "robustness_type": "out_of_domain_rejection",
                "source": "instagram_raw_metadata",
                "expected_category": "out_of_domain",
                "query": r['query_natural'],
                "parent_record_id": r['id']
            })
            
    # Identify cross-domain ambiguous cases (e.g. fashion post with fitness keywords)
    for r in formatted_test:
        cat = r['raw_source_category']
        if cat in ACTIVE_CATEGORIES:
            other_cats = [c for c in ACTIVE_CATEGORIES if c != cat]
            cross_matches = {}
            for oc in other_cats:
                matches = [kw for kw in CATEGORY_KEYWORDS[oc] if re.search(r'\b' + kw + r'\b', r['query_natural'], re.IGNORECASE)]
                if matches:
                    cross_matches[oc] = matches
            if len(cross_matches) >= 2:
                robust_records.append({
                    "id": f"ROB-AMBIG-{r['id']}",
                    "robustness_type": "cross_domain_ambiguity",
                    "source": "instagram_raw_metadata",
                    "expected_category": r['expected_category'],
                    "competing_domains": list(cross_matches.keys()),
                    "query": r['query_natural'],
                    "parent_record_id": r['id']
                })
                
    print(f"Total Robustness Set records: {len(robust_records)}")
    
    # 7. Write Datasets to Disk
    print(f"\n[6/6] Writing datasets and metadata report to {OUTPUT_DATA_DIR}...")
    calib_out_path = os.path.join(OUTPUT_DATA_DIR, "relevance_calibration_set.json")
    test_out_path = os.path.join(OUTPUT_DATA_DIR, "relevance_held_out_test_set.json")
    robust_out_path = os.path.join(OUTPUT_DATA_DIR, "relevance_robustness_set.json")
    
    with open(calib_out_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_calib, f, indent=2)
    with open(test_out_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_test, f, indent=2)
    with open(robust_out_path, 'w', encoding='utf-8') as f:
        json.dump(robust_records, f, indent=2)
        
    # Metadata and statistics
    def get_stats(dataset):
        df_d = pd.DataFrame(dataset)
        cat_counts = df_d['expected_category'].value_counts().to_dict()
        w_lens = df_d['word_count']
        c_lens = df_d['char_count']
        leakage_pct = (df_d['category_name_leakage'].sum() / len(df_d)) * 100
        return {
            "record_count": len(dataset),
            "category_distribution": cat_counts,
            "word_count_stats": {"mean": float(w_lens.mean()), "median": float(w_lens.median()), "min": int(w_lens.min()), "max": int(w_lens.max())},
            "char_count_stats": {"mean": float(c_lens.mean()), "median": float(c_lens.median()), "min": int(c_lens.min()), "max": int(c_lens.max())},
            "category_name_leakage_pct": float(leakage_pct)
        }
        
    meta = {
        "dataset_name": "Instagram Influencer Relevance Benchmark (Kim et al. WWW '20 subset)",
        "creation_timestamp_utc": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        "seed": SEED,
        "raw_data_sources": {
            "influencers_path": INFLUENCERS_PATH,
            "info_directory": INFO_DIR,
            "total_raw_info_files": len(info_files),
            "total_mapped_influencers": len(inf_map)
        },
        "filtering_criteria": {
            "min_words": 6,
            "max_words": 150,
            "min_latin_ratio": 0.60,
            "posts_per_account": 1,
            "post_selection_policy": "closest_to_median_length_35_words"
        },
        "exclusion_statistics": dict(exclusion_reasons),
        "split_methodology": "Account-level disjoint stratified split (40% Calibration, 60% Held-Out Test)",
        "calibration_set": get_stats(formatted_calib),
        "held_out_test_set": get_stats(formatted_test),
        "robustness_set": {
            "record_count": len(robust_records),
            "breakdown_by_type": Counter(r['robustness_type'] for r in robust_records)
        }
    }
    
    meta_path = os.path.join(OUTPUT_DATA_DIR, "relevance_curated_metadata.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
        
    print(f"\nSUCCESS: Generated datasets:")
    print(f"  Calibration Set: {len(formatted_calib)} records -> {calib_out_path}")
    print(f"  Test Set:        {len(formatted_test)} records -> {test_out_path}")
    print(f"  Robustness Set:  {len(robust_records)} records -> {robust_out_path}")
    print(f"  Metadata Report: {meta_path}")

if __name__ == '__main__':
    run_curation()
