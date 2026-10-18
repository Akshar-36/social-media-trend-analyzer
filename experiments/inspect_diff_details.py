import subprocess
import json

files = [
    'brand_1_lumiere_cosmetics.json',
    'brand_2_happy_tails.json',
    'brand_3_tech_innovate.json',
    'brand_4_nomad_eats.json',
    'brand_5_fitlife_gear.json',
    'brand_6_wanderlust_travels.json',
    'brand_7_cozy_home.json',
    'brand_8_urban_style.json'
]

for f in files:
    head_raw = subprocess.check_output(['git', 'show', f'HEAD:contract/sample_outputs/{f}']).decode('utf-8')
    head_json = json.loads(head_raw)
    with open(f'contract/sample_outputs/{f}', 'r', encoding='utf-8') as fh:
        curr_json = json.load(fh)
        
    old_rec = head_json['recommended_topic']
    new_rec = curr_json['recommended_topic']
    old_trends = head_json['trending_topics']
    new_trends = curr_json['trending_topics']
    old_er = head_json['predicted_engagement']['value']
    new_er = curr_json['predicted_engagement']['value']
    
    old_details = {h['hashtag'].replace('#',''): h['trend_direction'] for h in head_json['hashtag_details']}
    new_details = {h['hashtag'].replace('#',''): h['trend_direction'] for h in curr_json['hashtag_details']}
    
    old_rec_dir = old_details.get(old_rec, 'not_in_details')
    new_rec_dir = new_details.get(new_rec, 'not_in_details')
    
    old_trend_dirs = [old_details.get(t, 'not_in_details') for t in old_trends]
    new_trend_dirs = [new_details.get(t, 'not_in_details') for t in new_trends]
    
    n_old = len(head_json['hashtag_details'])
    n_new = len(curr_json['hashtag_details'])
    
    print(f'=== {f} ===')
    print(f'  Rec Topic: {old_rec} ({old_rec_dir}) -> {new_rec} ({new_rec_dir})')
    print(f'  Trending: {old_trends} ({old_trend_dirs}) -> {new_trends} ({new_trend_dirs})')
    print(f'  Predicted ER: {old_er:.6f} -> {new_er:.6f}')
    print(f'  Old Details Count: {n_old} | New Details Count: {n_new}')
    print()
