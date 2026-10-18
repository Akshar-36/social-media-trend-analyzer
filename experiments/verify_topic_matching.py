import json
import os

files = sorted([f for f in os.listdir('contract/sample_outputs') if f.endswith('.json')])
print("--- Checking topic presence in hashtag_details ---")
for f in files:
    with open(os.path.join('contract/sample_outputs', f), 'r', encoding='utf-8') as fh:
        d = json.load(fh)
    rec = d['recommended_topic']
    trends = d['trending_topics']
    details = {h['hashtag'].replace('#',''): h for h in d['hashtag_details']}
    
    assert rec in details, f"{f}: rec {rec} not in details"
    rec_record = details[rec]
    assert rec_record['trend_direction'] in ['rising', 'stable'], f"{f}: rec {rec} is {rec_record['trend_direction']}"
    
    trend_matches = []
    for t in trends:
        assert t in details, f"{f}: trend {t} not in details"
        t_record = details[t]
        assert t_record['trend_direction'] in ['rising', 'stable'], f"{f}: trend {t} is {t_record['trend_direction']}"
        trend_matches.append(f"{t} ({t_record['trend_direction']})")
        
    print(f"{f}: PASS | rec: {rec} ({rec_record['trend_direction']}) | trends: {trend_matches}")

print("\nAll 8 files: 100% of recommended_topic and trending_topics have matching hashtag_details records and NONE are falling!")
