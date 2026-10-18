import sys
import os
import json
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
from pipeline import RecommenderPipeline

p = RecommenderPipeline()

brands = [
    {'name': 'CampusWear', 'desc': 'We sell trendy college apparel, hoodies, and sweatpants for students.', 'ind': 'fashion', 'aud': 'university students and young adults'},
    {'name': 'FitForge', 'desc': 'Premium gym wear, lifting straps, and bodybuilding accessories.', 'ind': 'fitness', 'aud': 'bodybuilders and powerlifters'},
    {'name': 'GlowTheory', 'desc': 'Clean, cruelty-free skincare serums and mineral cosmetics.', 'ind': 'beauty', 'aud': 'skincare enthusiasts and clean beauty shoppers'},
    {'name': 'SpiceRoute', 'desc': 'Authentic artisan spice blends, curry pastes, and regional sauces.', 'ind': 'food & beverage', 'aud': 'home cooks and culinary enthusiasts'},
    {'name': 'UrbanNest', 'desc': 'Mid-century modern home decor, handcrafted ceramics, and modular storage.', 'ind': 'interior design & furniture', 'aud': 'apartment renters and interior designers'},
    {'name': 'PawPal', 'desc': 'Hypoallergenic dog treats, durable chew toys, and ergonomic leashes.', 'ind': 'pet supplies', 'aud': 'dog owners and pet lovers'},
    {'name': 'WanderWeekends', 'desc': 'Curated weekend travel itineraries, boutique stay bookings, and travel kits.', 'ind': 'travel & tourism', 'aud': 'young professionals and weekend travelers'},
    {'name': 'CreatorHub', 'desc': 'Coworking spaces, creator studios, and community networking events.', 'ind': 'technology & coworking', 'aud': 'content creators, podcasters, and digital nomads'}
]

rows = []
for b in brands:
    out = p.recommend({
        'brand_name': b['name'],
        'business_description': b['desc'],
        'industry': b['ind'],
        'target_audience': b['aud']
    })
    
    # Simulate before with follower=0
    cat = out['data_basis']['category']
    em = p.engagement_model
    inf_cat = 'other' if cat == 'all' else cat
    
    # Before prediction (with follower=0)
    pred_before = em.predict(
        category=cat,
        media_type=out['recommended_content_type']['value'],
        hour_of_day=int(out['recommended_posting_time']['start'].split(':')[0]),
        day_of_week=0,
        hashtag_count=len(out['recommended_hashtags']),
        follower_count=0
    )
    
    pred_after = out['predicted_engagement']
    
    rows.append({
        'Brand': b['name'],
        'Category': cat,
        'Match': out['data_basis']['category_match'],
        'Content Type': out['recommended_content_type']['value'],
        'Posting Time': f"{out['recommended_posting_time']['day']} {out['recommended_posting_time']['start']}",
        'Topics': ", ".join(out['trending_topics']),
        'Before ER': f"{pred_before['value']*100:.2f}% ({pred_before['label']})",
        'After ER': f"{pred_after['value']*100:.2f}% ({pred_after['label']})"
    })

df_res = pd.DataFrame(rows)
print("=" * 100)
print("SMOKE TEST & BEFORE/AFTER ENGAGEMENT FALLBACK COMPARISON")
print("=" * 100)
print(df_res.to_string(index=False))
