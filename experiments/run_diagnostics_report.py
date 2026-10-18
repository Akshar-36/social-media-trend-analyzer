import sys
import os
import json

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

total_falling = 0
for b in brands:
    out = p.recommend({
        'brand_name': b['name'],
        'business_description': b['desc'],
        'industry': b['ind'],
        'target_audience': b['aud']
    })
    cat = out['data_basis']['category']
    topics = out['trending_topics']
    details = {h['hashtag'].replace('#',''): h['trend_direction'] for h in out['hashtag_details']}
    t_dirs = [f"{t} ({details.get(t, 'unknown')})" for t in topics]
    has_falling = any(details.get(t) == 'falling' for t in topics)
    if has_falling:
        total_falling += 1
    print(f"| {b['name']} | {cat} | {t_dirs[0]} | {t_dirs[1]} | {t_dirs[2]} | {'Yes' if has_falling else 'No'} |")

print(f"\nTotal falling brands: {total_falling}/8")
