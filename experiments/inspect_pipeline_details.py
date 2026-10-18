import sys, os
sys.path.append('paper/src')
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

print("=== CURRENT PIPELINE RECOMMENDATIONS ===")
for b in brands:
    out = p.recommend({'brand_name': b['name'], 'business_description': b['desc'], 'industry': b['ind'], 'target_audience': b['aud']})
    details = out['hashtag_details']
    falling_in_hashtags = [h['hashtag'] for h in details if h['trend_direction'] == 'falling']
    rec_topic_dir = [h['trend_direction'] for h in details if h['hashtag'].replace('#','') == out['recommended_topic']]
    print(f"{b['name']}:")
    print(f"  recommended_topic: {out['recommended_topic']} ({rec_topic_dir[0] if rec_topic_dir else 'unknown'})")
    print(f"  trending_topics: {out['trending_topics']}")
    print(f"  falling in recommended_hashtags ({len(falling_in_hashtags)}/8): {falling_in_hashtags}")
