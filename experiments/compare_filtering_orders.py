import sys, os
sys.path.append('paper/src')
from pipeline import RecommenderPipeline
from hashtags import get_candidates_for_brand, calculate_historical_engagement, rank_hashtags

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

for b in brands:
    out = p.recommend({'brand_name': b['name'], 'business_description': b['desc'], 'industry': b['ind'], 'target_audience': b['aud']})
    best_cat = out['data_basis']['category']
    brand_text = b['desc'] + ' ' + b['ind'] + ' ' + b['aud']
    cands, _ = get_candidates_for_brand(brand_text, p.trends_df, p.h_emb_df, p.centroids, p.model, p.config, resolved_category=best_cat)
    hist = calculate_historical_engagement(best_cat, cands['hashtag'].tolist(), p.history_df)
    cands = cands.merge(hist, on='hashtag', how='left')
    cands['historical_engagement'] = cands['historical_engagement'].fillna(0.0)
    ranked = rank_hashtags(cands, p.config)
    
    top8 = ranked.head(8)
    opt1_eligible = top8[top8['trend_direction'] != 'falling']
    opt1_topics = opt1_eligible.sort_values('trend_score', ascending=False).head(3)['hashtag'].tolist()
    
    opt2_eligible = ranked[ranked['trend_direction'] != 'falling']
    opt2_topics = opt2_eligible.sort_values('trend_score', ascending=False).head(3)['hashtag'].tolist()
    
    opt3_ranked_eligible = ranked[ranked['trend_direction'] != 'falling'].head(8)
    opt3_topics = opt3_ranked_eligible.sort_values('trend_score', ascending=False).head(3)['hashtag'].tolist()
    
    print(f"{b['name']}:")
    print(f"  Opt1 (filter top8): {opt1_topics}")
    print(f"  Opt2 (filter all50 by trend_score): {opt2_topics}")
    print(f"  Opt3 (filter all50 by rank score, then top8): {opt3_topics}")
