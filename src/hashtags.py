import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from trend import calculate_trend_signals

def get_candidates_for_brand(brand_text, trends_df, h_emb_df, category_centroids, model, config, resolved_category=None, brand_emb=None):
    if brand_emb is None:
        brand_emb = model.encode([brand_text])[0]
    
    if resolved_category is not None:
        best_cat = resolved_category
    else:
        # Map to category consistently with predict_category (skipping 'other')
        best_cat = None
        best_sim = -1
        for cat, cent in category_centroids.items():
            if cat == 'other':
                continue
            sim = cosine_similarity([brand_emb], [cent])[0][0]
            if sim > best_sim:
                best_sim = sim
                best_cat = cat
                
        if best_sim < config['tau']:
            best_cat = 'all'
        
    pool_trends = trends_df[trends_df['pool'] == best_cat]
    if pool_trends.empty:
        return pd.DataFrame(), best_cat
        
    eligible_hashtags = pool_trends['hashtag'].tolist()
    
    pool_emb_df = h_emb_df[h_emb_df['hashtag'].isin(eligible_hashtags)]
    
    if pool_emb_df.empty:
        return pd.DataFrame(), best_cat
        
    emb_matrix = np.stack(pool_emb_df['embedding'].values)
    sims = cosine_similarity([brand_emb], emb_matrix)[0]
    
    pool_emb_df = pool_emb_df.copy()
    pool_emb_df['relevance'] = sims
    pool_emb_df = pool_emb_df.reset_index()
    
    candidates = pool_emb_df.sort_values('relevance', ascending=False).head(config['candidate_pool_size'])
    candidates = candidates.merge(pool_trends[['hashtag', 'trend_score', 'trend_direction']], on='hashtag', how='inner')
    
    return candidates, best_cat

def calculate_historical_engagement(pool, candidate_hashtags, history_df):
    pool_history = history_df[history_df['pool'] == pool]
    hist_eng = pool_history[pool_history['hashtag'].isin(candidate_hashtags)][['hashtag', 'historical_engagement']]
    return hist_eng

CAMPAIGN_PATTERNS = re.compile(
    r'.*(giveaway|giveaways|contest|challenge|prsearch|modelsearch)$|'
    r'.*(collab|collaboration|sponsored|promo|promotions|ambassador)$|'
    r'.*(xme|byme|withme|findme|popme|asseenonme)$|'
    r'^(picoftheday|photooftheday|instagood|instalike|follow|followme|like4like|likeforlike|tbt|repost|love)$',
    re.IGNORECASE
)

def is_campaign_risk(tag: str) -> bool:
    """Identify if a hashtag matches known campaign, giveaway, influencer UGC or vanity pod patterns."""
    return bool(CAMPAIGN_PATTERNS.match(tag.lstrip('#')))

def rank_hashtags(candidates, config, category=None):
    if candidates.empty:
        return candidates
        
    c = candidates.copy()
    
    # Relevance qualifying gate: ensure candidates meet minimum relative semantic relevance
    if category == 'all' or (isinstance(category, str) and 'fallback' in category):
        gate_ratio = config.get('relevance_qualifying_ratio_fallback', 0.65)
    else:
        gate_ratio = config.get('relevance_qualifying_ratio', 0.55)
    max_rel = c['relevance'].max()
    gate = max_rel * gate_ratio
    gated_candidates = c[c['relevance'] >= gate].copy()
    
    target_k = config.get('n_hashtags', 8)
    if len(gated_candidates) >= target_k:
        c = gated_candidates
    else:
        # If too few candidates qualify under the relative gate, preserve the relevance gate
        # by selecting the top candidates strictly by relevance rather than reverting to the unfiltered pool.
        c = c.sort_values('relevance', ascending=False).head(max(target_k, len(gated_candidates))).copy()
        
    def min_max_norm(series):
        if series.max() == series.min():
            return np.zeros(len(series))
        return (series - series.min()) / (series.max() - series.min())
        
    c['norm_relevance'] = min_max_norm(c['relevance'])
    c['norm_trend'] = min_max_norm(c['trend_score'])
    c['norm_history'] = min_max_norm(c['historical_engagement'])
    
    # Apply separate ranking weights to fallback_all results so global trend scores cannot dominate relevance.
    if category == 'all' or (isinstance(category, str) and 'fallback' in category):
        w_rel, w_trend, w_hist = config.get('weights_hashtag_fallback', [0.70, 0.20, 0.10])
    else:
        w_rel, w_trend, w_hist = config['weights_hashtag']
        
    c['score'] = (w_rel * c['norm_relevance']) + (w_trend * c['norm_trend']) + (w_hist * c['norm_history'])
    
    # Campaign risk signal: apply soft score penalty rather than outright entity removal
    c['campaign_risk'] = c['hashtag'].apply(is_campaign_risk)
    risk_penalty = config.get('campaign_risk_penalty', 0.80)
    c.loc[c['campaign_risk'], 'score'] *= risk_penalty
    
    c = c.sort_values('score', ascending=False).reset_index(drop=True)
    return c
