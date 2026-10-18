import os
import re
import json
import pandas as pd

INFO_DIR = r"C:\Users\singh\Downloads\metadata\info"
INFLUENCERS_PATH = r"C:\Users\singh\Downloads\influencers.txt"

# Sample 10 files
files = [f for f in os.listdir(INFO_DIR) if f.endswith('.info')][:10]

def clean_caption_text(caption):
    if not caption:
        return ""
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', caption)
    # Remove user handles
    text = re.sub(r'@\w+', '', text)
    # Convert hashtags to spaced words
    text = re.sub(r'#(\w+)', r'\1', text)
    # Normalize whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def mask_category_names(text, categories):
    pattern = r'\b(' + '|'.join(categories) + r')\b'
    return re.sub(pattern, '[MASK]', text, flags=re.IGNORECASE)

categories = ['beauty', 'family', 'fashion', 'fitness', 'food', 'interior', 'pet', 'travel']

for fn in files:
    with open(os.path.join(INFO_DIR, fn), 'r', encoding='utf-8', errors='replace') as f:
        d = json.load(f)
    edges = d.get('edge_media_to_caption', {}).get('edges', [])
    cap = edges[0].get('node', {}).get('text') if edges else ""
    cleaned = clean_caption_text(cap)
    masked = mask_category_names(cleaned, categories)
    print(f"File: {fn}")
    print(f"  Raw:    {cap[:60]}...")
    print(f"  Clean:  {cleaned[:60]}...")
    print(f"  Masked: {masked[:60]}...")
    print("-" * 50)
