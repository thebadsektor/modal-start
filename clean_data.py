import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load raw data
df = pd.read_csv('all_posts_raw.csv')

print(f"Starting with {len(df)} posts")

# Remove duplicates - BATCHED (memory efficient)
print("Removing duplicates (batched)...")

indices_to_keep = []
batch_size = 5000

for batch_start in range(0, len(df), batch_size):
    batch_end = min(batch_start + batch_size, len(df))
    batch = df.iloc[batch_start:batch_end]
    
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=1000)
    matrix = vectorizer.fit_transform(batch['post_text'].astype(str))
    similarity = cosine_similarity(matrix)
    
    for i in range(len(batch)):
        # Check against all previous posts
        is_duplicate = False
        for kept_idx in indices_to_keep[-100:]:  # Only check last 100 kept
            if batch_start + i > kept_idx:
                prev_text = df.iloc[kept_idx]['post_text']
                curr_text = batch.iloc[i]['post_text']
                if prev_text == curr_text:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            indices_to_keep.append(batch_start + i)
    
    print(f"Processed {batch_end}/{len(df)}")

df_dedup = df.iloc[indices_to_keep].reset_index(drop=True)
print(f"After deduplication: {len(df_dedup)} posts")

# Remove spam
print("Removing spam...")
df_clean = df_dedup[
    (df_dedup['post_text'].str.len() >= 30) &
    (df_dedup['post_text'].str.len() <= 3000) &
    (~df_dedup['post_text'].str.contains('http.*http', regex=True)) &
    (df_dedup['engagement_rate'] >= 0.0001)
].reset_index(drop=True)

print(f"After spam removal: {len(df_clean)} posts")

# Add engagement tiers
df_clean['engagement_percentile'] = df_clean['engagement_rate'].rank(pct=True) * 100

def assign_tier(percentile):
    if percentile >= 80:
        return 'viral'
    elif percentile >= 60:
        return 'high'
    else:
        return 'medium'

df_clean['performance_tier'] = df_clean['engagement_percentile'].apply(assign_tier)

# Save
df_clean.to_csv('dataset_clean.csv', index=False)

print(f"\n✅ Clean dataset saved")
print(f"Distribution:\n{df_clean['performance_tier'].value_counts()}")
print(f"Engagement rate range: {df_clean['engagement_rate'].min():.6f} - {df_clean['engagement_rate'].max():.6f}")
