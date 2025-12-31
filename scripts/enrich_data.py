import pandas as pd
import numpy as np
import anthropic
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

df = pd.read_csv('dataset_clean.csv')

# Get API key from environment
api_key = os.getenv('ANTHROPIC_API_KEY')
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in .env file")

client = anthropic.Anthropic(api_key=api_key)

def analyze_post(post_text):
    prompt = f"""Analyze this social media post for viral characteristics.

Post: "{post_text}"

Return ONLY valid JSON (no markdown, no code blocks, no explanation):
{{
  "hook_type": "question",
  "psychology_triggers": ["urgency"],
  "cta_type": "explicit",
  "content_pillar": "education"
}}

Valid hook_type values: question, statement, curiosity_gap, stat, story, personal
Valid psychology_triggers: urgency, FOMO, social_proof, inspiration, fear, joy, surprise, anger
Valid cta_type: explicit, implicit, none
Valid content_pillar: education, entertainment, inspiration, promotion"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",  # Use this stable model
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = message.content[0].text.strip()
        return json.loads(response_text)
    except Exception as e:
        print(f"Error analyzing post: {e}")
        return {
            "hook_type": "unknown",
            "psychology_triggers": ["unknown"],
            "cta_type": "unknown",
            "content_pillar": "unknown"
        }

# Analyze posts (limit to 200 to save cost, spread across dataset)
print(f"Analyzing {min(200, len(df))} posts with Claude...")

# Sample posts evenly across the dataset
sample_indices = np.linspace(0, len(df)-1, min(200, len(df)), dtype=int)

results = []
for idx, sample_idx in enumerate(sample_indices):
    row = df.iloc[sample_idx]
    analysis = analyze_post(row['post_text'])
    results.append({'post_id': sample_idx, **analysis})
    
    if (idx + 1) % 25 == 0:
        print(f"Analyzed {idx + 1}/{len(sample_indices)}")

print(f"Completed analysis of {len(results)} posts")

# Merge analysis back to dataset
df_enriched = df.copy()

# Add columns with defaults
for col in ['hook_type', 'psychology_triggers', 'cta_type', 'content_pillar']:
    df_enriched[col] = 'unknown'

# Update analyzed posts
for _, row in pd.DataFrame(results).iterrows():
    post_id = int(row['post_id'])
    if post_id < len(df_enriched):
        df_enriched.loc[post_id, 'hook_type'] = row['hook_type']
        df_enriched.loc[post_id, 'psychology_triggers'] = str(row['psychology_triggers'])
        df_enriched.loc[post_id, 'cta_type'] = row['cta_type']
        df_enriched.loc[post_id, 'content_pillar'] = row['content_pillar']

# For non-analyzed posts, forward-fill from analyzed ones
for col in ['hook_type', 'psychology_triggers', 'cta_type', 'content_pillar']:
    df_enriched[col] = df_enriched[col].replace('unknown', pd.NA)
    df_enriched[col] = df_enriched[col].ffill().bfill().fillna('unknown')

df_enriched.to_csv('dataset_enriched.csv', index=False)

print(f"\n✅ Enriched dataset saved to dataset_enriched.csv")
print(f"Total posts: {len(df_enriched)}")
print(f"Analyzed posts: {len(results)}")
print(f"\nHook type distribution:\n{df_enriched['hook_type'].value_counts()}")
print(f"\nCTA type distribution:\n{df_enriched['cta_type'].value_counts()}")
