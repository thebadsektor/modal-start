import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Load enriched data
df = pd.read_csv('dataset_enriched.csv')

print(f"Loading {len(df)} posts...")

# Split 80/20
train, val = train_test_split(
    df,
    test_size=0.2,
    stratify=df['performance_tier'],
    random_state=42
)

print(f"Training: {len(train)}, Validation: {len(val)}")

# Create JSONL for training
jsonl_lines = []

for _, row in train.iterrows():
    record = {
        "instruction": "Generate a high-engagement social media post based on viral patterns.",
        "input": f"""Platform: {row['platform']}
Hook type: {row['hook_type']}
Psychology: {row['psychology_triggers']}
CTA: {row['cta_type']}
Pillar: {row['content_pillar']}

Reference viral post:
"{row['post_text']}"

Engagement: {row['engagement_rate']:.2%}""",
        "output": row['post_text']
    }
    jsonl_lines.append(json.dumps(record))

# Write JSONL
with open('training_data.jsonl', 'w') as f:
    f.write('\n'.join(jsonl_lines))

print(f"✅ Created training_data.jsonl with {len(jsonl_lines)} examples")

# Count tokens estimate
total_chars = sum(len(line) for line in jsonl_lines)
total_tokens = total_chars / 4
print(f"Estimated tokens: {total_tokens:,.0f}")
print(f"Dataset ready for fine-tuning on Modal")