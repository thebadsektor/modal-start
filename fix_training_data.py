"""
Use Claude API to generate variations of viral posts
This creates better training data where the model learns patterns, not copying
"""
import anthropic
import json
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

def generate_variation(original_post, platform, hook_type, psychology, cta, pillar):
    """Generate a variation of a viral post using Claude"""
    
    prompt = f"""You are a viral social media content creator. Given this viral post and its characteristics, generate a NEW post that follows the SAME viral patterns but with DIFFERENT content.

Original viral post: "{original_post}"

Characteristics:
- Platform: {platform}
- Hook type: {hook_type}
- Psychology: {psychology}
- CTA: {cta}
- Pillar: {pillar}

Generate a COMPLETELY NEW post that:
1. Uses the same hook structure
2. Triggers the same psychology
3. Has similar CTA style
4. Fits the same content pillar
5. But is about a DIFFERENT topic

Return ONLY the new post text, nothing else."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text.strip()
    except Exception as e:
        print(f"Error: {e}")
        return None

# Load original training data
original_data = []
with open('training_data.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            original_data.append(json.loads(line))

print(f"Loaded {len(original_data)} original examples")
print("Generating variations (this will take a while)...")

# Generate 3 variations for first 1000 posts (adjust as needed)
augmented_data = []
sample_size = min(1000, len(original_data))

for i, example in enumerate(original_data[:sample_size]):
    # Extract characteristics
    input_lines = example['input'].split('\n')
    platform = input_lines[0].split(': ')[1] if len(input_lines) > 0 else 'twitter'
    hook_type = input_lines[1].split(': ')[1] if len(input_lines) > 1 else 'statement'
    psychology = input_lines[2].split(': ')[1] if len(input_lines) > 2 else 'engagement'
    cta = input_lines[3].split(': ')[1] if len(input_lines) > 3 else 'none'
    pillar = input_lines[4].split(': ')[1] if len(input_lines) > 4 else 'education'
    
    original_post = example['output']
    
    # Add original (with cleaned input - no reference)
    clean_input = '\n'.join(input_lines[:5])  # Just the characteristics
    augmented_data.append({
        "instruction": "Generate a viral social media post with these characteristics.",
        "input": clean_input,
        "output": original_post
    })
    
    # Generate 2 variations
    for v in range(2):
        variation = generate_variation(original_post, platform, hook_type, psychology, cta, pillar)
        if variation and variation != original_post:
            augmented_data.append({
                "instruction": "Generate a viral social media post with these characteristics.",
                "input": clean_input,
                "output": variation
            })
    
    if (i + 1) % 50 == 0:
        print(f"Processed {i + 1}/{sample_size} examples, generated {len(augmented_data)} total")

# Save augmented data
with open('training_data_augmented.jsonl', 'w', encoding='utf-8') as f:
    for example in augmented_data:
        f.write(json.dumps(example, ensure_ascii=False) + '\n')

print(f"\n✅ Created {len(augmented_data)} training examples")
print(f"Original: {sample_size}, With variations: {len(augmented_data)}")
print("\nSample variations:")
for i in range(min(3, len(augmented_data))):
    print(f"\n{i+1}. {augmented_data[i]['output'][:100]}...")