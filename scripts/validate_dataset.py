"""
A more robust validation script for the training data.
"""
import json
import pandas as pd
from transformers import AutoTokenizer

DATA_FILE = 'data/training_data_augmented.jsonl'
MODEL_NAME = "unsloth/mistral-7b-v0.3-bnb-4bit"

print("="*60)
print("RUNNING DEEP DATASET VALIDATION")
print("="*60)

# Load tokenizer
print(f"Loading tokenizer for '{MODEL_NAME}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load data
print(f"Loading data from '{DATA_FILE}'...")
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f if line.strip()]

print(f"Loaded {len(data)} examples.")

# --- Validation Checks ---
token_lengths = []
empty_fields = []
format_mismatches = 0

for i, ex in enumerate(data):
    # Check for empty fields
    for key, value in ex.items():
        if not str(value).strip():
            empty_fields.append((i, key))

    # Calculate token length
    text = f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Output:\n{ex['output']}{tokenizer.eos_token}"
    tokens = tokenizer.encode(text)
    token_lengths.append(len(tokens))

    # Check for chat template mismatch
    chat = [
        {"role": "user", "content": f"{ex['instruction']}\n\n{ex['input']}"},
        {"role": "assistant", "content": ex['output']}
    ]
    chat_template_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)

    if "###" in chat_template_text:
        format_mismatches += 1

# --- Report Results ---
print("\n" + "="*60)
print("VALIDATION RESULTS")
print("="*60)

# Empty fields
if empty_fields:
    print(f"❌ Found {len(empty_fields)} examples with empty fields:")
    for i, key in empty_fields:
        print(f"  - Example {i}, Field: '{key}'")
else:
    print("✅ No empty or whitespace-only fields found.")

# Token length distribution
if token_lengths:
    ds = pd.Series(token_lengths)
    print("\n📊 Token Length Distribution:")
    print(ds.describe())
else:
    print("\n⚠️ No token lengths were calculated.")

# Format mismatches
if format_mismatches > 0:
    print(f"\n❌ Found {format_mismatches} potential format mismatches with the model's chat template.")
else:
    print("\n✅ No obvious format mismatches found with the model's chat template.")

print("\n" + "="*60)
print("VALIDATION COMPLETE")
print("="*60)
