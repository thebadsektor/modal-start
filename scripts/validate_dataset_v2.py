"""
A more robust validation script for the training data (v2).
This version sets a default chat template for the Mistral tokenizer.
"""
import json
import pandas as pd
from transformers import AutoTokenizer
import numpy as np

DATA_FILE = 'data/training_data_augmented.jsonl'
MODEL_NAME = "unsloth/mistral-7b-v0.3-bnb-4bit"

print("="*60)
print("RUNNING DEEP DATASET VALIDATION (V2)")
print("="*60)

try:
    # Load tokenizer
    print(f"Loading tokenizer for '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Manually set the chat template for Mistral Instruct
    # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1#instruction-format
    if tokenizer.chat_template is None:
        print("Tokenizer has no default chat template. Setting a standard Mistral Instruct template.")
        tokenizer.chat_template = "{% if messages[0]['role'] == 'user' %}" \
                                  "<s>[INST] {{ messages[0]['content'] }} [/INST]" \
                                  "{% for message in messages[1:] %}" \
                                      "{% if message['role'] == 'assistant' %}" \
                                          "{{ message['content'] }}</s>" \
                                      "{% endif %}" \
                                  "{% endfor %}" \
                                  "{% endif %}"

    # Load data
    print(f"Loading data from '{DATA_FILE}'...")
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(data)} examples.")

    # --- Validation Checks ---
    manual_token_lengths = []
    template_token_lengths = []
    empty_fields = []

    for i, ex in enumerate(data):
        # Check for empty fields
        for key, value in ex.items():
            if isinstance(value, str) and not value.strip():
                empty_fields.append((i, key))

        # Calculate token length using the manual format
        manual_text = f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Output:\n{ex['output']}{tokenizer.eos_token}"
        manual_tokens = tokenizer.encode(manual_text)
        manual_token_lengths.append(len(manual_tokens))

        # Calculate token length using the chat template
        chat = [
            {"role": "user", "content": f"{ex['instruction']}\n\n{ex['input']}"},
            {"role": "assistant", "content": ex['output']}
        ]
        template_text = tokenizer.apply_chat_template(chat, tokenize=False)
        template_tokens = tokenizer.encode(template_text)
        template_token_lengths.append(len(template_tokens))


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

    # Token length distribution (Manual Format)
    if manual_token_lengths:
        print("\n📊 Token Length Distribution (Manual ### Format):")
        ds_manual = pd.Series(manual_token_lengths)
        print(ds_manual.describe())
        if ds_manual.max() > 2048:
            print(f"⚠️  WARNING: {sum(ds_manual > 2048)} examples exceed 2048 tokens and may be truncated.")
    else:
        print("\n⚠️ No manual token lengths were calculated.")

    # Token length distribution (Chat Template)
    if template_token_lengths:
        print("\n📊 Token Length Distribution (Mistral Chat Template):")
        ds_template = pd.Series(template_token_lengths)
        print(ds_template.describe())
        if ds_template.max() > 2048:
            print(f"⚠️  WARNING: {sum(ds_template > 2048)} examples exceed 2048 tokens and may be truncated.")
    else:
        print("\n⚠️ No template token lengths were calculated.")

    avg_diff = np.mean(np.array(manual_token_lengths) - np.array(template_token_lengths))
    print(f"\n💡 On average, the manual format uses {avg_diff:.2f} more tokens than the chat template.")

    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)

except FileNotFoundError:
    print(f"\n❌ ERROR: Could not find the data file at {DATA_FILE}")
    print("Please ensure the file exists and the path is correct.")

except Exception as e:
    print(f"\n❌ An unexpected error occurred: {e}")
