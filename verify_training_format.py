"""
Check if training data has the correct format
"""
import json

print("="*60)
print("CHECKING TRAINING DATA FORMAT")
print("="*60)

with open('training_data_augmented.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 5:  # Check first 5 examples
            break
        
        data = json.loads(line)
        
        print(f"\n{'='*60}")
        print(f"EXAMPLE {i+1}")
        print(f"{'='*60}")
        print(f"Instruction: {data['instruction']}")
        print(f"\nInput:\n{data['input']}")
        print(f"\nOutput:\n{data['output']}")
        
        # Check for issues
        if "Reference viral post" in data['input']:
            print("\n❌ PROBLEM: Input contains 'Reference viral post'")
        if len(data['output']) < 10:
            print(f"\n⚠️  WARNING: Output is very short ({len(data['output'])} chars)")
        if data['output'] == data['input']:
            print("\n❌ PROBLEM: Output is same as input!")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

# Count total examples and check for issues
total = 0
with_reference = 0
short_outputs = 0

with open('training_data_augmented.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            total += 1
            data = json.loads(line)
            if "Reference viral post" in data['input']:
                with_reference += 1
            if len(data['output']) < 10:
                short_outputs += 1

print(f"Total examples: {total}")
print(f"Examples with 'Reference viral post': {with_reference}")
print(f"Examples with short outputs: {short_outputs}")

if with_reference > 0:
    print("\n❌ TRAINING DATA HAS REFERENCES - NEEDS FIXING!")
else:
    print("\n✅ Training data looks clean!")