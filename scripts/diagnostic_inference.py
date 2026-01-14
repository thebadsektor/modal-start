import modal

app = modal.App("viral-post-diagnostic")

inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "transformers==4.54.0",
        "torch==2.5.1",
        "peft==0.16.0",
        "accelerate==1.9.0",
    )
)

checkpoint_vol = modal.Volume.from_name("viral-checkpoints", create_if_missing=False)

@app.function(
    image=inference_image,
    gpu="T4",
    volumes={"/checkpoints": checkpoint_vol},
)
def test_different_prompts():
    """Test various prompt formats to see what works"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch
    
    print("Loading model...")
    base_model = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    model = PeftModel.from_pretrained(model, "/checkpoints/final")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test different prompt formats
    test_prompts = [
        # Format 1: Original training format
        """### Instruction:
Generate a viral social media post with the following characteristics.

### Input:
Platform: twitter
Hook type: question
Psychology: urgency
CTA: explicit
Pillar: education

### Output:
""",
        # Format 2: Simpler format
        """Generate a viral Twitter post:
- Hook: question
- Psychology: urgency
- CTA: explicit
- Pillar: education

Post:""",
        
        # Format 3: Very simple
        """Create a viral Twitter post about education with a question hook:""",
        
        # Format 4: Direct instruction
        """Write a Twitter post that asks an urgent question about education and includes a clear call-to-action.""",
    ]
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}")
        print(f"{'='*60}")
        print(f"Prompt:\n{prompt[:100]}...")
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\nFull Output:\n{generated}")
        print(f"{'='*60}\n")
        
        results.append({
            "prompt_num": i,
            "output": generated
        })
    
    return results

@app.function(
    image=inference_image,
    volumes={"/checkpoints": checkpoint_vol},
)
def check_training_data_sample():
    """Check a sample from your training data to verify format"""
    import json
    import os
    
    data_path = "/data/training_data.jsonl"
    
    if not os.path.exists(data_path):
        return "Training data not found in volume"
    
    samples = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:  # Just get first 3 samples
                break
            samples.append(json.loads(line))
    
    return samples

@app.local_entrypoint()
def main():
    print("\n" + "="*60)
    print("DIAGNOSTIC: Testing Different Prompt Formats")
    print("="*60)
    
    results = test_different_prompts.remote()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for result in results:
        print(f"\nPrompt {result['prompt_num']}:")
        output = result['output']
        # Show first 200 chars
        print(output[:200] + ("..." if len(output) > 200 else ""))
        print("-" * 40)
    
    print("\n" + "="*60)
    print("Which format worked best?")
    print("="*60)
    
    # Try to check training data too
    print("\n" + "="*60)
    print("Checking training data format...")
    print("="*60)
    
    try:
        samples = check_training_data_sample.remote()
        if isinstance(samples, list) and len(samples) > 0:
            print(f"\nFound {len(samples)} training samples:")
            for i, sample in enumerate(samples, 1):
                print(f"\nSample {i}:")
                print(f"Instruction: {sample.get('instruction', 'N/A')[:100]}...")
                print(f"Input: {sample.get('input', 'N/A')[:100]}...")
                print(f"Output: {sample.get('output', 'N/A')[:100]}...")
        else:
            print(samples)
    except Exception as e:
        print(f"Could not check training data: {e}")