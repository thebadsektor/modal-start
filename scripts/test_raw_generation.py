import modal

app = modal.App("test-raw-generation")

inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "unsloth[cu128-torch270]==2025.7.8",
        "transformers==4.54.0",
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
def test_generation():
    """Test different prompt formats to see what works"""
    from unsloth import FastLanguageModel
    from peft import PeftModel
    import torch
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/mistral-7b-v0.3-bnb-4bit",
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    
    model = PeftModel.from_pretrained(model, "/checkpoints/final")
    FastLanguageModel.for_inference(model)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test 1: Exact training format
    prompt1 = """### Instruction:
Generate a viral social media post with these characteristics.

### Input:
Platform: twitter
Hook type: statement
Psychology: ['surprise', 'FOMO']
CTA: none
Pillar: education

### Output:
"""
    
    # Test 2: Just the example from training
    prompt2 = "Sif's crush on Thor is adorable but also sad"
    
    # Test 3: Very simple
    prompt3 = "Write a viral Twitter post:"
    
    prompts = [
        ("Training format", prompt1),
        ("Direct example", prompt2),
        ("Simple instruction", prompt3),
    ]
    
    for name, prompt in prompts:
        print("\n" + "="*60)
        print(f"TEST: {name}")
        print("="*60)
        print(f"Prompt: {prompt[:100]}...")
        print()
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Generated:")
        print(result)
        print("="*60)

@app.local_entrypoint()
def main():
    test_generation.remote()