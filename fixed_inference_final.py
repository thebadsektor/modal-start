import modal

app = modal.App("viral-post-inference-fixed")

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
    scaledown_window=300,
)
def generate_post(platform="twitter", hook_type="question", psychology="urgency",
                  cta_type="explicit", pillar="education"):
    """Generate a viral post"""
    from unsloth import FastLanguageModel
    from peft import PeftModel
    import torch
    
    print("🔄 Loading model...")
    
    # Load base model
    base_model = "unsloth/mistral-7b-v0.3-bnb-4bit"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Load adapter
    print("📦 Loading fine-tuned adapter...")
    model = PeftModel.from_pretrained(model, "/checkpoints/final")
    
    # Set for inference
    FastLanguageModel.for_inference(model)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Match the EXACT training format
    prompt = f"""### Instruction:
Generate a viral social media post with these characteristics.

### Input:
Platform: {platform}
Hook type: {hook_type}
Psychology: ['{psychology}']
CTA: {cta_type}
Pillar: {pillar}

### Output:
"""
    
    print("📝 Generating...")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n" + "="*60)
    print("RAW OUTPUT (for debugging):")
    print("="*60)
    print(generated)
    print("="*60)
    
    # Extract just the output
    if "### Output:" in generated:
        result = generated.split("### Output:")[-1].strip()
    else:
        result = generated
    
    # Remove any trailing special tokens
    if "<|endoftext|>" in result:
        result = result.split("<|endoftext|>")[0].strip()
    
    return result

@app.local_entrypoint()
def main(platform: str = "twitter"):
    print(f"\n{'='*60}")
    print(f"GENERATING {platform.upper()} POST")
    print(f"{'='*60}\n")
    
    post = generate_post.remote(platform=platform)
    
    print(f"\n{'='*60}")
    print("FINAL RESULT:")
    print(f"{'='*60}")
    print(post)
    print(f"{'='*60}\n")