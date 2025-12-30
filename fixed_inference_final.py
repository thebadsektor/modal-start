import modal
import json

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
def generate_post(platform: str = "twitter", hook_type: str = "question",
                  psychology: str = '["urgency"]',  # Pass as JSON string
                  cta_type: str = "explicit", pillar: str = "education"):
    """Generate a viral post"""
    from unsloth import FastLanguageModel
    import torch
    
    print("🔄 Loading model from checkpoint...")
    
    # Load Unsloth model directly from the fine-tuned checkpoint
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/checkpoints/final",
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Set for inference (Unsloth optimization)
    FastLanguageModel.for_inference(model)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Ensure psychology is a string that looks like a list, like in training
    try:
        # Re-format string to be a valid list representation
        psychology_list = json.loads(psychology.replace("'", "\""))
        psychology_formatted = str(psychology_list)
    except (json.JSONDecodeError, TypeError):
        # Fallback for single values
        psychology_formatted = f"['{psychology}']"

    # Match the EXACT training format
    prompt = f"""### Instruction:
Generate a viral social media post with these characteristics.

### Input:
Platform: {platform}
Hook type: {hook_type}
Psychology: {psychology_formatted}
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
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    print("\n" + "="*60)
    print("RAW OUTPUT (for debugging):")
    print("="*60)
    print(generated)
    print("="*60)
    
    # Extract just the output
    if "### Output:" in generated:
        result = generated.split("### Output:")[-1].strip()
    else:
        # Fallback if the model doesn't follow the format
        result = generated.split("### Input:")[-1].strip()
        if f"Pillar: {pillar}" in result:
             result = result.split(f"Pillar: {pillar}")[-1].strip()

    if tokenizer.eos_token in result:
        result = result.split(tokenizer.eos_token)[0].strip()
    
    return result

@app.local_entrypoint()
def main(
    platform: str = "twitter",
    hook_type: str = "statement",
    psychology: str = '["surprise", "FOMO"]',
    cta_type: str = "none",
    pillar: str = "education",
):
    print(f"\n{'='*60}")
    print(f"GENERATING {platform.upper()} POST")
    print(f"{'='*60}\n")
    
    post = generate_post.remote(
        platform=platform,
        hook_type=hook_type,
        psychology=psychology,
        cta_type=cta_type,
        pillar=pillar,
    )
    
    print(f"\n{'='*60}")
    print("FINAL RESULT:")
    print(f"{'='*60}")
    print(post)
    print(f"{'='*60}\n")