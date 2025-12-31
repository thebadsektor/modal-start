"""
Fast inference using Unsloth - same library used for training
Install: pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
"""
from unsloth import FastLanguageModel
import torch

# Load model with Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="KapPuroyX/mistral-7b-viral-posts",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)

# Enable fast inference
FastLanguageModel.for_inference(model)

def generate_post(platform, hook_type, psychology, cta_type, pillar):
    """Generate a viral post with given parameters"""
    
    prompt = f"""### Instruction:
Generate a viral social media post with the following characteristics.

### Input:
Platform: {platform}
Hook type: {hook_type}
Psychology: {psychology}
CTA: {cta_type}
Pillar: {pillar}

### Output:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,
        do_sample=True,
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the output
    if "### Output:" in result:
        result = result.split("### Output:")[1].strip()
    
    return result

# Example usage
if __name__ == "__main__":
    post = generate_post(
        platform="twitter",
        hook_type="question",
        psychology="urgency",
        cta_type="explicit",
        pillar="education"
    )
    
    print("\n" + "="*60)
    print("GENERATED VIRAL POST:")
    print("="*60)
    print(post)
    print("="*60)