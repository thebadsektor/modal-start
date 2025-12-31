from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

print("Loading base model...")
# Use the standard Mistral model instead of the 4-bit quantized one
base_model_name = "mistralai/Mistral-7B-v0.1"
adapter_model_name = "KapPuroyX/mistral-7b-viral-posts"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32,  # Use float32 for CPU
    device_map="cpu",
    low_cpu_mem_usage=True,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, adapter_model_name)

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_post(platform="twitter", hook_type="question", psychology="urgency", 
                  cta_type="explicit", pillar="education"):
    """Generate a viral post"""
    
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
    
    print(f"\n🔄 Generating post for {platform}...")
    print(f"   Hook: {hook_type} | Psychology: {psychology}")
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    # Generate with CPU-friendly settings
    with torch.no_grad():  # Saves memory
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the output part
    if "### Output:" in generated_text:
        result = generated_text.split("### Output:")[1].strip()
    else:
        result = generated_text
    
    return result

# Test generation
if __name__ == "__main__":
    print("\n" + "="*60)
    print("VIRAL POST GENERATOR")
    print("="*60)
    
    # Example 1: Twitter question
    post1 = generate_post(
        platform="twitter",
        hook_type="question",
        psychology="urgency",
        cta_type="explicit",
        pillar="education"
    )
    print("\n📱 POST 1:")
    print("-" * 60)
    print(post1)
    print("-" * 60)
    
    # Example 2: LinkedIn story
    post2 = generate_post(
        platform="linkedin",
        hook_type="story",
        psychology="inspiration",
        cta_type="implicit",
        pillar="education"
    )
    print("\n💼 POST 2:")
    print("-" * 60)
    print(post2)
    print("-" * 60)
    
    # Interactive mode
    print("\n" + "="*60)
    print("Want to generate more? (y/n)")
    if input().lower() == 'y':
        platform = input("Platform (twitter/linkedin/instagram): ")
        hook_type = input("Hook type (question/statement/story/curiosity_gap): ")
        psychology = input("Psychology (urgency/FOMO/inspiration/joy): ")
        
        custom_post = generate_post(
            platform=platform,
            hook_type=hook_type,
            psychology=psychology,
            cta_type="explicit",
            pillar="education"
        )
        
        print("\n📝 CUSTOM POST:")
        print("-" * 60)
        print(custom_post)
        print("-" * 60)