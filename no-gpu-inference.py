"""
Use HuggingFace Inference API - runs on their servers, no GPU needed locally
Get a free API token at: https://huggingface.co/settings/tokens
"""
from huggingface_hub import InferenceClient
import os

def generate_with_api(platform="twitter", hook_type="question", psychology="urgency", 
                      cta_type="explicit", pillar="education", api_token=None):
    """Generate using HuggingFace Inference API"""
    
    # Get token from environment or parameter
    token = api_token or os.getenv("HF_TOKEN")
    if not token:
        token = input("Enter your HuggingFace token: ").strip()
    
    client = InferenceClient(
        model="KapPuroyX/mistral-7b-viral-posts",
        token=token
    )
    
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
    
    print(f"🚀 Generating via API...")
    
    try:
        response = client.text_generation(
            prompt,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        
        # Extract just the output
        if "### Output:" in response:
            response = response.split("### Output:")[-1].strip()
        
        return response
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        print("\nNote: The model might need to 'wake up' if it hasn't been used recently.")
        print("Try again in a few seconds, or visit the model page to load it:")
        print("https://huggingface.co/KapPuroyX/mistral-7b-viral-posts")
        return None

if __name__ == "__main__":
    print("="*60)
    print("VIRAL POST GENERATOR (API Mode)")
    print("="*60)
    
    # Generate example posts
    examples = [
        {
            "platform": "twitter",
            "hook_type": "question",
            "psychology": "urgency",
            "cta_type": "explicit",
            "pillar": "education"
        },
        {
            "platform": "linkedin",
            "hook_type": "story",
            "psychology": "inspiration",
            "cta_type": "implicit",
            "pillar": "education"
        },
        {
            "platform": "instagram",
            "hook_type": "curiosity_gap",
            "psychology": "FOMO",
            "cta_type": "explicit",
            "pillar": "entertainment"
        }
    ]
    
    for i, params in enumerate(examples, 1):
        print(f"\n📱 POST {i}: {params['platform'].upper()}")
        print("-" * 60)
        
        post = generate_with_api(**params)
        
        if post:
            print(post)
        
        print("-" * 60)
        
        if i < len(examples):
            input("Press Enter for next post...")
    
    # Interactive mode
    print("\n" + "="*60)
    print("Generate custom post? (y/n)")
    if input().lower() == 'y':
        platform = input("Platform: ")
        hook_type = input("Hook type: ")
        psychology = input("Psychology: ")
        
        custom = generate_with_api(
            platform=platform,
            hook_type=hook_type,
            psychology=psychology
        )
        
        if custom:
            print("\n📝 CUSTOM POST:")
            print("-" * 60)
            print(custom)
            print("-" * 60)