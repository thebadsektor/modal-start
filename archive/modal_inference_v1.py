import modal
from dataclasses import dataclass

app = modal.App("viral-post-inference")

# Same image as training
inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "transformers==4.54.0",
        "torch==2.5.1",
        "peft==0.16.0",
        "accelerate==1.9.0",
    )
)

# Reference the checkpoint volume
checkpoint_vol = modal.Volume.from_name("viral-checkpoints", create_if_missing=False)

@dataclass
class PostRequest:
    platform: str = "twitter"
    hook_type: str = "question"
    psychology: str = "urgency"
    cta_type: str = "explicit"
    pillar: str = "education"

@app.cls(
    image=inference_image,
    gpu="T4",  # Cheaper GPU for inference
    volumes={"/checkpoints": checkpoint_vol},
    container_idle_timeout=300,  # Keep warm for 5 minutes
)
class ViralPostGenerator:
    
    @modal.enter()
    def load_model(self):
        """Load model once when container starts"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch
        
        print("🔄 Loading model...")
        
        # Load base model
        base_model = "mistralai/Mistral-7B-v0.1"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        # Load LoRA adapter from checkpoint volume
        checkpoint_path = "/checkpoints/final"
        self.model = PeftModel.from_pretrained(model, checkpoint_path)
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ Model loaded and ready!")
    
    @modal.method()
    def generate(self, request: PostRequest) -> str:
        """Generate a viral post"""
        import torch
        
        prompt = f"""### Instruction:
Generate a viral social media post with the following characteristics.

### Input:
Platform: {request.platform}
Hook type: {request.hook_type}
Psychology: {request.psychology}
CTA: {request.cta_type}
Pillar: {request.pillar}

### Output:
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the output
        if "### Output:" in generated_text:
            result = generated_text.split("### Output:")[1].strip()
        else:
            result = generated_text
        
        return result
    
    @modal.method()
    def generate_batch(self, requests: list[PostRequest]) -> list[str]:
        """Generate multiple posts at once"""
        return [self.generate(req) for req in requests]

@app.local_entrypoint()
def main(
    platform: str = "twitter",
    hook_type: str = "question",
    psychology: str = "urgency",
    cta_type: str = "explicit",
    pillar: str = "education",
    batch: bool = False,
):
    """Generate viral posts from the command line"""
    
    if not batch:
        # Single generation
        print(f"\n{'='*60}")
        print("VIRAL POST GENERATOR")
        print(f"{'='*60}")
        print(f"Platform: {platform}")
        print(f"Hook: {hook_type} | Psychology: {psychology}")
        print(f"CTA: {cta_type} | Pillar: {pillar}")
        print(f"{'='*60}\n")
        
        request = PostRequest(
            platform=platform,
            hook_type=hook_type,
            psychology=psychology,
            cta_type=cta_type,
            pillar=pillar,
        )
        
        print("🚀 Generating on Modal GPU...\n")
        
        generator = ViralPostGenerator()
        post = generator.generate.remote(request)
        
        print(f"{'='*60}")
        print("GENERATED POST:")
        print(f"{'='*60}")
        print(post)
        print(f"{'='*60}\n")
    
    else:
        # Batch generation - multiple examples
        print(f"\n{'='*60}")
        print("BATCH VIRAL POST GENERATION")
        print(f"{'='*60}\n")
        
        examples = [
            PostRequest(
                platform="twitter",
                hook_type="question",
                psychology="urgency",
                cta_type="explicit",
                pillar="education"
            ),
            PostRequest(
                platform="linkedin",
                hook_type="story",
                psychology="inspiration",
                cta_type="implicit",
                pillar="education"
            ),
            PostRequest(
                platform="instagram",
                hook_type="curiosity_gap",
                psychology="FOMO",
                cta_type="explicit",
                pillar="entertainment"
            ),
        ]
        
        print("🚀 Generating 3 posts on Modal GPU...\n")
        
        generator = ViralPostGenerator()
        posts = generator.generate_batch.remote(examples)
        
        for i, (req, post) in enumerate(zip(examples, posts), 1):
            print(f"{'='*60}")
            print(f"POST {i}: {req.platform.upper()}")
            print(f"Hook: {req.hook_type} | Psychology: {req.psychology}")
            print(f"{'='*60}")
            print(post)
            print(f"{'='*60}\n")