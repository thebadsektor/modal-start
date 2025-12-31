import modal
from dataclasses import dataclass
import os

app = modal.App("viral-post-inference")

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

@dataclass
class PostRequest:
    platform: str = "twitter"
    hook_type: str = "question"
    psychology: str = "urgency"
    cta_type: str = "explicit"
    pillar: str = "education"

@app.cls(
    image=inference_image,
    gpu="T4",
    volumes={"/checkpoints": checkpoint_vol},  # Match training mount!
    scaledown_window=300,
)
class ViralPostGenerator:

    @modal.enter()
    def load_model(self):
        """Load the fine-tuned model once per container."""
        from unsloth import FastLanguageModel
        from peft import PeftModel

        print("🔄 Loading fine-tuned model...")

        model_name = "unsloth/mistral-7b-v0.3-bnb-4bit"
        adapter_path = "/checkpoints/final"

        # Verify checkpoint exists
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")
        
        print(f"✅ Found checkpoint at {adapter_path}")

        # Load base model with Unsloth
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        )

        # Load fine-tuned adapter
        print("📦 Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        
        # Enable fast inference mode
        FastLanguageModel.for_inference(self.model)

        # Fix tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("✅ Model loaded and ready!")

    @modal.method()
    def generate(self, request: PostRequest) -> str:
        """Generate a viral post based on request parameters."""
        import torch

        # Match EXACT training format
        prompt = f"""### Instruction:
Generate a viral social media post with these characteristics.

### Input:
Platform: {request.platform}
Hook type: {request.hook_type}
Psychology: ['{request.psychology}']
CTA: {request.cta_type}
Pillar: {request.pillar}

### Output:
"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the output
        if "### Output:" in generated_text:
            result = generated_text.split("### Output:")[-1].strip()
        else:
            result = generated_text.strip()

        # Clean up
        if "<|endoftext|>" in result:
            result = result.split("<|endoftext|>")[0].strip()
        
        # Remove any remaining instruction/input text
        if "### Instruction" in result:
            result = result.split("### Instruction")[0].strip()
        if "### Input" in result:
            result = result.split("### Input")[0].strip()

        return result

    @modal.method()
    def generate_batch(self, requests: list[PostRequest]) -> list[str]:
        """Generate multiple posts"""
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
    """Generate viral posts"""
    
    generator = ViralPostGenerator()
    
    if not batch:
        # Single post generation
        print(f"\n{'='*60}")
        print(f"GENERATING {platform.upper()} POST")
        print(f"{'='*60}")
        print(f"Hook: {hook_type} | Psychology: {psychology}")
        print(f"CTA: {cta_type} | Pillar: {pillar}")
        print(f"{'='*60}\n")
        
        request = PostRequest(platform, hook_type, psychology, cta_type, pillar)
        post = generator.generate.remote(request)
        
        print(f"{'='*60}")
        print("GENERATED POST:")
        print(f"{'='*60}")
        print(post)
        print(f"{'='*60}\n")
    
    else:
        # Batch generation
        print(f"\n{'='*60}")
        print("BATCH GENERATION")
        print(f"{'='*60}\n")
        
        examples = [
            PostRequest("twitter", "question", "urgency", "explicit", "education"),
            PostRequest("linkedin", "story", "inspiration", "implicit", "education"),
            PostRequest("instagram", "curiosity_gap", "FOMO", "explicit", "entertainment"),
        ]
        
        posts = generator.generate_batch.remote(examples)
        
        for i, (req, post) in enumerate(zip(examples, posts), 1):
            print(f"{'='*60}")
            print(f"POST {i}: {req.platform.upper()}")
            print(f"Hook: {req.hook_type} | Psychology: {req.psychology}")
            print(f"{'='*60}")
            print(post)
            print(f"{'='*60}\n")