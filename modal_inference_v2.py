import modal
from dataclasses import dataclass

app = modal.App("viral-post-inference")

# 🔹 Same dependencies & Unsloth setup as training
inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "unsloth[cu128-torch270]==2025.7.8",
        "transformers==4.54.0",
        "peft==0.16.0",
        "accelerate==1.9.0",
    )
)

# 🔹 Reuse checkpoint volume from training
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
    volumes={"/checkpoints": checkpoint_vol},
    scaledown_window=300,
)
class ViralPostGenerator:

    @modal.enter()
    def load_model(self):
        """Load the fine-tuned model once per container."""
        from unsloth import FastLanguageModel
        from peft import PeftModel

        print("🔄 Loading Unsloth model for inference...")

        # Base model — same as training
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/mistral-7b-v0.3-bnb-4bit",
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        )

        # ✅ Correct way: load LoRA adapter using PEFT
        self.model = PeftModel.from_pretrained(
            self.model,
            "/checkpoints/final",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("✅ Model and tokenizer loaded successfully!")


    @modal.method()
    def generate(self, request: PostRequest) -> str:
        """Generate a viral post based on request parameters."""
        import torch

        # Match dataset prompt structure exactly
        prompt = (
            "### Instruction:\n"
            "Generate a viral social media post with these characteristics.\n\n"
            "### Input:\n"
            f"Platform: {request.platform}\n"
            f"Hook type: {request.hook_type}\n"
            f"Psychology: ['{request.psychology}']\n"
            f"CTA: {request.cta_type}\n"
            f"Pillar: {request.pillar}\n\n"
            "### Output:\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the model's response after "### Output:"
        if "### Output:" in generated_text:
            result = generated_text.split("### Output:")[1].strip()
        else:
            result = generated_text.strip()

        # Truncate if the model includes its own <|endoftext|>
        if "<|endoftext|>" in result:
            result = result.split("<|endoftext|>")[0].strip()

        return result

    @modal.method()
    def generate_batch(self, requests: list[PostRequest]) -> list[str]:
        """Generate multiple posts at once."""
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
    """Run generation locally (triggers Modal cloud inference)."""

    if not batch:
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
        print(f"\n{'='*60}")
        print("BATCH VIRAL POST GENERATION")
        print(f"{'='*60}\n")

        examples = [
            PostRequest(
                platform="twitter",
                hook_type="question",
                psychology="urgency",
                cta_type="explicit",
                pillar="education",
            ),
            PostRequest(
                platform="linkedin",
                hook_type="story",
                psychology="inspiration",
                cta_type="implicit",
                pillar="education",
            ),
            PostRequest(
                platform="instagram",
                hook_type="curiosity_gap",
                psychology="FOMO",
                cta_type="explicit",
                pillar="entertainment",
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