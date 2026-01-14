import modal
import json

app = modal.App("viral-post-inference-v6")

# Container image
inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate==0.21.0",
        "peft==0.4.0",
        "transformers==4.31.0",
        "unsloth[cu118-torch201]==2023.10",
    )
)

# Volumes
checkpoint_vol = modal.Volume.from_name("viral-checkpoints-v6", create_if_missing=False)

@app.function(
    image=inference_image,
    gpu="A100",
    volumes={"/checkpoints": checkpoint_vol},
)
def generate_post(platform: str = "twitter", hook_type: str = "question",
                  psychology: str = '["urgency"]',
                  cta_type: str = "explicit", pillar: str = "education"):
    """Generate a viral post using the v6 fine-tuned model."""
    from unsloth import FastLanguageModel
    import torch

    print("🔄 Loading model from v6 checkpoint...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/checkpoints/final_v6",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Manually set the chat template for Mistral Instruct
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% if messages[0]['role'] == 'user' %}" \
                                  "<s>[INST] {{ messages[0]['content'] }} [/INST]" \
                                  "{% for message in messages[1:] %}" \
                                      "{% if message['role'] == 'assistant' %}" \
                                          "{{ message['content'] }}</s>" \
                                      "{% endif %}" \
                                  "{% endfor %}" \
                                  "{% endif %}"

    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    instruction = "Generate a viral social media post with these characteristics."
    input_text = f"Platform: {platform}\nHook type: {hook_type}\nPsychology: {psychology}\nCTA: {cta_type}\nPillar: {pillar}"

    chat = [
        {"role": "user", "content": f"{instruction}\n\n{input_text}"}
    ]

    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    print("📝 Generating...")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the assistant's response
    if "[/INST]" in generated:
        result = generated.split("[/INST]")[-1].strip()
    else:
        result = generated

    return result

@app.local_entrypoint()
def main():
    post = generate_post.remote(
        platform="twitter",
        hook_type="statement",
        psychology='["surprise", "FOMO"]',
        cta_type="none",
        pillar="education",
    )
    print("\n" + "="*60)
    print("GENERATED POST:")
    print("="*60)
    print(post)
    print("="*60)
