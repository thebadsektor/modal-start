import modal
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

app = modal.App("viral-post-finetuner-v7")

# Container image with corrected dependencies
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate==1.9.0",
        "datasets==3.6.0",
        "peft==0.16.0",
        "transformers==4.54.0",
        "trl==0.19.1",
        "unsloth[cu128-torch270]==2025.7.8",
        "huggingface-hub==0.34.2",
    )
    .env({"HF_HOME": "/model_cache"})
)

with train_image.imports():
    import unsloth
    import datasets
    import torch
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

# Volumes
model_cache = modal.Volume.from_name("viral-model-cache", create_if_missing=True)
checkpoint_vol = modal.Volume.from_name("viral-checkpoints-v7", create_if_missing=True)
data_vol = modal.Volume.from_name("viral-data", create_if_missing=True)

@dataclass
class TrainingConfig:
    model_name: str
    dataset_path: str
    batch_size: int
    learning_rate: float
    lora_r: int
    lora_alpha: int
    max_seq_length: int = 2048
    num_epochs: int = 3
    experiment_name: Optional[str] = None

    def __post_init__(self):
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.experiment_name = f"mistral-viral-{timestamp}"

@app.function(
    image=train_image,
    gpu="A100",
    volumes={
        "/model_cache": model_cache,
        "/checkpoints": checkpoint_vol,
        "/data": data_vol,
    },
    timeout=6 * 60 * 60,
    retries=modal.Retries(max_retries=2),
)
def finetune(config: TrainingConfig):
    import json

    print(f"🚀 Starting: {config.experiment_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% if messages[0]['role'] == 'user' %}" \
                                  "<s>[INST] {{ messages[0]['content'] }} [/INST]" \
                                  "{% for message in messages[1:] %}" \
                                      "{% if message['role'] == 'assistant' %}" \
                                          "{{ message['content'] }}</s>" \
                                      "{% endif %}" \
                                  "{% endfor %}" \
                                  "{% endif %}"

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing=True,
        random_state=42,
    )

    dataset_file = f"/data/{config.dataset_path}"

    data = []
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"Loaded {len(data)} training examples")

    def format_example(ex):
        chat = [
            {"role": "user", "content": f"{ex['instruction']}\n\n{ex['input']}"},
            {"role": "assistant", "content": ex['output']}
        ]
        return {"text": tokenizer.apply_chat_template(chat, tokenize=False)}

    dataset = datasets.Dataset.from_list([format_example(ex) for ex in data])

    training_args = TrainingArguments(
        output_dir="/checkpoints/final_v7",
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=config.learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        packing=True,
    )

    trainer.train()

    print("Saving final model...")
    model.save_pretrained("/checkpoints/final_v7")
    tokenizer.save_pretrained("/checkpoints/final_v7")

    checkpoint_vol.commit()

    print(f"✅ Done: {config.experiment_name}")
    return config.experiment_name

@app.local_entrypoint()
def main():
    config = TrainingConfig(
        model_name="unsloth/mistral-7b-v0.3-bnb-4bit",
        dataset_path="training_data_augmented.jsonl",
        batch_size=2,
        learning_rate=2e-4,
        lora_r=16,
        lora_alpha=32,
        num_epochs=3,
    )

    finetune.remote(config)
