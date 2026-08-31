"""
QLoRA fine-tuning for PakGuide domain adaptation.

RUN THIS ON GOOGLE COLAB (free T4 GPU) -- NOT on a CPU-only machine.
QLoRA needs 4-bit CUDA quantization (bitsandbytes), which does not work on CPU.

Colab setup (first cell):
    !pip install -q -U transformers accelerate peft trl bitsandbytes datasets

Usage:
    python train_qlora.py

Files expected in the same folder:
    train.jsonl   (190 instruction-response pairs)
    val.jsonl     (20 held-out pairs, used for eval loss during training)

Output:
    ./pakguide-qlora-adapter/   <- LoRA adapter weights (small, a few MB)
"""

import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # small, ungated, works great with QLoRA on a T4
OUTPUT_DIR = "./pakguide-qlora-adapter"
MAX_SEQ_LEN = 512

SYSTEM_PROMPT = (
    "You are PakGuide, an assistant that helps users understand Pakistani "
    "government and educational service procedures clearly and accurately."
)

# ---------------------------------------------------------------------------
# 2. Load dataset and format into chat-style text
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def to_chat_text(example):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


dataset = load_dataset(
    "json",
    data_files={"train": "train.jsonl", "validation": "val.jsonl"},
)
dataset = dataset.map(to_chat_text, remove_columns=dataset["train"].column_names)

# ---------------------------------------------------------------------------
# 3. Load base model in 4-bit (QLoRA)
# ---------------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

# ---------------------------------------------------------------------------
# 4. LoRA config
# ---------------------------------------------------------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # sanity check: should be < 1-2% of total params

# ---------------------------------------------------------------------------
# 5. Training
# ---------------------------------------------------------------------------
sft_config_kwargs = dict(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    fp16=False,
    max_seq_length=MAX_SEQ_LEN,
    dataset_text_field="text",
    report_to="none",
)


def build_sft_config(kwargs):
    """Some kwargs get renamed/removed between trl versions. Drop any
    that this installed version doesn't accept and retry, instead of
    crashing the whole run."""
    kwargs = dict(kwargs)
    while True:
        try:
            return SFTConfig(**kwargs)
        except TypeError as e:
            msg = str(e)
            bad_key = None
            for k in list(kwargs.keys()):
                if k in msg:
                    bad_key = k
                    break
            if bad_key is None:
                raise
            print(f"[compat] Dropping unsupported SFTConfig arg: {bad_key}")
            kwargs.pop(bad_key)


sft_config = build_sft_config(sft_config_kwargs)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
)

trainer.train()

# ---------------------------------------------------------------------------
# 6. Save adapter (small, portable -- a few MB, not the full base model)
# ---------------------------------------------------------------------------
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to {OUTPUT_DIR}")

# Save final eval loss for the improvement summary
metrics = trainer.evaluate()
with open(f"{OUTPUT_DIR}/final_eval_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Final eval metrics:", metrics)
