"""
Before / After comparison for the PakGuide QLoRA adapter.

Runs the SAME set of test prompts through:
  (A) the base model alone (no fine-tuning)
  (B) the base model + PakGuide LoRA adapter

...and writes a side-by-side markdown report so you can see the improvement.

Run this AFTER train_qlora.py has produced ./pakguide-qlora-adapter/
(same Colab session, or reload the adapter later).

    python compare_before_after.py
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = "./pakguide-qlora-adapter"
SYSTEM_PROMPT = (
    "You are PakGuide, an assistant that helps users understand Pakistani "
    "government and educational service procedures clearly and accurately."
)

# Held-out style prompts -- phrased differently from the training templates,
# so this is a genuine test of generalization, not memorized wording.
TEST_PROMPTS = [
    "My CNIC has expired, what should I do now?",
    "What's the process to get my degree attested for a job abroad?",
    "I want to become a tax filer this year, walk me through it.",
    "How does someone in Lahore get a new gas connection for their house?",
    "I lost my passport in Dubai, what should I do?",
    "What documents does IBCC need for O-Level equivalence?",
    "How do university students in Pakistan usually apply for HEC scholarships?",
    "Steps to transfer a used car's registration to a new owner?",
]


def load_model(with_adapter: bool):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto"
    )
    if with_adapter:
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    return model


def generate(model, tokenizer, prompt, max_new_tokens=180):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[-1]
    text = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
    return text.strip()


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []

    print("Loading BASE model (no fine-tuning)...")
    base_model = load_model(with_adapter=False)
    for p in TEST_PROMPTS:
        ans = generate(base_model, tokenizer, p)
        results.append({"prompt": p, "before": ans})
        print(f"[BEFORE] {p}\n-> {ans}\n")
    del base_model
    torch.cuda.empty_cache()

    print("Loading FINE-TUNED model (base + PakGuide adapter)...")
    ft_model = load_model(with_adapter=True)
    for r in results:
        ans = generate(ft_model, tokenizer, r["prompt"])
        r["after"] = ans
        print(f"[AFTER] {r['prompt']}\n-> {ans}\n")

    # Write markdown report
    with open("before_after_report.md", "w", encoding="utf-8") as f:
        f.write("# PakGuide QLoRA -- Before vs After Comparison\n\n")
        for r in results:
            f.write(f"## Prompt: {r['prompt']}\n\n")
            f.write(f"**Before (base model):**\n\n{r['before']}\n\n")
            f.write(f"**After (fine-tuned):**\n\n{r['after']}\n\n")
            f.write("---\n\n")

    with open("before_after_raw.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Saved before_after_report.md and before_after_raw.json")


if __name__ == "__main__":
    main()
