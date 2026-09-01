"""
Task 3: Merge LoRA adapter with base model, then convert to GGUF format.
Run this in Google Colab with GPU runtime.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_path = "/content/drive/MyDrive/pakguide-qlora-adapter/pakguide-qlora-adapter"
merged_model_path = "/content/pakguide-merged"

print("Step A: Loading base model (full precision)...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Step B: Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

print("Step C: Applying LoRA adapter to base model...")
model_with_adapter = PeftModel.from_pretrained(base_model, adapter_path)

print("Step D: Merging adapter into base model...")
merged_model = model_with_adapter.merge_and_unload()

print("Step E: Saving merged model...")
merged_model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

print(f"Merge complete! Model saved at: {merged_model_path}")

# --- GGUF Conversion ---
# Run these separately in Colab after llama.cpp is cloned and built:
#
# !git clone https://github.com/ggerganov/llama.cpp.git
# !pip install -q -r llama.cpp/requirements.txt
# !cd llama.cpp && cmake -B build && cmake --build build --config Release -j 4
#
# import os
# os.environ["USE_TF"] = "0"
# os.environ["USE_FLAX"] = "0"
#
# !python llama.cpp/convert_hf_to_gguf.py /content/pakguide-merged \
#     --outfile /content/pakguide-fp16.gguf \
#     --outtype f16