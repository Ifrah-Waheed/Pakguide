"""
Task 3: Quantize the FP16 GGUF model into multiple precision levels.
Run this in Google Colab, after merge_and_convert.py has produced
/content/pakguide-fp16.gguf
"""

import subprocess

quant_levels = ["Q8_0", "Q5_K_M", "Q4_K_M"]

for level in quant_levels:
    output_path = f"/content/pakguide-{level.lower()}.gguf"
    print(f"Quantizing to {level}...")
    subprocess.run([
        "./llama.cpp/build/bin/llama-quantize",
        "/content/pakguide-fp16.gguf",
        output_path,
        level
    ])
    print(f"{level} quantization complete! Saved at {output_path}")