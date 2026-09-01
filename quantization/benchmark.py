"""
Task 3: Benchmark FP16 and quantized GGUF models.
Measures file size, RAM usage, and inference speed.
Run this in Google Colab after quantize.py has produced all .gguf files.
"""

import subprocess
import time
import os
import psutil

models = {
    "FP16": "/content/pakguide-fp16.gguf",
    "Q8_0": "/content/pakguide-q8_0.gguf",
    "Q5_K_M": "/content/pakguide-q5_k_m.gguf",
    "Q4_K_M": "/content/pakguide-q4_k_m.gguf",
}

test_prompt = "What documents are required for a Pakistani passport renewal?"

results = []

for name, path in models.items():
    print(f"\n{'='*50}")
    print(f"Testing: {name}")
    print(f"{'='*50}")

    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    ram_before = psutil.virtual_memory().used / (1024 * 1024)
    start_time = time.time()

    result = subprocess.run(
        [
            "./llama.cpp/build/bin/llama-cli",
            "-m", path,
            "-p", test_prompt,
            "-n", "100",
            "--temp", "0.7",
            "-st"
        ],
        capture_output=True,
        text=True,
        timeout=300
    )

    end_time = time.time()
    total_time = end_time - start_time
    ram_after = psutil.virtual_memory().used / (1024 * 1024)
    ram_used = ram_after - ram_before
    tokens_per_sec = 100 / total_time if total_time > 0 else 0

    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Time taken: {total_time:.2f} seconds")
    print(f"Approx speed: {tokens_per_sec:.2f} tokens/sec")
    print(f"RAM used (approx): {ram_used:.2f} MB")

    results.append({
        "Model": name,
        "File Size (MB)": round(file_size_mb, 2),
        "Time (s)": round(total_time, 2),
        "Speed (tokens/sec)": round(tokens_per_sec, 2),
        "RAM Used (MB)": round(ram_used, 2)
    })

print("\nBenchmarking complete for all models!")