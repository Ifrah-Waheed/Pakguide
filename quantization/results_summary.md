# Task 3: Quantization & Optimization — Results

## Model
Qwen2.5-1.5B-Instruct, fine-tuned with QLoRA on PakGuide dataset,
merged with base model, converted to GGUF, and quantized to
multiple precision levels using llama.cpp.

## Benchmark Results

| Model  | File Size (MB) | Time (s) | Speed (tokens/sec) | RAM Used (MB) |
|--------|----------------|----------|---------------------|----------------|
| FP16   | 2950.35        | 25.55    | 3.91                | ~2.0 (noisy)   |
| Q8_0   | 1570.29        | 26.49    | 3.78                | 25.92          |
| Q5_K_M | 1072.93        | 22.58    | 4.43                | 19.83          |
| Q4_K_M | 940.37         | 16.24    | 6.16                | 7.93           |

## Trade-off Analysis

- **FP16 → Q4_K_M**: ~68% reduction in file size (2950 MB → 940 MB)
- Lower precision consistently improves inference speed, with Q4_K_M
  being the fastest at 6.16 tokens/sec, nearly 1.6x faster than FP16.
- Q8_0 offers the highest fidelity to the original model but has
  minimal speed benefit over FP16, making it less attractive unless
  maximum accuracy is critical.
- Q5_K_M is a reasonable middle ground between size and quality.
- **Recommendation**: For PakGuide's deployment (offline, resource-
  constrained environments), Q4_K_M offers the best trade-off —
  smallest footprint and fastest inference, with acceptable quality
  loss for a instruction-following assistant task.