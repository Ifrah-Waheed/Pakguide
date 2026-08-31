# PakGuide QLoRA Fine-tuning Project

This package is ready to customize a small base model with QLoRA on your FYP
domain (Pakistani government/educational services guidance).

## ⚠️ Read this first (important)

QLoRA uses **4-bit CUDA quantization** (via the `bitsandbytes` library).
**This does not work on CPU** — you need a GPU. Since your machine is
CPU-only, run this entire pipeline on **Google Colab (free T4 GPU)**.
Exact steps are below.

You can also use Kaggle Notebooks (free GPU quota) if you prefer — the setup
is the same.

---

## Files in this package

| File | What it is |
|---|---|
| `pakguide_dataset.jsonl` | Full dataset — 210 instruction-response pairs (PakGuide domain: CNIC, passport, tax, university admission, HEC scholarships, etc.) |
| `train.jsonl` | Training split (190 pairs) |
| `val.jsonl` | Held-out validation split (20 pairs) |
| `generate_dataset.py` | Script that generated the dataset (edit this file if you want to add/edit facts) |
| `train_qlora.py` | Main QLoRA training script (run this on Colab) |
| `compare_before_after.py` | Generates a before/after comparison between the base model and the fine-tuned model |

---

## Step-by-step: how to run this on Colab

### Step 1 — Open a new Colab notebook
Go to `colab.research.google.com` → New Notebook → Runtime → Change runtime
type → select **T4 GPU** (free tier).

### Step 2 — Install libraries (first cell)
```python
!pip install -q -U transformers accelerate peft trl bitsandbytes datasets
```

### Step 3 — Upload your files
Left sidebar → Files icon → upload:
- `train.jsonl`
- `val.jsonl`
- `train_qlora.py`
- `compare_before_after.py`

(Or upload `pakguide_dataset.jsonl` and build your own split instead.)

### Step 4 — Run training
```python
!python train_qlora.py
```
With this small model (1.5B params) and ~190 examples, training on a T4
GPU should take roughly **10-20 minutes** (3 epochs). Output: LoRA adapter
weights get saved in `./pakguide-qlora-adapter/` (only a few MB — this is
the trained "delta" weights, not the full base model).

### Step 5 — Run the before/after comparison
```python
!python compare_before_after.py
```
This generates answers from both the base model and the fine-tuned model on
8 test prompts, and saves them side by side in `before_after_report.md`.

### Step 6 — Download the adapter
Zip the `./pakguide-qlora-adapter/` folder and download it to your PC —
this is your **deliverable adapter weights**.
```python
!zip -r pakguide-qlora-adapter.zip pakguide-qlora-adapter
```

---

## Dataset details

- **210 instruction-response pairs**, across 35 distinct PakGuide topics
  (NADRA/CNIC, passport, driving license, FBR tax/NTN, vehicle registration,
  property mutation, domicile, HEC scholarships, university admission,
  utility connections, EOBI, etc.)
- Each fact is rendered through 6 different phrasings (formal question,
  casual ask, "explain this", "guide me step by step", etc.) so the model
  learns to understand the topic rather than memorizing one exact wording.
- Format: `{"instruction": ..., "input": "", "output": ..., "topic": ...}`
  per line — TRL's `SFTTrainer` converts this into a chat template before
  training (see the `to_chat_text()` function in `train_qlora.py`).

**Note:** These facts are general/simplified guidance meant as a starting
point. Before your FYP submission, verify/update them against current
NADRA/FBR/HEC rules — especially details like fees and processing times,
which change over time.

---

## Improvement summary — template

After running training and the comparison, fill in this table for your FYP
documentation:

| Metric | Before (base model) | After (fine-tuned) |
|---|---|---|
| Domain-specific correctness (rate manually on the 8 test prompts, 1-5 scale) | | |
| Response specificity (does it name the correct department/portal/form?) | | |
| Final eval loss (from `train_qlora.py`'s `final_eval_metrics.json`) | N/A (base model has no eval loss) | |
| Avg response length (words) | | |
| Hallucination rate (wrong fees/portal names — check manually) | | |

Typical pattern to expect: the base model gives generic/vague answers
(sometimes referencing the wrong country/system), while the fine-tuned model
consistently mentions the correct Pakistani department names (NADRA, FBR,
HEC, DGIP, Excise & Taxation) and process steps — because it has seen those
190 examples repeatedly during training.

---

## Next steps (optional extensions)

1. Grow the dataset to 500 by adding more real processes to the `FACTS` list
   in `generate_dataset.py` (e.g., provincial variations, more HEC
   scholarship categories).
2. Experiment with `num_train_epochs` in `train_qlora.py` (watch the
   `val.jsonl` loss to avoid overfitting).
3. You can also merge the adapter into a standalone model
   (`model.merge_and_unload()`) if you want to avoid adapter-loading at
   deployment time.
