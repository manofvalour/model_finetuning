# 🧬 LoRA / QLoRA Fine-tuning Framework

> **Parameter-efficient fine-tuning of large language models for domain-specific tasks**  
> Supports: Medical QA · Code Generation · Legal Summarization · Any instruction-following task

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red?logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co)
[![PEFT](https://img.shields.io/badge/PEFT-0.9%2B-green)](https://github.com/huggingface/peft)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## What is LoRA / QLoRA?

**LoRA** (Low-Rank Adaptation) injects small, trainable rank-decomposition matrices into frozen transformer weights. Instead of updating all ~7 billion parameters in a 7B model, you train only ~0.1–1% of them — with competitive or superior task-specific performance.

**QLoRA** extends this by first quantizing the base model to 4-bit (using NF4 + double quantization), enabling fine-tuning of a 65B model on a single 48GB GPU, or a 7B model on a consumer 24GB GPU.

```
Base Model (frozen, 4-bit)          LoRA Adapters (trainable)
┌─────────────────────┐            ┌──────────────────────┐
│  W ∈ ℝ^(d×k)        │   +        │  A ∈ ℝ^(d×r)         │
│  (7B params, 4-bit) │            │  B ∈ ℝ^(r×k)         │
│  ~3.5 GB VRAM       │            │  r << min(d,k)       │
└─────────────────────┘            │  ~40M params, bf16   │
                                   └──────────────────────┘
          Forward: h = W₀x + (BA)x · (α/r)
```

---

## Project Structure

```
lora-finetune/
├── src/
│   ├── train.py          # Main training entry point
│   ├── inference.py      # Adapter loading, merging, generation
│   ├── evaluate.py       # Perplexity, ROUGE, BERTScore metrics
│   └── data_utils.py     # Dataset formatting & domain converters
├── configs/
│   ├── medical_qa.json          # Mistral-7B on MedQA
│   ├── code_generation.json     # CodeLlama-7B on Python instruct
│   └── legal_summarization.json # LLaMA-2-13B on LegalBench
├── notebooks/
│   └── lora_walkthrough.ipynb   # Full interactive walkthrough
├── docs/
│   └── portfolio.html           # Portfolio showcase page
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/manofvalour/lora-finetune
cd lora-finetune
uv pip install -r requirements.txt

# Optional: Flash Attention 2 (speeds up training ~2×)
pip install flash-attn --no-build-isolation
```

### 2. Train

```bash
# Medical QA fine-tuning (Mistral-7B + QLoRA)
python src/train.py --config configs/medical_qa.json

# Code generation (CodeLlama-7B + LoRA)
python src/train.py --config configs/code_generation.json

# Legal summarization (LLaMA-2-13B + QLoRA)
python src/train.py --config configs/legal_summarization.json
```

### 3. Inference

```python
from src.inference import LoraInference

model = LoraInference(
    base_model_name="mistralai/Mistral-7B-v0.1",
    adapter_path="./outputs/medical-lora",
    load_in_4bit=True,
)

response = model.generate(
    "### Instruction:\nWhat are the first-line treatments for Type 2 Diabetes?\n\n### Response:\n",
    max_new_tokens=256,
    temperature=0.3,
)
print(response)
```

### 4. Merge Adapter → Standalone Model

```python
from src.inference import merge_and_save

merge_and_save(
    base_model_name="mistralai/Mistral-7B-v0.1",
    adapter_path="./outputs/medical-lora",
    output_path="./outputs/medical-merged",
    push_to_hub=True,
    hub_repo="manofvalour/mistral-7b-medical",
)
```

### 5. Evaluate

```python
from src.evaluate import compare_models

results = compare_models(
    base_model_path="mistralai/Mistral-7B-v0.1",
    adapter_path="./outputs/medical-lora",
    eval_texts=test_corpus,
    output_json="eval_comparison.json",
)
# → prints delta perplexity, ROUGE, BERTScore
```

---

## Supported Tasks & Configurations

| Domain | Base Model | Dataset | r | VRAM | Adapter Size |
|---|---|---|---|---|---|
| Medical QA | Mistral-7B | medalpaca/medical_meadow_medqa | 64 | ~14 GB | ~120 MB |
| Code Gen | CodeLlama-7B | iamtarun/python_code_instructions_18k | 32 | ~12 GB | ~60 MB |
| Legal Summarization | LLaMA-2-13B | nguha/legalbench | 128 | ~24 GB | ~450 MB |

---

## Key Design Decisions

### Why QLoRA over full fine-tuning?

| Approach | VRAM (7B) | Train Time | Performance |
|---|---|---|---|
| Full fine-tune (fp16) | ~112 GB | 1× | Baseline |
| LoRA (r=64, bf16) | ~28 GB | 0.9× | ≈ Baseline |
| QLoRA (4-bit NF4 + LoRA) | **~14 GB** | 1.1× | ≈ Baseline |

QLoRA makes 7B–13B model fine-tuning accessible on a single consumer GPU (e.g. RTX 3090/4090).

### Completion-only training

The `DataCollatorForCompletionOnlyLM` masks loss on the prompt tokens — the model only learns to predict the response, not the instruction. This prevents "instruction collapse" where the model starts generating instructions instead of responses.

### Rank selection guidelines

| Use case | Recommended r | Notes |
|---|---|---|
| Quick domain adaptation | 8–16 | Low compute, often sufficient |
| General instruction tuning | 32–64 | Good balance |
| Complex reasoning tasks | 64–128 | More adapter capacity |
| Near full-finetune quality | 128–256 | Diminishing returns past 256 |

---

## Experiment Tracking

Set `report_to: "wandb"` in your config and run:

```bash
wandb login
python src/train.py --config configs/medical_qa.json
```

Tracked metrics: `train/loss`, `eval/loss`, `learning_rate`, `epoch`, `grad_norm`.

---

## GPU Requirements

| Model Size | Quantization | Minimum VRAM |
|---|---|---|
| 7B | 4-bit QLoRA | 12–16 GB |
| 13B | 4-bit QLoRA | 20–24 GB |
| 7B | LoRA (bf16) | 24–28 GB |
| 13B | LoRA (bf16) | 48 GB |
| 70B | 4-bit QLoRA | 48 GB |

For multi-GPU training, `accelerate` handles device mapping automatically via `device_map="auto"`.

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Citation

```bibtex
@misc{lora-finetune-2026,
  title   = {LoRA/QLoRA Fine-tuning Framework for Domain-Specific LLMs},
  author  = {Emmanuel Ajala},
  year    = {2026},
  url     = {https://github.com/manofvalour/lora-finetune}
}
```
