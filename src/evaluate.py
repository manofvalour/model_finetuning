"""
Evaluation Suite for Fine-tuned Models
Includes: perplexity, ROUGE, BERTScore, task-specific metrics
"""

import json
import math
import torch
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Perplexity
# ─────────────────────────────────────────────

@torch.inference_mode()
def compute_perplexity(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 1024,
    stride: int = 512,
    batch_size: int = 4,
) -> Dict[str, float]:
    """
    Sliding-window perplexity (handles texts longer than context window).
    Returns mean PPL and standard deviation across the corpus.
    """
    model.eval()
    loss_fn = CrossEntropyLoss(reduction="none")
    ppls = []

    for text in tqdm(texts, desc="Computing perplexity"):
        encodings = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = encodings.input_ids.to(model.device)
        seq_len = input_ids.size(1)

        nlls = []
        prev_end = 0

        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            target_len = end - prev_end
            input_chunk = input_ids[:, begin:end]
            target_chunk = input_chunk.clone()
            target_chunk[:, :-target_len] = -100  # mask previously seen tokens

            with torch.no_grad():
                outputs = model(input_chunk, labels=target_chunk)
                neg_ll = outputs.loss * target_len
            nlls.append(neg_ll)
            prev_end = end
            if end == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).sum() / prev_end).item()
        ppls.append(ppl)

    return {
        "perplexity_mean": round(float(np.mean(ppls)), 4),
        "perplexity_std": round(float(np.std(ppls)), 4),
        "perplexity_median": round(float(np.median(ppls)), 4),
    }


# ─────────────────────────────────────────────
# Generation Quality Metrics
# ─────────────────────────────────────────────

def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """ROUGE-1, ROUGE-2, ROUGE-L scores."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(predictions, references):
            s = scorer.score(ref, pred)
            for k in scores:
                scores[k].append(s[k].fmeasure)
        return {k: round(float(np.mean(v)), 4) for k, v in scores.items()}
    
    except ImportError:
        logger.warning("Install rouge-score: pip install rouge-score")
        return {}


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    lang: str = "en",
    model_type: str = "microsoft/deberta-xlarge-mnli",
) -> Dict[str, float]:
    """BERTScore F1 (semantic similarity)."""
    try:
        from bert_score import score
        P, R, F1 = score(predictions, references, lang=lang, model_type=model_type, verbose=False)
        return {
            "bertscore_precision": round(P.mean().item(), 4),
            "bertscore_recall": round(R.mean().item(), 4),
            "bertscore_f1": round(F1.mean().item(), 4),
        }
    except ImportError:
        logger.warning("Install bert-score: pip install bert-score")
        return {}


#Instruction-Following Benchmark
@dataclass
class BenchmarkResult:
    model_name: str
    task: str
    metrics: Dict[str, float] = field(default_factory=dict)
    num_samples: int = 0
    generation_config: Dict = field(default_factory=dict)


def run_generation_benchmark(
    model,
    tokenizer,
    dataset: Dataset,
    prompt_template: str,
    reference_column: str = "output",
    max_new_tokens: int = 256,
    batch_size: int = 8,
    temperature: float = 0.1,
) -> BenchmarkResult:
    """Run full generation benchmark on a dataset split."""
    predictions, references = [], []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        prompts = [prompt_template.format(**row) for row in batch]
        refs = [row[reference_column] for row in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # Strip prompt from output
        for prompt, dec, ref in zip(prompts, decoded, refs):
            predictions.append(dec[len(prompt):].strip())
            references.append(ref.strip())

    rouge = compute_rouge(predictions, references)
    bert = compute_bertscore(predictions, references)

    result = BenchmarkResult(
        model_name=getattr(model.config, "_name_or_path", "unknown"),
        task="generation",
        metrics={**rouge, **bert},
        num_samples=len(dataset),
        generation_config={"max_new_tokens": max_new_tokens, "temperature": temperature},
    )
    return result


# ─────────────────────────────────────────────
# Comparison: Base vs Fine-tuned
# ─────────────────────────────────────────────

def compare_models(
    base_model_path: str,
    adapter_path: str,
    eval_texts: List[str],
    output_json: str = "eval_comparison.json",
):
    """Side-by-side perplexity comparison of base vs fine-tuned model."""
    from inference import LoraInference

    logger.info("Evaluating base model...")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    base_ppl = compute_perplexity(base_model, base_tokenizer, eval_texts)

    logger.info("Evaluating fine-tuned model...")
    ft = LoraInference(base_model_path, adapter_path)
    ft_ppl = compute_perplexity(ft.model, ft.tokenizer, eval_texts)

    results = {
        "base_model": {"path": base_model_path, **base_ppl},
        "finetuned_model": {"path": adapter_path, **ft_ppl},
        "delta": {
            k: round(base_ppl[k] - ft_ppl[k], 4) for k in base_ppl
        },
    }

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_json}")
    return results
