"""
Dataset Preparation Utilities
Handles: Alpaca-format, ShareGPT, raw text, custom JSON/CSV
"""

import json
import csv
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)

ALPACA_PROMPT = (
    "Below is an instruction that describes a task{context_str}. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "{input_block}"
    "### Response:\n{output}"
)

CHAT_PROMPT = "{system}\n\n" + "\n\n".join([
    "Human: {human}",
    "Assistant: {assistant}",
])


#Format Converters
def text_to_alpaca(row: dict) -> str:
    
    context_str = ", using the input below as context" if row.get("input") else ""
    input_block = f"### Input:\n{row['input']}\n\n" if row.get("input") else ""
    
    return ALPACA_PROMPT.format(
        context_str=context_str,
        instruction=row["instruction"],
        input_block=input_block,
        output=row["output"],
    )


def text_to_sharegpt(row: dict, system_prompt: str = "") -> str:
    """Convert ShareGPT multi-turn conversation to text."""
    parts = []
    if system_prompt:
        parts.append(f"System: {system_prompt}\n")
    for turn in row.get("conversations", []):
        role = "Human" if turn["from"] in ("human", "user") else "Assistant"
        parts.append(f"{role}: {turn['value']}")
    return "\n\n".join(parts)


def raw_text_chunker(text: str, chunk_size: int = 1024,
    overlap: int = 128) -> List[str]:

    """Split long text into overlapping chunks for pre-training style datasets."""
    
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


# ─────────────────────────────────────────────
# Domain-Specific Dataset Builders
# ─────────────────────────────────────────────

def build_medical_qa_dataset(source_path: str, output_path: str):
    """
    Example: Convert MedQA / PubMedQA style data to instruction format.
    Input JSON: [{"question": ..., "answer": ..., "context": ...}]
    """
    with open(source_path) as f:
        data = json.load(f)

    records = []
    for item in data:
        records.append({
            "instruction": item["question"],
            "input": item.get("context", ""),
            "output": item["answer"],
            "text": text_to_alpaca({
                "instruction": item["question"],
                "input": item.get("context", ""),
                "output": item["answer"],
            }),
        })

    ds = Dataset.from_list(records)
    ds.save_to_disk(output_path)
    logger.info(f"Saved {len(ds)} medical QA samples to {output_path}")
    return ds


def build_code_instruct_dataset(source_path: str, output_path: str):
    """
    Example: Convert code instruction data to training format.
    Input JSON: [{"prompt": ..., "completion": ..., "language": ...}]
    """
    with open(source_path) as f:
        data = json.load(f)

    records = []
    for item in data:
        lang = item.get("language", "python")
        instruction = f"Write {lang} code for the following task:\n{item['prompt']}"
        records.append({
            "instruction": instruction,
            "input": "",
            "output": f"```{lang}\n{item['completion']}\n```",
            "text": text_to_alpaca({
                "instruction": instruction,
                "input": "",
                "output": f"```{lang}\n{item['completion']}\n```",
            }),
        })

    ds = Dataset.from_list(records)
    ds.save_to_disk(output_path)
    logger.info(f"Saved {len(ds)} code instruction samples to {output_path}")
    return ds


def build_legal_dataset(source_path: str, output_path: str):
    """
    Example: Legal document summarization dataset.
    Input JSON: [{"document": ..., "summary": ..., "jurisdiction": ...}]
    """
    with open(source_path) as f:
        data = json.load(f)

    records = []
    for item in data:
        jurisdiction = item.get("jurisdiction", "")
        instruction = (
            f"Summarize the following legal document"
            + (f" ({jurisdiction} jurisdiction)" if jurisdiction else "")
            + " in plain language:"
        )
        records.append({
            "instruction": instruction,
            "input": item["document"],
            "output": item["summary"],
            "text": text_to_alpaca({
                "instruction": instruction,
                "input": item["document"],
                "output": item["summary"],
            }),
        })

    ds = Dataset.from_list(records)
    ds.save_to_disk(output_path)
    logger.info(f"Saved {len(ds)} legal samples to {output_path}")
    return ds


# ─────────────────────────────────────────────
# Generic CSV / JSON Loader
# ─────────────────────────────────────────────

def load_csv_as_instruct(
    csv_path: str,
    instruction_col: str,
    output_col: str,
    input_col: Optional[str] = None,
    system_prefix: str = "",
) -> Dataset:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            instruction = (system_prefix + " " + row[instruction_col]).strip()
            inp = row.get(input_col, "") if input_col else ""
            out = row[output_col]
            rows.append({
                "instruction": instruction,
                "input": inp,
                "output": out,
                "text": text_to_alpaca({"instruction": instruction, "input": inp, "output": out}),
            })
    return Dataset.from_list(rows)


def train_val_test_split(
    dataset: Dataset,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> DatasetDict:
    """Split a dataset into train/val/test."""
    total = len(dataset)
    n_test = int(total * test_ratio)
    n_val = int(total * val_ratio)

    tmp = dataset.train_test_split(test_size=n_test, seed=seed)
    train_val = tmp["train"].train_test_split(test_size=n_val, seed=seed)

    return DatasetDict({
        "train": train_val["train"],
        "validation": train_val["test"],
        "test": tmp["test"],
    })