"""
LoRA/QLoRA Fine-tuning Trainer
Supports: causal LMs (LLaMA, Mistral, Falcon, GPT-2, etc.)
"""

import os
import json
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from .data_config import ModelConfig, QuantizationConfig, LoraAdapterConfig, DataConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────

def load_base_model(model_cfg: ModelConfig, quant_cfg: QuantizationConfig):
    """Load model with optional quantization."""
    bnb_config = None

    if quant_cfg.use_4bit:
        compute_dtype = getattr(torch, quant_cfg.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_cfg.bnb_4bit_use_double_quant,
        )
        logger.info("Using 4-bit QLoRA quantization (NF4 + double quant)")
    elif quant_cfg.use_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        logger.info("Using 8-bit quantization")

    model_kwargs: Dict[str, Any] = {
        "pretrained_model_name_or_path": model_cfg.model_name_or_path,
        "trust_remote_code": model_cfg.trust_remote_code,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
    if model_cfg.cache_dir:
        model_kwargs["cache_dir"] = model_cfg.cache_dir
    if model_cfg.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    if quant_cfg.use_4bit or quant_cfg.use_8bit:
        model = prepare_model_for_kbit_training(model)

    return model


def attach_lora(model, lora_cfg: LoraAdapterConfig):
    """Attach LoRA adapters to the model."""
    config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        target_modules=lora_cfg.target_modules,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=lora_cfg.modules_to_save,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────

def load_tokenizer(model_cfg: ModelConfig):
    name = model_cfg.tokenizer_name or model_cfg.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        trust_remote_code=model_cfg.trust_remote_code,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ─────────────────────────────────────────────
# Dataset Preparation
# ─────────────────────────────────────────────

def format_instruction(sample: dict, data_cfg: DataConfig) -> str:
    """Format a sample into instruction-following format."""
    instruction = sample.get(data_cfg.prompt_column, "")
    context = sample.get("input", "")
    response = sample.get(data_cfg.response_column, "")

    if context:
        return (
            f"{data_cfg.instruction_template}{instruction}\n\n"
            f"### Context:\n{context}\n\n"
            f"{data_cfg.response_template}{response}"
        )
    return (
        f"{data_cfg.instruction_template}{instruction}\n\n"
        f"{data_cfg.response_template}{response}"
    )


def load_and_prepare_data(data_cfg: DataConfig, tokenizer):
    """Load and tokenize dataset."""
    if data_cfg.dataset_name:
        raw = load_dataset(
            data_cfg.dataset_name,
            data_cfg.dataset_config,
            num_proc=data_cfg.num_proc,
        )
        if "validation" not in raw:
            split = raw["train"].train_test_split(test_size=data_cfg.val_split, seed=42)
            train_ds, val_ds = split["train"], split["test"]
        else:
            train_ds, val_ds = raw["train"], raw["validation"]
   
    elif data_cfg.train_file:
        train_ds = load_dataset("json", data_files=data_cfg.train_file)["train"]
        val_ds = (
            load_dataset("json", data_files=data_cfg.val_file)["train"]
            if data_cfg.val_file
            else train_ds.train_test_split(test_size=data_cfg.val_split)["test"]
        )
    else:
        raise ValueError("Provide dataset_name or train_file.")

    logger.info(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    return train_ds, val_ds


# ─────────────────────────────────────────────
# Main Training Entry Point
# ─────────────────────────────────────────────

def train(config_path: str):
    with open(config_path) as f:
        cfg = json.load(f)

    model_cfg = ModelConfig(**cfg.get("model", {}))
    quant_cfg = QuantizationConfig(**cfg.get("quantization", {}))
    lora_cfg = LoraAdapterConfig(**cfg.get("lora", {}))
    data_cfg = DataConfig(**cfg.get("data", {}))
    train_args_dict = cfg.get("training", {})

    # Load components
    tokenizer = load_tokenizer(model_cfg)
    model = load_base_model(model_cfg, quant_cfg)
    model = attach_lora(model, lora_cfg)

    # Load data
    train_ds, val_ds = load_and_prepare_data(data_cfg, tokenizer)

    # Format dataset
    def formatting_func(sample):
        return format_instruction(sample, data_cfg)

    # Training arguments
    training_args = TrainingArguments(
        **train_args_dict,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Completion-only collator (trains only on responses, not prompts)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=data_cfg.response_template,
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        formatting_func=formatting_func,
        data_collator=collator,
        max_seq_length=data_cfg.max_seq_length,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    logger.info(">>>> Starting fine-tuning...")
    trainer.train()

    # Save adapter
    output_dir = training_args.output_dir
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f">>>> Adapter saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()
    train(args.config)
