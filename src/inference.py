"""
Inference & Adapter Merging Utilities
"""

import torch
import logging
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel

logger = logging.getLogger(__name__)


class LoraInference:
    """Load a base model + LoRA adapter and run inference."""

    def __init__(
        self,
        base_model_name: str,
        adapter_path: str,
        device: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)

        kwargs = {"device_map": device, "torch_dtype": torch.bfloat16}

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        base = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()
        logger.info(f"✓ Loaded {base_model_name} + adapter from {adapter_path}")

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        stream: bool = False,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True) if stream else None

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            streamer=streamer,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if stream:
            return ""
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded[len(prompt):]

    def batch_generate(
        self, prompts: List[str], max_new_tokens: int = 256, **kwargs
    ) -> List[str]:
        return [self.generate(p, max_new_tokens=max_new_tokens, **kwargs) for p in prompts]


def merge_and_save(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
    push_to_hub: bool = False,
    hub_repo: Optional[str] = None,
):
    """
    Merge LoRA weights into base model and save as a standalone model.
    The merged model can be used without PEFT.
    """
    logger.info("Loading base model for merging...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_path)

    logger.info("Merging LoRA weights into base model...")
    merged = model.merge_and_unload()

    logger.info(f"Saving merged model to {output_path}...")
    merged.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    if push_to_hub and hub_repo:
        logger.info(f"Pushing to HuggingFace Hub: {hub_repo}")
        merged.push_to_hub(hub_repo)
        tokenizer.push_to_hub(hub_repo)

    logger.info("✅ Merge complete.")
    return merged, tokenizer


def compute_trainable_params(model) -> dict:
    """Report trainable vs total parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": round(100 * trainable / total, 4),
        "total_M": round(total / 1e6, 2),
        "trainable_M": round(trainable / 1e6, 2),
    }
