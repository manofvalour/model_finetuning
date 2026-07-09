from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# ─────────────────────────────────────────────
# Config Dataclasses
# ─────────────────────────────────────────────

@dataclass
class ModelConfig:
    model_name_or_path: str = "mistralai/Mistral-7B-v0.1"
    tokenizer_name: Optional[str] = None
    use_flash_attention: bool = False
    trust_remote_code: bool = False
    cache_dir: Optional[str] = None


@dataclass
class QuantizationConfig:
    use_4bit: bool = True
    use_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoraAdapterConfig:
    r: int = 64
    lora_alpha: int = 16
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]
    )
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    modules_to_save: Optional[List[str]] = None


@dataclass
class DataConfig:
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    text_column: str = "text"
    prompt_column: Optional[str] = "instruction"
    response_column: Optional[str] = "output"
    max_seq_length: int = 2048
    instruction_template: str = "### Instruction:\n"
    response_template: str = "### Response:\n"
    val_split: float = 0.05
    num_proc: int = 4
