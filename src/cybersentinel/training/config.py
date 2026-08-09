"""Training configuration.

Every knob the blueprint requires is exposed here and loadable from YAML, so a
run is reproducible from a config file rather than from remembered flags. No
machine-specific path is hardcoded: paths are resolved against the project root.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from cybersentinel.utils.config import get_settings


@dataclass
class LoRAConfig:
    """LoRA adapter hyperparameters."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    #: Attention and MLP projections. Adapting both is what lets a small rank
    #: change behaviour rather than only attention routing.
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class QuantizationConfig:
    """4-bit (QLoRA) quantisation settings."""

    load_in_4bit: bool = True
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "bfloat16"


@dataclass
class TrainingConfig:
    """Full fine-tuning configuration."""

    # --- model ---
    base_model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    output_dir: str = "models/cybersentinel-lora"
    max_seq_length: int = 1024

    # --- data ---
    train_file: str = "data/train/train_chat.jsonl"
    validation_file: str = "data/validation/validation_chat.jsonl"

    # --- optimisation ---
    learning_rate: float = 2e-4
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 0.3
    optim: str = "paged_adamw_8bit"
    seed: int = 20260808

    # --- memory ---
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False

    # --- logging / checkpointing ---
    logging_steps: int = 10
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    #: Only used when the corresponding strategy is "steps".
    eval_steps: int = 25
    save_steps: int = 25
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    report_to: str = "none"

    lora: LoRAConfig = field(default_factory=LoRAConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)

    def resolved_output_dir(self) -> Path:
        return get_settings().resolve(self.output_dir)

    def resolved_train_file(self) -> Path:
        return get_settings().resolve(self.train_file)

    def resolved_validation_file(self) -> Path:
        return get_settings().resolve(self.validation_file)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        payload = dict(data)
        lora = LoRAConfig(**payload.pop("lora", {}) or {})
        quantization = QuantizationConfig(**payload.pop("quantization", {}) or {})
        known = {key: value for key, value in payload.items() if key in cls.__dataclass_fields__}
        return cls(**known, lora=lora, quantization=quantization)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        resolved = get_settings().resolve(path)
        data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)

    def save_yaml(self, path: str | Path) -> Path:
        resolved = get_settings().resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return resolved


#: Lower-memory profile for GPUs that cannot hold the default configuration.
#: Shorter sequences and rank 8 are the two changes that actually move peak
#: memory; batch size is already 1 with accumulation carrying the effective size.
LOW_MEMORY_OVERRIDES: dict[str, Any] = {
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "max_seq_length": 768,
    "gradient_checkpointing": True,
    "lora": {"r": 8, "alpha": 16, "dropout": 0.05},
}


def low_memory_config(base: TrainingConfig | None = None) -> TrainingConfig:
    """Return the low-memory variant of a configuration."""
    config = base or TrainingConfig()
    data = config.to_dict()
    for key, value in LOW_MEMORY_OVERRIDES.items():
        if key == "lora":
            data["lora"].update(value)
        else:
            data[key] = value
    return TrainingConfig.from_dict(data)
