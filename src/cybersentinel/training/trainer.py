"""QLoRA fine-tuning.

    base model -> 4-bit quantisation -> freeze base -> attach LoRA adapters
                -> train adapters -> save adapter -> evaluate

Only the adapter is saved. The base model stays untouched on disk, so an
experiment costs tens of megabytes rather than a full model copy, and the
application can run with or without the adapter by changing one setting.

All heavy imports are deferred into the functions that need them, so this module
can be imported (and its configuration validated) on a machine with no GPU and
no ML stack installed.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from cybersentinel.training.config import TrainingConfig
from cybersentinel.utils.logging import get_logger

logger = get_logger(__name__)


class TrainingDependencyError(RuntimeError):
    """Raised when the ML extra is not installed."""


@contextmanager
def keep_system_awake() -> Iterator[None]:
    """Stop the machine sleeping for the duration of a training run.

    A laptop that suspends mid-run does not resume training: the CUDA context is
    lost, the process survives holding VRAM, and the job stalls indefinitely
    without an error. On Windows this asks the OS to stay awake (the display may
    still switch off); elsewhere it is a no-op.
    """
    if sys.platform != "win32":
        yield
        return

    import ctypes

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        logger.info("system sleep suspended for the duration of training")
    except Exception as exc:  # pragma: no cover - platform specific
        logger.warning(f"could not suspend system sleep: {type(exc).__name__}: {exc}")

    try:
        yield
    finally:
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            logger.info("normal sleep behaviour restored")
        except Exception:  # pragma: no cover - platform specific
            pass


def require_ml_stack() -> None:
    """Fail early, with a useful message, when training deps are missing."""
    missing: list[str] = []
    for module in ("torch", "transformers", "peft", "trl", "datasets"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise TrainingDependencyError(
            f"missing training dependencies: {', '.join(missing)}. "
            "Install with: pip install -e '.[ml]'"
        )


def describe_hardware() -> dict[str, Any]:
    """Report the training hardware so a run can be reproduced and compared."""
    try:
        import torch
    except ImportError:
        return {"torch": None, "cuda_available": False}

    info: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["total_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
        )
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
    return info


def validate_config(config: TrainingConfig) -> list[str]:
    """Check a configuration without loading anything. Returns problems found."""
    problems: list[str] = []

    train_file = config.resolved_train_file()
    if not train_file.exists():
        problems.append(f"training file not found: {train_file}")
    validation_file = config.resolved_validation_file()
    if not validation_file.exists():
        problems.append(f"validation file not found: {validation_file}")

    if config.learning_rate <= 0:
        problems.append("learning_rate must be positive")
    if config.num_train_epochs <= 0:
        problems.append("num_train_epochs must be positive")
    if config.lora.r <= 0:
        problems.append("lora.r must be positive")
    if config.bf16 and config.fp16:
        problems.append("bf16 and fp16 cannot both be enabled")

    return problems


def load_chat_dataset(config: TrainingConfig) -> Any:
    """Load the chat-format JSONL splits into a DatasetDict."""
    from datasets import load_dataset

    files = {
        "train": str(config.resolved_train_file()),
        "validation": str(config.resolved_validation_file()),
    }
    dataset = load_dataset("json", data_files=files)
    logger.info(
        f"loaded {len(dataset['train'])} training and "
        f"{len(dataset['validation'])} validation examples"
    )
    return dataset


def build_model_and_tokenizer(config: TrainingConfig) -> tuple[Any, Any]:
    """Load the base model in 4-bit and attach LoRA adapters."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from cybersentinel.utils.config import get_settings

    settings = get_settings()
    compute_dtype = getattr(torch, config.quantization.compute_dtype, torch.bfloat16)

    quantization_config = None
    if config.quantization.load_in_4bit and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.quantization.quant_type,
            bnb_4bit_use_double_quant=config.quantization.double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    elif config.quantization.load_in_4bit:
        logger.warning("CUDA unavailable: loading without 4-bit quantisation")

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name, token=settings.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Causal LM training pads on the right; generation pads on the left.
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        quantization_config=quantization_config,
        dtype=compute_dtype if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        token=settings.hf_token,
    )
    model.config.use_cache = False

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )

    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias=config.lora.bias,
        task_type="CAUSAL_LM",
        target_modules=config.lora.target_modules,
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    logger.info(
        f"trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)"
    )

    return model, tokenizer


def train(config: TrainingConfig, resume: bool = False) -> dict[str, Any]:
    """Run QLoRA fine-tuning and save the adapter.

    Set ``resume`` to continue from the most recent checkpoint in the output
    directory - useful after an interruption on a long run.
    """
    require_ml_stack()

    import torch
    from trl import SFTConfig, SFTTrainer

    problems = validate_config(config)
    if problems:
        raise ValueError("invalid training configuration: " + "; ".join(problems))

    output_dir = config.resolved_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    hardware = describe_hardware()
    logger.info(f"training hardware: {json.dumps(hardware)}")

    dataset = load_chat_dataset(config)
    model, tokenizer = build_model_and_tokenizer(config)

    use_bf16 = config.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = config.fp16 and torch.cuda.is_available() and not use_bf16

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        max_length=config.max_seq_length,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        optim=config.optim if torch.cuda.is_available() else "adamw_torch",
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=config.logging_steps,
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        eval_steps=config.eval_steps if config.eval_strategy == "steps" else None,
        save_steps=config.save_steps if config.save_strategy == "steps" else 500,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        report_to=config.report_to,
        seed=config.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    resume_from = str(checkpoints[-1]) if (resume and checkpoints) else None
    if resume_from:
        logger.info(f"resuming from {resume_from}")

    with keep_system_awake():
        result = trainer.train(resume_from_checkpoint=resume_from)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics: dict[str, Any] = {
        "train_runtime_seconds": round(result.metrics.get("train_runtime", 0.0), 2),
        "train_loss": round(result.metrics.get("train_loss", 0.0), 4),
        "epochs": config.num_train_epochs,
        "adapter_path": str(output_dir),
        "hardware": hardware,
        "config": config.to_dict(),
    }

    try:
        evaluation = trainer.evaluate()
        metrics["eval_loss"] = round(float(evaluation.get("eval_loss", 0.0)), 4)
    except Exception as exc:
        logger.warning(f"final evaluation failed: {type(exc).__name__}: {exc}")

    (output_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    logger.info(f"adapter saved to {output_dir}")
    return metrics


def adapter_exists(path: str | Path) -> bool:
    """True when a directory contains a saved PEFT adapter."""
    directory = Path(path)
    return (directory / "adapter_config.json").exists()
