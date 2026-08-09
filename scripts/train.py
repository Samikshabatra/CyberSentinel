"""QLoRA fine-tuning entry point.

Usage:
    python scripts/train.py --dry-run                 # validate without a GPU
    python scripts/train.py                           # default GPU profile
    python scripts/train.py --low-memory              # smaller-GPU profile
    python scripts/train.py --config configs/training.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cybersentinel.training.config import TrainingConfig, low_memory_config  # noqa: E402
from cybersentinel.training.trainer import (  # noqa: E402
    TrainingDependencyError,
    describe_hardware,
    train,
    validate_config,
)
from cybersentinel.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the CyberSentinel model with QLoRA.")
    parser.add_argument("--config", type=str, default="configs/training.yaml")
    parser.add_argument("--low-memory", action="store_true", help="Use the reduced-memory profile.")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration and exit.")
    parser.add_argument("--model", type=str, default=None, help="Override the base model name.")
    parser.add_argument("--output", type=str, default=None, help="Override the adapter output path.")
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the most recent checkpoint in the output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()

    config_path = Path(args.config)
    try:
        config = TrainingConfig.from_yaml(config_path)
        logger.info(f"loaded configuration from {config_path}")
    except FileNotFoundError:
        logger.warning(f"{config_path} not found; using built-in defaults")
        config = TrainingConfig()

    if args.low_memory:
        config = low_memory_config(config)
        logger.info("low-memory profile applied")

    if args.model:
        config.base_model_name = args.model
    if args.output:
        config.output_dir = args.output
    if args.epochs is not None:
        config.num_train_epochs = args.epochs
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.batch_size is not None:
        config.per_device_train_batch_size = args.batch_size
    if args.lora_r is not None:
        config.lora.r = args.lora_r

    problems = validate_config(config)
    hardware = describe_hardware()

    if args.dry_run:
        print(
            json.dumps(
                {
                    "config": config.to_dict(),
                    "hardware": hardware,
                    "problems": problems,
                    "ready": not problems,
                },
                indent=2,
            )
        )
        return 0 if not problems else 1

    if problems:
        for problem in problems:
            logger.error(problem)
        return 1

    if not hardware.get("cuda_available"):
        logger.warning(
            "no CUDA device detected: training will run on CPU and will be extremely slow. "
            "Use --dry-run to validate the setup, or run on a GPU machine."
        )

    try:
        metrics = train(config, resume=args.resume)
    except TrainingDependencyError as exc:
        logger.error(str(exc))
        return 1

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
