"""Generate the cybersecurity instruction dataset.

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --instances 20 --seed 7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cybersentinel.training.dataset import build_dataset, write_splits  # noqa: E402
from cybersentinel.utils.config import get_settings  # noqa: E402
from cybersentinel.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("prepare_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the CyberSentinel training dataset.")
    parser.add_argument(
        "--per-category",
        type=int,
        default=120,
        help="Target examples per attack category across all splits.",
    )
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--output", type=Path, default=None, help="Output directory (default: data/).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()
    settings = get_settings()

    output_dir = args.output or settings.data_dir
    splits = build_dataset(examples_per_category=args.per_category, seed=args.seed)
    paths = write_splits(splits, output_dir)

    summary = {
        "counts": splits.counts(),
        "templates_per_split": splits.card["templates_per_split"],
        "duplicates_removed": splits.card["duplicates_removed"],
        "class_distribution": splits.card["class_distribution"],
        "files": {name: str(path) for name, path in paths.items()},
    }
    print(json.dumps(summary, indent=2))

    # A category missing from the test set would make its metrics unreportable.
    train_labels = set(splits.card["class_distribution"]["train"])
    test_labels = set(splits.card["class_distribution"]["test"])
    missing = sorted(train_labels - test_labels)
    if missing:
        logger.warning(f"categories absent from the test split: {missing}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
