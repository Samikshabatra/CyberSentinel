"""Run the CyberSentinel evaluation suite.

Usage:
    python scripts/evaluate.py                       # rules baseline + pipeline studies
    python scripts/evaluate.py --limit 40            # quick pass
    python scripts/evaluate.py --with-hf --adapter models/cybersentinel-lora
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cybersentinel.evaluation.runner import (  # noqa: E402
    build_context,
    load_test_records,
    run_all,
    summarise,
)
from cybersentinel.utils.config import get_settings  # noqa: E402
from cybersentinel.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CyberSentinel.")
    parser.add_argument("--limit", type=int, default=None, help="Cap on test examples.")
    parser.add_argument(
        "--with-hf",
        action="store_true",
        help="Include Hugging Face base/fine-tuned arms (requires the ml extra).",
    )
    parser.add_argument("--adapter", type=str, default=None, help="LoRA adapter path.")
    parser.add_argument(
        "--use-qdrant", action="store_true", help="Use Qdrant instead of the local store."
    )
    parser.add_argument("--output", type=Path, default=None, help="Results directory.")
    parser.add_argument(
        "--save-predictions", action="store_true", help="Include per-example predictions."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()
    settings = get_settings()

    try:
        records = load_test_records(settings)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1

    logger.info(f"loaded {len(records)} test examples")

    context = build_context(
        adapter_path=args.adapter or settings.model_adapter_path,
        prefer_local_store=not args.use_qdrant,
        include_hf_arms=args.with_hf,
    )

    if context.retriever.store.count() == 0:
        logger.warning(
            "the vector store is empty; retrieval studies will score zero. "
            "Run: python scripts/ingest_knowledge_base.py --local"
        )

    output_dir = args.output or (settings.evaluation_dir / "results")
    results = run_all(
        context,
        records,
        limit=args.limit,
        output_dir=output_dir,
        include_predictions=args.save_predictions,
    )

    print(summarise(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
