"""Ingest the cybersecurity knowledge base into the vector store.

    load -> clean -> chunk -> deduplicate -> embed -> upsert

Usage:
    python scripts/ingest_knowledge_base.py
    python scripts/ingest_knowledge_base.py --local          # skip Qdrant
    python scripts/ingest_knowledge_base.py --fetch-attack   # add live ATT&CK STIX
    python scripts/ingest_knowledge_base.py --reset
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running the script directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cybersentinel.rag.chunking import chunk_documents, deduplicate_chunks  # noqa: E402
from cybersentinel.rag.embeddings import build_embedding_backend  # noqa: E402
from cybersentinel.rag.loaders import (  # noqa: E402
    KnowledgeBaseError,
    fetch_attack_stix,
    load_cve_records,
    load_directory,
    summarise_corpus,
)
from cybersentinel.rag.vectorstore import build_vector_store  # noqa: E402
from cybersentinel.utils.config import get_settings  # noqa: E402
from cybersentinel.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("ingest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the CyberSentinel knowledge base.")
    parser.add_argument("--local", action="store_true", help="Use the local store, skip Qdrant.")
    parser.add_argument("--reset", action="store_true", help="Delete existing vectors first.")
    parser.add_argument(
        "--fetch-attack",
        action="store_true",
        help="Also download the official MITRE ATT&CK STIX bundle (requires network).",
    )
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=120)
    parser.add_argument(
        "--kb-dir",
        type=Path,
        default=None,
        help="Knowledge-base directory (default: data/knowledge_base).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()
    settings = get_settings()

    kb_dir = args.kb_dir or settings.knowledge_base_dir
    started = time.perf_counter()

    try:
        documents = load_directory(kb_dir)
    except KnowledgeBaseError as exc:
        logger.error(str(exc))
        return 1

    cve_path = kb_dir / "cve.jsonl"
    if cve_path.exists():
        documents.extend(load_cve_records(cve_path))

    if args.fetch_attack:
        try:
            documents.extend(fetch_attack_stix())
        except Exception as exc:
            logger.warning(
                f"could not download ATT&CK STIX ({type(exc).__name__}: {exc}); "
                "continuing with the offline corpus"
            )

    logger.info(f"corpus: {json.dumps(summarise_corpus(documents))}")

    chunks = deduplicate_chunks(chunk_documents(documents, args.chunk_size, args.overlap))
    logger.info(f"produced {len(chunks)} unique chunks from {len(documents)} documents")

    embedder = build_embedding_backend(settings)
    store = build_vector_store(settings, prefer_local=args.local)
    logger.info(f"store={store.name} embedder={embedder.info}")

    if args.reset:
        store.reset()
        logger.info("existing vectors removed")

    vectors = embedder.embed_documents([chunk.content for chunk in chunks])
    store.ensure_collection(len(vectors[0]) if vectors else settings.embedding_dim)
    written = store.upsert(chunks, vectors)

    elapsed = round(time.perf_counter() - started, 2)
    print(
        json.dumps(
            {
                "documents": len(documents),
                "chunks": len(chunks),
                "vectors_written": written,
                "store": store.name,
                "collection": settings.qdrant_collection,
                "embedding": embedder.info,
                "total_points": store.count(),
                "seconds": elapsed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
