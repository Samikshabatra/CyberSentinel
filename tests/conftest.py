"""Shared test fixtures.

Tests run entirely on fallbacks: the mock LLM backend, the local vector store
and a temporary SQLite database. No GPU, no network, no Docker.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

# Pin the test environment BEFORE importing anything that reads settings.
# Environment variables take precedence over the .env file, so a developer whose
# .env points at a real model or a live Qdrant does not change what the suite
# exercises. Tests must be hermetic and identical on every machine.
os.environ["LLM_BACKEND"] = "mock"
os.environ["EMBEDDING_BACKEND"] = "hash"
os.environ["APP_ENV"] = "test"

import pytest  # noqa: E402

from cybersentinel.database.connection import build_engine  # noqa: E402
from cybersentinel.database.models import Base  # noqa: E402
from cybersentinel.graph.workflow import CyberSentinelWorkflow  # noqa: E402
from cybersentinel.llm.model import MockBackend  # noqa: E402
from cybersentinel.rag.chunking import Document, chunk_documents  # noqa: E402
from cybersentinel.rag.embeddings import HashEmbedding  # noqa: E402
from cybersentinel.rag.retriever import Retriever  # noqa: E402
from cybersentinel.rag.vectorstore import LocalVectorStore  # noqa: E402
from cybersentinel.utils.config import PROJECT_ROOT  # noqa: E402

KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "knowledge_base"


@pytest.fixture(scope="session")
def backend() -> MockBackend:
    """Deterministic LLM backend."""
    return MockBackend()


@pytest.fixture(scope="session")
def knowledge_documents() -> list[Document]:
    """Load the curated knowledge base from disk."""
    documents: list[Document] = []
    for path in sorted(KNOWLEDGE_BASE_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            content = record.pop("content")
            documents.append(Document(content=content, metadata=record))
    return documents


@pytest.fixture(scope="session")
def retriever(tmp_path_factory: pytest.TempPathFactory, knowledge_documents: list[Document]) -> Retriever:
    """A retriever backed by a freshly built temporary local store."""
    store_path: Path = tmp_path_factory.mktemp("vectors") / "store.json"
    store = LocalVectorStore(store_path)
    embedder = HashEmbedding(384)

    chunks = chunk_documents(knowledge_documents)
    vectors = embedder.embed_documents([chunk.content for chunk in chunks])
    store.ensure_collection(len(vectors[0]))
    store.upsert(chunks, vectors)

    return Retriever(store=store, embedder=embedder)


@pytest.fixture
def empty_retriever(tmp_path: Path) -> Retriever:
    """A retriever whose store contains nothing, for degradation tests."""
    return Retriever(store=LocalVectorStore(tmp_path / "empty.json"), embedder=HashEmbedding(384))


@pytest.fixture
def workflow(backend: MockBackend, retriever: Retriever) -> CyberSentinelWorkflow:
    """A workflow that pauses at the approval checkpoint."""
    return CyberSentinelWorkflow(backend=backend, retriever=retriever)


@pytest.fixture
def straight_through_workflow(backend: MockBackend, retriever: Retriever) -> CyberSentinelWorkflow:
    """A workflow that runs to completion without pausing (used for reporting tests)."""
    return CyberSentinelWorkflow(backend=backend, retriever=retriever, enable_interrupt=False)


@pytest.fixture
def session_factory(tmp_path: Path) -> Iterator[object]:
    """A session factory bound to a temporary SQLite database."""
    from sqlalchemy.orm import sessionmaker

    engine = build_engine(url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, expire_on_commit=False, future=True)
    engine.dispose()


@pytest.fixture
def service(workflow: CyberSentinelWorkflow, session_factory: object):
    """An analysis service wired to the temporary database."""
    from cybersentinel.service import AnalysisService

    return AnalysisService(
        workflow=workflow, session_factory=session_factory, auto_init_database=False
    )


# --- Sample events reused across tests ---------------------------------------
BRUTE_FORCE_EVENT = (
    "47 failed SSH login attempts from 198.51.100.23 within 3 minutes, all targeting the "
    "account root."
)

PHISHING_EMAIL = (
    "From: security-alert@example.com\n"
    "Subject: Urgent - verify your account within 24 hours\n\n"
    "Your mailbox will be suspended unless you confirm your credentials at "
    "http://secure-example.verify-now.example.net/login"
)

MULTI_EVENT = (
    "Event 1: Port scan from 203.0.113.45 against 1200 sequential ports\n"
    "Event 2: 20 failed SSH logins for user admin from 203.0.113.45\n"
    "Event 3: Successful SSH login for user admin from 203.0.113.45\n"
    "Event 4: User admin added to the administrators group shortly after login"
)

BENIGN_EVENT = (
    "User alice authenticated successfully to the VPN at 08:52 from the usual office address, "
    "matching their normal weekday pattern. Routine activity."
)

VAGUE_EVENT = "Something looks wrong with the server, please investigate."
