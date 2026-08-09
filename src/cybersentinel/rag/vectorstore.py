"""Vector store abstraction.

Primary store is Qdrant. A pure-Python local store is included as a fallback so
the RAG pipeline keeps working when Qdrant is unavailable - the blueprint
requires the system to degrade gracefully rather than fail the whole workflow.
Which store served a query is always reported, never hidden.
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from cybersentinel.rag.chunking import Chunk
from cybersentinel.rag.embeddings import cosine_similarity
from cybersentinel.schemas.analysis import RetrievedDocument
from cybersentinel.utils.config import Settings, get_settings
from cybersentinel.utils.logging import get_logger

logger = get_logger(__name__)

_NAMESPACE = uuid.UUID("6d3f0e2a-1b7c-4a5e-9f31-0c8a2b6d4e70")


def _point_uuid(point_id: str) -> str:
    """Deterministic UUID for a chunk id, so re-ingestion updates in place."""
    return str(uuid.uuid5(_NAMESPACE, point_id))


def _payload(chunk: Chunk) -> dict[str, Any]:
    payload = dict(chunk.metadata)
    payload["content"] = chunk.content
    payload["chunk_index"] = chunk.chunk_index
    return payload


def _to_document(payload: dict[str, Any], score: float) -> RetrievedDocument:
    return RetrievedDocument(
        content=payload.get("content", ""),
        score=round(float(score), 4),
        source=payload.get("source", "unknown"),
        document_id=payload.get("document_id"),
        title=payload.get("title"),
        url=payload.get("url"),
        category=payload.get("category"),
    )


class VectorStore(ABC):
    """Minimal vector-store interface."""

    name: str = "base"

    @abstractmethod
    def ensure_collection(self, dimension: int) -> None: ...

    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int: ...

    @abstractmethod
    def search(
        self, vector: list[float], top_k: int = 5, score_threshold: float = 0.0
    ) -> list[RetrievedDocument]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def reset(self) -> None: ...

    def health(self) -> dict[str, Any]:
        try:
            return {"store": self.name, "available": True, "points": self.count()}
        except Exception as exc:
            return {"store": self.name, "available": False, "error": f"{type(exc).__name__}: {exc}"}


class LocalVectorStore(VectorStore):
    """Brute-force cosine search over a JSON-persisted list of vectors.

    Adequate for a knowledge base of a few thousand chunks and requires no
    services at all.
    """

    name = "local"

    def __init__(self, path: Path) -> None:
        self.path = path
        self._points: list[dict[str, Any]] = []
        self._dimension: int | None = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._points = data.get("points", [])
            self._dimension = data.get("dimension")
            logger.debug(f"loaded {len(self._points)} points from {self.path}")
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"could not load local vector store: {exc}")
            self._points = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"dimension": self._dimension, "points": self._points}),
            encoding="utf-8",
        )

    def ensure_collection(self, dimension: int) -> None:
        if self._dimension is not None and self._dimension != dimension:
            logger.warning(
                f"embedding dimension changed ({self._dimension} -> {dimension}); clearing store"
            )
            self._points = []
        self._dimension = dimension

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")

        index = {point["id"]: position for position, point in enumerate(self._points)}
        for chunk, vector in zip(chunks, vectors, strict=True):
            point = {"id": _point_uuid(chunk.point_id), "vector": vector, "payload": _payload(chunk)}
            if point["id"] in index:
                self._points[index[point["id"]]] = point
            else:
                index[point["id"]] = len(self._points)
                self._points.append(point)

        self._save()
        return len(chunks)

    def search(
        self, vector: list[float], top_k: int = 5, score_threshold: float = 0.0
    ) -> list[RetrievedDocument]:
        scored = [
            (cosine_similarity(vector, point["vector"]), point["payload"]) for point in self._points
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            _to_document(payload, score)
            for score, payload in scored[:top_k]
            if score >= score_threshold
        ]

    def count(self) -> int:
        return len(self._points)

    def reset(self) -> None:
        self._points = []
        self._save()


class QdrantVectorStore(VectorStore):
    """Qdrant-backed store (server or embedded in-memory)."""

    name = "qdrant"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.collection = self.settings.qdrant_collection
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            from qdrant_client import QdrantClient

            if self.settings.qdrant_in_memory:
                logger.info("using in-memory Qdrant instance")
                self._client = QdrantClient(location=":memory:")
            else:
                self._client = QdrantClient(
                    url=self.settings.qdrant_url,
                    api_key=self.settings.qdrant_api_key,
                    timeout=10,
                )
        return self._client

    def ensure_collection(self, dimension: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            existing = info.config.params.vectors.size
            if existing == dimension:
                return
            logger.warning(
                f"collection '{self.collection}' has dimension {existing}, expected {dimension}; "
                "recreating"
            )
            self.client.delete_collection(self.collection)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
        logger.info(f"created Qdrant collection '{self.collection}' (dim={dimension})")

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        from qdrant_client.models import PointStruct

        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")

        points = [
            PointStruct(id=_point_uuid(chunk.point_id), vector=vector, payload=_payload(chunk))
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        # Batched so a large knowledge base does not build one huge request.
        batch_size = 128
        for start in range(0, len(points), batch_size):
            self.client.upsert(collection_name=self.collection, points=points[start : start + batch_size])
        return len(points)

    def search(
        self, vector: list[float], top_k: int = 5, score_threshold: float = 0.0
    ) -> list[RetrievedDocument]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            score_threshold=score_threshold or None,
            with_payload=True,
        )
        return [_to_document(point.payload or {}, point.score) for point in response.points]

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection, exact=True).count

    def reset(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)


def local_store_path(settings: Settings | None = None) -> Path:
    resolved = settings or get_settings()
    return resolved.data_dir / "processed" / f"{resolved.qdrant_collection}_local.json"


def build_vector_store(
    settings: Settings | None = None,
    prefer_local: bool = False,
) -> VectorStore:
    """Return a working vector store, falling back to the local one on failure."""
    resolved = settings or get_settings()

    if prefer_local:
        return LocalVectorStore(local_store_path(resolved))

    try:
        store = QdrantVectorStore(resolved)
        store.client.get_collections()  # forces a connection attempt
        return store
    except Exception as exc:
        logger.warning(
            f"Qdrant unavailable ({type(exc).__name__}: {exc}); using local vector store fallback"
        )
        return LocalVectorStore(local_store_path(resolved))
