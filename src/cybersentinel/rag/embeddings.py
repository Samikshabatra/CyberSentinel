"""Embedding backends.

Two implementations behind one interface:

* ``sentence-transformers`` - real dense embeddings, used for the reported
  results.
* ``hash`` - a deterministic hashed bag-of-ngrams projection. It needs no model
  download and no GPU, so the RAG pipeline (and its tests) run anywhere. It is
  weaker than a trained encoder; the evaluation reports both so the difference
  is visible rather than hidden.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

from cybersentinel.utils.config import Settings, get_settings
from cybersentinel.utils.logging import get_logger

logger = get_logger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*", re.IGNORECASE)

# Identifiers must survive tokenisation intact so that a query for "T1110"
# retrieves the T1110 chunk.
_ID_PATTERN = re.compile(r"\b(?:T\d{4}(?:\.\d{3})?|CVE-\d{4}-\d{4,7}|CWE-\d{1,5})\b", re.IGNORECASE)

_STOPWORDS = frozenset(
    """a an the and or of to in for on with is are was were be been being this that these those
    it its as at by from can may will would should could not no if then than there their they
    we you your our""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, with threat-intelligence identifiers preserved."""
    identifiers = [match.upper() for match in _ID_PATTERN.findall(text)]
    words = [
        token.lower()
        for token in _TOKEN_PATTERN.findall(text)
        if len(token) > 1 and token.lower() not in _STOPWORDS
    ]
    return identifiers + words


class EmbeddingBackend(ABC):
    """Turns text into fixed-length vectors."""

    name: str = "base"
    dimension: int = 0

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query. Defaults to the document path."""
        return self.embed_documents([text])[0]

    @property
    def info(self) -> dict[str, Any]:
        return {"embedding_backend": self.name, "dimension": self.dimension}


class HashEmbedding(EmbeddingBackend):
    """Hashed TF-IDF-style projection with sublinear term weighting.

    Each token (plus each adjacent bigram) is hashed into a bucket; the value is
    a signed, sublinearly scaled term frequency. Vectors are L2-normalised so
    cosine similarity is a dot product.
    """

    name = "hash"

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % self.dimension
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return index, sign

    def _embed_one(self, text: str) -> list[float]:
        tokens = tokenize(text)
        if not tokens:
            return [0.0] * self.dimension

        bigrams = [f"{first}_{second}" for first, second in zip(tokens, tokens[1:], strict=False)]
        counts: dict[str, float] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0.0) + 1.0
        for bigram in bigrams:
            counts[bigram] = counts.get(bigram, 0.0) + 0.5

        vector = [0.0] * self.dimension
        for token, count in counts.items():
            index, sign = self._bucket(token)
            weight = 1.0 + math.log(count)
            # Identifiers are the highest-signal tokens in this domain.
            if _ID_PATTERN.fullmatch(token):
                weight *= 3.0
            vector[index] += sign * weight

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]


class SentenceTransformerEmbedding(EmbeddingBackend):
    """Dense embeddings from a sentence-transformers model (lazy loaded)."""

    name = "sentence-transformers"

    def __init__(self, model_name: str, expected_dimension: int | None = None) -> None:
        self.model_name = model_name
        self._model: Any = None
        self.dimension = expected_dimension or 0

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "sentence-transformers is not installed. Install with: pip install -e '.[ml]' "
                "or set EMBEDDING_BACKEND=hash"
            ) from exc

        logger.info(f"loading embedding model {self.model_name}")
        self._model = SentenceTransformer(self.model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._load()
        vectors = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )
        return [vector.tolist() for vector in vectors]

    @property
    def info(self) -> dict[str, Any]:
        return {
            "embedding_backend": self.name,
            "model": self.model_name,
            "dimension": self.dimension,
        }


def build_embedding_backend(settings: Settings | None = None) -> EmbeddingBackend:
    """Construct the configured embedding backend, falling back to hashing."""
    resolved = settings or get_settings()

    if resolved.embedding_backend == "sentence-transformers":
        try:
            backend = SentenceTransformerEmbedding(
                resolved.embedding_model_name, resolved.embedding_dim
            )
            backend._load()
            return backend
        except RuntimeError as exc:
            logger.warning(f"falling back to hash embeddings: {exc}")

    return HashEmbedding(resolved.embedding_dim)


@lru_cache(maxsize=2)
def get_embedding_backend() -> EmbeddingBackend:
    """Return a cached embedding backend."""
    return build_embedding_backend()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
