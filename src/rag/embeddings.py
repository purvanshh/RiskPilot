import logging
from typing import Iterable, List

logger = logging.getLogger(__name__)


class MockEmbeddings:
    """Fallback embeddings when the real sentence transformer is unavailable."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 768 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 768


class SentenceTransformerEmbeddings:
    """Wrapper around sentence-transformers for direct local embedding generation."""

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError(f"sentence-transformers is required for {model_name}: {e}") from e

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False)
        return [list(map(float, emb)) for emb in embeddings]

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode([text], batch_size=1, show_progress_bar=False)[0]
        return list(map(float, embedding))


def get_embedding_provider(model_name: str = "BAAI/bge-small-en-v1.5") -> object:
    """Returns an embedding provider object exposing embed_documents/embed_query.

    Falls back to MockEmbeddings if sentence-transformers cannot be loaded.
    This is the single canonical entry point for embeddings across the RAG subsystem.
    """
    try:
        logger.info("Initializing sentence-transformers for %s", model_name)
        return SentenceTransformerEmbeddings(model_name=model_name)
    except Exception as first_error:
        logger.warning(
            "sentence-transformers initialization failed: %s. Using MockEmbeddings.",
            first_error,
        )
        return MockEmbeddings()
