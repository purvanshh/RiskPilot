import logging

logger = logging.getLogger(__name__)


class MockEmbeddings:
    """Fallback embeddings when the real sentence transformer is unavailable."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        logger.warning(
            f"FALLING BACK TO MockEmbeddings — all embeddings will be zero vectors (dimension={dimension}). "
            "Install sentence-transformers for semantic retrieval."
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dimension for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * self.dimension


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


_EMBEDDINGS_CACHE = {}


def get_embeddings(model_name: str = "BAAI/bge-small-en-v1.5") -> object:
    """Returns an embeddings object for local BGE usage with sensible fallbacks."""
    if model_name in _EMBEDDINGS_CACHE:
        return _EMBEDDINGS_CACHE[model_name]

    try:
        logger.info("Initializing sentence-transformers for %s", model_name)
        embedder = SentenceTransformerEmbeddings(model_name=model_name)
        _EMBEDDINGS_CACHE[model_name] = embedder
        return embedder
    except Exception as first_error:
        logger.warning(
            "sentence-transformers initialization failed: %s. Using MockEmbeddings.",
            first_error,
        )
        return MockEmbeddings()


# Backward-compatible alias (policy_loader.py uses the pre-rename name)
get_embedding_provider = get_embeddings
