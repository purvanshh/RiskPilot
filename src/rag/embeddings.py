import logging
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

class MockEmbeddings(Embeddings):
    """
    Fallback mock embeddings class in case sentence-transformers takes too long to load
    or is not installed, to keep the boilerplate runnable immediately.
    """
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 384

def get_embeddings():
    """
    Tries to initialize SentenceTransformers embeddings,
    falling back to MockEmbeddings if initialization fails.
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        logger.info("Initializing HuggingFaceEmbeddings with all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        logger.warning(f"Could not load HuggingFaceEmbeddings: {str(e)}. Using MockEmbeddings fallback.")
        return MockEmbeddings()
