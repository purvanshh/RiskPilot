import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def get_vector_store(
    persist_directory: str = "./data/chroma_db",
    collection_name: str = "lending_policy",
) -> Any:
    """Initializes or loads a persistent ChromaDB collection."""
    os.makedirs(persist_directory, exist_ok=True)

    try:
        import chromadb
        from chromadb.config import Settings

        logger.info(
            "Initializing local ChromaDB collection '%s' at %s",
            collection_name,
            persist_directory,
        )
        client = chromadb.Client(
            settings=Settings(is_persistent=True, persist_directory=persist_directory)
        )
        return client.get_or_create_collection(name=collection_name)
    except Exception as exc:
        logger.warning(
            "ChromaDB initialization failed: %s. Falling back to mock vector store.",
            exc,
        )

        class MockCollection:
            def __init__(self):
                self._ids: List[str] = []
                self._documents: List[str] = []
                self._metadatas: List[Dict[str, Any]] = []

            def add(self, ids, documents=None, metadatas=None, embeddings=None):
                documents = documents or []
                metadatas = metadatas or [{}] * len(documents)
                self._ids.extend(ids)
                self._documents.extend(documents)
                self._metadatas.extend(metadatas)

            def query(
                self,
                query_embeddings=None,
                query_texts=None,
                n_results=10,
                include=None,
            ):
                if query_texts is None:
                    query_texts = [""]
                query_text = query_texts[0].lower()
                scores = []
                for idx, document in enumerate(self._documents):
                    score = sum(1 for token in set(query_text.split()) if token in document.lower())
                    if score > 0:
                        scores.append((score, idx))
                scores.sort(key=lambda item: item[0], reverse=True)
                selected = [idx for _, idx in scores[:n_results]]
                return {
                    "ids": [[self._ids[idx] for idx in selected]],
                    "documents": [[self._documents[idx] for idx in selected]],
                    "metadatas": [[self._metadatas[idx] for idx in selected]],
                    "distances": [[0.0 for _ in selected]],
                }

            def count(self):
                return len(self._documents)

            def persist(self):
                logger.info("MockCollection persist called.")

        return MockCollection()
