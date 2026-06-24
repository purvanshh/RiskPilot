import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.rag.embeddings import get_embedding_provider
from src.rag.policy_loader import load_and_index_policies
from src.rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedPolicyChunk:
    text: str
    metadata: Dict[str, Any]
    score: float

    def citation(self) -> str:
        document = self.metadata.get("document") or self.metadata.get("source_file") or "unknown"
        section = self.metadata.get("section", "unknown")
        chunk_index = self.metadata.get("chunk_index", "0")
        return f"{document} | {section} | chunk {chunk_index}"


def _distance_to_similarity(distance: Optional[float]) -> float:
    if distance is None:
        return 1.0

    try:
        distance_value = float(distance)
    except (TypeError, ValueError):
        return 1.0

    if 0.0 <= distance_value <= 1.0:
        return 1.0 - distance_value
    return 1.0 / (1.0 + distance_value)


class PolicyRetriever:
    def __init__(
        self,
        policy_docs_dir: str = "./data/policy_docs",
        persist_dir: str = "./data/chroma_db",
        collection_name: str = "lending_policy",
        top_k: int = 3,
        similarity_threshold: float = 0.5,
    ):
        self.policy_docs_dir = policy_docs_dir
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.embedding_provider = get_embedding_provider()
        self._db = None

    def _get_db(self):
        if self._db is None:
            self._db = get_vector_store(
                persist_directory=self.persist_dir,
                collection_name=self.collection_name,
            )
        return self._db

    def _ensure_index(self) -> None:
        db = self._get_db()
        try:
            if hasattr(db, "count") and db.count() == 0:
                load_and_index_policies(
                    policy_docs_dir=self.policy_docs_dir,
                    persist_dir=self.persist_dir,
                    collection_name=self.collection_name,
                )
        except Exception:
            load_and_index_policies(
                policy_docs_dir=self.policy_docs_dir,
                persist_dir=self.persist_dir,
                collection_name=self.collection_name,
            )

    def retrieve(self, query: str) -> List[RetrievedPolicyChunk]:
        self._ensure_index()
        db = self._get_db()

        query_embedding = self.embedding_provider.embed_query(query)
        try:
            raw_results = db.query(
                query_embeddings=[query_embedding],
                n_results=self.top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.warning("Chroma query failed: %s", exc)
            return []

        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        results: List[RetrievedPolicyChunk] = []
        for idx, text in enumerate(documents):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            distance = distances[idx] if idx < len(distances) else None
            score = _distance_to_similarity(distance)
            if score < self.similarity_threshold:
                continue
            results.append(RetrievedPolicyChunk(text=text, metadata=metadata or {}, score=score))

        return results


def policy_retriever(
    query: str,
    top_k: int = 3,
    similarity_threshold: float = 0.5,
) -> List[RetrievedPolicyChunk]:
    retriever = PolicyRetriever(top_k=top_k, similarity_threshold=similarity_threshold)
    return retriever.retrieve(query)
