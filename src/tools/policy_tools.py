from typing import Any, Dict, List, Optional

from src.rag.embeddings import get_embedding_provider
from src.rag.policy_loader import load_and_index_policies
from src.rag.vector_store import get_vector_store


def ltv_calculator(loan_amount: float, property_value: float) -> float:
    """Computes the Loan-to-Value (LTV) ratio."""
    if property_value <= 0:
        return 0.0
    return loan_amount / property_value


class RetrievedPolicyChunk:
    def __init__(self, text: str, metadata: Dict[str, Any], score: float):
        self.text = text
        self.metadata = metadata
        self.score = score

    def citation(self) -> str:
        document = self.metadata.get("document", self.metadata.get("source_file", "unknown"))
        section = self.metadata.get("section", "unknown")
        chunk_index = self.metadata.get("chunk_index", "0")
        return f"{document} | {section} | chunk {chunk_index}"


_db_cache: dict = {}


def _ensure_collection(
    policy_docs_dir: str = "./data/policy_docs",
    persist_dir: str = "./data/chroma_db",
    collection_name: str = "lending_policy",
) -> Any:
    cache_key = f"{persist_dir}:{collection_name}"
    if cache_key in _db_cache:
        return _db_cache[cache_key]
    db = get_vector_store(persist_directory=persist_dir, collection_name=collection_name)
    try:
        if hasattr(db, "count") and db.count() == 0:
            load_and_index_policies(policy_docs_dir, persist_dir, collection_name)
    except Exception:
        load_and_index_policies(policy_docs_dir, persist_dir, collection_name)
    _db_cache[cache_key] = db
    return db


def policy_retriever(
    query: str,
    top_k: int = 3,
    similarity_threshold: float = 0.5,
    policy_docs_dir: str = "./data/policy_docs",
    persist_dir: str = "./data/chroma_db",
    collection_name: str = "lending_policy",
) -> List[RetrievedPolicyChunk]:
    """Retrieves nearest policy chunks from ChromaDB for the given query."""
    db = _ensure_collection(policy_docs_dir, persist_dir, collection_name)
    embedder = get_embedding_provider()

    try:
        query_embedding = embedder.embed_query(query)
        raw_results = db.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        raise RuntimeError(f"Policy retriever failed: {e}") from e

    documents = raw_results.get("documents", [[]])[0]
    metadatas = raw_results.get("metadatas", [[]])[0]
    distances = raw_results.get("distances", [[]])[0]

    results: List[RetrievedPolicyChunk] = []
    for idx, text in enumerate(documents):
        metadata = metadatas[idx] if idx < len(metadatas) else {}
        raw_distance = distances[idx] if idx < len(distances) else None
        # Convert L2 distance to a [0,1] similarity score
        try:
            d = float(raw_distance) if raw_distance is not None else 0.0
            score = 1.0 - d if 0.0 <= d <= 1.0 else 1.0 / (1.0 + d)
        except (TypeError, ValueError):
            score = 1.0
        if score < similarity_threshold:
            continue
        results.append(RetrievedPolicyChunk(text=text, metadata=metadata or {}, score=score))

    return results


def policy_validator(
    credit_score: int,
    dti: float,
    ltv: float,
    employment_months: int,
    policy_chunks: List[RetrievedPolicyChunk],
    income_document_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Validates applicant metrics against deterministic lending policy rules."""
    passed = True
    violations: List[str] = []
    min_credit_met = True
    max_dti_threshold = 0.45

    if credit_score < 650:
        min_credit_met = False
        passed = False
        violations.append(
            "Credit score below minimum requirement of 650. "
            f"Citation: {policy_chunks[0].citation() if policy_chunks else 'unknown'}"
        )

    if dti > 0.50:
        passed = False
        violations.append(
            "DTI exceeds hard cap of 50%. "
            f"Citation: {policy_chunks[0].citation() if policy_chunks else 'unknown'}"
        )
    elif dti > max_dti_threshold:
        passed = False
        violations.append(
            "DTI exceeds standard maximum threshold of 45%. "
            f"Citation: {policy_chunks[0].citation() if policy_chunks else 'unknown'}"
        )

    if ltv > 0.85:
        passed = False
        violations.append(
            "LTV exceeds hard cap of 85%. "
            f"Citation: {policy_chunks[0].citation() if policy_chunks else 'unknown'}"
        )
    elif ltv > 0.80:
        if credit_score <= 720 or dti >= 0.35:
            passed = False
            violations.append(
                "LTV exceeds standard threshold of 80% and exception conditions are not met. "
                f"Citation: {policy_chunks[0].citation() if policy_chunks else 'unknown'}"
            )

    if employment_months < 6:
        passed = False
        violations.append(
            "Employment tenure below 6 months is a hard denial condition. "
            f"Citation: {policy_chunks[0].citation() if policy_chunks else 'unknown'}"
        )
    elif employment_months < 12:
        passed = False
        violations.append(
            "Employment tenure below 12 months requires review. "
            f"Citation: {policy_chunks[0].citation() if policy_chunks else 'unknown'}"
        )

    if income_document_count is not None and income_document_count < 2:
        passed = False
        violations.append(
            "Income verification does not include two required documents. "
            f"Citation: {policy_chunks[0].citation() if policy_chunks else 'unknown'}"
        )

    reasoning = (
        f"Checked credit score ({credit_score}), DTI ({dti:.2%}), LTV ({ltv:.2%}), "
        f"employment months ({employment_months}), income docs ({income_document_count}). "
        f"Policy pass status: {'PASSED' if passed else 'FAILED'}."
    )

    return {
        "passed": passed,
        "violations": violations,
        "min_credit_met": min_credit_met,
        "max_dti_threshold": max_dti_threshold,
        "reasoning": reasoning,
    }
