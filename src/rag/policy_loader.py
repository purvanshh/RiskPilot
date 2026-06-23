import logging
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document

from src.rag.embeddings import get_embedding_provider
from src.rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


def _find_markdown_files(policy_docs_dir: str) -> List[Path]:
    policy_dir = Path(policy_docs_dir)
    return sorted(policy_dir.glob("*.md"))


def _build_chunk_metadata(
    source_file: Path, document_name: str, section: str, chunk_index: int
) -> Dict[str, str]:
    return {
        "document": document_name,
        "section": section,
        "source_file": source_file.name,
        "policy_type": document_name.lower().replace(" ", "_").replace("-", "_"),
        "chunk_index": str(chunk_index),
    }


def _split_markdown_sections(text: str) -> List[Dict[str, str]]:
    sections: List[Dict[str, str]] = []
    current_section = "General"
    buffer: List[str] = []

    for line in text.splitlines(keepends=True):
        if line.startswith("#"):
            if buffer:
                sections.append({"section": current_section, "text": "".join(buffer).strip()})
                buffer = []
            current_section = line.lstrip("# ").strip() or "General"
            buffer.append(line)
            continue
        buffer.append(line)

    if buffer:
        sections.append({"section": current_section, "text": "".join(buffer).strip()})

    return sections


def _split_document(file_path: Path) -> List[Document]:
    source_text = file_path.read_text(encoding="utf-8")
    document_name = file_path.stem.replace("_", " ").title()

    sections = _split_markdown_sections(source_text)
    documents: List[Document] = []
    chunk_index = 0

    for section in sections:
        section_name = section["section"]
        text = section["text"]
        start = 0
        while start < len(text):
            end = min(len(text), start + CHUNK_SIZE)
            chunk = text[start:end].strip()
            if chunk:
                metadata = _build_chunk_metadata(
                    file_path, document_name, section_name, chunk_index
                )
                documents.append(Document(page_content=chunk, metadata=metadata))
                chunk_index += 1
            start += CHUNK_SIZE - CHUNK_OVERLAP

    return documents


def load_and_index_policies(
    policy_docs_dir: str = "./data/policy_docs",
    persist_dir: str = "./data/chroma_db",
    collection_name: str = "lending_policy",
) -> None:
    logger.info("Starting policy loader")
    policy_files = _find_markdown_files(policy_docs_dir)
    if not policy_files:
        raise FileNotFoundError(f"No markdown policy files found in {policy_docs_dir}")

    provider = get_embedding_provider()
    db = get_vector_store(persist_directory=persist_dir, collection_name=collection_name)
    documents: List[Document] = []

    for file_path in policy_files:
        logger.info(f"Chunking policy file: {file_path.name}")
        documents.extend(_split_document(file_path))

    if not documents:
        raise ValueError("Policy loader did not generate any document chunks.")

    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    ids = [f"{doc.metadata['source_file']}_{doc.metadata['chunk_index']}" for doc in documents]

    logger.info("Generating embeddings for %d policy chunks.", len(texts))
    embeddings = provider.embed_documents(texts)
    logger.info("Indexing %d policy chunks into ChromaDB.", len(texts))
    db.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)

    try:
        if hasattr(db, "persist"):
            db.persist()
            logger.info("ChromaDB policy collection persisted successfully.")
    except Exception:
        logger.warning("ChromaDB persist() failed, continuing without persistence.")
