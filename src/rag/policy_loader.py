import glob
import logging
import os

from src.rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def load_and_index_policies(
    policy_docs_dir: str = "./data/policy_docs", persist_dir: str = "./data/chroma_db"
):
    """
    Reads all policy files (*.md), chunks them (500 chars with 50 chars overlap),
    and embeds them into the Chroma vector store.
    """
    logger.info(f"Scanning for policy files in {policy_docs_dir}")

    policy_files = glob.glob(os.path.join(policy_docs_dir, "*.md"))
    if not policy_files:
        logger.warning("No markdown policy files found to index!")
        return

    db = get_vector_store(persist_directory=persist_dir)

    all_chunks = []
    all_metadatas = []

    for file_path in policy_files:
        filename = os.path.basename(file_path)
        logger.info(f"Processing policy file: {filename}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Standard Chunking Strategy: 500 characters, 50 overlap
        chunk_size = 500
        overlap = 50

        start = 0
        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]

            all_chunks.append(chunk)
            all_metadatas.append(
                {
                    "source": filename,
                    "start_char": start,
                    "end_char": min(end, len(content)),
                }
            )

            start += chunk_size - overlap

    if all_chunks:
        logger.info(f"Adding {len(all_chunks)} chunks to vector store.")
        db.add_texts(all_chunks, metadatas=all_metadatas)
        logger.info("Indexing completed successfully.")
    else:
        logger.warning("No text extracted for indexing.")
