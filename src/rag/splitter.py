import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List

from langchain.docstore.document import Document

logger = logging.getLogger(__name__)

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


def _simple_recursive_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text.strip()]

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks


def _extract_markdown_sections(text: str) -> List[Dict[str, str]]:
    pattern = re.compile(r"^(#{1,4})\s*(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return [{"section": "General", "text": text.strip()}]

    sections = []
    for index, match in enumerate(matches):
        start = match.start()
        header = match.group(2).strip()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        sections.append({"section": header, "text": section_text})

    return sections


def split_markdown_document(file_path: Path) -> List[Document]:
    """Splits a markdown policy document into metadata-rich chunks."""
    source_text = file_path.read_text(encoding="utf-8")
    document_name = file_path.stem.replace("_", " ").title()

    sections = _extract_markdown_sections(source_text)
    documents: List[Document] = []
    chunk_index = 0

    for section in sections:
        section_name = section["section"]
        for chunk_text in _simple_recursive_split(
            section["text"], CHUNK_SIZE, CHUNK_OVERLAP
        ):
            metadata = {
                "document": document_name,
                "section": section_name,
                "policy_type": document_name.lower().replace(" ", "_").replace("-", "_"),
                "source_file": file_path.name,
                "chunk_index": str(chunk_index),
            }
            documents.append(Document(page_content=chunk_text, metadata=metadata))
            chunk_index += 1

    logger.debug(
        "Split markdown document %s into %d chunks.", file_path.name, len(documents)
    )
    return documents
