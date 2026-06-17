import os
import logging
from src.rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)

def get_vector_store(persist_directory: str = "./data/chroma_db", collection_name: str = "policy_documents"):
    """
    Initializes ChromaDB vector store.
    Tries to import chromadb. If missing, prints warnings and returns a mock vector store interface.
    """
    embeddings = get_embeddings()
    os.makedirs(persist_directory, exist_ok=True)
    
    try:
        from langchain_community.vectorstores import Chroma
        logger.info(f"Initializing Chroma vector store at {persist_directory}")
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
    except Exception as e:
        logger.warning(f"Could not initialize Chroma vector store: {str(e)}. Mocking database actions.")
        class MockChroma:
            def add_texts(self, texts, metadatas=None):
                logger.info(f"MockChroma: Added {len(texts)} texts to vector store.")
                return [str(i) for i in range(len(texts))]
            def similarity_search(self, query, k=3):
                logger.info(f"MockChroma: Searching for query '{query}'")
                # Fallback to simple policy retriever tool mock logic
                from src.tools.policy_tools import policy_retriever
                from langchain_core.documents import Document
                chunks = policy_retriever(query)
                return [Document(page_content=c, metadata={"source": "mock"}) for c in chunks[:k]]
        return MockChroma()
