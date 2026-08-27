"""
Qdrant vector store: connects to an existing collection if one exists,
otherwise creates it and embeds+upserts the given documents.
"""

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from finance_rag.config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME
from finance_rag.indexing.embeddings import embeddings


def get_vectorstore(langchain_docs: list = None) -> QdrantVectorStore:
    """
    Returns a QdrantVectorStore connected to COLLECTION_NAME.
    If the collection already exists, connects without re-embedding.
    If it doesn't exist, creates it and embeds `langchain_docs` (required in that case).
    """
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    collection_exists = client.collection_exists(COLLECTION_NAME)

    if collection_exists:
        print(f"Collection '{COLLECTION_NAME}' already exists — connecting without re-embedding.")
        return QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=COLLECTION_NAME,
        )

    if not langchain_docs:
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' doesn't exist yet and no documents were "
            "provided to create it. Run ingestion first (scripts/ingest.py)."
        )

    print(f"Collection '{COLLECTION_NAME}' not found — creating and embedding documents now.")
    return QdrantVectorStore.from_documents(
        documents=langchain_docs,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
    )