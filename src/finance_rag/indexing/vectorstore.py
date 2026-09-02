"""
Qdrant vector store: connects to an existing collection if one exists,
otherwise creates it and embeds+upserts the given documents.
"""

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from finance_rag.config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME
from finance_rag.indexing.embeddings import embeddings


def get_vectorstore(langchain_docs: list = None) -> QdrantVectorStore:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    collection_exists = client.collection_exists(COLLECTION_NAME)

    if not collection_exists:
        if not langchain_docs:
            raise ValueError(f"Collection '{COLLECTION_NAME}' doesn't exist and no documents were provided to create it.")
        print(f"Collection '{COLLECTION_NAME}' not found — creating and embedding first document.")
        return QdrantVectorStore.from_documents(
            documents=langchain_docs,
            embedding=embeddings,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=COLLECTION_NAME,
        )

    store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
    )

    if langchain_docs:
        print(f"Collection exists — embedding and adding {len(langchain_docs)} new document(s).")
        store.add_documents(langchain_docs)

    return store