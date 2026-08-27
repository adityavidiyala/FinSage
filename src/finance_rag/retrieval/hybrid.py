"""
Hybrid retrieval: combines BM25 (keyword) and vector (semantic) retrievers
via LangChain's EnsembleRetriever, which does weighted Reciprocal Rank Fusion.
"""

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from finance_rag.config import BM25_K, VECTOR_K, HYBRID_WEIGHTS


def build_hybrid_retriever(langchain_docs: list, vectorstore) -> EnsembleRetriever:
    """
    Builds the BM25 + vector ensemble retriever.
    `langchain_docs` — needed to build the BM25 index (keyword search needs raw text).
    `vectorstore` — the QdrantVectorStore from indexing/vectorstore.py.
    """
    bm25_retriever = BM25Retriever.from_documents(langchain_docs)
    bm25_retriever.k = BM25_K

    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": VECTOR_K},
    )

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=HYBRID_WEIGHTS,  # [BM25, vector] — tilt toward semantic, keyword still counts
    )