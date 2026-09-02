"""
Hybrid retrieval: combines BM25 (keyword) and vector (semantic) retrievers
via LangChain's EnsembleRetriever, which does weighted Reciprocal Rank Fusion.
"""

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from qdrant_client.models import Filter, FieldCondition, MatchAny
from finance_rag.config import BM25_K, VECTOR_K, HYBRID_WEIGHTS


def build_hybrid_retriever(langchain_docs: list, vectorstore, document_ids: list[str]) -> EnsembleRetriever:
    """
    Builds the BM25 + vector ensemble retriever, scoped to `document_ids`.
    `langchain_docs` — already pre-filtered to just these documents by the
    caller (build_retriever), used to build the BM25 index.
    `vectorstore` — the QdrantVectorStore, connected to the FULL shared
    collection — filtered here via `document_ids` so it only searches
    chunks belonging to this conversation's documents.
    """
    if not document_ids:
        raise ValueError("build_hybrid_retriever() requires at least one document_id to filter on.")

    bm25_retriever = BM25Retriever.from_documents(langchain_docs)
    bm25_retriever.k = BM25_K

    qdrant_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.document_id",
                match=MatchAny(any=document_ids),
            )
        ]
    )

    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": VECTOR_K, "filter": qdrant_filter},
    )

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=HYBRID_WEIGHTS,
    )