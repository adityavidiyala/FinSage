"""
Reranking: re-scores the hybrid retriever's candidates and keeps only the
top-N most relevant chunks for generation.

Switched from a local cross-encoder (BAAI/bge-reranker-base, CPU-bound,
the dominant latency cost in earlier sessions) to Cohere's cloud reranker
(rerank-v3.5) — removes the local model load/inference cost entirely and
scales better now that BM25_K/VECTOR_K widened to 30 (more candidates to sort).
"""

from langchain_cohere import CohereRerank
from langchain_classic.retrievers import ContextualCompressionRetriever

from finance_rag.config import COHERE_API_KEY, RERANK_TOP_N


def build_reranked_retriever(hybrid_retriever) -> ContextualCompressionRetriever:
    """
    Wraps `hybrid_retriever` with Cohere cloud reranking.
    Returns the final retriever your pipeline should actually query.
    """
    reranker = CohereRerank(
        cohere_api_key=COHERE_API_KEY,
        model="rerank-v3.5",
        top_n=RERANK_TOP_N,
    )

    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=hybrid_retriever,
    )