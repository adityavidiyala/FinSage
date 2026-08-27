"""
Cross-encoder reranking: re-scores the hybrid retriever's candidates using a
cross-encoder model, keeping only the top-N most relevant chunks for generation.
"""

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever

from finance_rag.config import RERANK_MODEL, RERANK_TOP_N

_cross_encoder = HuggingFaceCrossEncoder(model_name=RERANK_MODEL)


def build_reranked_retriever(hybrid_retriever) -> ContextualCompressionRetriever:
    """
    Wraps `hybrid_retriever` with cross-encoder reranking.
    Returns the final retriever your pipeline should actually query.
    """
    reranker = CrossEncoderReranker(
        model=_cross_encoder,
        top_n=RERANK_TOP_N,
    )

    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=hybrid_retriever,
    )