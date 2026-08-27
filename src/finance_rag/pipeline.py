"""
Top-level orchestration. Two entry points:
  build_index()  — run once: parse -> chunk -> build docs -> embed+upsert -> cache docs for BM25.
  answer_query() — run per question: hybrid retrieve -> rerank -> dedup -> generate with citations.
"""

import os

from finance_rag.config import DOCS_CACHE_PATH
from finance_rag.ingestion.parser import parse_document
from finance_rag.ingestion.chunker import build_semantic_chunks
from finance_rag.indexing.documents import build_langchain_documents, save_documents, load_documents
from finance_rag.indexing.vectorstore import get_vectorstore
from finance_rag.retrieval.hybrid import build_hybrid_retriever
from finance_rag.retrieval.rerank import build_reranked_retriever
from finance_rag.retrieval.dedup import dedup_chunks
from finance_rag.generation.answer import generate_answer


def build_index() -> None:
    """One-time (or re-run on new document) ingestion: parse, chunk, embed, upsert to Qdrant."""
    result = parse_document()
    semantic_chunks = build_semantic_chunks(result.document)
    langchain_docs = build_langchain_documents(semantic_chunks)

    get_vectorstore(langchain_docs)          # creates Qdrant collection + upserts if not already present
    save_documents(langchain_docs, DOCS_CACHE_PATH)  # so BM25 can be rebuilt later without re-ingesting

    print(f"Index built: {len(langchain_docs)} documents. BM25 cache saved to {DOCS_CACHE_PATH}")


def _get_retriever():
    if not os.path.exists(DOCS_CACHE_PATH):
        raise RuntimeError(
            f"No document cache found at {DOCS_CACHE_PATH}. Run build_index() first (scripts/ingest.py)."
        )

    langchain_docs = load_documents(DOCS_CACHE_PATH)
    vectorstore = get_vectorstore()  # connects to existing collection, no re-embedding

    hybrid_retriever = build_hybrid_retriever(langchain_docs, vectorstore)
    return build_reranked_retriever(hybrid_retriever)


def answer_query(question: str) -> dict:
    """
    Runs the full query-time pipeline: hybrid retrieval -> rerank -> dedup -> generate.
    Returns {"answer": str, "citations": list[dict]}.
    """
    retriever = _get_retriever()
    retrieved_docs = retriever.invoke(question)
    retrieved_docs = dedup_chunks(retrieved_docs)

    return generate_answer(question, retrieved_docs)