"""
Top-level orchestration. Three entry points:
  build_index()      — run once: parse -> chunk -> build docs -> embed+upsert -> cache docs for BM25.
  build_retriever()  — run once per process: rebuilds BM25 + connects Qdrant + loads reranker.
  answer_query()      — run per question, given an already-built retriever.
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

    get_vectorstore(langchain_docs)
    save_documents(langchain_docs, DOCS_CACHE_PATH)

    print(f"Index built: {len(langchain_docs)} documents. BM25 cache saved to {DOCS_CACHE_PATH}")


def build_retriever():
    """
    Expensive setup, meant to be called ONCE per process — not once per question.
    Rebuilds BM25 from the cached documents, connects to Qdrant, loads the reranker model.
    Reuse the returned retriever across as many answer_query() calls as you need.
    """
    if not os.path.exists(DOCS_CACHE_PATH):
        raise RuntimeError(
            f"No document cache found at {DOCS_CACHE_PATH}. Run build_index() first (scripts/ingest.py)."
        )

    langchain_docs = load_documents(DOCS_CACHE_PATH)
    vectorstore = get_vectorstore()

    hybrid_retriever = build_hybrid_retriever(langchain_docs, vectorstore)
    return build_reranked_retriever(hybrid_retriever)


def answer_query(question: str, retriever) -> dict:
    """
    Cheap, per-question part: retrieve -> dedup -> generate.
    `retriever` must come from build_retriever() — build it once, reuse it across calls.
    Returns {"answer": str, "citations": list[dict]}.
    """
    retrieved_docs = retriever.invoke(question)
    retrieved_docs = dedup_chunks(retrieved_docs)

    return generate_answer(question, retrieved_docs)