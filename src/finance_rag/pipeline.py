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
from finance_rag.observability.tracing import track_stage
from finance_rag.caching.semantic_cache import load_cache, find_cached_answer, store_answer
from finance_rag.config import SEMANTIC_CACHE_ENABLED
from finance_rag.generation.rewrite import rewrite_standalone_question
from finance_rag.guardrails.input_guardrails import classify_input

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


def answer_query(
    question: str,
    retriever,
    use_cache: bool = SEMANTIC_CACHE_ENABLED,
    history: list[dict] | None = None,
) -> dict:
    if GUARDRAILS_ENABLED:
        with track_stage(question, "guardrail_input") as extra:
            classification = classify_input(question)
            extra.update(classification)

        if classification["jailbreak_attempt"]:
            return {
                "answer": "I'm designed to answer questions about financial filings, and I maintain "
                          "consistent guidelines regardless of how a request is phrased. What would "
                          "you like to know about the document?",
                "citations": [],
                "from_cache": False,
                "blocked_reason": "jailbreak_attempt",
            }

        if not classification["on_topic"]:
            return {
                "answer": "I can only answer questions grounded in the financial documents I've been "
                          "given — things like financial statements, segment results, or risk factors. "
                          "That question falls outside that scope.",
                "citations": [],
                "from_cache": False,
                "blocked_reason": "off_topic",
            }

    if history:
        with track_stage(question, "rewrite"):
            question = rewrite_standalone_question(history, question)

    if use_cache:
        cache_entries = load_cache()
        cached = find_cached_answer(question, cache_entries)
        if cached is not None:
            with track_stage(question, "cache_hit") as extra:
                extra["cache_similarity"] = cached["cache_similarity"]
            return {
                "answer": cached["answer"],
                "citations": cached["citations"],
                "from_cache": True,
                "cache_similarity": cached["cache_similarity"],
            }

    with track_stage(question, "retrieval"):
        retrieved_docs = retriever.invoke(question)

    with track_stage(question, "dedup"):
        retrieved_docs = dedup_chunks(retrieved_docs)

    with track_stage(question, "generation") as extra:
        result = generate_answer(question, retrieved_docs)
        extra.update(result.get("usage", {}))

    if use_cache:
        store_answer(question, result["answer"], result["citations"], cache_entries)

    result["from_cache"] = False
    return result