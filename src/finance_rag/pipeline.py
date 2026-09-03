"""
Top-level orchestration. Three entry points:
  build_index(document_id, pdf_path)  — run once per NEW document (by content_hash):
                                          parse -> chunk -> tag with document_id -> embed+upsert -> cache docs for BM25.
  build_retriever(document_ids)       — run per request: merges BM25 docs for the given documents,
                                          connects Qdrant filtered to those document_ids, loads reranker.
  answer_query()                       — run per question, given an already-built retriever + conversation_id.
"""
import os
from finance_rag.config import get_document_paths, get_conversation_paths, GUARDRAILS_ENABLED, SEMANTIC_CACHE_ENABLED
from finance_rag.ingestion.parser import parse_document
from finance_rag.ingestion.chunker import build_semantic_chunks
from finance_rag.indexing.documents import build_langchain_documents, save_documents, load_documents
from finance_rag.indexing.vectorstore import get_vectorstore
from finance_rag.retrieval.hybrid import build_hybrid_retriever
from finance_rag.retrieval.rerank import build_reranked_retriever
from finance_rag.retrieval.dedup import dedup_chunks
from finance_rag.generation.answer import generate_answer
from finance_rag.observability.tracing import track_stage
from finance_rag.caching.semantic_cache import load_cache, save_cache, find_cached_answer, store_answer
from finance_rag.generation.rewrite import rewrite_standalone_question
from finance_rag.guardrails.input_guardrails import classify_input
from finance_rag.guardrails.pii_check import detect_pii
from finance_rag.retrieval.decompose import decompose_query


def build_index(document_id: str, pdf_path: str) -> None:
    """
    Ingests exactly ONE document, scoped to document_id. Called once per new
    document (i.e. only when content_hash dedup finds no existing row) —
    never re-run for a document that's already been ingested.
    """
    paths = get_document_paths(document_id)
    os.makedirs(paths["cache_dir"], exist_ok=True)

    result = parse_document(pdf_path=pdf_path, cache_path=paths["docling_json"])
    semantic_chunks = build_semantic_chunks(result.document, pdf_path=pdf_path)
    langchain_docs = build_langchain_documents(semantic_chunks)

    # Stamp every chunk with document_id so Qdrant filtering and per-conversation
    # BM25 merging (build_retriever) can scope correctly downstream.
    for doc in langchain_docs:
        doc.metadata["document_id"] = str(document_id)

    get_vectorstore(langchain_docs)
    save_documents(langchain_docs, paths["docs_json"])

    print(f"Index built for document {document_id}: {len(langchain_docs)} chunks. Cached to {paths['docs_json']}")


def build_retriever(document_ids: list[str]):
    """
    Called per request (or cached in memory per conversation, invalidated when
    a document is added — that caching decision lives in the API layer, not here).
    Merges BM25 docs from every document_id given, and filters Qdrant retrieval
    to only those document_ids so a conversation never sees another document's chunks.
    """
    if not document_ids:
        raise ValueError("build_retriever() requires at least one document_id.")

    langchain_docs = []
    for doc_id in document_ids:
        paths = get_document_paths(str(doc_id))
        if not os.path.exists(paths["docs_json"]):
            raise RuntimeError(f"No document cache found for {doc_id} at {paths['docs_json']}. Run build_index() for it first.")
        langchain_docs.extend(load_documents(paths["docs_json"]))

    vectorstore = get_vectorstore()  # connects only, no langchain_docs passed — nothing new to embed here

    hybrid_retriever = build_hybrid_retriever(
        langchain_docs, vectorstore, document_ids=[str(d) for d in document_ids]
    )
    return build_reranked_retriever(hybrid_retriever)


def answer_query(
    question: str,
    retriever,
    conversation_id: str,
    use_cache: bool = SEMANTIC_CACHE_ENABLED,
    history: list[dict] | None = None,
    use_decomposition: bool = True,
) -> dict:
    conv_paths = get_conversation_paths(conversation_id)

    if GUARDRAILS_ENABLED:
        pii_found = detect_pii(question)
        if pii_found:
            return {
                "answer": f"Your message appears to contain sensitive information "
                          f"({', '.join(pii_found)}). Please remove it and try again.",
                "citations": [], "from_cache": False, "blocked_reason": "pii_detected",
            }
        with track_stage(question, "guardrail_input") as extra:
            classification = classify_input(question)
            extra.update(classification)

        if classification.get("jailbreak_attempt"):
            return {
                "answer": "I'm designed to answer questions about financial filings, and I maintain "
                          "consistent guidelines regardless of how a request is phrased. What would "
                          "you like to know about the document?",
                "citations": [], "from_cache": False, "blocked_reason": "jailbreak_attempt",
            }

        # NEW: Handle polite greetings and pleasantries cleanly
        if classification.get("is_greeting"):
            return {
                "answer": "Hello! I am FinSage. Ask me any question about your uploaded financial filings, and I'll retrieve the answers with source citations.",
                "citations": [],
                "from_cache": False,
            }

        if not classification.get("on_topic"):
            return {
                "answer": "I can only answer questions grounded in the financial documents I've been "
                          "given — things like financial statements, segment results, or risk factors. "
                          "That question falls outside that scope.",
                "citations": [], "from_cache": False, "blocked_reason": "off_topic",
            }

    if history:
        with track_stage(question, "rewrite"):
            question = rewrite_standalone_question(history, question)

    cache_entries = []
    if use_cache:
        cache_entries = load_cache(conv_paths["semantic_cache_json"])
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

    if use_decomposition:
        with track_stage(question, "decompose") as extra:
            sub_queries = decompose_query(question)
            extra["sub_queries"] = sub_queries

        with track_stage(question, "retrieval") as extra:
            retrieved_docs = []
            for sq in sub_queries:
                retrieved_docs.extend(retriever.invoke(sq))
            extra["sub_query_count"] = len(sub_queries)
    else:
        with track_stage(question, "retrieval"):
            retrieved_docs = retriever.invoke(question)

    with track_stage(question, "dedup"):
        retrieved_docs = dedup_chunks(retrieved_docs)

    with track_stage(question, "generation") as extra:
        result = generate_answer(question, retrieved_docs)
        extra.update(result.get("usage", {}))

    if use_cache:
        store_answer(question, result["answer"], result["citations"], cache_entries)
        save_cache(cache_entries, conv_paths["semantic_cache_json"])

    result["from_cache"] = False
    return result