import logging
import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from finance_rag.pipeline import answer_query, build_retriever  # noqa: E402
from finance_rag.generation.llm import llm
from finance_rag.guardrails.pii_check import detect_pii
from api.schemas import HealthResponse, QueryRequest, QueryResponse  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Nothing document-specific to build up front anymore — the retriever
    # depends on request-time document_ids now. This lifespan hook is kept
    # only in case future model warm-up (e.g. pre-loading the embedder or
    # reranker independent of any retriever) is worth adding later.
    logger.info("Model API starting up.")
    yield
    logger.info("Model API shutting down.")


app = FastAPI(
    title="FinSage Model API",
    description="Wraps the FinSage RAG pipeline (retrieval + generation) as an HTTP service.",
    version="0.1.0",
    lifespan=lifespan,
)

_allowed_origins = [
    origin.strip()
    for origin in os.environ.get(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:8000"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_GUIDE_PROMPT = """You are FinSage, an AI financial document analysis assistant.
There are currently NO documents attached to this conversation.

Your job:
1. Politely respond to greetings, introductions, and questions about FinSage's capabilities or how to use the app.
2. If the user asks general trivia, creative writing, jokes, or random non-application topics, politely refuse and instruct them to keep questions focused on financial filings.
3. If the user asks a specific financial or company question (e.g., 'What is Google's revenue?'), inform them that they must upload the relevant financial document (10-K, 10-Q, transcript) using the '+' button before you can answer.

User question: {question}
Answer:"""


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    # retriever_ready no longer means "loaded once at startup" - there's no
    # single global retriever anymore. Reporting True here just confirms the
    # process is up; per-request build_retriever() failures surface as 404s
    # below instead.
    return HealthResponse(status="ok", retriever_ready=True)

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    # 1. No documents attached -> Guide the user or handle pleasantries
    if not request.document_ids:
        pii = detect_pii(request.question)
        if pii:
            return QueryResponse(
                answer=f"Your message appears to contain sensitive information ({', '.join(pii)}). Please remove it.",
                citations=[],
                from_cache=False,
            )

        resp = llm.invoke(SYSTEM_GUIDE_PROMPT.format(question=request.question))
        content = (
            resp.content
            if isinstance(resp.content, str)
            else resp.content[0].get("text", "")
        )
        return QueryResponse(
            answer=content.strip(),
            citations=[],
            from_cache=False,
        )

    # 2. Documents attached -> Run the full RAG pipeline
    history_dicts = (
        [turn.model_dump() for turn in request.history]
        if request.history
        else None
    )

    retriever = build_retriever(request.document_ids)
    result = answer_query(
        question=request.question,
        retriever=retriever,
        conversation_id=str(request.conversation_id),
        use_cache=request.use_cache,
        history=history_dicts,
        use_decomposition=request.use_decomposition,
    )
    return QueryResponse(**result)

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    try:
        retriever = build_retriever(request.document_ids)
    except RuntimeError as e:
        # raised when a document_id hasn't been indexed yet
        logger.warning("build_retriever() failed: %s", e)
        raise HTTPException(status_code=404, detail=str(e))

    history = (
        [turn.model_dump() for turn in request.history] if request.history else None
    )

    try:
        result = answer_query(
            question=request.question,
            retriever=retriever,
            conversation_id=request.conversation_id,
            use_cache=request.use_cache,
            history=history,
            use_decomposition=request.use_decomposition,
        )
    except Exception:
        logger.exception("answer_query() failed for question: %r", request.question)
        raise HTTPException(status_code=500, detail="Failed to generate an answer.")

    return QueryResponse(
        answer=result.get("answer", ""),
        citations=result.get("citations", []),
        usage=result.get("usage"),
    )

