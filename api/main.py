import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Same sys.path pattern used by scripts/ask.py and scripts/ingest.py,
# since the editable-install (pip install -e .) approach was abandoned
# earlier in the project (OneDrive write-lock issue on Windows).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from finance_rag.pipeline import answer_query, build_retriever  # noqa: E402

from api.schemas import HealthResponse, QueryRequest, QueryResponse  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Populated once at startup, reused for every request. Simple module-level
# state is enough here since this process only ever holds one retriever.
_state: dict = {"retriever": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Building retriever (loading embedder + reranker)...")
    _state["retriever"] = build_retriever()
    logger.info("Retriever ready. Model API is up.")
    yield
    _state.clear()
    logger.info("Model API shutting down.")


app = FastAPI(
    title="FinSage Model API",
    description="Wraps the FinSage RAG pipeline (retrieval + generation) as an HTTP service.",
    version="0.1.0",
    lifespan=lifespan,
)

# Only web/server (and, during local dev, web/client directly) should ever
# call this service - it's an internal service, not a public-facing API.
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


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", retriever_ready=_state["retriever"] is not None)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    retriever = _state["retriever"]
    if retriever is None:
        # Should never happen once lifespan startup has completed, but
        # fail loudly rather than silently returning a bad answer.
        raise HTTPException(status_code=503, detail="Retriever not ready yet.")

    history = (
        [turn.model_dump() for turn in request.history] if request.history else None
    )

    try:
        result = answer_query(
            question=request.question,
            retriever=retriever,
            use_cache=request.use_cache,
            history=history,
        )
    except Exception:
        logger.exception("answer_query() failed for question: %r", request.question)
        raise HTTPException(status_code=500, detail="Failed to generate an answer.")

    return QueryResponse(
        answer=result.get("answer", ""),
        citations=result.get("citations", []),
        usage=result.get("usage"),
    )