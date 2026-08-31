"""Pydantic request/response models for the FinSage model API.

This API is intentionally stateless: it does not know about users, sessions,
or Mongo. `web/server` owns conversation history and passes the last few
turns on every request, matching the `history: list[dict] | None` shape
`pipeline.answer_query()` already expects.
"""

from pydantic import BaseModel, Field


class HistoryTurn(BaseModel):
    question: str
    answer: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question.")
    history: list[HistoryTurn] | None = Field(
        default=None,
        description="Last few turns of the conversation, oldest first. "
        "Omit or leave empty for the first message in a conversation.",
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to check/populate the semantic cache for this query.",
    )
    use_decomposition: bool = Field(
        default=False,
        description="Whether to break the question into sub-queries for retrieval.",
    )


class Citation(BaseModel):
    chunk_id: str | int | None = None
    source_filename: str | None = None
    page_number: int | str | None = None
    section: str | None = None
    doc_type: str | None = None
    chunk_kind: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    usage: dict | None = None


class HealthResponse(BaseModel):
    status: str
    retriever_ready: bool