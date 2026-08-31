"""
Converts semantic chunk dicts (from chunker.py) into LangChain Document objects,
ready for embedding. Text chunks and table-row chunks both pass through with
their content unchanged — table rows are already self-contained (heading +
header + single row), so no LLM summarization step is needed here.
"""
import json
from langchain_core.documents import Document


def build_langchain_documents(semantic_chunks: list[dict]) -> list[Document]:
    """
    Converts semantic chunk dicts into LangChain Document objects.
    Each row mini-table or text chunk becomes a standalone Document, with
    parent metadata (e.g. full_table_markdown, table_id) preserved for
    retrieval-time hydration back to the full table (see generation/answer.py).
    """
    langchain_docs = []

    for chunk in semantic_chunks:
        metadata = chunk["metadata"].copy()
        raw_content = chunk["content"]

        if metadata.get("contains_table"):
            metadata["doc_type"] = "table_row"
        else:
            metadata["doc_type"] = "text"

        # Transient key, not needed once page_content is set.
        metadata.pop("raw_chunk", None)

        doc = Document(page_content=raw_content, metadata=metadata)
        langchain_docs.append(doc)

    return langchain_docs


def save_documents(docs: list[Document], path: str) -> None:
    """Persist LangChain Documents to disk so BM25 can be rebuilt without re-running ingestion."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
            f,
        )


def load_documents(path: str) -> list[Document]:
    """Load Documents previously saved by save_documents()."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Document(page_content=item["page_content"], metadata=item["metadata"]) for item in data]