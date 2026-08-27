"""
Converts semantic chunk dicts (from chunker.py) into LangChain Document objects,
ready for embedding. Table chunks get an LLM summary + deterministic text rendering
merged into the page content; text chunks pass through unchanged.
"""

from langchain_core.documents import Document

from finance_rag.ingestion.tables import summarize_table, markdown_table_to_text


def build_langchain_documents(semantic_chunks: list[dict]) -> list[Document]:
    langchain_docs = []

    for chunk in semantic_chunks:
        metadata = chunk["metadata"].copy()

        if metadata["contains_table"]:
            raw_table = metadata["raw_chunk"]

            table_summary = summarize_table(raw_table)
            table_text = markdown_table_to_text(raw_table)

            metadata["doc_type"] = "table"

            page_content = f"""
Section: {metadata.get('section', '')}

TABLE SUMMARY:
{table_summary}

TABLE CONTENT:
{table_text}
""".strip()

            doc = Document(page_content=page_content, metadata=metadata)

        else:
            metadata["doc_type"] = "text"
            raw_text = chunk["content"]
            metadata.pop("raw_chunk", None)

            doc = Document(page_content=raw_text, metadata=metadata)

        langchain_docs.append(doc)

    return langchain_docs