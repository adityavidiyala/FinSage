"""
Final answer generation with inline citations: takes a question + retrieved
Document objects, hydrates any table-row chunks back to their full parent
table, numbers everything as sources, and asks the LLM to cite which
source(s) support each part of its answer using [n] markers. The markers are
then resolved back to real chunk metadata for display.
"""

import re
import time
import random

from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from finance_rag.generation.llm import llm

ANSWER_PROMPT = PromptTemplate.from_template("""
You are answering a question about financial documents.
Use ONLY the information in the numbered sources below. Do not use outside knowledge.
If the sources do not contain the answer, say so explicitly rather than guessing.

Cite the source(s) supporting each claim using bracketed numbers, e.g. [1] or [2][3],
placed right after the claim. Only cite source numbers that are listed below.
Do not add a citation if the sentence is not directly supported by a specific source.

When identifying a company or reporting entity, mention both the official SEC
registrant name and common parent/subsidiary brand names if applicable
(e.g., 'Alphabet Inc. (Google)').

Multi-Document & Fiscal Calendar Rules:
- Companies often report on different fiscal year schedules. Do NOT assume two documents share the same reporting dates or quarters.
- Extract metrics for each company based on whatever reporting periods are present in its own filing (e.g. Q2 2026 for one, Q3 2025 for another).
- Never dismiss a company's data simply because its filing date differs from another attached document.
- Clearly state the company, filing date, and period corresponding to each metric.
- For comparative or synthesis questions, contrast the reported figures and trends from each document without extrapolating beyond the provided data.
Sources:
{context}

Question:
{question}

Answer:
""")


def _hydrate_and_dedup_sources(retrieved_docs: list[Document]) -> list[Document]:
    """
    Expands table-row chunks back to their full parent table (via
    full_table_markdown, set at chunking time) and deduplicates by table_id
    so a table retrieved via multiple rows only appears once in the prompt.
    Non-table chunks pass through unchanged.
    """
    seen_tables = set()
    hydrated_docs = []

    for doc in retrieved_docs:
        metadata = doc.metadata.copy()

        if metadata.get("contains_table") and metadata.get("full_table_markdown"):
            table_id = metadata.get("table_id")
            if table_id not in seen_tables:
                seen_tables.add(table_id)
                hydrated_docs.append(
                    Document(
                        page_content=metadata["full_table_markdown"],
                        metadata=metadata,
                    )
                )
            # else: already added this table via an earlier row — skip.
        else:
            hydrated_docs.append(doc)

    return hydrated_docs


def _build_numbered_context(hydrated_docs: list[Document]) -> str:
    blocks = []
    for i, doc in enumerate(hydrated_docs, start=1):
        section_tag = doc.metadata.get("section", "")
        page_tag = doc.metadata.get("page_number")
        header = f"[{i}]"
        if section_tag or page_tag:
            header += f" (Section: {section_tag}, Page: {page_tag})"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks)


def _extract_citations(answer_text: str, hydrated_docs: list[Document]) -> list[dict]:
    """
    Finds all [n] markers actually used in the answer, deduplicates them,
    and resolves each to its source chunk's metadata. Silently skips any
    marker number the LLM hallucinated outside the valid range.
    """
    marker_numbers = sorted(set(int(n) for n in re.findall(r"\[(\d+)\]", answer_text)))

    citations = []
    for n in marker_numbers:
        if 1 <= n <= len(hydrated_docs):
            doc = hydrated_docs[n - 1]
            citations.append({
                "marker": n,
                "chunk_id": doc.metadata.get("chunk_id"),
                "source_filename": doc.metadata.get("source_filename"),
                "page_number": doc.metadata.get("page_number"),
                "section": doc.metadata.get("section"),
                "doc_type": doc.metadata.get("doc_type"),
                "chunk_kind": doc.metadata.get("chunk_kind"),
            })
    return citations


def generate_answer(question: str, retrieved_docs: list[Document]) -> dict:
    """
    Returns {"answer": str, "citations": list[dict], "usage": dict}.
    `answer` contains inline [n] markers as written by the LLM.
    `citations` is the resolved, deduplicated list of sources those markers point to.
    """
    hydrated_docs = _hydrate_and_dedup_sources(retrieved_docs)

    context_text = _build_numbered_context(hydrated_docs)
    prompt = ANSWER_PROMPT.format(context=context_text, question=question)

    for attempt in range(5):
        try:
            response = llm.invoke(prompt)
            content = response.content

            if isinstance(content, list):
                content = "".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )

            usage = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = {
                    "input_tokens": response.usage_metadata.get("input_tokens"),
                    "output_tokens": response.usage_metadata.get("output_tokens"),
                    "total_tokens": response.usage_metadata.get("total_tokens"),
                }

            answer_text = str(content).strip()
            citations = _extract_citations(answer_text, hydrated_docs)

            return {"answer": answer_text, "citations": citations, "usage": usage}

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 20 + random.uniform(1, 5)
                print(f"Rate limit reached. Waiting {wait:.1f}s...")
                time.sleep(wait)
            else:
                raise

    return {"answer": "GENERATION FAILED after retries", "citations": [], "usage": {}}