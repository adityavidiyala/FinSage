"""
Final answer generation with inline citations: takes a question + retrieved
Document objects, numbers them as sources, and asks the LLM to cite which
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
You are answering a question about a company's financial filing (10-Q).
Use ONLY the information in the numbered sources below. Do not use outside knowledge.
If the sources do not contain the answer, say so explicitly rather than guessing.

Cite the source(s) supporting each claim using bracketed numbers, e.g. [1] or [2][3],
placed right after the claim. Only cite source numbers that are listed below.
Do not add a citation if the sentence is not directly supported by a specific source.

Sources:
{context}

Question:
{question}

Answer:
""")


def _build_numbered_context(retrieved_docs: list[Document]) -> str:
    blocks = []
    for i, doc in enumerate(retrieved_docs, start=1):
        blocks.append(f"[{i}] {doc.page_content}")
    return "\n\n".join(blocks)


def _extract_citations(answer_text: str, retrieved_docs: list[Document]) -> list[dict]:
    """
    Finds all [n] markers actually used in the answer, deduplicates them,
    and resolves each to its source chunk's metadata. Silently skips any
    marker number the LLM hallucinated outside the valid range.
    """
    marker_numbers = sorted(set(int(n) for n in re.findall(r"\[(\d+)\]", answer_text)))

    citations = []
    for n in marker_numbers:
        if 1 <= n <= len(retrieved_docs):
            doc = retrieved_docs[n - 1]
            citations.append({
                "marker": n,
                "chunk_id": doc.metadata.get("chunk_id"),
                "source_filename": doc.metadata.get("source_filename"),
                "page_number": doc.metadata.get("page_number"),
                "section": doc.metadata.get("section"),
                "doc_type": doc.metadata.get("doc_type"),
            })
    return citations


def generate_answer(question: str, retrieved_docs: list[Document]) -> dict:
    """
    Returns {"answer": str, "citations": list[dict]}.
    `answer` contains inline [n] markers as written by the LLM.
    `citations` is the resolved, deduplicated list of sources those markers point to.
    """
    context_text = _build_numbered_context(retrieved_docs)
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

            answer_text = content.strip()
            citations = _extract_citations(answer_text, retrieved_docs)

            return {"answer": answer_text, "citations": citations}

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 20 + random.uniform(1, 5)
                print(f"Rate limit reached. Waiting {wait:.1f}s...")
                time.sleep(wait)
            else:
                raise

    return {"answer": "GENERATION FAILED after retries", "citations": []}