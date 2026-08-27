"""
Final answer generation: takes a question + retrieved contexts, builds a
grounded prompt, and calls the LLM — with retry/backoff on rate limits.
"""

import time
import random

from langchain_core.prompts import PromptTemplate

from finance_rag.generation.llm import llm

ANSWER_PROMPT = PromptTemplate.from_template("""
You are answering a question about a company's financial filing (10-Q).
Use ONLY the information in the provided context. Do not use outside knowledge.
If the context does not contain the answer, say so explicitly rather than guessing.

Context:
{context}

Question:
{question}

Answer:
""")


def generate_answer(question: str, contexts: list[str]) -> str:
    context_text = "\n\n---\n\n".join(contexts)
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

            return content.strip()

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 20 + random.uniform(1, 5)
                print(f"Rate limit reached. Waiting {wait:.1f}s...")
                time.sleep(wait)
            else:
                raise

    return "GENERATION FAILED after retries"