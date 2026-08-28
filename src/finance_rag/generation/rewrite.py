from finance_rag.config import MAX_HISTORY_TURNS
from finance_rag.generation.llm import llm


REWRITE_PROMPT = """Given the recent conversation history and a new question, rewrite the new question \
into a fully standalone question that includes all necessary context (dates, entities, topics) \
from the history. If the new question is already standalone, return it unchanged. \
Do not answer the question — only rewrite it.

Conversation history:
{history_block}

New question: {question}

Standalone question:"""


def rewrite_standalone_question(history: list[dict], question: str) -> str:
    if not history:
        return question

    recent = history[-MAX_HISTORY_TURNS:]
    history_block = "\n".join(
        f"Q: {turn['question']}\nA: {turn['answer']}" for turn in recent
    )

    prompt = REWRITE_PROMPT.format(history_block=history_block, question=question)
    response = llm.invoke(prompt)
    return response.content.strip()