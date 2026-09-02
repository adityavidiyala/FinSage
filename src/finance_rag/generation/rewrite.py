from finance_rag.config import MAX_HISTORY_TURNS
from finance_rag.generation.llm import llm


REWRITE_PROMPT = """Given the recent conversation history and a new question, rewrite the new question \
into a fully standalone question that includes all necessary context (dates, entities, topics) \
from the history. If the new question is already standalone, return it unchanged. \
Do not answer the question — only rewrite it.

"Recognize common company synonyms and tickers (such as Google <-> Alphabet Inc., "
"Meta <-> Facebook) when reformulating queries."

Conversation history:
{history_block}

New question: {question}

Standalone question:"""


def rewrite_standalone_question(history: list[dict], question: str) -> str:
    if not history:
        return question

    recent = history[-MAX_HISTORY_TURNS:]
    def _extract_qa(turn):
        if hasattr(turn, "question") and hasattr(turn, "answer"):
            return turn.question, turn.answer
        if isinstance(turn, dict):
            return turn.get("question", ""), turn.get("answer", "")
        return getattr(turn, "question", ""), getattr(turn, "answer", "")

    history_block = "\n".join(
        f"Q: {q}\nA: {a}" for q, a in (_extract_qa(turn) for turn in recent)
    )

    prompt = REWRITE_PROMPT.format(history_block=history_block, question=question)
    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        content = "".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in content
        )
    return content.strip()