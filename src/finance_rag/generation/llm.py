"""
Shared Gemini LLM client, used by both table summarization and answer generation.
"""

from langchain_google_genai import ChatGoogleGenerativeAI

from finance_rag.config import GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE

llm = ChatGoogleGenerativeAI(
    google_api_key=GOOGLE_API_KEY,
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
)