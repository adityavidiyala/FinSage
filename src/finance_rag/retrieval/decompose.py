"""
Query decomposition: breaks a complex, multi-part question into 2-4 focused
sub-queries via an LLM planning call, so each sub-part can be retrieved for
independently rather than relying on one retrieval pass to surface everything
a compound question needs.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from finance_rag.generation.llm import llm

DECOMPOSE_PROMPT = PromptTemplate.from_template("""
You are a financial query planner. Break down the user's complex question into 2 to 4 independent ('only if needed'), highly specific sub-queries designed to retrieve precise financial data from text and tables.
Return a valid JSON object with a single key "sub_queries" containing a list of strings. Do not include markdown code fences or extra text; return only the JSON.

User Question: {question}
""")

_decomposition_chain = DECOMPOSE_PROMPT | llm | JsonOutputParser()


def decompose_query(master_query: str) -> list[str]:
    """
    Returns a list of sub-queries for `master_query`. Falls back to
    [master_query] unchanged if the LLM output isn't in the expected shape —
    fail-open, so a bad decomposition never blocks retrieval entirely.
    """
    result = _decomposition_chain.invoke({"question": master_query})

    if isinstance(result, dict):
        return result.get("sub_queries", [master_query])
    elif isinstance(result, list):
        return result
    else:
        return [master_query]