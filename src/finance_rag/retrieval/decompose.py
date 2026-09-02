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
You are a financial query planner. Your job is to break down a user's question into standalone sub-queries that can be retrieved independently from SEC filings.

Instructions:
1. Simplicity: If the user's question is straightforward and asks about only one company or metric, return a list containing just the original question.
2. Decomposition: If the question involves comparisons, multiple companies, or multiple distinct metrics, break it into 2 to 4 simple, focused sub-queries.
3. Decouple Entities & Dates: If multiple companies are mentioned, generate separate sub-queries for each company. Never carry over fiscal dates, periods, or metrics from one company into another company's sub-query.
4. Metric Normalization: Use broad, canonical financial terminology (e.g., "HP Commercial PC unit volume and ASP performance", "Alphabet capital expenditures and infrastructure investments").

Output Format:
Return strictly a valid JSON object with the following structure:
{{"sub_queries": ["query 1", "query 2"]}}

User Question: {question}
""")

_decomposition_chain = DECOMPOSE_PROMPT | llm | JsonOutputParser()


def decompose_query(master_query: str) -> list[str]:
    """
    Returns a list of sub-queries for `master_query`. Falls back to
    [master_query] unchanged if the LLM output isn't in the expected shape —
    fail-open, so a bad decomposition never blocks retrieval entirely.
    """
    try:
        result = _decomposition_chain.invoke({"question": master_query})

        if isinstance(result, dict):
            sub_queries = result.get("sub_queries", [master_query])
            return sub_queries if isinstance(sub_queries, list) and sub_queries else [master_query]
        elif isinstance(result, list):
            return result if result else [master_query]
        else:
            return [master_query]
    except Exception:
        # Prevents OutputParserException or network glitches from breaking the search
        return [master_query]