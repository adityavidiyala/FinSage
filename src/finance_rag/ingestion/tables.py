"""
Table handling: converts docling's markdown tables into two complementary forms —
an LLM-generated natural-language summary (for retrieval matching), and a
deterministic row-by-row text rendering (for preserving exact values).
"""

import time
import random
from io import StringIO

import pandas as pd
from langchain_core.prompts import PromptTemplate

from finance_rag.generation.llm import llm  # shared LLM client, defined in generation/llm.py

# =========================================================
# TABLE SUMMARIZATION PROMPT
# =========================================================
TABLE_SUMMARY_PROMPT = PromptTemplate.from_template("""
You are summarizing a financial table extracted from a document.

STRICT RULES:
1. Use ONLY information explicitly present in the table.
2. Preserve exact important numbers.
3. Mention totals, percentages, comparisons, and time periods ONLY if explicitly shown.
4. Do NOT perform arithmetic calculations or derive new values.
5. Do NOT infer context not shown in the table.
6. Do NOT hallucinate headings or business meaning.
7. If table context is ambiguous, say so.
8. Output concise factual prose only and Avoid unnecessary verbosity.
9. Include specific row-level values when they are important for understanding the financial meaning of the table.

Table:
{table}

Summary:
""")


def summarize_table(table_markdown: str) -> str:
    """LLM-generated natural-language summary of a markdown table, with 429 retry/backoff."""
    if not table_markdown or not table_markdown.strip():
        return ""

    prompt = TABLE_SUMMARY_PROMPT.format(table=table_markdown)

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

    raise RuntimeError("Gemini free-tier quota exceeded after retries.")


def markdown_table_to_text(markdown_table: str) -> str:
    """
    Deterministic fallback: converts a markdown table into 'Column is Value.' sentences
    per row, handling single- and multi-row headers. Falls back to the raw markdown
    on any parsing failure.
    """
    try:
        lines = [line for line in markdown_table.splitlines() if "|" in line]
        lines = [line for line in lines if "---" not in line]

        if len(lines) < 2:
            return markdown_table

        def parse_row(line):
            return [cell.strip() for cell in line.split("|")[1:-1]]

        first_row = parse_row(lines[0])
        second_row = parse_row(lines[1]) if len(lines) > 1 else []

        empty_in_first = sum(1 for c in first_row if c == "")
        is_multi_header = (
            len(second_row) > 0
            and empty_in_first >= len(first_row) // 2
        )

        if is_multi_header:
            combined_headers = []
            for top, bot in zip(first_row, second_row):
                if top and bot:
                    combined_headers.append(f"{top} {bot}".strip())
                elif bot:
                    combined_headers.append(bot)
                elif top:
                    combined_headers.append(top)
                else:
                    combined_headers.append("")
            data_lines = lines[2:]
        else:
            combined_headers = first_row
            data_lines = lines[1:]

        if not data_lines:
            return markdown_table

        df = pd.read_csv(StringIO("\n".join(data_lines)), sep="|", engine="python", header=None)
        df = df.dropna(axis=1, how="all")
        df = df.iloc[:, :len(combined_headers)]
        df.columns = combined_headers[:len(df.columns)]

        for col in df.columns:
            df[col] = df[col].astype(str).str.strip()

        records = []
        for _, row in df.iterrows():
            parts = []
            for col in df.columns:
                value = str(row[col]).strip()
                if value in ("", ".", "-", "nan"):
                    continue
                if value == "X":
                    value = "Yes"
                parts.append(f"{col} is {value}")
            if parts:
                records.append(". ".join(parts) + ".")

        return "\n\n".join(records)

    except Exception:
        return markdown_table