"""
Semantic chunking: walks a parsed docling document and produces token-bounded,
heading-aware chunks. Text is chunked normally; tables are split into
per-row "mini-table" chunks (parent-child pattern) so retrieval can match a
specific row precisely, while the full table is preserved in metadata for
hydration back to full context at generation time.
"""

import re
from pathlib import Path

from transformers import AutoTokenizer

from finance_rag.config import EMBED_MODEL, PDF_PATH, MAX_TOKENS, OVERLAP_TOKENS

_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)


def _token_count(text: str) -> int:
    return len(_tokenizer.encode(text, add_special_tokens=False))


def _clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip())


def _extract_page_number(item):
    try:
        if hasattr(item, "prov") and item.prov:
            prov = item.prov[0]
            if hasattr(prov, "page_no"):
                return prov.page_no
            if hasattr(prov, "page"):
                return prov.page
    except Exception:
        pass
    return None


def _make_overlap(parts, overlap_tokens=OVERLAP_TOKENS):
    overlap = []
    running = 0
    for part in reversed(parts):
        t = _token_count(part)
        if running >= overlap_tokens:
            break
        overlap.insert(0, part)
        running += t
    return overlap


def _split_markdown_table_to_row_chunks(table_markdown: str, heading: str = "") -> list[str]:
    """
    Splits a markdown table into mini-tables, each containing:
    [Heading + Table Header Lines + a single data row]
    This lets retrieval match one specific row (e.g. one debt tranche, one
    quarter's figure) instead of competing against the whole table's text.
    """
    lines = [line.strip() for line in table_markdown.splitlines() if line.strip() and "|" in line]

    # Too short to split meaningfully — keep it intact as one chunk.
    if len(lines) < 3:
        return [f"### {heading}\n\n{table_markdown}".strip()] if heading else [table_markdown]

    header_lines = [lines[0], lines[1]]  # column headers + separator row
    header_block = "\n".join(header_lines)
    data_lines = lines[2:]

    row_chunks = []
    for row in data_lines:
        prefix = f"### {heading}\n" if heading else ""
        mini_table = f"{prefix}{header_block}\n{row}"
        row_chunks.append(mini_table)

    return row_chunks


def build_semantic_chunks(document, pdf_path: str = PDF_PATH) -> list[dict]:
    """
    Walk a docling `document` and produce a list of chunk dicts:
    {"content": str, "metadata": {...}}

    Text chunks: {chunk_id, source_filename, doc_title, page_number, section,
    contains_table=False, chunk_kind="text", token_count, raw_chunk,
    full_table_markdown=None}

    Table row chunks: {chunk_id, source_filename, doc_title, page_number, section,
    contains_table=True, chunk_kind="table_row", table_id, row_index,
    token_count, raw_chunk, full_table_markdown}
    """
    source_filename = Path(pdf_path).name if isinstance(pdf_path, str) else "unknown.pdf"
    doc_title = source_filename

    semantic_chunks = []
    chunk_counter = 0
    table_counter = 0

    def flush_chunk(parts, section, contains_table=False, page_number=None, raw_table=None):
        nonlocal chunk_counter
        content = "\n\n".join(parts).strip()
        if not content:
            return []

        semantic_chunks.append({
            "content": content,
            "metadata": {
                "chunk_id": chunk_counter,
                "source_filename": source_filename,
                "doc_title": doc_title,
                "page_number": page_number,
                "section": section,
                "contains_table": contains_table,
                "chunk_kind": "text",
                "token_count": _token_count(content),
                "raw_chunk": raw_table if raw_table is not None else content,
                "full_table_markdown": None,
            }
        })
        chunk_counter += 1
        return _make_overlap(parts)

    current_parts: list[str] = []
    current_tokens = 0
    current_page = None
    current_section = "UNKNOWN"
    last_heading = None

    for item, level in document.iterate_items():
        item_type = type(item).__name__

        page = _extract_page_number(item)
        if page is not None:
            current_page = page

        # --- Headings ---
        if item_type in ("SectionHeaderItem", "TitleItem"):
            heading = _clean_text(getattr(item, "text", ""))
            if heading:
                if current_parts:
                    overlap = flush_chunk(current_parts, current_section, False, current_page)
                    current_parts = overlap
                    current_tokens = _token_count("\n\n".join(current_parts))
                current_section = heading
                last_heading = heading
            continue

        # --- Tables: split into per-row mini-table chunks ---
        if item_type == "TableItem":
            try:
                table_content = item.export_to_markdown()
            except Exception:
                table_content = str(item)
            table_content = table_content.strip()
            if not table_content:
                continue

            if current_parts:
                flush_chunk(current_parts, current_section, False, current_page)
                current_parts = []
                current_tokens = 0

            context_heading = last_heading if last_heading else current_section

            row_mini_tables = _split_markdown_table_to_row_chunks(table_content, context_heading)
            table_unique_id = f"tbl_pg{current_page}_{table_counter}"
            table_counter += 1

            for row_idx, row_text in enumerate(row_mini_tables):
                semantic_chunks.append({
                    "content": row_text,
                    "metadata": {
                        "chunk_id": chunk_counter,
                        "source_filename": source_filename,
                        "doc_title": doc_title,
                        "page_number": current_page,
                        "section": current_section,
                        "contains_table": True,
                        "chunk_kind": "table_row",
                        "table_id": table_unique_id,
                        "row_index": row_idx,
                        "token_count": _token_count(row_text),
                        "raw_chunk": row_text,
                        "full_table_markdown": table_content,
                    }
                })
                chunk_counter += 1

            continue

        # --- Text ---
        if hasattr(item, "text") and item.text:
            content = _clean_text(item.text)
            if not content:
                continue

            content_tokens = _token_count(content)

            if current_tokens + content_tokens > MAX_TOKENS:
                overlap = flush_chunk(current_parts, current_section, False, current_page)
                current_parts = overlap
                current_tokens = _token_count("\n\n".join(current_parts))

            if not current_parts and last_heading:
                current_parts.append(last_heading)
                current_tokens += _token_count(last_heading)

            current_parts.append(content)
            current_tokens += content_tokens

    # Final flush
    if current_parts:
        flush_chunk(current_parts, current_section, False, current_page)

    return semantic_chunks