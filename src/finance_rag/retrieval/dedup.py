"""
Deduplicates near-identical retrieved chunks (e.g. the same table appearing
in both a table-summary chunk and an adjacent text chunk) before generation.
"""

from difflib import SequenceMatcher

from finance_rag.config import DEDUP_SIMILARITY_THRESHOLD


def dedup_chunks(docs: list, similarity_threshold: float = DEDUP_SIMILARITY_THRESHOLD) -> list:
    """
    Keeps the first occurrence of each chunk; drops any subsequent chunk whose
    page_content is more than `similarity_threshold` similar to one already kept.
    """
    kept = []
    for doc in docs:
        is_dup = False
        for kept_doc in kept:
            ratio = SequenceMatcher(None, doc.page_content, kept_doc.page_content).ratio()
            if ratio > similarity_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(doc)
    return kept
