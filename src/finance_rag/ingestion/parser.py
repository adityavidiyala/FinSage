"""
PDF parsing via docling, with local JSON caching so re-runs skip the slow parse step.
"""

from collections import Counter

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import DoclingDocument

from finance_rag.config import PDF_PATH, CACHE_PATH


class _CachedResult:
    """Wraps a loaded DoclingDocument so it matches converter.convert()'s return shape."""
    def __init__(self, document):
        self.document = document


def parse_document(pdf_path: str = PDF_PATH, cache_path: str = CACHE_PATH):
    """
    Parse a PDF with docling, or load a previously cached parse if one exists.
    Returns an object with a `.document` attribute (DoclingDocument).
    """
    import os

    if os.path.exists(cache_path):
        print(f"Loading cached parse from {cache_path} ...")
        doc = DoclingDocument.load_from_json(cache_path)
        print("Loaded from cache. Skipped re-parsing.")
        return _CachedResult(doc)

    print("No cache found. Running docling parse (this takes a while)...")

    pdf_options = PdfPipelineOptions()
    pdf_options.do_ocr = False

    converter = DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=pdf_options)}
    )

    result = converter.convert(pdf_path)
    result.document.save_as_json(cache_path)
    print(f"Parse complete. Cached to {cache_path}")

    return result


def split_text_and_tables(document) -> tuple[list, list]:
    """
    Walk the parsed document and split items into text-like items vs table items.
    Returns (text_items, table_items).
    """
    text_items = []
    table_items = []

    for item, level in document.iterate_items():
        if type(item).__name__ == "TableItem":
            table_items.append(item)
        else:
            text_items.append(item)

    return text_items, table_items


def item_type_counts(document) -> Counter:
    """Quick diagnostic: counts of each docling item type in the document."""
    counts = Counter()
    for item, level in document.iterate_items():
        counts[type(item).__name__] += 1
    return counts