import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Anchor to the repo root (two levels up from this file, adjust if config.py
# lives somewhere else), NOT the process's working directory — so it's
# consistent regardless of which folder api/ or web/server/ get launched from.

# Updated: resolves to .../FinSage
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE_ROOT = os.getenv("CACHE_ROOT", os.path.join(_REPO_ROOT, "cache"))
UPLOAD_ROOT = os.getenv("UPLOAD_ROOT", os.path.join(_REPO_ROOT, "data", "uploads"))

os.makedirs(CACHE_ROOT, exist_ok=True)
os.makedirs(UPLOAD_ROOT, exist_ok=True)


def get_document_paths(document_id: str) -> dict:
    """
    Returns all filesystem paths scoped to one document, keyed by its
    UUID (from the `documents` table). Replaces the old fixed
    PDF_PATH/CACHE_DIR globals — every document gets its own folder,
    isolated from every other document and every other user.
    """
    doc_cache_dir = os.path.join(CACHE_ROOT, str(document_id))
    return {
        "pdf_path": os.path.join(UPLOAD_ROOT, f"{document_id}.pdf"),
        "cache_dir": doc_cache_dir,
        "docling_json": os.path.join(doc_cache_dir, "parsed.json"),
        "docs_json": os.path.join(doc_cache_dir, "docs.json"),
        "semantic_cache_json": os.path.join(doc_cache_dir, "semantic_cache.json"),
    }

def get_conversation_paths(conversation_id: str) -> dict:
    """
    Returns paths scoped to one conversation — currently just the semantic
    cache, since retrieval (BM25 + Qdrant filter) is built fresh per request
    from whichever documents are attached to the conversation, not cached
    to disk the way the semantic cache is.
    """
    conv_cache_dir = os.path.join(CACHE_ROOT, "conversations", str(conversation_id))
    return {
        "cache_dir": conv_cache_dir,
        "semantic_cache_json": os.path.join(conv_cache_dir, "semantic_cache.json"),
    }


# os.makedirs(CACHE_DIR, exist_ok=True)

# # name the cache file after the source PDF, so different filings get separate caches
# _pdf_filename = os.path.basename(PDF_PATH)
# CACHE_PATH = os.path.join(CACHE_DIR, os.path.splitext(_pdf_filename)[0] + ".json")
# DOCS_CACHE_PATH = os.path.join(CACHE_DIR, os.path.splitext(_pdf_filename)[0] + "_docs.json")
# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
MAX_TOKENS = 450
OVERLAP_TOKENS = 50
MERGE_PEER_CHUNKS = True

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
BM25_K = 30
VECTOR_K = 30
HYBRID_WEIGHTS = [0.4, 0.6]  # [BM25, vector] — tilt toward semantic, keyword still counts
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
RERANK_TOP_N = 25
DEDUP_SIMILARITY_THRESHOLD = 0.85

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY_FALLBACK = os.environ.get("GOOGLE_API_KEY_FALLBACK")
LLM_MODEL = "gemini-3.1-flash-lite"
LLM_TEMPERATURE = 0

JINA_API_KEY = os.getenv("JINA_API_KEY")

# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "finance_rag_exp_2"

# ---------------------------------------------------------------------------
# Observability (LangSmith)
# ---------------------------------------------------------------------------
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "finsage-rag")


LOG_DIR = os.path.join(_REPO_ROOT, "logs")
LATENCY_LOG_PATH = os.path.join(LOG_DIR, "latency.csv")

# Semantic cache — threshold/enabled flag are still global settings (apply to
# every document equally); the actual cache FILE path is per-document now,
# from get_document_paths()["semantic_cache_json"], not a fixed constant here.
SEMANTIC_CACHE_THRESHOLD = 0.92
SEMANTIC_CACHE_ENABLED = True

MAX_HISTORY_TURNS = 5

GUARDRAILS_ENABLED = True