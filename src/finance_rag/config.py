import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PDF_PATH = os.getenv("PDF_PATH", "data/raw/walmart_10q.pdf")
CACHE_DIR = os.getenv("CACHE_DIR", "cache/walmart_10Q_parsed")

os.makedirs(CACHE_DIR, exist_ok=True)

# name the cache file after the source PDF, so different filings get separate caches
_pdf_filename = os.path.basename(PDF_PATH)
CACHE_PATH = os.path.join(CACHE_DIR, os.path.splitext(_pdf_filename)[0] + ".json")
DOCS_CACHE_PATH = os.path.join(CACHE_DIR, os.path.splitext(_pdf_filename)[0] + "_docs.json")
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
BM25_K = 10
VECTOR_K = 10
HYBRID_WEIGHTS = [0.4, 0.6]  # [BM25, vector] — tilt toward semantic, keyword still counts
RERANK_MODEL = "BAAI/bge-reranker-base"
RERANK_TOP_N = 10
DEDUP_SIMILARITY_THRESHOLD = 0.85

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL = "gemini-3.1-flash-lite"
LLM_TEMPERATURE = 0

# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "finance_rag"

# ---------------------------------------------------------------------------
# Observability (LangSmith)
# ---------------------------------------------------------------------------
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "finsage-rag")


LOG_DIR = "logs"
LATENCY_LOG_PATH = os.path.join(LOG_DIR, "latency.csv")