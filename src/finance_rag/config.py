import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PDF_PATH = os.getenv("PDF_PATH", r"data\raw\Google 10Q.pdf")
CACHE_DIR = os.getenv("CACHE_DIR", r"cache\google_10Q_parsed")

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


LOG_DIR = "logs"
LATENCY_LOG_PATH = os.path.join(LOG_DIR, "latency.csv")

# Semantic cache
SEMANTIC_CACHE_PATH = os.path.join(CACHE_DIR, "semantic_cache.json")
SEMANTIC_CACHE_THRESHOLD = 0.92
SEMANTIC_CACHE_ENABLED = True

MAX_HISTORY_TURNS = 5

GUARDRAILS_ENABLED = True