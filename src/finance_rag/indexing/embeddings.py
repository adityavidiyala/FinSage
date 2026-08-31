"""
Shared embedding model instance, used for both indexing (Qdrant upserts)
and querying (retrieval-time embedding of the user's question).

Cloud-hosted via Jina AI (jina-embeddings-v3) instead of local
HuggingFaceEmbeddings — local CPU embedding took 6+ minutes for ~655 chunks;
the cloud API removes that bottleneck. Note: this changes vector dimensionality
(1024-dim vs. bge-base-en-v1.5's 768-dim) — an existing Qdrant collection built
with the old model cannot be reused; a new collection is required.
"""

from langchain_community.embeddings import JinaEmbeddings

from finance_rag.config import JINA_API_KEY

embeddings = JinaEmbeddings(
    model_name="jina-embeddings-v3",
    jina_api_key=JINA_API_KEY,
)