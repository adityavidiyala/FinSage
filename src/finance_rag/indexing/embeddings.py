"""
Shared embedding model instance, used for both indexing (Qdrant upserts)
and querying (retrieval-time embedding of the user's question).
"""

from langchain_huggingface import HuggingFaceEmbeddings

from finance_rag.config import EMBED_MODEL

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)