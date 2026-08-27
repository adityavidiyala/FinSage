import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from finance_rag.pipeline import build_index

if __name__ == "__main__":
    print("Starting ingestion: parse -> chunk -> summarize tables -> embed -> upsert to Qdrant...")
    build_index()
    print("Ingestion complete.")