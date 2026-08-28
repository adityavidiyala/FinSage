import sys
import os
from finance_rag.pipeline import build_retriever, answer_query

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python scripts/ask.py "your question here"')
        sys.exit(1)

    question = sys.argv[1]

    print("Loading retriever (BM25 + Qdrant + reranker)...")
    retriever = build_retriever()

    print(f"\nQuestion: {question}\n")
    result = answer_query(question, retriever)

    print("Answer:")
    if result.get("from_cache"):
       print(f"[cache hit, similarity={result['cache_similarity']:.3f}]")

    print(result["answer"])

    if result["citations"]:
        print("\nSources:")
        for c in result["citations"]:
            print(f"  [{c['marker']}] {c['source_filename']} — page {c['page_number']}, section: {c['section']}")
    else:
        print("\n(No citations resolved.)")