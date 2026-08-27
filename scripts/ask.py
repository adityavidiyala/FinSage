import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from finance_rag.pipeline import build_retriever, answer_query

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
    print(result["answer"])

    if result["citations"]:
        print("\nSources:")
        for c in result["citations"]:
            print(f"  [{c['marker']}] {c['source_filename']} — page {c['page_number']}, section: {c['section']}")
    else:
        print("\n(No citations resolved.)")