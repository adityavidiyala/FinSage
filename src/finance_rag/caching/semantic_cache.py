import json
import os
import numpy as np
from finance_rag.indexing.embeddings import embeddings
from datetime import datetime, timezone
from finance_rag.config import SEMANTIC_CACHE_THRESHOLD


def load_cache(cache_path: str) -> list[dict]:
    if not os.path.exists(cache_path):
        return []
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(entries: list[dict], cache_path: str) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def find_cached_answer(question: str, cache_entries: list[dict]) -> dict | None:
    if not cache_entries:
        return None

    query_embedding = embeddings.embed_query(question)

    best_entry = None
    best_score = -1.0
    for entry in cache_entries:
        score = _cosine_similarity(query_embedding, entry["embedding"])
        if score > best_score:
            best_score = score
            best_entry = entry
    print(f"[cache debug] best similarity = {best_score:.4f} (threshold = {SEMANTIC_CACHE_THRESHOLD})")  # temporary

    if best_score >= SEMANTIC_CACHE_THRESHOLD:
        return {**best_entry, "cache_similarity": best_score}


def store_answer(question: str, answer: str, citations: list[dict], cache_entries: list[dict]) -> None:
    entry = {
        "question": question,
        "embedding": embeddings.embed_query(question),
        "answer": answer,
        "citations": citations,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    cache_entries.append(entry)