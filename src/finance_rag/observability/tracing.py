"""
Local latency + token/cost logging: wraps a pipeline stage, timing it and
optionally recording token usage, writing one CSV row per call.
Complementary to LangSmith (Step 26) — this is data you own and can chart
without needing API access to your LangSmith project.
"""

import os
import csv
import time
from datetime import datetime, timezone
from contextlib import contextmanager

from finance_rag.config import LATENCY_LOG_PATH

_FIELDNAMES = ["timestamp", "question", "stage", "latency_ms", "input_tokens", "output_tokens", "total_tokens"]


def _ensure_log_file():
    if not os.path.exists(LATENCY_LOG_PATH):
        os.makedirs(os.path.dirname(LATENCY_LOG_PATH), exist_ok=True)
        with open(LATENCY_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=_FIELDNAMES).writeheader()


@contextmanager
def track_stage(question: str, stage: str):
    """
    Usage:
        with track_stage(question, "retrieval"):
            ...

    Or, to also log token usage (generation stage only):
        with track_stage(question, "generation") as extra:
            result = generate_answer(...)
            extra.update(result.get("usage", {}))
    """
    _ensure_log_file()
    start = time.perf_counter()
    extra: dict = {}
    try:
        yield extra
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "stage": stage,
            "latency_ms": round(latency_ms, 2),
            "input_tokens": extra.get("input_tokens", ""),
            "output_tokens": extra.get("output_tokens", ""),
            "total_tokens": extra.get("total_tokens", ""),
        }
        with open(LATENCY_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=_FIELDNAMES).writerow(row)