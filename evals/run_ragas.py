"""
RAGAS evaluation: retrieval quality (context precision, context recall) and
generation quality (faithfulness, answer relevancy) against the 40-question
eval set in eval_dataset.json.

Reuses a single retriever build across all 40 questions (see pipeline.build_retriever) —
NOT rebuilt per question, which is what the earlier retriever-reload fix in pipeline.py
was specifically for.
"""

import sys
import os
import json
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ---------------------------------------------------------------------------
# RAGAS import workaround: ragas has an internal dependency on
# langchain_community.chat_models.vertexai which isn't always installed.
# We never use VertexAI (we use Gemini via langchain_google_genai), so this
# only needs to exist well enough that ragas's import line doesn't crash.
# ---------------------------------------------------------------------------
fake_vertexai_module = types.ModuleType("langchain_community.chat_models.vertexai")

class ChatVertexAI:
    pass

fake_vertexai_module.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = fake_vertexai_module

# NOW it's safe to import ragas
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from finance_rag.pipeline import build_retriever, answer_query
from finance_rag.retrieval.dedup import dedup_chunks
from finance_rag.generation.llm import llm
from finance_rag.indexing.embeddings import embeddings

EVAL_DATASET_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Questions the filing genuinely doesn't answer — used for the answerable/
# unanswerable split, since blended answer_relevancy is misleading otherwise
# (a correct "I don't know" scores 0.0 on relevancy, dragging the average down).
NEGATIVE_CASE_QUESTIONS = [
    "What was Walmart's total marketing and advertising expenditure for the three months ended April 30, 2026?",
    "What long-term debt matured during the three months ended April 30, 2026, and what were the repayment amounts?",
    "How much did Walmart pay in dividends (cash flow statement) during the three months ended April 30, 2026 vs 2025?",
    "What tariff/trade-related risks does Walmart flag as affecting its business in the MD&A 'Recent Developments' section?",
    "According to the MD&A, what share of Walmart's U.S. imports come from China, Vietnam, Mexico, India, and Canada, and what fraction of total U.S. sales is imported?",
    "What was Walmart's total headcount or number of employees as of April 30, 2026?",
    "What credit rating agencies have rated Walmart's debt, and what were the specific ratings as of April 30, 2026?",
]


def load_eval_dataset() -> list[dict]:
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_retrieval_and_generation(eval_dataset: list[dict], retriever) -> list[dict]:
    """
    For each question: retrieve -> dedup -> take top 5 -> generate answer.
    Mutates and returns eval_dataset with "contexts" and "answer" added.
    """
    for i, item in enumerate(eval_dataset):
        print(f"[{i+1}/{len(eval_dataset)}] {item['question'][:70]}...")

        raw_results = retriever.invoke(item["question"])
        deduped = dedup_chunks(raw_results)[:5]  # cap at 5 contexts, same as notebook

        item["contexts"] = [doc.page_content for doc in deduped]

        result = answer_query(item["question"], retriever)
        item["answer"] = result["answer"]  # generate_answer now returns a dict — see Step 20

    return eval_dataset


def run_ragas_eval(eval_dataset: list[dict]):
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    custom_run_config = RunConfig(
        timeout=180,
        max_retries=3,
        max_wait=60,
        max_workers=2,  # keep concurrency low — Gemini free-tier rate limits
    )

    ragas_dataset = Dataset.from_dict({
        "question": [item["question"] for item in eval_dataset],
        "contexts": [item["contexts"] for item in eval_dataset],
        "answer": [item["answer"] for item in eval_dataset],
        "ground_truth": [item["ground_truth"] for item in eval_dataset],
    })

    # strictness=1: RAGAS's answer_relevancy default (strictness=3) requires multi-candidate
    # generation, which Gemini doesn't support cleanly here. strictness=1 trades a bit of
    # robustness for consistent, complete results — same fix applied in the notebook (session 3).
    answer_relevancy_single = AnswerRelevancy(strictness=1)

    results = evaluate(
        dataset=ragas_dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy_single],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=custom_run_config,
    )

    return results.to_pandas()


def summarize(results_df, eval_dataset: list[dict]):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "ragas_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")

    print(f"\nALL ({len(results_df)} questions):")
    print(f"  context_precision: {results_df['context_precision'].mean():.4f}")
    print(f"  context_recall:    {results_df['context_recall'].mean():.4f}")
    print(f"  faithfulness:      {results_df['faithfulness'].mean():.4f}")
    print(f"  answer_relevancy:  {results_df['answer_relevancy'].mean():.4f}")

    answerable_df = results_df[~results_df["user_input"].isin(NEGATIVE_CASE_QUESTIONS)]
    unanswerable_df = results_df[results_df["user_input"].isin(NEGATIVE_CASE_QUESTIONS)]

    print(f"\nANSWERABLE (n={len(answerable_df)}):")
    print(f"  faithfulness:      {answerable_df['faithfulness'].mean():.4f}")
    print(f"  answer_relevancy:  {answerable_df['answer_relevancy'].mean():.4f}")

    print(f"\nUNANSWERABLE/MISS (n={len(unanswerable_df)}):")
    print(f"  faithfulness:      {unanswerable_df['faithfulness'].mean():.4f}")
    print(f"  answer_relevancy:  {unanswerable_df['answer_relevancy'].mean():.4f}")


if __name__ == "__main__":
    eval_dataset = load_eval_dataset()[:3]
    print(f"Loaded {len(eval_dataset)} eval questions.\n")

    print("Building retriever once (BM25 + Qdrant + reranker)...")
    retriever = build_retriever()

    print("\nRunning retrieval + generation for all questions...")
    eval_dataset = run_retrieval_and_generation(eval_dataset, retriever)

    print("\nRunning RAGAS evaluation...")
    results_df = run_ragas_eval(eval_dataset)

    summarize(results_df, eval_dataset)