"""
Offline evaluation: runs every question in test_set.json through every
combination of (prompting strategy x LLM provider), scores each answer
with a simple keyword-match heuristic, and prints/saves a comparison
table.

This is the core deliverable of the project - a measured answer to
"which prompting strategy actually works better," not a guess.

Run from the project root:
    python -m evaluation.evaluate
"""
import json
import re
import time
from pathlib import Path

import pandas as pd

from app.prompts import Strategy, build_prompt
from app.providers import Provider, call_llm
from app.vectorstore import get_client, search

TEST_SET_PATH = Path(__file__).resolve().parent / "test_set.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results.csv"


def keyword_score(answer: str, expected_keywords: list[str]) -> float:
    """
    Simple, transparent scoring: fraction of expected keywords found in
    the answer (case-insensitive). Not perfect, but explainable - which
    matters more here than a fancier metric you can't fully justify.
    """
    if not expected_keywords:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if re.search(kw.lower(), answer_lower))
    return hits / len(expected_keywords)


def run_evaluation():
    with open(TEST_SET_PATH) as f:
        test_cases = json.load(f)

    client = get_client()
    rows = []

    for strategy in Strategy:
        for provider in Provider:
            for case in test_cases:
                question = case["question"]
                expected = case["expected_keywords"]

                retrieved = search(client, question)
                if not retrieved:
                    print(f"[skip] No indexed docs for question: {question}")
                    continue

                prompt = build_prompt(strategy, question, retrieved)
                result = None
                for attempt in range(4):
                    try:
                        result = call_llm(prompt, provider)
                        break
                    except Exception as e:
                        wait = 15 * (attempt + 1)  # 15s, 30s, 45s, 60s
                        print(f"[retry] {provider}/{strategy} attempt {attempt+1}: {e}")
                        time.sleep(wait)
                if result is None:
                    print(f"[error] {provider}/{strategy}: failed after retries, skipping")
                    continue

                # Pace requests to stay under Gemini's free-tier rate limit
                if provider == Provider.GEMINI:
                    time.sleep(13)  # ~5 requests/minute allowed → 1 every 12+ sec

                score = keyword_score(result.text, expected)

                rows.append(
                    {
                        "strategy": strategy.value,
                        "provider": result.provider,
                        "model": result.model,
                        "question": question,
                        "score": score,
                        "latency_seconds": result.latency_seconds,
                        "estimated_cost_usd": result.estimated_cost_usd,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        print("No results generated - check that /ingest has been run "
              "and API keys are set in .env")
        return

    summary = (
        df.groupby(["strategy", "provider"])
        .agg(
            avg_score=("score", "mean"),
            avg_latency_s=("latency_seconds", "mean"),
            avg_cost_usd=("estimated_cost_usd", "mean"),
        )
        .round(3)
        .reset_index()
        .sort_values("avg_score", ascending=False)
    )

    df.to_csv(RESULTS_PATH, index=False)
    print("\n=== Per-question results saved to evaluation/results.csv ===\n")
    print("=== Summary: strategy x provider comparison ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    run_evaluation()
