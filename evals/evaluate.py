"""
evals/evaluate.py
──────────────────
Evaluate the RAG system using RAGAS metrics.

Metrics:
  - faithfulness      : does the answer stick to context? (no hallucination)
  - answer_relevancy  : does the answer address the question?
  - context_precision : were the retrieved chunks relevant?
  - context_recall    : were all relevant chunks found?

Usage:
    python -m evals.evaluate --questions evals/test_questions.json

Output:
    evals/results/report_<timestamp>.json
    evals/results/report_<timestamp>.csv
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from loguru import logger


SAMPLE_TEST_QUESTIONS = [
    {
        "question": "What is the main topic of the document?",
        "ground_truth": "The document covers the key subject matter outlined in its introduction."
    },
    {
        "question": "What are the key findings or conclusions?",
        "ground_truth": "The key findings are summarized in the conclusion section."
    },
    {
        "question": "What methodology was used?",
        "ground_truth": "The methodology is described in the methods section."
    },
]


def run_evaluation(
    test_questions: list[dict],
    api_url: str = "http://localhost:8000",
) -> dict:
    """
    Run RAGAS evaluation against the live API.

    Args:
        test_questions: list of {"question": ..., "ground_truth": ..., "contexts": [...]}
        api_url: base URL of the FastAPI backend
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    logger.info(f"Evaluating {len(test_questions)} questions...")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, item in enumerate(test_questions):
        q = item["question"]
        gt = item.get("ground_truth", "")

        logger.info(f"  [{i+1}/{len(test_questions)}] {q[:60]}...")

        # Query the API
        try:
            resp = requests.post(
                f"{api_url}/query/sync",
                json={"question": q, "use_hyde": True, "use_rerank": True},
                timeout=60,
            )
            if not resp.ok:
                logger.warning(f"API error for question {i}: {resp.text}")
                continue

            data = resp.json()
            answer = data["answer"]
            source_chunks = data.get("sources", [])

            # Get the actual context text (re-query to get content)
            # In production, store context text in the /query response
            context_texts = [
                f"[{s['chunk_id']}] {s['filename']} p.{s['page']}"
                for s in source_chunks
            ]

            questions.append(q)
            answers.append(answer)
            contexts.append(context_texts if context_texts else ["No context retrieved"])
            ground_truths.append(gt)

        except Exception as e:
            logger.error(f"Failed to evaluate question {i}: {e}")
            continue

    if not questions:
        logger.error("No questions evaluated successfully")
        return {}

    # Build RAGAS dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # Run RAGAS evaluation
    logger.info("Running RAGAS metrics...")
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    # Convert to dict
    scores = results.to_pandas().mean().to_dict()

    report = {
        "timestamp": datetime.now().isoformat(),
        "n_questions": len(questions),
        "scores": {k: round(float(v), 4) for k, v in scores.items()},
        "raw_results": results.to_pandas().to_dict(orient="records"),
    }

    # Save results
    output_dir = Path("evals/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(output_dir / f"report_{ts}.json", "w") as f:
        json.dump(report, f, indent=2)

    results.to_pandas().to_csv(output_dir / f"report_{ts}.csv", index=False)

    logger.success(f"Evaluation complete:")
    for metric, score in report["scores"].items():
        emoji = "✓" if score > 0.8 else "⚠" if score > 0.6 else "✗"
        logger.info(f"  {emoji} {metric}: {score:.3f}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        help="Path to JSON file with test questions",
    )
    args = parser.parse_args()

    if args.questions:
        with open(args.questions) as f:
            test_questions = json.load(f)
    else:
        logger.info("No test questions provided — using sample questions")
        test_questions = SAMPLE_TEST_QUESTIONS

    report = run_evaluation(test_questions)

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for metric, score in report.get("scores", {}).items():
        bar = "█" * int(score * 20)
        print(f"{metric:<25} {score:.3f}  {bar}")
