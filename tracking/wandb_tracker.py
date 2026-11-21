"""
tracking/wandb_tracker.py
──────────────────────────
Weights & Biases experiment tracking for DocIQ.

What gets tracked:
  - Every query: latency, chunks retrieved, rerank scores
  - Every feedback: rating, question, answer quality
  - RAGAS eval scores over time
  - Model configuration per run

Why this matters:
  W&B tracking appears in 68% of ML Engineer JDs.
  A dashboard screenshot in your README signals production
  ML thinking — not just "it works on my machine".

Setup (one-time):
  1. pip install wandb
  2. wandb login    (paste your API key from wandb.ai)
  3. Set WANDB_PROJECT=dociq in .env

Usage:
  from tracking.wandb_tracker import tracker
  tracker.log_query(question, chunks, latency, rating)
"""

import os
import time
from functools import wraps
from typing import Optional

from loguru import logger


class WandBTracker:
    """
    Lightweight W&B wrapper. Gracefully degrades if W&B
    is not installed or not logged in — app keeps working.
    """

    def __init__(self):
        self._enabled = False
        self._run = None
        self._init()

    def _init(self):
        """Initialize W&B. Silently disable if unavailable."""
        try:
            import wandb
            from config import settings

            project = os.getenv("WANDB_PROJECT", "dociq")
            api_key = os.getenv("WANDB_API_KEY", "")

            if not api_key:
                logger.warning(
                    "W&B disabled — set WANDB_API_KEY in .env to enable tracking. "
                    "Get free key at: https://wandb.ai/authorize"
                )
                return

            self._run = wandb.init(
                project=project,
                config={
                    "gemini_model":       settings.gemini_model,
                    "embed_model":        settings.gemini_embed_model,
                    "top_k_retrieval":    settings.top_k_retrieval,
                    "top_k_rerank":       settings.top_k_rerank,
                    "chunk_size":         settings.chunk_size,
                    "chunk_overlap":      settings.chunk_overlap,
                    "embedding_dim":      settings.embedding_dim,
                },
                reinit=True,
            )
            self._enabled = True
            logger.success(
                f"W&B tracking enabled — project: {project} "
                f"| dashboard: {self._run.url}"
            )

        except ImportError:
            logger.warning(
                "W&B not installed — run: pip install wandb. "
                "App will work without tracking."
            )
        except Exception as e:
            logger.warning(f"W&B init failed (continuing without tracking): {e}")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def log_query(
        self,
        question: str,
        chunks: list[dict],
        latency_ms: float,
        use_hyde: bool = True,
        use_rerank: bool = True,
        answer_length: int = 0,
    ):
        """Log a single query event."""
        if not self._enabled:
            return
        try:
            import wandb

            top_rerank = max(
                (c.get("rerank_score", c.get("score", 0)) for c in chunks),
                default=0.0,
            )
            avg_score = (
                sum(c.get("rerank_score", c.get("score", 0)) for c in chunks)
                / len(chunks)
                if chunks else 0.0
            )

            # Count chunk types
            type_counts = {"text": 0, "image": 0, "table": 0}
            for c in chunks:
                t = c.get("chunk_type", "text")
                type_counts[t] = type_counts.get(t, 0) + 1

            wandb.log({
                "query/latency_ms":          round(latency_ms, 1),
                "query/chunks_retrieved":    len(chunks),
                "query/top_rerank_score":    round(top_rerank, 4),
                "query/avg_rerank_score":    round(avg_score, 4),
                "query/hyde_enabled":        int(use_hyde),
                "query/rerank_enabled":      int(use_rerank),
                "query/answer_length_chars": answer_length,
                "query/chunks_text":         type_counts["text"],
                "query/chunks_image":        type_counts["image"],
                "query/chunks_table":        type_counts["table"],
                "query/question_length":     len(question),
            })
        except Exception as e:
            logger.debug(f"W&B log_query failed: {e}")

    def log_feedback(
        self,
        question: str,
        rating: int,
        has_correction: bool = False,
    ):
        """Log user feedback (thumbs up/down)."""
        if not self._enabled:
            return
        try:
            import wandb
            wandb.log({
                "feedback/rating":          rating,        # 1 or -1
                "feedback/is_positive":     int(rating > 0),
                "feedback/has_correction":  int(has_correction),
                "feedback/question_length": len(question),
            })
        except Exception as e:
            logger.debug(f"W&B log_feedback failed: {e}")

    def log_ingest(
        self,
        filename: str,
        n_text: int,
        n_image: int,
        n_table: int,
        n_chunks: int,
        latency_ms: float,
    ):
        """Log a document ingestion event."""
        if not self._enabled:
            return
        try:
            import wandb
            wandb.log({
                "ingest/n_text_chunks":   n_text,
                "ingest/n_image_chunks":  n_image,
                "ingest/n_table_chunks":  n_table,
                "ingest/total_chunks":    n_chunks,
                "ingest/latency_ms":      round(latency_ms, 1),
                "ingest/filename_length": len(filename),
            })
        except Exception as e:
            logger.debug(f"W&B log_ingest failed: {e}")

    def log_ragas(self, scores: dict):
        """
        Log RAGAS evaluation scores.
        Call this after running evals/evaluate.py.

        Args:
            scores: {
                "faithfulness": 0.94,
                "answer_relevancy": 0.91,
                "context_precision": 0.88,
                "context_recall": 0.85,
            }
        """
        if not self._enabled:
            return
        try:
            import wandb
            wandb.log({f"eval/{k}": v for k, v in scores.items()})
            logger.info(f"W&B: logged RAGAS scores: {scores}")
        except Exception as e:
            logger.debug(f"W&B log_ragas failed: {e}")

    def log_dpo_training(
        self,
        n_pairs: int,
        train_loss: float,
        eval_loss: Optional[float] = None,
        epoch: int = 1,
    ):
        """Log DPO fine-tuning metrics."""
        if not self._enabled:
            return
        try:
            import wandb
            data = {
                "dpo/n_preference_pairs": n_pairs,
                "dpo/train_loss":         round(train_loss, 4),
                "dpo/epoch":              epoch,
            }
            if eval_loss is not None:
                data["dpo/eval_loss"] = round(eval_loss, 4)
            wandb.log(data)
        except Exception as e:
            logger.debug(f"W&B log_dpo_training failed: {e}")

    def finish(self):
        """Close the W&B run cleanly."""
        if self._enabled and self._run:
            try:
                self._run.finish()
            except Exception:
                pass


# ── Singleton ─────────────────────────────────────────────────────────────
# Import this in api/main.py and call tracker.log_*() wherever needed
tracker = WandBTracker()


# ── Decorator for automatic query tracking ────────────────────────────────
def track_query(func):
    """
    Decorator that automatically tracks latency and chunks for any
    function that takes (question, chunks, ...) and returns a string.

    Usage:
        @track_query
        def my_generate_function(question, chunks):
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        latency_ms = (time.time() - start) * 1000

        # Try to extract question and chunks from args
        question = args[0] if args else ""
        chunks = args[1] if len(args) > 1 else []

        tracker.log_query(
            question=str(question),
            chunks=chunks if isinstance(chunks, list) else [],
            latency_ms=latency_ms,
            answer_length=len(str(result)),
        )
        return result
    return wrapper
