"""
generation/prompt_engineer.py
──────────────────────────────
Advanced prompt engineering techniques for DocIQ.

Techniques implemented:
  1. Chain-of-Thought (CoT)     — step-by-step reasoning before answering
  2. Few-shot prompting          — example Q&A pairs before the real question
  3. Self-consistency            — generate N answers, return the best one
  4. Structured output (Pydantic)— typed JSON responses instead of free text
  5. Hypothetical Document Embedding (HyDE) — already in retriever.py

Why this matters:
  "Prompt engineering" appears in virtually every GenAI JD.
  Having NAMED TECHNIQUES you can discuss shows depth — not just
  "I tweaked the system prompt until it worked."

Usage:
    from generation.prompt_engineer import PromptEngineer
    pe = PromptEngineer()

    # Chain-of-thought
    answer = pe.generate_cot(question, chunks)

    # Structured output
    result = pe.generate_structured(question, chunks)
    print(result.confidence)  # 0.87
    print(result.citations)   # ["a3f2b1c4", "b7e9d2f1"]
"""

from __future__ import annotations
import json
import time
from typing import Optional

import google.generativeai as genai
from loguru import logger
from pydantic import BaseModel, Field

from config import settings


# ── Pydantic schema for structured output ─────────────────────────────────

class RAGAnswer(BaseModel):
    """Structured answer from the RAG system."""
    answer: str = Field(description="The complete answer to the question")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score 0-1 based on context quality"
    )
    citations: list[str] = Field(
        description="List of chunk_ids cited in the answer"
    )
    needs_more_context: bool = Field(
        description="True if the documents don't fully answer the question"
    )
    reasoning: str = Field(
        description="Brief explanation of how the answer was derived"
    )


# ── Prompt templates ──────────────────────────────────────────────────────

COT_TEMPLATE = """You are an expert document analyst. Think step by step.

STEP 1 — UNDERSTAND the question:
What exactly is being asked? What type of answer is needed (fact, explanation, comparison)?

STEP 2 — IDENTIFY relevant context:
Which chunks contain information relevant to this question?

STEP 3 — REASON from evidence:
What does the evidence say? Are there contradictions? Is the information complete?

STEP 4 — FORMULATE answer:
State your answer clearly with inline citations [chunk_id].

STEP 5 — VERIFY:
Does the answer directly address the question? Is every claim supported?

---
CONTEXT CHUNKS:
{context}

QUESTION: {question}

Begin your step-by-step reasoning:"""


FEW_SHOT_TEMPLATE = """You are a precise document Q&A assistant.
Answer ONLY from the provided context. Always cite sources as [chunk_id].

--- EXAMPLES ---

Example 1:
Context: [a1b2c3d4] SQL Notes p.3 | A primary key uniquely identifies each record in a table.
Question: What is a primary key?
Answer: A primary key uniquely identifies each record in a table [a1b2c3d4].
Sources: [a1b2c3d4]

Example 2:
Context: [e5f6g7h8] ML Guide p.12 | Supervised learning uses labeled training data.
Question: What was the revenue in Q3?
Answer: The provided documents don't contain information about Q3 revenue.
Sources: none

--- YOUR TURN ---

Context:
{context}

Question: {question}

Answer:"""


STRUCTURED_TEMPLATE = """Analyze the question and context, then respond with a JSON object.

Context chunks:
{context}

Question: {question}

Respond with ONLY valid JSON in this exact format (no markdown, no explanation):
{{
  "answer": "your complete answer with inline [chunk_id] citations",
  "confidence": 0.85,
  "citations": ["chunk_id_1", "chunk_id_2"],
  "needs_more_context": false,
  "reasoning": "brief explanation of how you derived the answer"
}}

Rules:
- confidence: 0.9+ if context directly answers, 0.5-0.9 if partial, 0.0-0.5 if insufficient
- needs_more_context: true if the question cannot be fully answered from context
- citations: only chunk_ids that you actually used in your answer"""


# ── Prompt Engineer class ─────────────────────────────────────────────────

class PromptEngineer:
    """
    Advanced prompt engineering for DocIQ.
    Uses the same Gemini model as the main generator.
    """

    def __init__(self):
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model)
        logger.info("PromptEngineer ready")

    # ── 1. Chain-of-Thought ───────────────────────────────────────────────

    def generate_cot(self, question: str, chunks: list[dict]) -> str:
        """
        Chain-of-thought prompting.
        Forces the model to reason through steps before answering.
        Reduces hallucination by 20-30% on complex questions.
        """
        context = self._format_context(chunks)
        prompt = COT_TEMPLATE.format(context=context, question=question)

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"CoT generation failed: {e}")
            return f"Error: {e}"

    # ── 2. Few-shot prompting ────────────────────────────────────────────

    def generate_few_shot(
        self,
        question: str,
        chunks: list[dict],
        custom_examples: list[dict] | None = None,
    ) -> str:
        """
        Few-shot prompting with example Q&A pairs.
        Forces consistent citation format and answer style.

        Args:
            custom_examples: list of {"question": ..., "answer": ..., "context": ...}
        """
        context = self._format_context(chunks)

        if custom_examples:
            # Build custom few-shot header
            examples = "\n\n".join(
                f"Example {i+1}:\n"
                f"Context: {ex.get('context', '')}\n"
                f"Question: {ex['question']}\n"
                f"Answer: {ex['answer']}"
                for i, ex in enumerate(custom_examples)
            )
            prompt = (
                f"Answer questions using ONLY the context. Cite as [chunk_id].\n\n"
                f"--- EXAMPLES ---\n{examples}\n\n"
                f"--- YOUR TURN ---\nContext:\n{context}\n\n"
                f"Question: {question}\n\nAnswer:"
            )
        else:
            prompt = FEW_SHOT_TEMPLATE.format(
                context=context, question=question
            )

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Few-shot generation failed: {e}")
            return f"Error: {e}"

    # ── 3. Self-consistency ──────────────────────────────────────────────

    def generate_self_consistent(
        self,
        question: str,
        chunks: list[dict],
        n_samples: int = 3,
    ) -> str:
        """
        Self-consistency prompting — generate N answers, return the best.

        How it works:
          1. Generate n_samples answers independently
          2. Pick the one that is most "average" in length
             (simple proxy for consensus — production version
              would use embedding similarity clustering)

        Research: Wang et al. (2022) showed self-consistency reduces
        reasoning errors by 17.9% on complex tasks.
        """
        answers = []
        for i in range(n_samples):
            try:
                context = self._format_context(chunks)
                prompt = FEW_SHOT_TEMPLATE.format(
                    context=context, question=question
                )
                response = self.model.generate_content(prompt)
                answers.append(response.text)
                time.sleep(0.5)  # gentle rate limiting
            except Exception as e:
                logger.warning(f"Sample {i+1} failed: {e}")

        if not answers:
            return "Generation failed"

        if len(answers) == 1:
            return answers[0]

        # Select the median-length answer as consensus proxy
        answers.sort(key=len)
        consensus = answers[len(answers) // 2]
        logger.info(
            f"Self-consistency: {len(answers)} samples, "
            f"lengths: {[len(a) for a in answers]}"
        )
        return consensus

    # ── 4. Structured output ─────────────────────────────────────────────

    def generate_structured(
        self,
        question: str,
        chunks: list[dict],
    ) -> RAGAnswer:
        """
        Structured output — returns a typed Pydantic object instead of a string.

        Use this when downstream code needs to:
          - Display a confidence score in the UI
          - Check if more context is needed
          - Extract citations programmatically
          - Route low-confidence answers for human review

        Returns:
            RAGAnswer(
                answer="...",
                confidence=0.94,
                citations=["a3f2b1c4"],
                needs_more_context=False,
                reasoning="..."
            )
        """
        context = self._format_context(chunks)
        prompt = STRUCTURED_TEMPLATE.format(
            context=context, question=question
        )

        try:
            response = self.model.generate_content(prompt)
            raw = response.text.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            data = json.loads(raw)
            return RAGAnswer(**data)

        except json.JSONDecodeError as e:
            logger.warning(f"Structured output JSON parse failed: {e}")
            # Fallback: return a basic structured response
            return RAGAnswer(
                answer=self.generate_few_shot(question, chunks),
                confidence=0.5,
                citations=[],
                needs_more_context=True,
                reasoning="Structured output parsing failed — using fallback",
            )
        except Exception as e:
            logger.error(f"Structured generation failed: {e}")
            return RAGAnswer(
                answer=f"Error: {e}",
                confidence=0.0,
                citations=[],
                needs_more_context=True,
                reasoning=str(e),
            )

    # ── Utility ──────────────────────────────────────────────────────────

    def _format_context(self, chunks: list[dict]) -> str:
        parts = []
        for c in chunks:
            cid   = c.get("chunk_id", "unknown")[:8]
            ctype = c.get("chunk_type", "text")
            fname = c.get("filename", "")
            page  = c.get("page_number", "?")
            text  = c.get("content", "").strip()
            score = c.get("rerank_score", c.get("score", 0))
            parts.append(
                f"[{cid}] {ctype.upper()} | {fname} p.{page} "
                f"| score={score:.2f}\n{text}"
            )
        return "\n\n---\n\n".join(parts)

    def compare_techniques(
        self,
        question: str,
        chunks: list[dict],
    ) -> dict:
        """
        Run all techniques on the same question and return results.
        Useful for README demo and interview demos.
        """
        logger.info(f"Comparing prompt techniques on: {question[:60]}")
        results = {}

        for name, fn in [
            ("few_shot",        lambda: self.generate_few_shot(question, chunks)),
            ("chain_of_thought", lambda: self.generate_cot(question, chunks)),
            ("structured",      lambda: self.generate_structured(question, chunks).answer),
        ]:
            try:
                start = time.time()
                results[name] = fn()
                results[f"{name}_latency_ms"] = round((time.time()-start)*1000, 1)
            except Exception as e:
                results[name] = f"Error: {e}"

        return results
