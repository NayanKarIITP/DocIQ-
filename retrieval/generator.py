"""
generation/generator.py
────────────────────────
Generates answers from retrieved chunks using Gemini.
Uses google.genai (new SDK, not deprecated google.generativeai).
"""

from typing import Generator

import google.genai as genai
import google.genai.types as genai_types
from loguru import logger

from config import settings


SYSTEM_PROMPT = """You are a precise, expert AI assistant that answers questions using ONLY the provided context chunks.

RULES:
1. Answer ONLY from the provided context. Never use outside knowledge.
2. After every factual claim, cite the source using [chunk_id] format.
3. If the context doesn't contain enough information, say: "The provided documents don't contain sufficient information to answer this question."
4. For tables, refer to specific rows/columns.
5. For images, reference what was described in the image.
6. Be concise but complete. Use bullet points for lists.
7. If asked about something not in context, say so clearly.

FORMAT:
- Use markdown for structure
- Cite inline: "Revenue grew 12% in Q3 [a3f2b1c4]"
- End with: "Sources: [id1], [id2], ..." listing all cited chunks
"""


class RAGGenerator:
    """Generates answers with Gemini, streaming + citations."""

    def __init__(self):
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model
        self.conversation_history = []

    def generate(self, question: str, chunks: list[dict]) -> str:
        """Non-streaming generation. Returns complete answer string."""
        prompt = self._build_prompt(question, chunks)
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,
                ),
            )
            answer = response.text
            self._update_history(question, answer)
            return answer
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error generating answer: {e}"

    def generate_stream(
        self, question: str, chunks: list[dict]
    ) -> Generator[str, None, None]:
        """Streaming generation — yields tokens as they arrive."""
        prompt = self._build_prompt(question, chunks)
        full_answer = []

        try:
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,
                ),
            ):
                token = chunk.text
                if token:
                    full_answer.append(token)
                    yield token

            self._update_history(question, "".join(full_answer))

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"\n\nError: {e}"

    def reset_history(self) -> None:
        """Clear conversation memory."""
        self.conversation_history = []

    def _build_prompt(self, question: str, chunks: list[dict]) -> str:
        context_block = self._format_context(chunks)
        return (
            f"CONTEXT CHUNKS:\n{context_block}\n\n"
            f"QUESTION: {question}"
        )

    def _format_context(self, chunks: list[dict]) -> str:
        parts = []
        for c in chunks:
            chunk_id = c.get("chunk_id", "unknown")[:8]
            chunk_type = c.get("chunk_type", "text")
            filename = c.get("filename", "")
            page = c.get("page_number", "?")
            content = c.get("content", "").strip()
            score = c.get("rerank_score", c.get("score", 0))
            header = f"[{chunk_id}] {chunk_type.upper()} | {filename} p.{page} | relevance={score:.2f}"
            parts.append(f"{header}\n{content}")
        return "\n\n---\n\n".join(parts)

    def _update_history(self, question: str, answer: str) -> None:
        self.conversation_history.append({"role": "user", "parts": [question]})
        self.conversation_history.append({"role": "model", "parts": [answer]})
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]