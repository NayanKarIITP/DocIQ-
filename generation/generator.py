"""generation/generator.py"""
from typing import Generator
import google.generativeai as genai
from loguru import logger
from config import settings

SYSTEM_PROMPT = """You are a precise expert AI assistant. Answer ONLY from the provided context.
After every factual claim cite the source as [chunk_id].
If context is insufficient say so. Never hallucinate.
End with: Sources: [id1], [id2]"""

class RAGGenerator:
    def __init__(self):
        genai.configure(api_key=settings.gemini_api_key)
        # self.model = genai.GenerativeModel(settings.gemini_model)
        self.model = genai.GenerativeModel(f"models/{settings.gemini_model}")
        self.conversation_history = []
        logger.info(f"Generator ready — model: {settings.gemini_model}")

    def generate(self, question: str, chunks: list[dict]) -> str:
        prompt = self._build_prompt(question, chunks)
        try:
            response = self.model.generate_content(prompt)
            answer = response.text
            self._update_history(question, answer)
            return answer
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error generating answer: {e}"

    def generate_stream(self, question: str, chunks: list[dict]) -> Generator[str, None, None]:
        prompt = self._build_prompt(question, chunks)
        full_answer = []
        try:
            for chunk in self.model.generate_content(prompt, stream=True):
                token = chunk.text
                if token:
                    full_answer.append(token)
                    yield token
            self._update_history(question, "".join(full_answer))
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"\n\nError: {e}"

    def reset_history(self):
        self.conversation_history = []

    def _build_prompt(self, question: str, chunks: list[dict]) -> str:
        return (f"{SYSTEM_PROMPT}\n\n"
                f"CONTEXT CHUNKS:\n{self._format_context(chunks)}\n\n"
                f"QUESTION: {question}")

    def _format_context(self, chunks: list[dict]) -> str:
        parts = []
        for c in chunks:
            cid   = c.get("chunk_id", "unknown")[:8]
            ctype = c.get("chunk_type", "text")
            fname = c.get("filename", "")
            page  = c.get("page_number", "?")
            score = c.get("rerank_score", c.get("score", 0))
            text  = c.get("content", "").strip()
            parts.append(f"[{cid}] {ctype.upper()} | {fname} p.{page} | score={score:.2f}\n{text}")
        return "\n\n---\n\n".join(parts)

    def _update_history(self, question, answer):
        self.conversation_history.append({"role": "user",  "parts": [question]})
        self.conversation_history.append({"role": "model", "parts": [answer]})
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
