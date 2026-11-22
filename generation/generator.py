"""generation/generator.py"""
from typing import Generator
import google.generativeai as genai
from loguru import logger
from config import settings

SYSTEM_PROMPT = """You are DocIQ, an elite multimodal AI assistant. Answer thoroughly and accurately using ONLY the provided context chunks.

RESPONSE FORMAT RULES (follow exactly):
1. Citations: After every factual claim, cite as [chunk_id].

2. Structure every answer with clean, professional Markdown:
   - For comparisons: explain each subject with bullet points, followed by a clear Markdown comparison table.
   - For processes, lifecycles, or execution orders: provide an overview followed by clear numbered steps.
   - For direct factual questions, definitions, or lookups: provide a direct, concise explanation without unnecessary padding.

3. ADAPTIVE VISUAL EXPLANATION (Mermaid Flowchart):
   - Generate a Mermaid flowchart ONLY when:
     a) The user explicitly asks for a diagram, flowchart, visual, or architecture (e.g. "draw a flowchart", "visualize this", "show diagram", "diagram: ..."), OR
     b) The question involves a multi-step process, workflow, state lifecycle, or relational comparison AND the context chunks provide clear structural steps to justify a diagram.
   - DO NOT generate a diagram for simple lookups, single facts, direct definitions, numbers, or when the context lacks structural/procedural flow. Never invent artificial diagrams.
   - When including a diagram, place it under a "### 🗺️ Visual Architecture & Flow" heading right before Sources.

   MERMAID RULES — READ CAREFULLY:
   * Start EXACTLY like this with newlines:
     ```mermaid
     flowchart TD
     A["Step 1 / Component A"] -->|Action / Relation| B["Step 2 / Component B"]
     ```
   * Put EACH statement on its OWN line. NEVER merge statements on the same line.
   * Arrow label MUST NOT have quotes: -->|Referenced by|  (NEVER -->|"Referenced by"|).
   * NO subgraphs, NO -.- dotted lines, NO & symbol (use 'and').
   * Max 3-6 simple rectangular nodes with clear concise text. Closing ``` on its own line.

4. If context is insufficient: state honestly, do not hallucinate.

5. End with: Sources: [id1], [id2]"""

class RAGGenerator:
    def __init__(self):
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(f"models/{settings.gemini_model}")
        self.conversation_history = []
        logger.info(f"Generator ready — model: {settings.gemini_model}")

    def generate(self, question: str, chunks: list[dict]) -> str:
        prompt = self._build_prompt(question, chunks)
        for attempt in range(2):
            try:
                response = self.model.generate_content(prompt, request_options={"timeout": 60})
                answer = response.text
                self._update_history(question, answer)
                return answer
            except Exception as e:
                err_str = str(e).lower()
                if attempt == 0 and ("503" in err_str or "504" in err_str or "overloaded" in err_str or "high demand" in err_str):
                    logger.warning(f"Transient Gemini overload ({e}), retrying generate in 1.5s...")
                    import time; time.sleep(1.5)
                    continue
                logger.error(f"Generation failed: {e}")
                return f"Error: {e}"

    def generate_stream(self, question: str, chunks: list[dict]) -> Generator[str, None, None]:
        prompt = self._build_prompt(question, chunks)
        full_answer = []
        for attempt in range(2):
            try:
                for chunk in self.model.generate_content(prompt, stream=True, request_options={"timeout": 60}):
                    token = chunk.text
                    if token:
                        full_answer.append(token)
                        yield token
                self._update_history(question, "".join(full_answer))
                return
            except Exception as e:
                err_str = str(e).lower()
                if attempt == 0 and ("503" in err_str or "504" in err_str or "overloaded" in err_str or "high demand" in err_str):
                    logger.warning(f"Transient Gemini overload ({e}), retrying stream in 1.5s...")
                    import time; time.sleep(1.5)
                    full_answer = []
                    continue
                logger.error(f"Streaming failed: {e}")
                if not full_answer:
                    yield f"Error: {e}"
                else:
                    yield f"\n\n*(Generation interrupted: {e}. Please retry the query.)*"
                return

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
