"""retrieval/retriever.py"""
import json
import re
import google.generativeai as genai
from loguru import logger
from config import settings
from retrieval.embedder import GeminiEmbedder
from retrieval.vector_store import VectorStore


class Retriever:
    def __init__(self, embedder: GeminiEmbedder, vector_store: VectorStore):
        self.embedder     = embedder
        self.vector_store = vector_store
        genai.configure(api_key=settings.gemini_api_key)
        # self.model = genai.GenerativeModel(settings.gemini_model)
        self.model = genai.GenerativeModel(f"models/{settings.gemini_model}")
        logger.info(f"Retriever ready — model: {settings.gemini_model}")

    def retrieve(self, question: str, top_k: int = None,
                 use_hyde: bool = True, rerank: bool = True) -> list:
        top_k     = top_k or settings.top_k_rerank
        initial_k = settings.top_k_retrieval
        query_emb = self._hyde_embed(question) if use_hyde else self.embedder.embed_query(question)
        candidates = self.vector_store.search(query_embedding=query_emb, top_k=initial_k)
        if not candidates:
            logger.warning("No candidates — upload and index a document first")
            return []
        if rerank and len(candidates) > top_k:
            return self._rerank(question, candidates, top_k)
        return candidates[:top_k]

    def _hyde_embed(self, question: str) -> list:
        try:
            resp = self.model.generate_content(
                f"Write a 2-sentence factual answer to: {question}")
            return self.embedder.embed_query(resp.text.strip())
        except Exception as e:
            logger.warning(f"HyDE failed, direct embed: {e}")
            return self.embedder.embed_query(question)

    def _rerank(self, question: str, candidates: list, top_k: int) -> list:
        try:
            texts  = [f"[{i}] {c['chunk_type']} p.{c['page_number']}\n{c['content'][:300]}"
                      for i, c in enumerate(candidates)]
            prompt = (f"Question: {question}\nRate each 0-10 for relevance. "
                      f"Reply ONLY as JSON array e.g. [8,3,9]\n\n" + "\n\n".join(texts))
            resp  = self.model.generate_content(prompt)
            match = re.search(r'\[[\d\s,\.]+\]', resp.text)
            if not match:
                return candidates[:top_k]
            scores = json.loads(match.group())
            for i, c in enumerate(candidates):
                c["rerank_score"] = scores[i] if i < len(scores) else 0.0
            return sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)[:top_k]
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return candidates[:top_k]
