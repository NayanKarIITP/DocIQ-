"""
retrieval/retriever.py
──────────────────────
Hybrid Search Retriever for DocIQ:
Combines Dense Vector Search (Qdrant) + Sparse Keyword Search (BM25)
using Reciprocal Rank Fusion (RRF) and optional Gemini Reranking.
"""

from __future__ import annotations
import json
import re
from typing import Optional, List, Dict, Any
import google.generativeai as genai
from loguru import logger

from config import settings
from retrieval.embedder import GeminiEmbedder
from retrieval.vector_store import VectorStore
from retrieval.bm25_search import BM25Searcher, reciprocal_rank_fusion


class Retriever:
    def __init__(self, embedder: GeminiEmbedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

        genai.configure(api_key=settings.gemini_api_key, transport="rest")
        self.model = genai.GenerativeModel(f"models/{settings.gemini_model}")

        # Initialize BM25 searcher
        self.bm25_searcher = BM25Searcher()
        self._sync_bm25_from_store()

        logger.info(f"Retriever ready — hybrid search active (model: {settings.gemini_model})")

    def _sync_bm25_from_store(self) -> None:
        """Load existing chunks from Qdrant into BM25 index."""
        try:
            chunks = self.vector_store.get_all_chunks(limit=10000)
            if chunks:
                self.bm25_searcher.build_index(chunks)
                logger.info(f"Synced {len(chunks)} chunks from Qdrant into BM25 index")
        except Exception as e:
            logger.warning(f"Could not sync BM25 index from Qdrant: {e}")

    def update_bm25(self, new_chunks: List[Dict[str, Any]]) -> None:
        """Add newly indexed chunks into the in-memory BM25 index."""
        try:
            self.bm25_searcher.add_chunks(new_chunks)
            logger.info(f"Updated BM25 index with {len(new_chunks)} new chunks")
        except Exception as e:
            logger.error(f"Failed to update BM25 index: {e}")

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        use_hyde: bool = False,
        rerank: bool = False,
        use_hybrid: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks using Hybrid Search (Dense + BM25 with RRF).
        """
        top_k = top_k or settings.top_k_rerank
        initial_k = settings.top_k_retrieval

        # --------------------------------------------------------
        # 1. Dense Vector Search (Qdrant)
        # --------------------------------------------------------
        query_emb = self._hyde_embed(question) if use_hyde else self.embedder.embed_query(question)
        dense_candidates = []
        if query_emb:
            try:
                dense_candidates = self.vector_store.search(
                    query_embedding=query_emb,
                    top_k=initial_k,
                )
            except Exception as e:
                logger.error(f"Dense vector search failed: {e}")

        # --------------------------------------------------------
        # 2. Sparse Keyword Search (BM25)
        # --------------------------------------------------------
        if use_hybrid and self.bm25_searcher and self.bm25_searcher.chunks:
            bm25_candidates = self.bm25_searcher.search(query=question, top_k=initial_k)

            # ----------------------------------------------------
            # 3. Reciprocal Rank Fusion (RRF)
            # ----------------------------------------------------
            candidates = reciprocal_rank_fusion(
                dense_results=dense_candidates,
                bm25_results=bm25_candidates,
                top_k=initial_k,
                rrf_k=60,
            )
            logger.info(
                f"Hybrid RRF: {len(dense_candidates)} dense + {len(bm25_candidates)} BM25 "
                f"-> {len(candidates)} fused candidates"
            )
        else:
            candidates = dense_candidates

        if not candidates:
            logger.warning("No candidates retrieved from vector store or BM25")
            return []

        # --------------------------------------------------------
        # 4. Optional Gemini Reranker
        # --------------------------------------------------------
        if rerank and len(candidates) > top_k:
            return self._rerank(question, candidates, top_k)

        return candidates[:top_k]

    def _hyde_embed(self, question: str) -> list:
        try:
            resp = self.model.generate_content(
                f"Write a 2-sentence factual answer to: {question}",
                request_options={"timeout": 15},
            )
            return self.embedder.embed_query(resp.text.strip())
        except Exception as e:
            logger.warning(f"HyDE failed, falling back to direct embed: {e}")
            return self.embedder.embed_query(question)

    def _rerank(self, question: str, candidates: list, top_k: int) -> list:
        try:
            texts = [
                f"[{i}] {c.get('chunk_type', 'text')} p.{c.get('page_number', '?')}\n{str(c.get('content', ''))[:300]}"
                for i, c in enumerate(candidates)
            ]
            prompt = (
                f"Question: {question}\nRate each 0-10 for relevance. "
                f"Reply ONLY as JSON array e.g. [8,3,9]\n\n" + "\n\n".join(texts)
            )
            resp = self.model.generate_content(prompt, request_options={"timeout": 15})
            match = re.search(r"\[[\d\s,\.]+\]", resp.text)
            if not match:
                return candidates[:top_k]
            scores = json.loads(match.group())
            for i, c in enumerate(candidates):
                c["rerank_score"] = scores[i] if i < len(scores) else 0.0
            return sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)[:top_k]
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return candidates[:top_k]
