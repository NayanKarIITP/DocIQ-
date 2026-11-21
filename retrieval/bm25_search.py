"""
retrieval/bm25_search.py
────────────────────────
BM25 Keyword Search Engine with Reciprocal Rank Fusion (RRF) for DocIQ.

Implements:
1. BM25Okapi indexing over chunk text/tables/image descriptions.
2. Fast regex-based alphanumeric tokenization.
3. RRF (Reciprocal Rank Fusion) algorithm to merge Dense Vector & BM25 rankings.
"""

from __future__ import annotations
import re
from typing import List, Dict, Any, Optional
from loguru import logger
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


class BM25Searcher:
    """In-memory BM25 index over document chunks."""

    def __init__(self, chunks: Optional[List[Dict[str, Any]]] = None):
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        if chunks:
            self.build_index(chunks)

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Build or rebuild BM25 index from a list of chunk payload dictionaries."""
        valid_chunks = [c for c in chunks if c.get("content") and str(c.get("content")).strip()]
        self.chunks = valid_chunks
        if not valid_chunks:
            self.bm25 = None
            return

        tokenized_corpus = [tokenize(str(c.get("content", ""))) for c in valid_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built with {len(valid_chunks)} chunks.")

    def add_chunks(self, new_chunks: List[Dict[str, Any]]) -> None:
        """Append new chunks to the BM25 index and rebuild."""
        all_chunks = self.chunks + [c for c in new_chunks if c.get("content") and str(c.get("content")).strip()]
        # Remove duplicates by chunk_id if present
        seen_ids = set()
        unique_chunks = []
        for c in all_chunks:
            cid = c.get("chunk_id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                unique_chunks.append(c)
            elif not cid:
                unique_chunks.append(c)
        self.build_index(unique_chunks)

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Search BM25 index for keyword matches."""
        if not self.bm25 or not self.chunks:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        scored_indices = [
            (idx, score) for idx, score in enumerate(scores) if score > 0
        ]
        if not scored_indices:
            return []

        scored_indices.sort(key=lambda x: x[1], reverse=True)
        top_matches = scored_indices[:top_k]

        results = []
        for idx, score in top_matches:
            chunk = dict(self.chunks[idx])
            chunk["bm25_score"] = float(score)
            results.append(chunk)

        return results


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    top_k: int = 10,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) algorithm to combine Dense Vector and BM25 results.

    RRF_score(d) = sum(1 / (rrf_k + rank)) across ranking systems.
    Standard constant rrf_k = 60 prevents high rankings from overly dominating.
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    # Dense rankings
    for rank, doc in enumerate(dense_results, start=1):
        doc_id = doc.get("chunk_id") or str(doc.get("id", "")) or str(hash(str(doc.get("content", ""))))
        doc_map[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

    # BM25 rankings
    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc.get("chunk_id") or str(doc.get("id", "")) or str(hash(str(doc.get("content", ""))))
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        else:
            if "bm25_score" in doc:
                doc_map[doc_id]["bm25_score"] = doc["bm25_score"]
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

    # Sort documents by RRF score descending
    sorted_doc_ids = sorted(scores.keys(), key=lambda did: scores[did], reverse=True)

    fused_results = []
    for did in sorted_doc_ids[:top_k]:
        fused_chunk = dict(doc_map[did])
        fused_chunk["rrf_score"] = scores[did]
        fused_results.append(fused_chunk)

    return fused_results
