"""
retrieval/hf_embedder.py
─────────────────────────
Local HuggingFace embedder using sentence-transformers.

Why add this:
  - HuggingFace appears in 70%+ of GenAI Engineer JDs
  - Runs entirely on CPU — no API key needed, no quota limits
  - Shows you can deploy models locally, not just call APIs
  - Useful when Gemini API is slow, expensive, or unavailable

Supported models (all free, run on CPU):
  BAAI/bge-small-en-v1.5   → 384-dim, fastest, ~130MB
  BAAI/bge-base-en-v1.5    → 768-dim, better quality, ~430MB
  BAAI/bge-large-en-v1.5   → 1024-dim, best quality, ~1.3GB
  all-MiniLM-L6-v2         → 384-dim, classic choice, ~90MB

Usage:
    from retrieval.hf_embedder import HuggingFaceEmbedder
    embedder = HuggingFaceEmbedder("BAAI/bge-small-en-v1.5")
    embedding = embedder.embed_query("What is SQL?")
    print(len(embedding))  # 384
"""

from __future__ import annotations
from typing import Optional
from loguru import logger


# Model registry — maps short name to HuggingFace model id + output dim
MODEL_REGISTRY = {
    "bge-small":  ("BAAI/bge-small-en-v1.5",  384),
    "bge-base":   ("BAAI/bge-base-en-v1.5",   768),
    "bge-large":  ("BAAI/bge-large-en-v1.5",  1024),
    "minilm":     ("all-MiniLM-L6-v2",         384),
    "e5-small":   ("intfloat/e5-small-v2",     384),
}


class HuggingFaceEmbedder:
    """
    Local sentence-transformers embedder.
    Drop-in replacement for GeminiEmbedder — same interface.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Args:
            model_name: HuggingFace model ID or short name from MODEL_REGISTRY.
                        First run downloads the model (~130MB for bge-small).
        """
        # Resolve short names
        if model_name in MODEL_REGISTRY:
            hf_id, dim = MODEL_REGISTRY[model_name]
            logger.info(f"Resolved '{model_name}' → {hf_id}")
            model_name = hf_id
            self.embed_dim = dim
        else:
            self.embed_dim = None  # will detect after loading

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed.\n"
                "Run: pip install sentence-transformers"
            )

        logger.info(f"Loading HuggingFace model: {model_name}")
        logger.info("(First run downloads model — subsequent runs use cache)")

        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

        # Detect embedding dim if not known
        if self.embed_dim is None:
            test_emb = self.model.encode("test")
            self.embed_dim = len(test_emb)

        logger.success(
            f"HuggingFace embedder ready — "
            f"model: {model_name}, dim: {self.embed_dim}"
        )

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a search query.
        BGE models work best with a query prefix.
        """
        # BGE models improve with "Represent this sentence: " prefix for queries
        if "bge" in self.model_name.lower():
            query = f"Represent this sentence: {query}"

        embedding = self.model.encode(
            query,
            normalize_embeddings=True,  # cosine similarity ready
            convert_to_numpy=True,
        )
        return embedding.tolist()

    def embed_chunks(self, chunks: list) -> list[tuple]:
        """
        Embed a list of DocumentChunk objects.
        Returns (chunk, embedding) pairs — same as GeminiEmbedder.
        """
        from ingestion.parser import ChunkType

        results = []
        texts = []
        valid_chunks = []

        for chunk in chunks:
            text = self._get_text(chunk)
            if text and len(text.strip()) > 10:
                texts.append(text)
                valid_chunks.append(chunk)

        if not texts:
            return []

        logger.info(f"Batch embedding {len(texts)} chunks with HuggingFace...")

        # Batch encode — much faster than one-by-one
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=len(texts) > 20,
            convert_to_numpy=True,
        )

        for chunk, emb in zip(valid_chunks, embeddings):
            results.append((chunk, emb.tolist()))

        logger.success(f"HuggingFace embedded {len(results)} chunks")
        return results

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of plain strings. Useful for batch operations."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            convert_to_numpy=True,
        )
        return [e.tolist() for e in embeddings]

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts. Useful for testing."""
        import numpy as np
        e1 = np.array(self.embed_query(text1))
        e2 = np.array(self.embed_query(text2))
        return float(np.dot(e1, e2))

    def _get_text(self, chunk) -> str:
        from ingestion.parser import ChunkType
        if chunk.chunk_type == ChunkType.TEXT:
            return chunk.content
        elif chunk.chunk_type == ChunkType.TABLE:
            return f"Table from {chunk.filename}:\n{chunk.content}"
        elif chunk.chunk_type == ChunkType.IMAGE:
            return chunk.content if chunk.content else (
                f"Image from {chunk.filename} page {chunk.page_number}"
            )
        return chunk.content


def compare_embedders(text: str = "What is machine learning?"):
    """
    Utility: compare HuggingFace vs Gemini embedding quality.
    Prints similarity scores for a test query.
    """
    print(f"\nComparing embedders on: '{text}'")
    print("─" * 50)

    hf = HuggingFaceEmbedder("bge-small")
    similar_text = "Machine learning is a subset of AI that learns from data."
    different_text = "The weather today is sunny and warm."

    sim1 = hf.similarity(text, similar_text)
    sim2 = hf.similarity(text, different_text)

    print(f"HuggingFace bge-small:")
    print(f"  Similar text similarity:   {sim1:.3f}  (should be high)")
    print(f"  Different text similarity: {sim2:.3f}  (should be low)")
    print(f"  Embedding dimension: {hf.embed_dim}")


if __name__ == "__main__":
    compare_embedders()
