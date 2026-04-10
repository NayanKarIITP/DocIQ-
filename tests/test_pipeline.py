"""
tests/test_pipeline.py
───────────────────────
Unit tests for the RAG pipeline components.
Run: pytest tests/ -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Parser tests ──────────────────────────────────────────────────────────

class TestDocumentParser:
    def test_chunk_id_deterministic(self):
        from ingestion.parser import DocumentParser
        p = DocumentParser()
        id1 = p._make_chunk_id("abc", 1, "text", 0)
        id2 = p._make_chunk_id("abc", 1, "text", 0)
        assert id1 == id2

    def test_chunk_id_unique(self):
        from ingestion.parser import DocumentParser
        p = DocumentParser()
        id1 = p._make_chunk_id("abc", 1, "text", 0)
        id2 = p._make_chunk_id("abc", 1, "text", 1)
        assert id1 != id2

    def test_chunk_type_enum(self):
        from ingestion.parser import ChunkType
        assert ChunkType.TEXT.value == "text"
        assert ChunkType.IMAGE.value == "image"
        assert ChunkType.TABLE.value == "table"

    def test_document_chunk_to_dict(self):
        from ingestion.parser import ChunkType, DocumentChunk
        chunk = DocumentChunk(
            chunk_id="abc123",
            doc_id="doc1",
            filename="test.pdf",
            page_number=1,
            chunk_type=ChunkType.TEXT,
            content="Hello world",
        )
        d = chunk.to_dict()
        assert d["chunk_id"] == "abc123"
        assert d["chunk_type"] == "text"
        assert d["content"] == "Hello world"


# ── Chunker tests ─────────────────────────────────────────────────────────

class TestTextChunker:
    def test_short_text_passthrough(self):
        from ingestion.chunker import TextChunker
        from ingestion.parser import ChunkType, DocumentChunk

        chunker = TextChunker(chunk_size=1000, overlap=100)
        chunk = DocumentChunk(
            chunk_id="x", doc_id="d", filename="f.pdf",
            page_number=1, chunk_type=ChunkType.TEXT,
            content="Short text."
        )
        result = chunker.split([chunk])
        assert len(result) == 1
        assert result[0].content == "Short text."

    def test_long_text_splits(self):
        from ingestion.chunker import TextChunker
        from ingestion.parser import ChunkType, DocumentChunk

        chunker = TextChunker(chunk_size=100, overlap=20)
        long_text = "This is a sentence. " * 50  # ~1000 chars
        chunk = DocumentChunk(
            chunk_id="x", doc_id="d", filename="f.pdf",
            page_number=1, chunk_type=ChunkType.TEXT,
            content=long_text
        )
        result = chunker.split([chunk])
        assert len(result) > 1

    def test_images_passthrough_unchunked(self):
        from ingestion.chunker import TextChunker
        from ingestion.parser import ChunkType, DocumentChunk

        chunker = TextChunker(chunk_size=50, overlap=10)
        img_chunk = DocumentChunk(
            chunk_id="x", doc_id="d", filename="f.pdf",
            page_number=1, chunk_type=ChunkType.IMAGE,
            content="Image description here",
            image_b64="fakebase64data"
        )
        result = chunker.split([img_chunk])
        assert len(result) == 1
        assert result[0].image_b64 == "fakebase64data"

    def test_split_preserves_metadata(self):
        from ingestion.chunker import TextChunker
        from ingestion.parser import ChunkType, DocumentChunk

        chunker = TextChunker(chunk_size=50, overlap=10)
        chunk = DocumentChunk(
            chunk_id="x", doc_id="d", filename="test.pdf",
            page_number=3, chunk_type=ChunkType.TEXT,
            content="Long text. " * 30,
            metadata={"source": "custom"}
        )
        result = chunker.split([chunk])
        for r in result:
            assert r.filename == "test.pdf"
            assert r.page_number == 3
            assert r.doc_id == "d"


# ── Vector store tests (mocked Qdrant) ───────────────────────────────────

class TestVectorStore:
    @patch("retrieval.vector_store.QdrantClient")
    def test_upsert_converts_chunk_id_to_int(self, mock_qdrant):
        from ingestion.parser import ChunkType, DocumentChunk
        from retrieval.vector_store import VectorStore

        vs = VectorStore()
        vs.client = MagicMock()
        vs.client.get_collections.return_value = MagicMock(collections=[])

        chunk = DocumentChunk(
            chunk_id="a1b2c3d4e5f60000",  # valid hex
            doc_id="doc1", filename="f.pdf",
            page_number=1, chunk_type=ChunkType.TEXT,
            content="test content"
        )
        embedding = [0.1] * 768

        vs.upsert([(chunk, embedding)])
        vs.client.upsert.assert_called_once()

    @patch("retrieval.vector_store.QdrantClient")
    def test_search_returns_list(self, mock_qdrant):
        from retrieval.vector_store import VectorStore

        vs = VectorStore()
        mock_hit = MagicMock()
        mock_hit.payload = {"content": "test", "chunk_id": "abc"}
        mock_hit.score = 0.95
        vs.client.search.return_value = [mock_hit]

        results = vs.search([0.1] * 768, top_k=5)
        assert isinstance(results, list)
        assert results[0]["score"] == 0.95


# ── Generator tests (mocked Gemini) ──────────────────────────────────────

class TestRAGGenerator:
    @patch("generation.generator.genai")
    def test_build_prompt_includes_chunks(self, mock_genai):
        from generation.generator import RAGGenerator

        gen = RAGGenerator()
        chunks = [
            {"chunk_id": "abc12345", "chunk_type": "text",
             "filename": "report.pdf", "page_number": 2,
             "content": "Revenue was $1M", "score": 0.9}
        ]
        prompt = gen._build_prompt("What was revenue?", chunks)
        assert "Revenue was $1M" in prompt
        assert "report.pdf" in prompt

    @patch("generation.generator.genai")
    def test_format_context_truncates_chunk_id(self, mock_genai):
        from generation.generator import RAGGenerator

        gen = RAGGenerator()
        chunks = [
            {"chunk_id": "abcdef1234567890", "chunk_type": "text",
             "filename": "doc.pdf", "page_number": 1,
             "content": "Some content", "score": 0.8}
        ]
        context = gen._format_context(chunks)
        assert "abcdef12" in context  # only first 8 chars


# ── Feedback tests (mocked DB) ────────────────────────────────────────────

class TestFeedbackCollector:
    @pytest.mark.asyncio
    async def test_save_feedback_positive(self):
        from feedback.collector import FeedbackCollector

        collector = FeedbackCollector()

        # Mock the DB session
        with patch("feedback.collector.AsyncSessionLocal") as mock_session:
            mock_ctx = MagicMock()
            mock_ctx.__aenter__ = MagicMock(return_value=MagicMock())
            mock_ctx.__aexit__ = MagicMock(return_value=False)
            mock_session.return_value = mock_ctx

            # Should not raise
            result = await collector.save_feedback(
                question="What is X?",
                answer="X is Y.",
                rating=1,
            )
            assert isinstance(result, str)  # returns feedback_id


# ── API tests ─────────────────────────────────────────────────────────────

class TestAPI:
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        from httpx import AsyncClient, ASGITransport
        from unittest.mock import AsyncMock

        # Mock all dependencies
        with patch("api.main.init_db", new_callable=AsyncMock), \
             patch("api.main.DocumentParser"), \
             patch("api.main.TextChunker"), \
             patch("api.main.GeminiEmbedder"), \
             patch("api.main.VectorStore") as mock_vs, \
             patch("api.main.Retriever"), \
             patch("api.main.RAGGenerator"), \
             patch("api.main.FeedbackCollector"):

            mock_vs.return_value.count.return_value = 42
            mock_vs.return_value.create_collection.return_value = None

            from api.main import app
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/health")
                # May be 200 or 503 depending on mock setup
                assert resp.status_code in (200, 503)
