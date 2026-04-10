"""
api/main.py
────────────
FastAPI backend for the Multimodal RAG system.

Endpoints:
  POST /ingest          — upload and index a PDF
  POST /query           — ask a question (streaming)
  POST /feedback        — submit thumbs up/down
  GET  /documents       — list indexed documents
  DELETE /documents/{doc_id} — remove a document
  GET  /health          — system health check
  GET  /stats           — feedback statistics

Run:
    uvicorn api.main:app --reload --port 8000
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from config import settings
from feedback.collector import FeedbackCollector, init_db
from generation.generator import RAGGenerator
from ingestion.chunker import TextChunker
from ingestion.parser import DocumentParser
from retrieval.embedder import GeminiEmbedder
from retrieval.retriever import Retriever
from retrieval.vector_store import VectorStore


# ── App lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all components at startup."""
    logger.info("Starting Multimodal RAG API...")

    # Init database
    await init_db()

    # Init components (stored on app.state for access in routes)
    app.state.parser = DocumentParser()
    app.state.chunker = TextChunker()
    app.state.embedder = GeminiEmbedder()
    app.state.vector_store = VectorStore()
    app.state.retriever = Retriever(app.state.embedder, app.state.vector_store)
    app.state.generator = RAGGenerator()
    app.state.feedback = FeedbackCollector()

    # Ensure Qdrant collection exists
    app.state.vector_store.create_collection()

    logger.success("API ready!")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Multimodal RAG API",
    description="Production-grade RAG system with Gemini + Qdrant + feedback loop",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    use_hyde: bool = True
    use_rerank: bool = True
    top_k: Optional[int] = None


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: int                   # 1 = good, -1 = bad
    correction: Optional[str] = None
    chunk_ids: Optional[list[str]] = None
    session_id: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Quick health check."""
    try:
        count = app.state.vector_store.count()
        return {
            "status": "healthy",
            "chunks_indexed": count,
            "model": settings.gemini_model,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Upload a PDF and index it.
    Returns: doc_id, filename, chunks_indexed
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    file_bytes = await file.read()
    if len(file_bytes) > max_bytes:
        raise HTTPException(413, f"File too large (max {settings.max_upload_size_mb}MB)")

    try:
        # 1. Parse
        logger.info(f"Ingesting: {file.filename}")
        raw_chunks = app.state.parser.parse_bytes(file_bytes, file.filename)

        # 2. Chunk text
        chunks = app.state.chunker.split(raw_chunks)

        # 3. Embed (this calls Gemini — takes a few seconds)
        chunks_with_embeddings = app.state.embedder.embed_chunks(chunks)

        # 4. Store in Qdrant
        count = app.state.vector_store.upsert(chunks_with_embeddings)

        doc_id = chunks[0].doc_id if chunks else "unknown"
        return {
            "success": True,
            "doc_id": doc_id,
            "filename": file.filename,
            "chunks_indexed": count,
        }

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(500, f"Ingestion failed: {e}")


@app.post("/query")
async def query(request: QueryRequest):
    """
    Ask a question. Returns a Server-Sent Event stream of tokens.

    In your frontend:
        const es = new EventSource('/query')  ← use fetch + ReadableStream
    """
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    async def token_stream():
        try:
            # Retrieve relevant chunks
            chunks = app.state.retriever.retrieve(
                question=request.question,
                use_hyde=request.use_hyde,
                rerank=request.use_rerank,
                top_k=request.top_k,
            )

            if not chunks:
                yield "data: No relevant documents found. Please upload documents first.\n\n"
                return

            # Stream the answer
            # Run sync generator in executor to avoid blocking event loop
            # loop = asyncio.get_event_loop()
            gen = app.state.generator.generate_stream(request.question, chunks)

            for token in gen:
                if token:
                    # SSE format: "data: <token>\n\n"
                    yield f"data: {token}\n\n"

            # Send the source chunks at the end as metadata
            sources = [
                {
                    "chunk_id": c.get("chunk_id", "")[:8],
                    "filename": c.get("filename", ""),
                    "page": c.get("page_number"),
                    "type": c.get("chunk_type"),
                }
                for c in chunks
            ]
            import json
            yield f"data: [SOURCES]{json.dumps(sources)}[/SOURCES]\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Query failed: {e}")
            yield f"data: Error: {e}\n\n"

    return EventSourceResponse(token_stream())


@app.post("/query/sync")
async def query_sync(request: QueryRequest):
    """Non-streaming query endpoint (for simple clients)."""
    chunks = app.state.retriever.retrieve(
        question=request.question,
        use_hyde=request.use_hyde,
        rerank=request.use_rerank,
        top_k=request.top_k,
    )
    if not chunks:
        return {"answer": "No relevant documents found.", "sources": []}

    answer = app.state.generator.generate(request.question, chunks)
    sources = [
        {
            "chunk_id": c.get("chunk_id", "")[:8],
            "filename": c.get("filename"),
            "page": c.get("page_number"),
            "type": c.get("chunk_type"),
        }
        for c in chunks
    ]
    return {"answer": answer, "sources": sources}


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Save user feedback for the RLHF loop."""
    if request.rating not in (1, -1):
        raise HTTPException(400, "Rating must be 1 (good) or -1 (bad)")

    feedback_id = await app.state.feedback.save_feedback(
        question=request.question,
        answer=request.answer,
        rating=request.rating,
        correction=request.correction,
        chunk_ids=request.chunk_ids,
        session_id=request.session_id,
    )
    return {"success": True, "feedback_id": feedback_id}


@app.get("/documents")
async def list_documents():
    """List all indexed documents."""
    docs = app.state.vector_store.list_documents()
    return {"documents": docs, "total": len(docs)}


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Remove all chunks for a document from the index."""
    app.state.vector_store.delete_document(doc_id)
    return {"success": True, "doc_id": doc_id}


@app.get("/stats")
async def get_stats():
    """Feedback and system statistics."""
    feedback_stats = await app.state.feedback.get_stats()
    return {
        "chunks_indexed": app.state.vector_store.count(),
        "documents": len(app.state.vector_store.list_documents()),
        **feedback_stats,
    }
