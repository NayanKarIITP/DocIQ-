"""
api/main.py  —  UPGRADED
New endpoints added:
  POST /query/langchain   → LangChain RAG pipeline
  POST /query/agent       → LangGraph autonomous agent
  POST /query/structured  → Pydantic structured output
  POST /query/cot         → Chain-of-thought prompting
  GET  /health            → includes W&B status
  GET  /techniques        → lists available prompt techniques
"""

import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import json
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
    logger.info("Starting Multimodal RAG API (upgraded)...")
    await init_db()

    app.state.parser      = DocumentParser()
    app.state.chunker     = TextChunker()
    app.state.embedder    = GeminiEmbedder()
    app.state.vector_store = VectorStore()
    app.state.retriever   = Retriever(app.state.embedder, app.state.vector_store)
    app.state.generator   = RAGGenerator()
    app.state.feedback    = FeedbackCollector()

    # Init W&B tracker (gracefully disabled if no API key)
    try:
        from tracking.wandb_tracker import tracker
        app.state.tracker = tracker
        logger.info(f"W&B tracking: {'enabled' if tracker.enabled else 'disabled'}")
    except Exception:
        app.state.tracker = None

    app.state.vector_store.create_collection()
    logger.success("API ready!")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="DocIQ — Multimodal RAG API (Upgraded)",
    description="RAG + LangChain + LangGraph Agent + Structured Output + W&B",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response models ───────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    use_hyde: bool = True
    use_rerank: bool = True
    use_hybrid: bool = True
    top_k: Optional[int] = None


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: int
    correction: Optional[str] = None
    chunk_ids: Optional[list[str]] = None
    session_id: Optional[str] = None


# ── Helper: track query to W&B ────────────────────────────────────────────

def _track(app, question, chunks, start_time, answer=""):
    try:
        if app.state.tracker:
            app.state.tracker.log_query(
                question=question,
                chunks=chunks,
                latency_ms=(time.time() - start_time) * 1000,
                answer_length=len(answer),
            )
    except Exception:
        pass


# ── Core endpoints ────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    tracker_status = "disabled"
    if hasattr(app.state, "tracker") and app.state.tracker:
        tracker_status = "enabled" if app.state.tracker.enabled else "no_api_key"
    return {
        "status":         "healthy",
        "chunks_indexed": app.state.vector_store.count(),
        "model":          settings.gemini_model,
        "embed_model":    settings.gemini_embed_model,
        "wandb":          tracker_status,
        "version":        "2.0.0",
    }


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    file_bytes = await file.read()
    if len(file_bytes) > max_bytes:
        raise HTTPException(413, f"File too large (max {settings.max_upload_size_mb}MB)")

    try:
        start = time.time()
        logger.info(f"Ingesting: {file.filename}")

        raw_chunks = await asyncio.to_thread(app.state.parser.parse_bytes, file_bytes, file.filename)
        chunks     = await asyncio.to_thread(app.state.chunker.split, raw_chunks)
        embedded   = await asyncio.to_thread(app.state.embedder.embed_chunks, chunks)
        count      = await asyncio.to_thread(app.state.vector_store.upsert, embedded)

        # Update BM25 keyword index for Hybrid Search
        try:
            new_payloads = [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "filename": c.filename,
                    "page_number": c.page_number,
                    "chunk_type": c.chunk_type.value,
                    "content": c.content,
                }
                for c, _ in embedded
            ]
            app.state.retriever.update_bm25(new_payloads)
        except Exception as bm25_err:
            logger.warning(f"Could not update BM25 index: {bm25_err}")

        # W&B tracking
        text_n  = sum(1 for c in chunks if c.chunk_type.value == "text")
        image_n = sum(1 for c in chunks if c.chunk_type.value == "image")
        table_n = sum(1 for c in chunks if c.chunk_type.value == "table")
        if app.state.tracker:
            app.state.tracker.log_ingest(
                filename=file.filename,
                n_text=text_n, n_image=image_n, n_table=table_n,
                n_chunks=count,
                latency_ms=(time.time() - start) * 1000,
            )

        doc_id = chunks[0].doc_id if chunks else "unknown"
        return {
            "success":        True,
            "doc_id":         doc_id,
            "filename":       file.filename,
            "chunks_indexed": count,
            "breakdown":      {"text": text_n, "image": image_n, "table": table_n},
        }
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(500, f"Ingestion failed: {e}")


@app.post("/query/sync")
async def query_sync(request: QueryRequest):
    """Standard RAG query — custom HyDE + Qdrant pipeline."""
    start = time.time()
    chunks = await asyncio.to_thread(
        app.state.retriever.retrieve,
        question=request.question,
        use_hyde=request.use_hyde,
        rerank=request.use_rerank,
        use_hybrid=request.use_hybrid,
        top_k=request.top_k,
    )
    if not chunks:
        return {"answer": "No relevant documents found.", "sources": []}

    answer = await asyncio.to_thread(app.state.generator.generate, request.question, chunks)
    _track(app, request.question, chunks, start, answer)

    return {
        "answer":  answer,
        "sources": [
            {
                "chunk_id": c.get("chunk_id", "")[:8],
                "filename": c.get("filename"),
                "page":     c.get("page_number"),
                "type":     c.get("chunk_type"),
                "snippet":  str(c.get("content", "")).strip()[:180],
            }
            for c in chunks
        ],
        "mode": "custom",
    }


# ── NEW: LangChain endpoint ───────────────────────────────────────────────

@app.post("/query/langchain")
async def query_langchain(request: QueryRequest):
    """
    LangChain RetrievalQA pipeline.
    Uses the same Qdrant collection but goes through LangChain's
    document loading and chain-of-thought summarization.
    """
    try:
        from retrieval.langchain_retriever import build_langchain_rag, run_langchain_query
    except ImportError as e:
        raise HTTPException(501, f"LangChain not installed: {e}")

    start = time.time()
    try:
        chain  = build_langchain_rag(top_k=request.top_k or settings.top_k_rerank)
        result = run_langchain_query(chain, request.question)
        _track(app, request.question, [], start, result["answer"])
        return result
    except Exception as e:
        logger.error(f"LangChain query failed: {e}")
        raise HTTPException(500, f"LangChain query failed: {e}")


# ── NEW: LangGraph agent endpoint ─────────────────────────────────────────

@app.post("/query/agent")
async def query_agent(request: QueryRequest):
    """
    LangGraph autonomous agent.
    The agent decides whether to search documents, search the web,
    or calculate — based on the question.
    """
    try:
        from agents.dociq_agent import run_agent
    except ImportError as e:
        raise HTTPException(501, f"LangGraph not installed: {e}")

    start = time.time()
    try:
        answer = run_agent(request.question)
        _track(app, request.question, [], start, answer)
        return {
            "answer": answer,
            "mode":   "agent",
            "note":   "Agent autonomously chose which tools to use",
        }
    except Exception as e:
        logger.error(f"Agent query failed: {e}")
        raise HTTPException(500, f"Agent failed: {e}")


# ── NEW: Structured output endpoint ──────────────────────────────────────

@app.post("/query/structured")
async def query_structured(request: QueryRequest):
    """
    Returns a structured Pydantic response with:
    - answer (str)
    - confidence (float 0-1)
    - citations (list of chunk_ids)
    - needs_more_context (bool)
    - reasoning (str)
    """
    from generation.prompt_engineer import PromptEngineer

    start = time.time()
    chunks = await asyncio.to_thread(
        app.state.retriever.retrieve,
        question=request.question,
        use_hyde=request.use_hyde,
        rerank=request.use_rerank,
        use_hybrid=request.use_hybrid,
        top_k=request.top_k,
    )
    if not chunks:
        return {
            "answer": "No relevant documents found.",
            "confidence": 0.0,
            "citations": [],
            "needs_more_context": True,
            "reasoning": "No chunks retrieved from vector store.",
        }

    pe     = PromptEngineer()
    result = pe.generate_structured(request.question, chunks)
    _track(app, request.question, chunks, start, result.answer)

    return result.model_dump()


# ── NEW: Chain-of-Thought endpoint ────────────────────────────────────────

@app.post("/query/cot")
async def query_cot(request: QueryRequest):
    """
    Chain-of-thought prompting.
    Forces the model to reason step-by-step before answering.
    Better for complex multi-step questions.
    """
    from generation.prompt_engineer import PromptEngineer

    start = time.time()
    chunks = await asyncio.to_thread(
        app.state.retriever.retrieve,
        question=request.question,
        use_hyde=request.use_hyde,
        rerank=request.use_rerank,
        use_hybrid=request.use_hybrid,
        top_k=request.top_k,
    )
    if not chunks:
        return {"answer": "No relevant documents found.", "mode": "cot"}

    pe     = PromptEngineer()
    answer = pe.generate_cot(request.question, chunks)
    _track(app, request.question, chunks, start, answer)

    return {
        "answer": answer,
        "mode":   "chain_of_thought",
        "sources": [
            {"chunk_id": c.get("chunk_id","")[:8],
             "filename": c.get("filename"),
             "page":     c.get("page_number")}
            for c in chunks
        ],
    }


# ── Existing endpoints (unchanged) ────────────────────────────────────────

@app.post("/query")
async def query_stream(request: QueryRequest):
    """Streaming SSE endpoint with JSON packets to preserve newlines and structure."""
    async def token_stream():
        start = time.time()
        full_tokens = []
        try:
            chunks = await asyncio.to_thread(
                app.state.retriever.retrieve,
                question=request.question,
                use_hyde=request.use_hyde,
                rerank=request.use_rerank,
                use_hybrid=request.use_hybrid,
                top_k=request.top_k,
            )
            if not chunks:
                yield json.dumps({"token": "No relevant documents found."})
                yield json.dumps({"done": True})
                return

            gen = app.state.generator.generate_stream(request.question, chunks)
            for token in gen:
                if token:
                    full_tokens.append(token)
                    yield json.dumps({"token": token})

            sources = [
                {"chunk_id": c.get("chunk_id","")[:8], "filename": c.get("filename"),
                 "page": c.get("page_number"), "type": c.get("chunk_type"),
                 "snippet": str(c.get("content", "")).strip()[:180]}
                for c in chunks
            ]
            yield json.dumps({"sources": sources})
            yield json.dumps({"done": True})
            _track(app, request.question, chunks, start, "".join(full_tokens))
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield json.dumps({"token": f"\n\nError: {e}"})
            yield json.dumps({"done": True})

    return EventSourceResponse(token_stream())


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    if request.rating not in (1, -1):
        raise HTTPException(400, "Rating must be 1 or -1")
    feedback_id = await app.state.feedback.save_feedback(
        question=request.question, answer=request.answer,
        rating=request.rating, correction=request.correction,
        chunk_ids=request.chunk_ids, session_id=request.session_id,
    )
    if app.state.tracker:
        app.state.tracker.log_feedback(
            question=request.question, rating=request.rating,
            has_correction=bool(request.correction),
        )
    return {"success": True, "feedback_id": feedback_id}


@app.get("/documents")
async def list_documents():
    docs = app.state.vector_store.list_documents()
    return {"documents": docs, "total": len(docs)}


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    app.state.vector_store.delete_document(doc_id)
    return {"success": True, "doc_id": doc_id}


@app.get("/stats")
async def get_stats():
    feedback_stats = await app.state.feedback.get_stats()
    return {
        "chunks_indexed": app.state.vector_store.count(),
        "documents":      len(app.state.vector_store.list_documents()),
        **feedback_stats,
    }


@app.get("/techniques")
async def list_techniques():
    """List all available query modes."""
    return {
        "modes": [
            {"endpoint": "/query/sync",       "name": "Custom RAG",       "description": "HyDE + Qdrant + Gemini reranking"},
            {"endpoint": "/query/langchain",  "name": "LangChain RAG",    "description": "LangChain RetrievalQA pipeline"},
            {"endpoint": "/query/agent",      "name": "LangGraph Agent",  "description": "Autonomous ReAct agent with 3 tools"},
            {"endpoint": "/query/structured", "name": "Structured Output","description": "Pydantic response with confidence score"},
            {"endpoint": "/query/cot",        "name": "Chain-of-Thought", "description": "Step-by-step reasoning before answer"},
            {"endpoint": "/query",            "name": "Streaming",        "description": "Token-by-token SSE streaming"},
        ]
    }
