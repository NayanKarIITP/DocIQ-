"""
quickstart.py
──────────────
End-to-end smoke test. Run this FIRST after setup to verify everything works.

Usage:
    python quickstart.py

What it does:
  1. Creates a tiny sample PDF in memory
  2. Parses + chunks it
  3. Embeds it with Gemini (real API call)
  4. Stores in Qdrant
  5. Runs a query with HyDE + reranking
  6. Prints the streamed answer
  7. Saves a feedback record
  8. Prints system stats

If this runs without errors, your full pipeline is working.
"""

import asyncio
import os
import sys
from pathlib import Path

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).parent))


def check_env():
    """Verify required environment variables are set."""
    required = ["GEMINI_API_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"ERROR: Missing environment variables: {missing}")
        print("Copy .env.example to .env and fill in your GEMINI_API_KEY")
        sys.exit(1)
    print("✓ Environment variables OK")


def create_sample_pdf() -> bytes:
    """Create a tiny in-memory PDF for testing."""
    try:
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text(
            (50, 100),
            "Multimodal RAG System — Test Document\n\n"
            "This document is about artificial intelligence and machine learning.\n\n"
            "Key findings:\n"
            "- RAG systems improve answer accuracy by grounding LLMs in retrieved context.\n"
            "- Multimodal parsing enables understanding of text, images, and tables.\n"
            "- The HyDE technique improves retrieval recall by approximately 30 percent.\n"
            "- Direct Preference Optimization (DPO) enables the model to learn from corrections.\n\n"
            "The system was evaluated on 100 questions and achieved:\n"
            "- Faithfulness score: 0.94\n"
            "- Answer relevancy: 0.91\n"
            "- Context precision: 0.88\n"
            "- Context recall: 0.85\n",
            fontsize=12,
        )
        return doc.tobytes()
    except Exception as e:
        print(f"WARNING: Could not create sample PDF: {e}")
        return None


async def run_quickstart():
    from dotenv import load_dotenv
    load_dotenv()
    check_env()

    print("\n" + "=" * 55)
    print("  MULTIMODAL RAG — QUICKSTART TEST")
    print("=" * 55)

    # ── Step 1: Init components ───────────────────────────────
    print("\n[1/7] Initializing components...")
    from ingestion.chunker import TextChunker
    from ingestion.parser import DocumentParser
    from retrieval.embedder import GeminiEmbedder
    from retrieval.retriever import Retriever
    from retrieval.vector_store import VectorStore
    from generation.generator import RAGGenerator
    from feedback.collector import FeedbackCollector, init_db

    parser    = DocumentParser()
    chunker   = TextChunker()
    embedder  = GeminiEmbedder()
    vs        = VectorStore()
    retriever = Retriever(embedder, vs)
    generator = RAGGenerator()
    feedback  = FeedbackCollector()
    print("  ✓ All components initialized")

    # ── Step 2: Init DB + Qdrant ──────────────────────────────
    print("\n[2/7] Initializing databases...")
    try:
        await init_db()
        print("  ✓ PostgreSQL tables created")
    except Exception as e:
        print(f"  ⚠ PostgreSQL not available ({e}) — feedback loop will be disabled")

    try:
        vs.create_collection()
        print("  ✓ Qdrant collection ready")
    except Exception as e:
        print(f"  ✗ Qdrant not available: {e}")
        print("  Start Qdrant with: docker run -p 6333:6333 qdrant/qdrant")
        sys.exit(1)

    # ── Step 3: Parse sample PDF ──────────────────────────────
    print("\n[3/7] Parsing sample document...")
    pdf_bytes = create_sample_pdf()
    if not pdf_bytes:
        print("  ✗ Could not create sample PDF")
        sys.exit(1)

    raw_chunks = parser.parse_bytes(pdf_bytes, "quickstart_test.pdf")
    chunks = chunker.split(raw_chunks)
    print(f"  ✓ Parsed → {len(chunks)} chunks "
          f"({sum(1 for c in chunks if c.chunk_type.value=='text')} text, "
          f"{sum(1 for c in chunks if c.chunk_type.value=='image')} image, "
          f"{sum(1 for c in chunks if c.chunk_type.value=='table')} table)")

    # ── Step 4: Embed ─────────────────────────────────────────
    print("\n[4/7] Embedding with Gemini...")
    print("  (This calls the Gemini API — may take 10-30 seconds)")
    try:
        chunks_with_embeddings = embedder.embed_chunks(chunks)
        print(f"  ✓ Embedded {len(chunks_with_embeddings)} chunks "
              f"(dim={len(chunks_with_embeddings[0][1])})")
    except Exception as e:
        print(f"  ✗ Embedding failed: {e}")
        print("  Check your GEMINI_API_KEY in .env")
        sys.exit(1)

    # ── Step 5: Store in Qdrant ───────────────────────────────
    print("\n[5/7] Storing in Qdrant...")
    count = vs.upsert(chunks_with_embeddings)
    print(f"  ✓ Stored {count} chunks  |  Total in index: {vs.count()}")

    # ── Step 6: Query ─────────────────────────────────────────
    print("\n[6/7] Running test query with HyDE + reranking...")
    question = "What evaluation scores did the system achieve?"
    print(f"  Question: {question}\n")

    retrieved = retriever.retrieve(question, use_hyde=True, rerank=True, top_k=3)
    print(f"  Retrieved {len(retrieved)} chunks\n")

    print("  Answer (streaming):")
    print("  " + "-" * 50)
    full_answer = ""
    for token in generator.generate_stream(question, retrieved):
        print(token, end="", flush=True)
        full_answer += token
    print("\n  " + "-" * 50)

    # ── Step 7: Save feedback ─────────────────────────────────
    print("\n[7/7] Saving test feedback...")
    try:
        fb_id = await feedback.save_feedback(
            question=question,
            answer=full_answer,
            rating=1,
            chunk_ids=[c.get("chunk_id", "") for c in retrieved],
        )
        stats = await feedback.get_stats()
        print(f"  ✓ Feedback saved (id={fb_id[:8]})")
        print(f"  ✓ Stats: {stats}")
    except Exception as e:
        print(f"  ⚠ Feedback not saved (DB may be offline): {e}")

    # ── Done ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  QUICKSTART COMPLETE — pipeline is working!")
    print("=" * 55)
    print("\nNext steps:")
    print("  1. Start the API:  uvicorn api.main:app --reload")
    print("  2. Start the UI:   streamlit run ui/app.py")
    print("  3. Open browser:   http://localhost:8501")
    print("  4. Upload your PDFs and start asking questions!\n")


if __name__ == "__main__":
    asyncio.run(run_quickstart())
