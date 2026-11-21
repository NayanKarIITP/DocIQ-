"""
retrieval/langchain_retriever.py
─────────────────────────────────
LangChain-based RAG pipeline as an alternative to the custom pipeline.

Why add this:
  - LangChain appears in 90%+ of GenAI Engineer JDs
  - Shows you know the industry-standard framework
  - Recruiters recognize it instantly

The API exposes both modes:
  POST /query/sync          → custom pipeline (HyDE + Qdrant)
  POST /query/langchain     → LangChain pipeline (same Qdrant collection)

Usage:
    from retrieval.langchain_retriever import build_langchain_rag, run_langchain_query
    chain = build_langchain_rag()
    answer = run_langchain_query(chain, "What is a primary key?")
"""

from __future__ import annotations
from typing import Optional
from loguru import logger

from config import settings


def build_langchain_rag(
    top_k: int = 5,
    chain_type: str = "stuff",
):
    """
    Build a LangChain RetrievalQA chain backed by the same Qdrant
    collection that the custom pipeline uses.

    Args:
        top_k      : number of chunks to retrieve
        chain_type : 'stuff' (concat all chunks) or 'map_reduce'

    Returns:
        A LangChain RetrievalQA chain ready for .invoke()
    """
    try:
        from langchain_google_genai import (
            GoogleGenerativeAIEmbeddings,
            ChatGoogleGenerativeAI,
        )
        from langchain_qdrant import QdrantVectorStore
        from langchain.chains import RetrievalQA
        from langchain_core.prompts import PromptTemplate
    except ImportError as e:
        raise ImportError(
            f"LangChain packages missing. Run:\n"
            f"pip install langchain langchain-google-genai langchain-qdrant\n"
            f"Original error: {e}"
        )

    logger.info("Building LangChain RAG chain...")

    # ── Embeddings ────────────────────────────────────────────────
    embeddings = GoogleGenerativeAIEmbeddings(
        model=f"models/{settings.gemini_embed_model}",
        google_api_key=settings.gemini_api_key,
    )

    # ── Vector store (connects to existing Qdrant collection) ─────
    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=settings.qdrant_collection,
        url=f"http://{settings.qdrant_host}:{settings.qdrant_port}",
    )

    # ── LLM ───────────────────────────────────────────────────────
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.1,
    )

    # ── Prompt with citation instruction ─────────────────────────
    prompt_template = """You are a precise document assistant.
Answer ONLY from the provided context. Cite sources as [doc: filename, page: N].
If the context doesn't contain the answer, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    # ── RetrievalQA chain ─────────────────────────────────────────
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        ),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    logger.success("LangChain RAG chain ready")
    return chain


def run_langchain_query(chain, question: str) -> dict:
    """
    Run a question through the LangChain chain.

    Returns:
        {
          "answer": str,
          "sources": [{"filename": ..., "page": ..., "content": ...}],
          "mode": "langchain"
        }
    """
    result = chain.invoke({"query": question})

    sources = []
    for doc in result.get("source_documents", []):
        sources.append({
            "filename": doc.metadata.get("filename", "unknown"),
            "page":     doc.metadata.get("page_number", "?"),
            "content":  doc.page_content[:200],
            "chunk_id": doc.metadata.get("chunk_id", "")[:8],
        })

    return {
        "answer":  result["result"],
        "sources": sources,
        "mode":    "langchain",
    }
