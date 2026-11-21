"""
ui/app.py  —  UPGRADED DocIQ v2.5
Features:
  - Modern Glassmorphism & Developer Dark Theme
  - Interactive Starter Prompt Chips
  - Rich Source Citation Cards (Text, Table, Image)
  - Real-Time Token Streaming with live cursor
  - Hybrid Search (BM25 + Dense RRF) toggle & status
  - Document Manager & Health Monitor
"""

import json
import os
from pathlib import Path
import re
import sys
import uuid
from typing import Optional

# Ensure project root and ui/ directory are in sys.path for Streamlit Cloud
_current_dir = Path(__file__).resolve().parent
_root_dir = _current_dir.parent
for _p in [str(_root_dir), str(_current_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import requests
import streamlit as st
import streamlit.components.v1 as components

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="DocIQ — Multimodal Document Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load External CSS & UI Templates ──────────────────────────────────────
try:
    from ui.templates import (
        HERO_BANNER_HTML,
        get_confidence_badge_html,
        get_health_pill_html,
        get_mermaid_html_container,
        get_mode_badge_html,
        get_source_card_html,
        load_stylesheet,
    )
except ImportError:
    from templates import (
        HERO_BANNER_HTML,
        get_confidence_badge_html,
        get_health_pill_html,
        get_mermaid_html_container,
        get_mode_badge_html,
        get_source_card_html,
        load_stylesheet,
    )

st.markdown(load_stylesheet(), unsafe_allow_html=True)


def format_error_message(resp) -> str:
    """Format an HTTP response error cleanly into a concise, human-readable one-liner."""
    try:
        data = resp.json()
        if isinstance(data, dict) and "detail" in data:
            detail = data["detail"]
            if isinstance(detail, list) and detail:
                detail = detail[0].get("msg", str(detail))
            return str(detail)[:120]
    except Exception:
        pass
    if resp.status_code == 502:
        return "Server is temporarily busy or restarting (502). Please retry in a moment."
    if resp.status_code == 504:
        return "Request timed out on server (504)."
    if resp.status_code == 413:
        return "File exceeds upload size limit (413)."
    return f"Server returned HTTP {resp.status_code}."


# ── Session state ─────────────────────────────────────────────────────────
for key, default in [
    ("messages", []),
    ("session_id", str(uuid.uuid4())),
    ("last_answer", None),
    ("last_question", None),
    ("last_sources", []),
    ("show_correction", False),
    ("starter_query", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ◈ DocIQ Intelligence")
    st.caption("Multimodal RAG · Hybrid Search · Gemini · Qdrant")
    st.divider()

    # Upload Documents
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF files", type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        if st.button("🚀 Index Documents (Batch)", type="primary", use_container_width=True):
            for f in uploaded_files:
                with st.spinner(f"Indexing {f.name} in batches..."):
                    try:
                        resp = requests.post(
                            f"{API_URL}/ingest",
                            files={"file": (f.name, f.getvalue(), "application/pdf")},
                            timeout=1200,
                        )
                        if resp.ok:
                            data = resp.json()
                            breakdown = data.get("breakdown", {})
                            st.success(
                                f"✅ **{f.name}**\n\n"
                                f"{data['chunks_indexed']} chunks indexed "
                                f"(📝 {breakdown.get('text',0)} · "
                                f"🖼 {breakdown.get('image',0)} · "
                                f"📊 {breakdown.get('table',0)})"
                            )
                        else:
                            st.error(f"❌ **{f.name}**: {format_error_message(resp)}")
                    except requests.exceptions.Timeout:
                        st.error(f"❌ **{f.name}**: Request timed out. Please retry.")
                    except requests.exceptions.ConnectionError:
                        st.error(f"❌ **{f.name}**: Server unavailable. Please check backend status.")
                    except Exception as e:
                        err_str = str(e)
                        if len(err_str) > 100:
                            err_str = err_str[:100] + "..."
                        st.error(f"❌ **{f.name}**: {err_str}")

    st.divider()

    # Query mode selector
    st.subheader("Query Pipeline")
    mode = st.radio(
        "Select pipeline",
        options=["Custom RAG", "LangChain RAG", "Agent (LangGraph)", "Chain-of-Thought", "Structured Output"],
        index=0,
        label_visibility="collapsed",
        help=(
            "Custom RAG: Hybrid BM25 + Qdrant Dense + Live Streaming\n"
            "LangChain RAG: Industry-standard LangChain retrieval\n"
            "Agent: Autonomous tool-calling (web search + docs + math)\n"
            "CoT: Step-by-step reasoning\n"
            "Structured: Typed response with confidence scoring"
        )
    )

    MODE_ENDPOINT = {
        "Custom RAG":          "/query/sync",
        "LangChain RAG":       "/query/langchain",
        "Agent (LangGraph)":   "/query/agent",
        "Chain-of-Thought":    "/query/cot",
        "Structured Output":   "/query/structured",
    }
    endpoint = MODE_ENDPOINT[mode]

    st.divider()

    # Retrieval settings
    st.subheader("Retrieval Engine")
    use_hybrid = st.toggle("Hybrid search (BM25 + Dense RRF)", value=True,
                           disabled=mode in ["Agent (LangGraph)", "LangChain RAG"],
                           help="Fuses keyword search (BM25) with dense vector search (Qdrant) via Reciprocal Rank Fusion.")
    use_hyde   = st.toggle("HyDE query expansion", value=False,
                           disabled=mode in ["Agent (LangGraph)", "LangChain RAG"],
                           help="Generates a hypothetical document before embedding for broader conceptual recall.")
    use_rerank = st.toggle("Gemini reranking", value=False,
                           disabled=mode in ["Agent (LangGraph)", "LangChain RAG"],
                           help="Reranks candidates with Gemini LLM prior to generation.")
    top_k      = st.slider("Sources per answer", 3, 10, 4,
                           disabled=mode == "Agent (LangGraph)")

    st.divider()

    # System status
    st.subheader("System Health")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.markdown(
            get_health_pill_html(True, health.get("chunks_indexed", 0)),
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        col1.caption(f"Model: `{health.get('model', '?')}`")
        col2.caption(f"Embed: `{health.get('embed_model', '?')}`")
    except:
        st.markdown(get_health_pill_html(False), unsafe_allow_html=True)

    st.divider()

    # Indexed documents list
    st.subheader("Indexed Documents")
    try:
        docs = requests.get(f"{API_URL}/documents", timeout=3).json().get("documents", [])
        if docs:
            for doc in docs:
                c1, c2 = st.columns([5, 1])
                c1.caption(f"📄 {doc['filename']}")
                if c2.button("✕", key=f"del_{doc['doc_id']}", help="Delete this document"):
                    requests.delete(f"{API_URL}/documents/{doc['doc_id']}")
                    st.rerun()
        else:
            st.caption("No documents indexed yet")
    except:
        pass

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_answer = None
        st.session_state.last_sources = []
        st.rerun()


# ── Main Header ───────────────────────────────────────────────────────────
c_header, c_mode = st.columns([5, 2])
c_header.markdown("## ◈ Multimodal Document Assistant")
c_mode.markdown(get_mode_badge_html(mode), unsafe_allow_html=True)

# ── Welcome Hero & Starter Pills (Shown when chat is empty) ───────────────
if not st.session_state.messages:
    st.markdown(HERO_BANNER_HTML, unsafe_allow_html=True)

    st.caption("Try asking a sample question:")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("💡 Difference between WHERE & HAVING", use_container_width=True):
            st.session_state.starter_query = "What is the difference between WHERE and HAVING in SQL?"
            st.rerun()
    with col_b:
        if st.button("📊 Explain GROUP BY with examples", use_container_width=True):
            st.session_state.starter_query = "Explain how the GROUP BY clause works with examples."
            st.rerun()
    with col_c:
        if st.button("🔍 What are primary & foreign keys?", use_container_width=True):
            st.session_state.starter_query = "What is the difference between primary key and foreign key?"
            st.rerun()
    st.divider()


# ── Mermaid Visual Renderer Helper ────────────────────────────────────────
def sanitize_mermaid_code(diagram_code: str) -> str:
    """Clean and fix common LLM formatting quirks in Mermaid code."""
    code = diagram_code.strip()
    
    # 1. Strip markdown fences if still present
    code = re.sub(r"^```(?:mermaid|mermaid\w+)?[ \t]*\n?", "", code, flags=re.IGNORECASE)
    code = re.sub(r"```$", "", code).strip()
    
    # 2. Strip leading 'mermaid' token
    if code.lower().startswith("mermaid"):
        code = code[7:].strip()
        
    # 3. Ensure valid diagram directive
    valid_starts = ("flowchart", "graph", "sequencediagram", "statediagram", "erdiagram", "classdiagram")
    if not any(code.lower().startswith(s) for s in valid_starts):
        code = "flowchart TD\n" + code

    # 4. Put flowchart TD on its own line if merged
    code = re.sub(r'^(flowchart\s+\w+)\s+', r'\1\n', code, flags=re.IGNORECASE)

    # 5. Separate connections merged onto one line: ] X --> into ]\nX -->
    code = re.sub(r'(\])\s+([A-Za-z0-9_]+\s*-->)', r'\1\n\2', code)

    # 6. Remove quotes inside arrow labels: |"text"| or |'text'| -> |text|
    code = re.sub(r'\|\s*["\']([^"\'|\n]+)["\']\s*\|', r'|\1|', code)

    # 7. Fix illegal or malformed arrows (dotted with pipes, spaces, etc.)
    code = re.sub(r'-\s*\.\s*-\s*\|', '-->|', code)
    code = re.sub(r'-\s*\.\s*->\s*\|', '-->|', code)
    code = re.sub(r'--\s*\|', '-->|', code)
    code = re.sub(r'-\s*\.\s*->', '-->', code)

    # 8. Replace unescaped & with 'and' (Mermaid reserved operator)
    code = re.sub(r'\s*&\s*', ' and ', code)

    # 9. Strip incomplete trailing arrows or dangling node ids (e.g. 'B -->' or 'B --' or 'B')
    code = re.sub(r'\b[A-Za-z0-9_]+\s*(-->|-->\|[^|\n]+\||--|->)\s*$', '', code).strip()

    return code.strip()


def render_mermaid(diagram_code: str, key: str = "mermaid"):
    """Render a Mermaid diagram using modern render with safe fallback."""
    clean_code = sanitize_mermaid_code(diagram_code)
    lines = len(clean_code.split("\n"))
    height = max(240, min(560, lines * 45 + 100))
    html_code = get_mermaid_html_container(clean_code, key)
    components.html(html_code, height=height, scrolling=True)


def extract_mermaid_and_clean_text(content: str) -> tuple[list[str], str]:
    """Extract all Mermaid code blocks and return the cleaned explanation text."""
    blocks = re.findall(r"```(?:mermaid|mermaid\w+)[ \t]*\n?([\s\S]*?)```", content, re.IGNORECASE)
    clean_text = re.sub(r"```(?:mermaid|mermaid\w+)[ \t]*\n?([\s\S]*?)```", "", content, flags=re.IGNORECASE).strip()
    
    if not blocks:
        unclosed = re.search(r"```(?:mermaid|mermaid\w+)[ \t]*\n?([\s\S]+)$", content, re.IGNORECASE)
        if unclosed:
            blocks = [unclosed.group(1)]
            clean_text = re.sub(r"```(?:mermaid|mermaid\w+)[ \t]*\n?([\s\S]+)$", "", content, flags=re.IGNORECASE).strip()
            
    return blocks, clean_text


def render_assistant_content(content: str, key_prefix: str = "hist"):
    """Renders text and any embedded Mermaid visual diagrams in clean tabs."""
    mermaid_blocks, clean_text = extract_mermaid_and_clean_text(content)

    if mermaid_blocks:
        tab_text, tab_visual = st.tabs(["💬 Explanation", "📊 Visual Diagram"])
        with tab_text:
            st.markdown(clean_text, unsafe_allow_html=True)
        with tab_visual:
            st.markdown("##### 📊 Interactive Architecture & Flow Diagram")
            for idx, code in enumerate(mermaid_blocks):
                render_mermaid(code, key=f"{key_prefix}_m_{idx}")
    else:
        st.markdown(content, unsafe_allow_html=True)


def render_source_citations(sources: list[dict]):
    """Renders formatted source citation cards inside an expander."""
    if not sources:
        return
    with st.expander(f"📑 Sources Cited ({len(sources)} chunks)"):
        for s in sources:
            st.markdown(
                get_source_card_html(
                    chunk_type=s.get("type", "text"),
                    filename=s.get("filename", "Document"),
                    page=str(s.get("page", "?")),
                    chunk_id=str(s.get("chunk_id", "?")),
                ),
                unsafe_allow_html=True,
            )
            if s.get("snippet"):
                st.caption(f"*\"{s['snippet']}...\"*")


# ── Render Message History ────────────────────────────────────────────────
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_assistant_content(msg["content"], key_prefix=f"hist_{idx}")
        else:
            st.markdown(msg["content"])

        # Show confidence badge for structured mode
        if msg.get("confidence") is not None:
            st.markdown(get_confidence_badge_html(msg["confidence"]), unsafe_allow_html=True)

        # Render rich source citation cards
        render_source_citations(msg.get("sources", []))

        if msg.get("agent_note"):
            st.info(f"🤖 {msg['agent_note']}")


# ── SSE Stream Generator Helper ───────────────────────────────────────────
def stream_sse_query(payload: dict, ctx: dict):
    """Generator that yields streaming text tokens from the SSE endpoint with full newline preservation."""
    resp = requests.post(
        f"{API_URL}/query",
        json=payload,
        stream=True,
        timeout=120,
    )
    if not resp.ok:
        yield f"Error ({resp.status_code}): {format_error_message(resp)}"
        return

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            payload_str = line[6:]
        elif line.startswith("data:"):
            payload_str = line[5:]
        else:
            continue

        try:
            data = json.loads(payload_str)
            if data.get("done"):
                break
            if "sources" in data:
                ctx["sources"] = data["sources"]
                continue
            if "token" in data:
                yield data["token"]
        except (json.JSONDecodeError, TypeError):
            if payload_str == "[DONE]":
                break
            yield payload_str


# ── Chat Input & Execution ────────────────────────────────────────────────
placeholder_texts = {
    "Custom RAG":        "Ask about your indexed documents (streaming)...",
    "LangChain RAG":     "Ask using the LangChain pipeline...",
    "Agent (LangGraph)": "Ask anything — agent will search docs, web, or calculate...",
    "Chain-of-Thought":  "Ask a complex question for step-by-step reasoning...",
    "Structured Output": "Ask a question to get a confidence-scored answer...",
}

input_prompt = st.chat_input(placeholder_texts.get(mode, "Ask a question..."))

# Handle clicked starter queries
if not input_prompt and st.session_state.starter_query:
    input_prompt = st.session_state.pop("starter_query")

if input_prompt:
    st.session_state.messages.append({"role": "user", "content": input_prompt})
    with st.chat_message("user"):
        st.markdown(input_prompt)

    with st.chat_message("assistant"):
        full_response = ""
        sources = []
        confidence = None
        agent_note = None

        if mode == "Custom RAG":
            ctx = {"sources": []}
            payload = {
                "question":   input_prompt,
                "session_id": st.session_state.session_id,
                "use_hyde":   use_hyde,
                "use_rerank": use_rerank,
                "use_hybrid": use_hybrid,
                "top_k":      top_k,
            }
            try:
                # Real-time token streaming with Streamlit's native write_stream
                full_response = st.write_stream(stream_sse_query(payload, ctx))
                sources = ctx.get("sources", [])

                # If Mermaid diagrams are present, seamlessly format into tabs via rerun
                streamed_mermaid_blocks, _ = extract_mermaid_and_clean_text(full_response)
                if streamed_mermaid_blocks:
                    st.session_state.messages.append({
                        "role":       "assistant",
                        "content":    full_response,
                        "sources":    sources,
                        "confidence": confidence,
                        "agent_note": agent_note,
                    })
                    st.session_state.last_question = input_prompt
                    st.session_state.last_answer   = full_response
                    st.session_state.last_sources  = sources
                    st.rerun()

            except requests.exceptions.ConnectionError:
                full_response = (
                    "Cannot connect to API.\n"
                    "Start it with: `python -m uvicorn api.main:app --reload`"
                )
                st.markdown(full_response)
            except Exception as e:
                full_response = f"Error: {e}"
                st.markdown(full_response)
        else:
            placeholder = st.empty()
            mode_spinner = {
                "Agent (LangGraph)": "Agent is thinking (searching web + docs)...",
                "LangChain RAG":     "Running LangChain pipeline...",
                "Chain-of-Thought":  "Reasoning step by step...",
                "Structured Output": "Generating structured response...",
            }.get(mode, "Thinking...")

            try:
                with st.spinner(mode_spinner):
                    resp = requests.post(
                        f"{API_URL}{endpoint}",
                        json={
                            "question":   input_prompt,
                            "session_id": st.session_state.session_id,
                            "use_hyde":   use_hyde,
                            "use_rerank": use_rerank,
                            "use_hybrid": use_hybrid,
                            "top_k":      top_k,
                        },
                        timeout=120,
                    )

                if resp.ok:
                    data = resp.json()
                    full_response = data.get("answer", "No answer returned")
                    sources       = data.get("sources", [])
                    confidence    = data.get("confidence")
                    agent_note    = data.get("note")

                    if mode == "Structured Output":
                        if data.get("needs_more_context"):
                            full_response += "\n\n⚠️ *The documents may not fully cover this topic.*"
                        if data.get("reasoning"):
                            full_response += f"\n\n**Reasoning:** {data['reasoning']}"

                else:
                    full_response = f"Error: {format_error_message(resp)}"

            except requests.exceptions.ConnectionError:
                full_response = (
                    "Cannot connect to API.\n"
                    "Start it with: `python -m uvicorn api.main:app --reload`"
                )
            except Exception as e:
                full_response = f"Error: {e}"

            blocks, _ = extract_mermaid_and_clean_text(full_response)
            if blocks:
                st.session_state.messages.append({
                    "role":       "assistant",
                    "content":    full_response,
                    "sources":    sources,
                    "confidence": confidence,
                    "agent_note": agent_note,
                })
                st.session_state.last_question = input_prompt
                st.session_state.last_answer   = full_response
                st.session_state.last_sources  = sources
                st.rerun()
            else:
                placeholder.markdown(full_response)

        # Confidence badge
        if confidence is not None:
            st.markdown(get_confidence_badge_html(confidence), unsafe_allow_html=True)

        # Agent note
        if agent_note:
            st.info(f"🤖 {agent_note}")

        # Rich source cards beneath assistant message
        render_source_citations(sources)

    st.session_state.messages.append({
        "role":       "assistant",
        "content":    full_response,
        "sources":    sources,
        "confidence": confidence,
        "agent_note": agent_note,
    })
    st.session_state.last_question = input_prompt
    st.session_state.last_answer   = full_response
    st.session_state.last_sources  = sources


# ── Feedback Loop ─────────────────────────────────────────────────────────
if st.session_state.last_answer:
    st.divider()
    col_fb_text, col_thumb_up, col_thumb_down, _ = st.columns([2, 1, 1, 6])
    col_fb_text.caption("Was this answer accurate?")

    with col_thumb_up:
        if st.button("👍 Yes", use_container_width=True):
            try:
                requests.post(f"{API_URL}/feedback", json={
                    "question":  st.session_state.last_question,
                    "answer":    st.session_state.last_answer,
                    "rating":    1,
                    "chunk_ids": [s.get("chunk_id","") for s in st.session_state.last_sources],
                    "session_id":st.session_state.session_id,
                })
            except:
                pass
            st.toast("Thanks for your feedback!", icon="👍")

    with col_thumb_down:
        if st.button("👎 No", use_container_width=True):
            st.session_state.show_correction = True

    if st.session_state.show_correction:
        with st.form("correction_form"):
            correction = st.text_area(
                "What would be the correct answer?",
                placeholder="Provide correction to train the feedback loop..."
            )
            if st.form_submit_button("Submit Correction"):
                try:
                    requests.post(f"{API_URL}/feedback", json={
                        "question":   st.session_state.last_question,
                        "answer":     st.session_state.last_answer,
                        "rating":     -1,
                        "correction": correction,
                        "chunk_ids":  [s.get("chunk_id","") for s in st.session_state.last_sources],
                        "session_id": st.session_state.session_id,
                    })
                except:
                    pass
                st.session_state.show_correction = False
                st.toast("Correction saved for RLHF fine-tuning!", icon="💾")
                st.rerun()
