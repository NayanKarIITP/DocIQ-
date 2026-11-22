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
    ("active_insights", None),
    ("active_page_view", None),
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

    # ── Uploaded Documents List & Management ──────────────────────────────
    docs_list = []
    try:
        doc_resp = requests.get(f"{API_URL}/documents", timeout=5)
        if doc_resp.ok:
            docs_list = doc_resp.json().get("documents", [])
    except Exception:
        pass

    with st.expander(f"📚 Indexed Documents ({len(docs_list)})", expanded=True):
        if docs_list:
            for d in docs_list:
                fname = d.get("filename", "Unknown")
                doc_id = d.get("doc_id", "")
                c_txt, c_ins, c_del = st.columns([0.64, 0.18, 0.18])
                with c_txt:
                    st.caption(f"📄 **{fname}**")
                with c_ins:
                    if st.button("✨", key=f"ins_{doc_id}", help=f"Generate Executive Summary, Cheat Sheet & Quiz for {fname}"):
                        target = doc_id or fname
                        with st.spinner(f"Analyzing {fname}..."):
                            try:
                                resp = requests.get(f"{API_URL}/documents/{target}/insights", timeout=45)
                                if resp.ok:
                                    st.session_state.active_insights = resp.json()
                                    st.rerun()
                                else:
                                    st.toast(f"⚠️ {format_error_message(resp)}")
                            except Exception as ex:
                                st.toast(f"⚠️ Error: {ex}")
                with c_del:
                    if st.button("🗑️", key=f"del_{doc_id}", help=f"Delete {fname} from index"):
                        try:
                            del_resp = requests.delete(f"{API_URL}/documents/{doc_id}", timeout=15)
                            if del_resp.ok:
                                st.toast(f"Deleted {fname}")
                                if st.session_state.active_insights and st.session_state.active_insights.get("filename") == fname:
                                    st.session_state.active_insights = None
                                st.rerun()
                            else:
                                st.error("Failed to delete")
                        except Exception as ex:
                            st.error(f"Error: {ex}")
        else:
            st.caption("No documents indexed yet. Upload a PDF above.")

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

    # Visual diagrams feature banner
    st.markdown(
        "<div style='background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.22); border-radius:8px; padding:10px 12px; margin-bottom:14px; font-size:12px; color:#cbd5e1;'>"
        "📊 <b>Adaptive Visual Diagrams:</b><br>"
        "DocIQ automatically generates interactive flowcharts for workflows, comparisons & architectures. You can also ask <i>'draw a flowchart'</i>."
        "</div>",
        unsafe_allow_html=True
    )

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

    # Export Study Notes / Q&A
    if st.session_state.messages:
        export_md = "# DocIQ Study Notes & Q&A\n\n"
        for m in st.session_state.messages:
            role_title = "🧑‍💻 User" if m["role"] == "user" else "◈ DocIQ Assistant"
            export_md += f"### {role_title}\n\n{m['content']}\n\n"
            if m.get("sources"):
                export_md += "**Sources Cited:**\n"
                for s in m["sources"]:
                    export_md += f"- `{s.get('filename', 'Doc')}` (Page {s.get('page', '?')}): {s.get('snippet', '')[:120]}...\n"
                export_md += "\n"
            export_md += "---\n\n"

        st.download_button(
            label="📥 Export Notes / Q&A (.md)",
            data=export_md,
            file_name="dociq_study_notes.md",
            mime="text/markdown",
            use_container_width=True,
        )

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_answer = None
        st.session_state.last_sources = []
        st.session_state.active_insights = None
        st.rerun()


# ── Main Header ───────────────────────────────────────────────────────────
c_header, c_mode = st.columns([5, 2])
c_header.markdown("## ◈ Multimodal Document Assistant")
c_mode.markdown(get_mode_badge_html(mode), unsafe_allow_html=True)

# ── Document Intelligence Card (When Activated) ──────────────────────────
if st.session_state.active_insights:
    insights = st.session_state.active_insights
    fname = insights.get("filename", "Document")
    
    with st.container():
        c_title, c_close = st.columns([0.92, 0.08])
        with c_title:
            st.markdown(f"### ✨ Document Intelligence: **{fname}**")
        with c_close:
            if st.button("✕", key="close_insights_panel", help="Close insights"):
                st.session_state.active_insights = None
                st.rerun()

        t_sum, t_cheat, t_quiz = st.tabs(["📌 Executive Summary", "🔑 Key Concepts Cheat Sheet", "🎯 Practice & Interview Quiz"])
        
        with t_sum:
            st.markdown(insights.get("summary", "No summary available."))
            
        with t_cheat:
            concepts = insights.get("key_concepts", [])
            if concepts:
                for c in concepts:
                    st.markdown(f"**{c.get('term', '')}**")
                    st.markdown(f"> {c.get('definition', '')}")
            else:
                st.caption("No key concepts extracted.")
                
        with t_quiz:
            quiz = insights.get("practice_quiz", [])
            if quiz:
                for q_idx, item in enumerate(quiz):
                    q_col, btn_col = st.columns([0.78, 0.22])
                    with q_col:
                        st.markdown(f"**Q{q_idx+1}:** {item.get('question', '')}")
                    with btn_col:
                        if st.button("💬 Ask in Chat", key=f"ask_quiz_{q_idx}"):
                            st.session_state.starter_query = item.get("question", "")
                            st.session_state.active_insights = None
                            st.rerun()
                    with st.expander("💡 Show Answer & Explanation"):
                        if item.get("hint"):
                            st.caption(f"*Hint:* {item['hint']}")
                        st.markdown(item.get("answer", ""))
                    st.write("")
            else:
                st.caption("No practice questions generated.")
        st.divider()

# ── Interactive Page Visualizer (When Activated) ──────────────────────────
if st.session_state.active_page_view:
    pv = st.session_state.active_page_view
    pv_fname = pv.get("filename", "Document")
    pv_page = pv.get("page_number", "?")
    pv_chunks = pv.get("chunks", [])

    with st.container():
        c_title, c_close = st.columns([0.92, 0.08])
        with c_title:
            st.markdown(f"### 📖 Page Inspector: **{pv_fname}** (Page {pv_page})")
            st.caption(f"Displaying all {len(pv_chunks)} extracted multimodal blocks from this page")
        with c_close:
            if st.button("✕", key="close_page_view", help="Close Page View"):
                st.session_state.active_page_view = None
                st.rerun()

        for c_idx, c in enumerate(pv_chunks):
            c_type = c.get("type", "text")
            c_content = c.get("content", "")

            if c_type == "table" or "[TABLE]" in c_content:
                st.markdown(f"**📊 Extracted Table #{c_idx+1}**")
                st.markdown(c_content.replace("[TABLE]", "").strip())
            elif c_type == "image":
                st.markdown(f"**🖼️ Visual / Figure #{c_idx+1}**")
                if c.get("image_b64"):
                    try:
                        import base64
                        st.image(base64.b64decode(c["image_b64"]), caption=f"Extracted from Page {pv_page}")
                    except Exception:
                        pass
                if c_content:
                    st.caption(f"**Vision Caption:** {c_content}")
            else:
                st.markdown(f"**📝 Text Excerpt #{c_idx+1}**")
                st.markdown(c_content)
            st.divider()

        col_q, col_cls = st.columns([0.78, 0.22])
        with col_q:
            if st.button(f"💬 Ask in Chat: Explain Page {pv_page} in depth", key="ask_page_deep"):
                st.session_state.starter_query = f"Can you summarize and explain all the core concepts and tables on page {pv_page} of {pv_fname}?"
                st.session_state.active_page_view = None
                st.rerun()
        with col_cls:
            if st.button("Close Page View", key="btn_close_pv"):
                st.session_state.active_page_view = None
                st.rerun()
        st.divider()

# ── Welcome Hero & Starter Pills (Shown when chat is empty) ───────────────
if not st.session_state.messages:
    st.markdown(HERO_BANNER_HTML, unsafe_allow_html=True)

    # ── Dynamic Sample Questions (tailored to uploaded documents) ────────
    sample_questions = []
    doc_names = [d.get("filename", "") for d in docs_list]
    doc_names_lower = " ".join(doc_names).lower()

    # Balanced primary questions across uploaded topics
    if "sql" in doc_names_lower or "db" in doc_names_lower:
        sample_questions.append(("🔍 Primary vs Foreign Keys", "What is the difference between primary key and foreign key?"))
    if "os" in doc_names_lower or "operating" in doc_names_lower or "process" in doc_names_lower:
        sample_questions.append(("💻 Process vs Thread", "Explain the difference between a process and a thread with a comparison table."))
    if "dsa" in doc_names_lower or "algo" in doc_names_lower or "data structure" in doc_names_lower:
        sample_questions.append(("🌳 BFS vs DFS Traversals", "Explain the difference between BFS and DFS with time and space complexity."))

    # Secondary questions per topic
    if "sql" in doc_names_lower or "db" in doc_names_lower:
        sample_questions.append(("💡 WHERE vs HAVING", "What is the difference between WHERE and HAVING in SQL?"))
        sample_questions.append(("📊 GROUP BY Clause", "Explain how the GROUP BY clause works with practical examples."))

    if "os" in doc_names_lower or "operating" in doc_names_lower or "process" in doc_names_lower:
        sample_questions.append(("🔄 CPU Scheduling Algorithms", "Explain CPU scheduling algorithms like Round Robin and FCFS."))
        sample_questions.append(("🧠 Virtual Memory & Paging", "How does virtual memory and demand paging work in operating systems?"))

    if "dsa" in doc_names_lower or "algo" in doc_names_lower or "data structure" in doc_names_lower:
        sample_questions.append(("⚡ Dynamic Programming", "When should you use Dynamic Programming vs a Greedy approach?"))
        sample_questions.append(("⏱️ Big-O Complexities", "Explain common Big-O time and space complexities with examples."))

    # For custom documents, generate dynamic questions tailored to their filenames
    for d in docs_list:
        raw_fname = d.get("filename", "")
        clean_name = raw_fname.replace(".pdf", "").replace("_", " ").strip().title()
        if not clean_name:
            continue
        q1 = (f"📝 Summary of {clean_name[:18]}", f"Provide a comprehensive summary and key takeaways from {raw_fname}.")
        q2 = (f"🔍 Key Concepts in {clean_name[:18]}", f"Explain the core concepts and definitions discussed in {raw_fname}.")
        if q1 not in sample_questions:
            sample_questions.append(q1)
        if q2 not in sample_questions:
            sample_questions.append(q2)

    # General default sample questions if no documents are indexed yet
    if not sample_questions:
        sample_questions = [
            ("📄 Summarize Key Concepts", "Summarize the key concepts and main ideas from the indexed documents."),
            ("📊 Compare Core Topics", "Create a structured comparison table of the major topics discussed in the documents."),
            ("🔍 What are the Main Takeaways?", "What are the most important takeaways and definitions from the uploaded documents?"),
        ]

    st.caption("Try asking a sample question:")
    display_q = sample_questions[:3]
    cols = st.columns(len(display_q))
    for col, (label, query) in zip(cols, display_q):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state.starter_query = query
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
        st.caption("🗺️ **Interactive Visual Diagram Included** · Switch tabs to explore the flowchart.")
        tab_text, tab_visual = st.tabs(["💬 Explanation", "📊 Visual Diagram"])
        with tab_text:
            st.markdown(clean_text, unsafe_allow_html=True)
        with tab_visual:
            st.markdown("##### 📊 Interactive Architecture & Flow Diagram")
            for idx, code in enumerate(mermaid_blocks):
                render_mermaid(code, key=f"{key_prefix}_m_{idx}")
    else:
        st.markdown(content, unsafe_allow_html=True)
        # Gentle hint informing the user about the visual diagram capability
        if len(content.strip()) > 80 and not content.startswith("Error") and "No relevant documents" not in content:
            st.caption("💡 *Tip: Want a visual flowchart? Ask 'Explain this with a diagram' or 'Draw a flowchart'.*")


def render_source_citations(sources: list[dict], key_prefix: str = "src"):
    """Renders formatted source citation cards inside an expander with deep-dive inspection & page visualizer."""
    if not sources:
        return
    with st.expander(f"📑 Sources Cited ({len(sources)} chunks)"):
        for i, s in enumerate(sources):
            fname = s.get("filename", "Document")
            page = s.get("page", 1)
            doc_id = s.get("doc_id", "")
            ctype = s.get("type", "text")
            snippet = s.get("snippet", "")
            content = s.get("content", snippet)
            score = s.get("score")

            col_card, col_act = st.columns([0.76, 0.24])
            with col_card:
                st.markdown(
                    get_source_card_html(
                        chunk_type=ctype,
                        filename=fname,
                        page=str(page),
                        chunk_id=str(s.get("chunk_id", "?")),
                    ),
                    unsafe_allow_html=True,
                )
            with col_act:
                if score is not None:
                    try:
                        st.caption(f"🎯 Match: `{float(score):.2f}`")
                    except (ValueError, TypeError):
                        pass

            # If it's a table chunk and has markdown table formatting, render as a nice table
            if ctype == "table" or "[TABLE]" in content:
                table_clean = content.replace("[TABLE]", "").strip()
                st.markdown(table_clean)
            elif snippet:
                st.markdown(f"> *\"{snippet.strip()}\"*")

            # Interactive citation inspection actions
            col_b1, col_b2 = st.columns([0.5, 0.5])
            with col_b1:
                if st.button(f"📖 Examine Page {page}", key=f"{key_prefix}_vp_{i}", help=f"View all extracted content and tables from Page {page}"):
                    target_id = doc_id if doc_id else fname
                    with st.spinner(f"Loading Page {page} from {fname}..."):
                        try:
                            resp = requests.get(f"{API_URL}/documents/{target_id}/page/{page}", timeout=15)
                            if resp.ok:
                                st.session_state.active_page_view = resp.json()
                                st.rerun()
                            else:
                                st.error(f"Could not load Page {page}.")
                        except Exception as ex:
                            st.error(f"Error loading page: {ex}")
            with col_b2:
                if st.button(f"❓ Ask about this excerpt", key=f"{key_prefix}_ask_{i}", help="Ask a question about this excerpt in chat"):
                    clean_excerpt = snippet[:140].replace("\n", " ").strip()
                    st.session_state.starter_query = f"Can you explain the following excerpt from page {page} of {fname} in detail:\n\n\"{clean_excerpt}\""
                    st.rerun()

            if i < len(sources) - 1:
                st.markdown("<hr style='margin:8px 0; border:none; border-top:1px solid rgba(255,255,255,0.08);'>", unsafe_allow_html=True)


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
        render_source_citations(msg.get("sources", []), key_prefix=f"hist_{idx}")

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
        render_source_citations(sources, key_prefix="live")

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
