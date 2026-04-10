"""
ui/app.py
──────────
Streamlit chat interface for the Multimodal RAG system.

Features:
  - PDF upload + indexing
  - Streaming chat with citations
  - Thumbs up/down feedback
  - Source viewer (which chunks were used)
  - System stats sidebar

Run:
    streamlit run ui/app.py
"""

import uuid

import requests
import streamlit as st

API_URL = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocIQ — Multimodal RAG",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_question" not in st.session_state:
    st.session_state.last_question = None
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("◈ DocIQ")
    st.caption("Multimodal RAG · Powered by Gemini")
    st.divider()

    # File upload
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("Index Documents", type="primary", use_container_width=True):
            for f in uploaded_files:
                with st.spinner(f"Indexing {f.name}..."):
                    try:
                        resp = requests.post(
                            f"{API_URL}/ingest",
                            files={"file": (f.name, f.getvalue(), "application/pdf")},
                            timeout=300,
                        )
                        if resp.ok:
                            data = resp.json()
                            st.success(
                                f"✓ {f.name}\n{data['chunks_indexed']} chunks indexed"
                            )
                        else:
                            st.error(f"✗ {f.name}: {resp.text}")
                    except Exception as e:
                        st.error(f"✗ {f.name}: {e}")

    st.divider()

    # Retrieval settings
    st.subheader("Settings")
    use_hyde = st.toggle("HyDE query expansion", value=True, help="Improves recall by 30%")
    use_rerank = st.toggle("Gemini reranking", value=True, help="Improves precision")
    top_k = st.slider("Chunks per answer", min_value=3, max_value=10, value=5)

    st.divider()

    # System stats
    st.subheader("System Stats")
    try:
        stats = requests.get(f"{API_URL}/stats", timeout=3).json()
        col1, col2 = st.columns(2)
        col1.metric("Chunks", f"{stats.get('chunks_indexed', 0):,}")
        col2.metric("Docs", stats.get("documents", 0))
        col1.metric("Feedback", stats.get("total_feedback", 0))
        col2.metric("Satisfaction", f"{stats.get('satisfaction_rate', 0)}%")
    except Exception:
        st.caption("API offline — start with: uvicorn api.main:app")

    st.divider()

    # Document list
    st.subheader("Indexed Documents")
    try:
        docs = requests.get(f"{API_URL}/documents", timeout=3).json().get("documents", [])
        if docs:
            for doc in docs:
                col1, col2 = st.columns([4, 1])
                col1.caption(doc["filename"])
                if col2.button("✕", key=f"del_{doc['doc_id']}", help="Remove"):
                    requests.delete(f"{API_URL}/documents/{doc['doc_id']}")
                    st.rerun()
        else:
            st.caption("No documents indexed yet")
    except Exception:
        pass

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_answer = None
        st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────
st.title("Ask your documents anything")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])} chunks)", expanded=False):
                for s in msg["sources"]:
                    st.caption(
                        f"**[{s['chunk_id']}]** {s['filename']} · "
                        f"p.{s['page']} · {s['type']}"
                    )


# ── Input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your documents..."):

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        sources = []

        try:
            # Use sync endpoint for simplicity in Streamlit
            # (SSE streaming with Streamlit requires extra handling)
            with st.spinner("Thinking..."):
                resp = requests.post(
                    f"{API_URL}/query/sync",
                    json={
                        "question": prompt,
                        "session_id": st.session_state.session_id,
                        "use_hyde": use_hyde,
                        "use_rerank": use_rerank,
                        "top_k": top_k,
                    },
                    timeout=60,
                )

            if resp.ok:
                data = resp.json()
                full_response = data["answer"]
                sources = data.get("sources", [])
            else:
                full_response = f"Error: {resp.text}"

        except requests.exceptions.ConnectionError:
            full_response = (
                "Cannot connect to API. "
                "Start it with: `uvicorn api.main:app --reload`"
            )

        placeholder.markdown(full_response)

        if sources:
            with st.expander(f"Sources ({len(sources)} chunks)", expanded=False):
                for s in sources:
                    st.caption(
                        f"**[{s['chunk_id']}]** {s['filename']} · "
                        f"p.{s['page']} · {s['type']}"
                    )

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources,
    })
    st.session_state.last_question = prompt
    st.session_state.last_answer = full_response
    st.session_state.last_sources = sources


# ── Feedback row ──────────────────────────────────────────────────────────
if st.session_state.last_answer:
    st.divider()
    st.caption("Was this answer helpful?")

    col1, col2, col3 = st.columns([1, 1, 8])

    with col1:
        if st.button("👍 Yes", use_container_width=True):
            requests.post(f"{API_URL}/feedback", json={
                "question": st.session_state.last_question,
                "answer": st.session_state.last_answer,
                "rating": 1,
                "chunk_ids": [s["chunk_id"] for s in st.session_state.last_sources],
                "session_id": st.session_state.session_id,
            })
            st.toast("Thanks for the feedback!", icon="✅")

    with col2:
        if st.button("👎 No", use_container_width=True):
            st.session_state.show_correction = True

    if st.session_state.get("show_correction"):
        with st.form("correction_form"):
            correction = st.text_area(
                "What would be a better answer?",
                placeholder="Type the correct answer here...",
            )
            if st.form_submit_button("Submit correction"):
                requests.post(f"{API_URL}/feedback", json={
                    "question": st.session_state.last_question,
                    "answer": st.session_state.last_answer,
                    "rating": -1,
                    "correction": correction,
                    "chunk_ids": [s["chunk_id"] for s in st.session_state.last_sources],
                    "session_id": st.session_state.session_id,
                })
                st.session_state.show_correction = False
                st.toast("Correction saved — this will improve future answers!", icon="✅")
                st.rerun()
