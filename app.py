import os
import streamlit as st
from dotenv import load_dotenv

from src.pdf_processor import process_pdfs
from src.rag_chain import build_rag_chain, get_answer
from src.utils import format_sources

load_dotenv()


st.set_page_config(
    page_title="DocuChat – PDF RAG Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
    /* ── Layout ── */
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    header[data-testid="stHeader"] { background: transparent; }

    /* ── App title — inherits Streamlit's foreground color ── */
    .app-title {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: inherit;
    }
    .app-subtitle {
        font-size: 0.95rem;
        opacity: 0.6;
        margin-top: -0.3rem;
        margin-bottom: 1.5rem;
        color: inherit;
    }

    /* ── Chat bubbles — use semi-transparent tints so they work in both modes ── */
    .chat-message {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
        line-height: 1.6;
        font-size: 0.95rem;
        color: inherit;
    }
    .user-msg {
        background: rgba(79, 70, 229, 0.08);
        border-left: 3px solid #4f46e5;
    }
    .assistant-msg {
        background: rgba(22, 163, 74, 0.08);
        border-left: 3px solid #16a34a;
    }

    /* ── Source citation cards ── */
    .source-card {
        background: rgba(234, 179, 8, 0.08);
        border: 1px solid rgba(234, 179, 8, 0.3);
        border-radius: 8px;
        padding: 0.6rem 0.9rem;
        margin: 0.3rem 0;
        font-size: 0.82rem;
        color: inherit;
    }
    .source-label {
        font-weight: 600;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.7;
    }

    /* ── Status badges ── */
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-ready  { background: rgba(22, 163, 74, 0.15);  color: #16a34a; }
    .badge-waiting{ background: rgba(234, 179, 8, 0.15);  color: #b45309; }

    /* ── Sidebar section headers — opacity instead of hardcoded grey ── */
    .sidebar-header {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.5;
        margin-bottom: 0.4rem;
        color: inherit;
    }

    /* ── Metric mini-cards ── */
    .metric-mini {
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 10px;
        padding: 0.8rem;
        text-align: center;
    }
    .metric-mini .val { font-size: 1.4rem; font-weight: 700; color: #4f46e5; }
    .metric-mini .lbl { font-size: 0.72rem; opacity: 0.55; margin-top: 0.1rem; }

    /* ── Dividers ── */
    .section-divider {
        border: none;
        border-top: 1px solid rgba(128,128,128,0.2);
        margin: 1rem 0;
    }

    /* ── Empty state ── */
    .empty-state {
        text-align: center;
        padding: 3rem 2rem;
        opacity: 0.6;
        color: inherit;
    }
    .empty-state .icon  { font-size: 3rem; margin-bottom: 0.8rem; }
    .empty-state .title { font-size: 1.1rem; font-weight: 600; opacity: 0.9; }
    .empty-state .hint  { font-size: 0.88rem; margin-top: 0.4rem; }
</style>
""", unsafe_allow_html=True)



def init_session() -> None:
    """Set default values for all session state keys on first run."""
    defaults: dict = {
        "chat_history": [],   # list of {"role": str, "content": str, "sources": list}
        "vector_store": None,
        "rag_chain": None,
        "processed_files": [],
        "total_chunks": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session()


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 DocuChat")
    st.markdown("*RAG-powered PDF Q&A*")
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # API key
    st.markdown("<div class='sidebar-header'>Configuration</div>", unsafe_allow_html=True)
    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        placeholder="gsk_...",
        help="Get your free key at console.groq.com",
    )
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # PDF upload
    st.markdown("<div class='sidebar-header'>Upload PDFs</div>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Model settings
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-header'>Model Settings</div>", unsafe_allow_html=True)

    model_name = st.selectbox(
        "LLM",
        [
            "llama-3.1-8b-instant",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "qwen/qwen3-32b",
        ],
        help="Free-tier models on Groq. llama-3.1-8b-instant is fastest & most permissive.",
    )
    chunk_size = st.slider(
        "Chunk Size", 300, 1200, 600, 100,
        help="Characters per text chunk. Smaller = more precise, Larger = more context.",
    )
    top_k = st.slider(
        "Sources to Retrieve", 2, 8, 4,
        help="How many document chunks to retrieve per question.",
    )

    
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    process_btn = st.button("⚡ Process PDFs", use_container_width=True, type="primary")

    if process_btn:
        if not uploaded_files:
            st.error("Please upload at least one PDF first.")
        elif not os.environ.get("GROQ_API_KEY"):
            st.error("Please enter your Groq API key.")
        else:
            with st.spinner("Processing PDFs..."):
                try:
                    vector_store, chunks_count, file_names = process_pdfs(
                        uploaded_files, chunk_size=chunk_size
                    )
                    rag_chain = build_rag_chain(
                        vector_store, model_name=model_name, top_k=top_k
                    )
                    st.session_state.vector_store = vector_store
                    st.session_state.rag_chain = rag_chain
                    st.session_state.processed_files = file_names
                    st.session_state.total_chunks = chunks_count
                    st.session_state.chat_history = []
                    st.success(
                        f"✅ Ready! Indexed {chunks_count} chunks "
                        f"from {len(file_names)} file(s)."
                    )
                except Exception as e:
                    st.error(f"Processing failed: {e}")

    
    if st.session_state.vector_store:
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-header'>Session Info</div>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                f"<div class='metric-mini'>"
                f"<div class='val'>{st.session_state.total_chunks}</div>"
                f"<div class='lbl'>Chunks</div></div>",
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown(
                f"<div class='metric-mini'>"
                f"<div class='val'>{len(st.session_state.chat_history)}</div>"
                f"<div class='lbl'>Messages</div></div>",
                unsafe_allow_html=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Loaded files:**")
        for fname in st.session_state.processed_files:
            st.markdown(f"• `{fname}`")

    # Clear chat
    if st.session_state.chat_history:
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


st.markdown("<div class='app-title'>📄 DocuChat</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='app-subtitle'>Ask anything about your PDFs — powered by LangChain + Groq</div>",
    unsafe_allow_html=True,
)


if st.session_state.vector_store:
    st.markdown(
        f"<span class='status-badge badge-ready'>"
        f"✓ Ready — {len(st.session_state.processed_files)} doc(s) loaded</span>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<span class='status-badge badge-waiting'>⏳ Upload & process PDFs to start</span>",
        unsafe_allow_html=True,
    )

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


if not st.session_state.chat_history:
    if st.session_state.vector_store:
        st.markdown("""
        <div class='empty-state'>
            <div class='icon'>💬</div>
            <div class='title'>Documents ready. Start asking!</div>
            <div class='hint'>Try: "Summarise the main points" or "What does this say about X?"</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='empty-state'>
            <div class='icon'>📤</div>
            <div class='title'>Upload your PDFs to get started</div>
            <div class='hint'>Sidebar → Upload PDFs → click "Process PDFs"</div>
        </div>
        """, unsafe_allow_html=True)
else:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f"<div class='chat-message user-msg'>🧑 {msg['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='chat-message assistant-msg'>🤖 {msg['content']}</div>",
                unsafe_allow_html=True,
            )
            if msg.get("sources"):
                with st.expander(f"📚 Sources ({len(msg['sources'])} chunks)", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(
                            f"<div class='source-card'>"
                            f"<div class='source-label'>"
                            f"Source {i} — {src['file']} · Page {src['page']}"
                            f"</div>"
                            f"<div style='margin-top:0.3rem;'>{src['snippet']}</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )


st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


with st.form(key="question_form", clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_input(
            "Ask a question",
            placeholder="e.g. What is the main conclusion of this paper?",
            label_visibility="collapsed",
            disabled=st.session_state.vector_store is None,
        )
    with col2:
        send_btn = st.form_submit_button(
            "Send",
            use_container_width=True,
            type="primary",
            disabled=st.session_state.vector_store is None,
        )


if send_btn and user_input.strip():
    question = user_input.strip()
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.spinner("🔍 Retrieving relevant chunks and generating answer..."):
        try:
            answer, sources = get_answer(
                st.session_state.rag_chain,
                question,
                st.session_state.chat_history[:-1],  # history excludes the just-added Q
            )
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": format_sources(sources),
            })
        except Exception as e:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"⚠️ Error: {e}",
                "sources": [],
            })

    st.rerun()
