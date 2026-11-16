"""RBI Compliance Copilot – Streamlit frontend.

Talks to the FastAPI backend only over HTTP (``API_BASE_URL``), so the UI and
the RAG service can be deployed and scaled independently.

Run:  streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import html
import os
import re

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("FRONTEND_TIMEOUT_SECONDS", "180"))

SUGGESTED_QUESTIONS = [
    "What are the requirements for periodic updation of KYC for high-risk customers?",
    "Within what time must a bank report a fraud, and to whom?",
    "What is the role of the Internal Ombudsman in a commercial bank?",
    "What controls must banks apply when outsourcing financial services?",
    "What are the conditions for issuing a credit card to a customer?",
    "How should a bank classify an account as a Non-Performing Asset?",
]

st.set_page_config(page_title="RBI Compliance Copilot", page_icon="⚖️", layout="wide")

st.markdown(
    """
<style>
:root { --navy:#0b2545; --teal:#13827a; --ink:#1d2433; --muted:#5b6475; --line:#dfe3ea; --bg-soft:#f5f7fa; }
.block-container { padding-top: 1.6rem; max-width: 1180px; }
.app-header { background: linear-gradient(120deg, var(--navy) 0%, #134074 60%, var(--teal) 100%);
  color: #fff; padding: 1.1rem 1.4rem; border-radius: 12px; margin-bottom: 1rem; }
.app-header h1 { color:#fff; font-size: 1.55rem; margin: 0; letter-spacing: .2px; }
.app-header p { color: #d9e4f2; margin: .25rem 0 0; font-size: .92rem; }
.badge { display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .76rem; font-weight: 600;
  margin-right: 6px; border: 1px solid transparent; }
.b-high { background:#e3f5ec; color:#11643d; border-color:#b8e4cc; }
.b-medium { background:#fff5dd; color:#7a5300; border-color:#f3dc9e; }
.b-low, .b-none { background:#fdeaea; color:#8a1f1f; border-color:#f3c2c2; }
.b-neutral { background: var(--bg-soft); color: var(--muted); border-color: var(--line); }
.cite-card { border:1px solid var(--line); border-left: 4px solid var(--teal); border-radius: 8px;
  padding: .65rem .85rem; margin: .45rem 0; background:#fff; }
.cite-title { font-weight: 650; color: var(--ink); font-size: .92rem; }
.cite-meta { color: var(--muted); font-size: .8rem; margin-top: 2px; }
.cite-id { display:inline-block; min-width: 2.1rem; text-align:center; background: var(--navy); color:#fff;
  border-radius: 6px; font-size: .75rem; font-weight: 700; padding: 1px 6px; margin-right: 6px; }
.excerpt { font-size: .86rem; color: var(--ink); background: var(--bg-soft); border-radius: 6px;
  padding: .55rem .7rem; border: 1px dashed var(--line); white-space: pre-wrap; }
.disclaimer { color: var(--muted); font-size: .76rem; margin-top: .8rem; }
section[data-testid="stSidebar"] .stButton button { width: 100%; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------


class BackendError(Exception):
    pass


def api(method: str, path: str, **kwargs):
    try:
        resp = requests.request(method, f"{API_BASE_URL}{path}", timeout=kwargs.pop("timeout", REQUEST_TIMEOUT), **kwargs)
    except requests.ConnectionError as exc:
        raise BackendError(
            f"Cannot reach the backend at {API_BASE_URL}. Start it with `python main.py`."
        ) from exc
    except requests.Timeout as exc:
        raise BackendError("The backend took too long to respond. Please try again.") from exc
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        if isinstance(detail, list):  # FastAPI validation errors
            detail = "; ".join(f"{'.'.join(map(str, d.get('loc', [])[1:]))}: {d.get('msg')}" for d in detail)
        raise BackendError(f"{resp.status_code}: {detail}")
    return resp.json()


@st.cache_data(ttl=15, show_spinner=False)
def fetch_health():
    return api("GET", "/health", timeout=10)


@st.cache_data(ttl=30, show_spinner=False)
def fetch_documents():
    return api("GET", "/documents", timeout=10)


def refresh_backend_cache() -> None:
    fetch_health.clear()
    fetch_documents.clear()


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def esc(value) -> str:
    return html.escape(str(value)) if value is not None else ""


def highlight_markers(answer: str) -> str:
    """Render [S1] markers as bold inline references in Markdown.

    Clause numbers at the start of a bullet ("- 1. A bank shall…") are escaped
    so Markdown does not turn them into nested numbered lists.
    """
    answer = re.sub(r"^(\s*[-*•]\s+)(\(?\d+(?:\.\d+)*)\.", r"\1\2\\.", answer, flags=re.MULTILINE)
    return re.sub(r"\[(S\d+)\]", r"**[\1]**", answer)


def pdf_link(doc_id: str, page: int) -> str:
    return f"{API_BASE_URL}/documents/{doc_id}/pdf#page={page}"


def render_citation(c: dict, key_prefix: str) -> None:
    meta = [f"Page {c['page_number']}"]
    meta.append(f"Ref: {esc(c['circular_number'])}" if c.get("circular_number") else "Ref: not printed in document")
    if c.get("publication_date"):
        meta.append(f"Dated {esc(c['publication_date'])}")
    if c.get("section"):
        meta.append(esc(c["section"][:90]))
    st.markdown(
        f"""<div class="cite-card"><div class="cite-title"><span class="cite-id">{esc(c['source_id'])}</span>{esc(c['title'])}</div>
<div class="cite-meta">{' · '.join(meta)}</div></div>""",
        unsafe_allow_html=True,
    )
    with st.expander(f"Supporting excerpt · {c['source_id']} · page {c['page_number']}"):
        st.markdown(f'<div class="excerpt">{esc(c["excerpt"])}</div>', unsafe_allow_html=True)
        links = [f"[Open PDF at page {c['page_number']}]({pdf_link(c['doc_id'], c['page_number'])})"]
        if c.get("source_url"):
            links.append(f"[Official RBI source]({c['source_url']})")
        st.markdown(" · ".join(links))
        st.caption(f"File: {c['source_filename']} · chunk {c['chunk_id']} · relevance {c.get('relevance_score')}")


def render_answer(resp: dict, idx: int) -> None:
    conf = resp["confidence"]
    level = conf["level"]
    status_badge = (
        '<span class="badge b-high">Answered from sources</span>'
        if resp["status"] == "answered"
        else '<span class="badge b-none">Insufficient evidence</span>'
    )
    provider = resp["provider"] + (f" · {resp['model']}" if resp.get("model") else "")
    st.markdown(
        f"""{status_badge}<span class="badge b-{level}">Evidence: {level.upper()} ({conf['score']:.2f})</span>
<span class="badge b-neutral">{len(resp['citations'])} citation(s)</span>
<span class="badge b-neutral">{resp['latency']['total_ms'] / 1000:.1f}s</span>
<span class="badge b-neutral">{esc(provider)}</span>""",
        unsafe_allow_html=True,
    )
    st.markdown(highlight_markers(resp["answer"]))

    if resp["citations"]:
        st.markdown("##### Sources")
        for c in resp["citations"]:
            render_citation(c, f"{idx}-{c['source_id']}")

    with st.expander("Evidence quality & retrieval details"):
        st.markdown("**Why this evidence level:** " + "; ".join(conf["reasons"]))
        g = resp.get("grounding")
        if g:
            cols = st.columns(4)
            cols[0].metric("Citation coverage", f"{g['citation_coverage']:.0%}")
            cols[1].metric("Lexical support", f"{g['lexical_support']:.0%}")
            cols[2].metric("Cited passages", g["cited_sources"])
            cols[3].metric("Invalid markers", len(g["invalid_citation_markers"]))
        lat = resp["latency"]
        st.caption(f"Retrieval {lat['retrieval_ms']:.0f} ms · generation {lat['generation_ms']:.0f} ms · query id {resp['query_id']}")
        if resp["retrieved"]:
            st.markdown("**All retrieved passages**")
            for r in resp["retrieved"]:
                st.markdown(f"- `{r['source_id']}` p.{r['page_number']} · {esc(r['title'][:110])} · relevance `{r.get('relevance_score')}`")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "response"?}

with st.sidebar:
    st.markdown("### ⚖️ RBI Compliance Copilot")
    try:
        health = fetch_health()
        ok = health["index_ready"]
        st.markdown(
            f"{'🟢' if ok else '🟠'} **Backend {'ready' if ok else 'needs documents'}**  \n"
            f"{health['documents']} documents · {health['chunks']} passages  \n"
            f"LLM: `{health['llm_provider']}`{' · `' + health['llm_model'] + '`' if health.get('llm_model') else ''}"
        )
        if not health["llm_configured"]:
            st.warning("No LLM API key configured. Answers are verbatim extracts, not synthesised answers.", icon="🔑")
        for msg in health.get("messages", []):
            if "ANTHROPIC_API_KEY" not in msg:  # already shown as the warning above
                st.caption(f"ℹ️ {msg}")
    except BackendError as exc:
        health = None
        st.error(str(exc))

    st.divider()
    st.markdown("**Search scope**")
    doc_filter: list[str] = []
    documents = []
    if health:
        try:
            documents = fetch_documents()["documents"]
        except BackendError as exc:
            st.error(str(exc))
    if documents:
        labels = {d["doc_id"]: d["title"] for d in documents}
        doc_filter = st.multiselect(
            "Limit to documents (optional)",
            options=list(labels),
            format_func=lambda d: labels[d][:70],
            placeholder="All indexed documents",
        )
    top_k = st.slider("Passages to use", 2, 10, 5)

    st.divider()
    with st.expander(f"📚 Document library ({len(documents)})"):
        for d in documents:
            st.markdown(
                f"**{esc(d['title'])}**  \n"
                f"<span style='color:#5b6475;font-size:.8rem'>{esc(d.get('circular_number') or 'Ref not detected')}"
                f" · {esc(d.get('publication_date') or 'date not detected')} · {d['page_count']} pages · {d['chunk_count']} passages</span>",
                unsafe_allow_html=True,
            )
        if not documents:
            st.caption("No documents indexed yet.")

    with st.expander("➕ Add documents"):
        uploads = st.file_uploader("Upload RBI PDFs", type=["pdf"], accept_multiple_files=True)
        if st.button("Upload & index", disabled=not uploads):
            with st.spinner("Uploading and indexing…"):
                try:
                    for f in uploads:
                        api("POST", "/documents/upload", files={"file": (f.name, f.getvalue(), "application/pdf")})
                    result = api("POST", "/ingest", json={"force": False})
                    st.success(f"Added {len(result['added'])}, updated {len(result['updated'])}. "
                               f"Total {result['total_documents']} documents.")
                    if result["skipped"]:
                        st.warning("Skipped: " + "; ".join(f"{k} ({v})" for k, v in result["skipped"].items()))
                except BackendError as exc:
                    st.error(str(exc))
            refresh_backend_cache()
        if st.button("Re-scan data/raw/"):
            with st.spinner("Indexing new or changed PDFs…"):
                try:
                    result = api("POST", "/ingest", json={"force": False})
                    st.success(f"Added {len(result['added'])}, updated {len(result['updated'])}, "
                               f"removed {len(result['removed'])}, unchanged {len(result['unchanged'])}.")
                except BackendError as exc:
                    st.error(str(exc))
            refresh_backend_cache()

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.markdown(
    """<div class="app-header"><h1>RBI Compliance Copilot</h1>
<p>Regulatory research grounded in Reserve Bank of India directions and circulars. Every answer cites the document, reference number and page.</p></div>""",
    unsafe_allow_html=True,
)

pending_question = None
if not st.session_state.messages:
    st.markdown("##### Suggested questions")
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        if cols[i % 2].button(q, key=f"suggest-{i}", use_container_width=True):
            pending_question = q

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "⚖️"):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        elif msg.get("response"):
            render_answer(msg["response"], i)
        else:
            st.error(msg["content"])

typed = st.chat_input("Ask a question about RBI regulations…", disabled=health is None)
question = typed or pending_question

if question:
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m["role"] == "user" or m.get("response")
    ][-6:]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="⚖️"):
        with st.spinner("Searching RBI documents and drafting a grounded answer…"):
            try:
                payload = {"question": question, "top_k": top_k, "history": history}
                if doc_filter:
                    payload["doc_ids"] = doc_filter
                resp = api("POST", "/query", json=payload)
                st.session_state.messages.append({"role": "assistant", "content": resp["answer"], "response": resp})
            except BackendError as exc:
                st.session_state.messages.append({"role": "assistant", "content": f"Request failed: {exc}"})
    st.rerun()

st.markdown(
    '<div class="disclaimer">Research aid only — not legal advice. Always verify against the original RBI document '
    "before acting; directions are amended frequently.</div>",
    unsafe_allow_html=True,
)
