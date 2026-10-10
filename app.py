"""Streamlit interface: how the answer is built (left) and the assistant (right)."""
import os

import streamlit as st

from access import role_for
from audit import record
from rag import answer
from uploads import save_and_index, validate

st.set_page_config(page_title="HR Assistant", layout="wide")

STAGES = [
    ("question", "Question", "You type the question"),
    ("retrieve", "Search", "Embeddings + FAISS + BM25, only the documents your role can read"),
    ("filter", "Filter", "Minimum relevance and validity date"),
    ("prompt", "Prompt", "Numbered excerpts and answer rules"),
    ("llm", "LLM", "Writes the answer with citations"),
    ("done", "Answer", "Text with its sources"),
]
ORDER = [key for key, _, _ in STAGES]
COLORS = {"idle": "#6b7385", "active": "#2563eb", "done": "#16a34a"}


def load_secrets_into_env():
    """On Streamlit Community Cloud the settings come from st.secrets; llm.py reads environment variables."""
    try:
        for key in ("LLM_PROVIDER", "GROQ_API_KEY", "GROQ_MODEL"):
            if key in st.secrets and key not in os.environ:
                os.environ[key] = str(st.secrets[key])
    except Exception:
        pass  # no secrets file on this machine: environment variables are used instead


def auth_enabled():
    """Sign-in is optional: it needs an [auth] block in the secrets and a Streamlit version that has st.login."""
    try:
        return "auth" in st.secrets and hasattr(st, "login")
    except Exception:
        return False


def current_role():
    if not auth_enabled() or not st.user.is_logged_in:
        return role_for(False, None, [])
    hr_emails = list(st.secrets["HR_EMAILS"]) if "HR_EMAILS" in st.secrets else []
    return role_for(True, st.user.email, hr_emails)


def render_diagram(active=None, done=()):
    html = []
    for i, (key, title, text) in enumerate(STAGES):
        state = "done" if key in done else "active" if key == active else "idle"
        glow = "box-shadow:0 0 0 4px #2563eb33;" if state == "active" else ""
        html.append(
            f'<div style="border:2px solid {COLORS[state]};border-radius:12px;padding:10px 14px;{glow}">'
            f'<b>{i + 1}. {title}</b><br><span style="font-size:0.85rem;opacity:0.8">{text}</span></div>')
        if i < len(STAGES) - 1:
            arrow = COLORS["done"] if key in done else COLORS["idle"]
            html.append(f'<div style="text-align:center;font-size:20px;line-height:1.2;color:{arrow}">↓</div>')
    return "".join(html)


load_secrets_into_env()
role = current_role()

with st.sidebar:
    st.subheader("Profile")
    if auth_enabled():
        if st.user.is_logged_in:
            st.write(f"Signed in as {st.user.email} · role **{role}**")
            st.button("Sign out", on_click=st.logout)
        else:
            st.write("Visitor: public documents only.")
            st.button("Sign in", on_click=st.login)
    else:
        st.write(f"Role **{role}**: public documents only. HR sign-in is not configured.")

    if role == "hr":
        st.subheader("Upload a document")
        st.caption("A .md file with the metadata block. It is indexed right away. "
                   "On Streamlit Cloud the upload disappears when the app restarts; "
                   "commit it to the repository to keep it.")
        upload = st.file_uploader("Markdown file (.md)", type=["md"])
        if upload is not None and st.button("Index document"):
            ok, reason = validate(upload.getvalue(), upload.name)
            if ok:
                st.success("Document indexed.")
                st.code(save_and_index(upload.getvalue(), upload.name))
            else:
                st.error(reason)

left, right = st.columns([1, 1.2], gap="large")

with left:
    st.subheader("How the answer is built")
    diagram = st.empty()
    diagram.markdown(render_diagram(), unsafe_allow_html=True)

with right:
    st.title("HR Assistant · DataFlow Brasil")
    st.caption("Answers come from the internal HR policies. Each statement shows its source section.")
    question = st.text_input("Ask a question", placeholder="e.g. How many vacation days do I get after 1 year?")

    if question.strip():
        def on_step(name):
            idx = ORDER.index(name)
            diagram.markdown(render_diagram(active=name, done=ORDER[:idx]), unsafe_allow_html=True)

        with st.spinner("Checking the policies..."):
            result = answer(question.strip(), role=role, on_step=on_step)
        diagram.markdown(render_diagram(active="done", done=ORDER[:-1]), unsafe_allow_html=True)

        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander(f"Sources ({len(result['sources'])})"):
                for i, (score, chunk) in enumerate(result["sources"], 1):
                    st.markdown(f"**[{i}] {chunk['title']}** · {chunk['section']} · "
                                f"version {chunk['version']} · relevance {score:.2f}")
                    st.caption(chunk["text"])
        record(role, question.strip(), result["answer"], result["sources"], result["refused"])
