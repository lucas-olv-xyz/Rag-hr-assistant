"""Streamlit interface: how the answer is built (left) and the assistant (right)."""
import os

import streamlit as st

from access import role_for
from audit import record
from rag import answer
from uploads import save_and_index, validate

st.set_page_config(page_title="Assistente de RH", layout="wide")

STAGES = [
    ("question", "Pergunta", "Você digita a dúvida"),
    ("retrieve", "Busca", "Embedding + FAISS + BM25, só nos documentos do seu perfil"),
    ("filter", "Filtro", "Relevância mínima e vigência"),
    ("prompt", "Prompt", "Trechos numerados e regras de resposta"),
    ("llm", "LLM", "Escreve a resposta com citações"),
    ("done", "Resposta", "Texto com as fontes"),
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
    """Login is optional: it needs an [auth] block in the secrets and a Streamlit version that has st.login."""
    try:
        return "auth" in st.secrets and hasattr(st, "login")
    except Exception:
        return False


def current_role():
    if not auth_enabled() or not st.user.is_logged_in:
        return role_for(False, None, [])
    rh_emails = list(st.secrets["RH_EMAILS"]) if "RH_EMAILS" in st.secrets else []
    return role_for(True, st.user.email, rh_emails)


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
    st.subheader("Perfil")
    if auth_enabled():
        if st.user.is_logged_in:
            st.write(f"{st.user.email} · perfil **{role}**")
            st.button("Sair", on_click=st.logout)
        else:
            st.write("Visitante: só documentos públicos.")
            st.button("Entrar", on_click=st.login)
    else:
        st.write(f"Perfil **{role}**: só documentos públicos. Login de RH não está configurado.")

    if role == "rh":
        st.subheader("Enviar documento")
        st.caption("Arquivo .md com o bloco de metadados. Ele é indexado na hora. "
                   "No Streamlit Cloud, o envio some quando o app reinicia; para manter, commite no repositório.")
        upload = st.file_uploader("Arquivo .md", type=["md"])
        if upload is not None and st.button("Indexar documento"):
            ok, reason = validate(upload.getvalue(), upload.name)
            if ok:
                st.success("Documento indexado.")
                st.code(save_and_index(upload.getvalue(), upload.name))
            else:
                st.error(reason)

left, right = st.columns([1, 1.2], gap="large")

with left:
    st.subheader("Como a resposta é construída")
    diagram = st.empty()
    diagram.markdown(render_diagram(), unsafe_allow_html=True)

with right:
    st.title("Assistente de RH · DataFlow Brasil")
    st.caption("Respostas baseadas nas políticas internas. Cada afirmação indica a seção de origem.")
    question = st.text_input("Faça uma pergunta", placeholder="Ex.: Quantos dias de férias eu tenho após 1 ano?")

    if question.strip():
        def on_step(name):
            idx = ORDER.index(name)
            diagram.markdown(render_diagram(active=name, done=ORDER[:idx]), unsafe_allow_html=True)

        with st.spinner("Consultando as políticas..."):
            result = answer(question.strip(), role=role, on_step=on_step)
        diagram.markdown(render_diagram(active="done", done=ORDER[:-1]), unsafe_allow_html=True)

        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander(f"Fontes ({len(result['sources'])})"):
                for i, (score, chunk) in enumerate(result["sources"], 1):
                    st.markdown(f"**[{i}] {chunk['titulo']}** · {chunk['secao']} · "
                                f"versão {chunk['versao']} · relevância {score:.2f}")
                    st.caption(chunk["texto"])
        record(role, question.strip(), result["answer"], result["sources"], result["refused"])
