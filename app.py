"""Streamlit interface for the HR assistant. Run with: streamlit run app.py"""
import os

import streamlit as st

from rag import answer


def load_secrets_into_env():
    """On Streamlit Community Cloud, secrets come from st.secrets; llm.py reads environment variables."""
    try:
        for key in ("LLM_PROVIDER", "GROQ_API_KEY", "GROQ_MODEL"):
            if key in st.secrets and key not in os.environ:
                os.environ[key] = str(st.secrets[key])
    except Exception:
        pass  # no secrets file on this machine: the environment variables are used instead


load_secrets_into_env()

st.set_page_config(page_title="Assistente de RH", layout="centered")
st.title("Assistente de RH · DataFlow Brasil")
st.caption("Respostas baseadas nas políticas internas. Cada afirmação indica a seção de origem.")

question = st.text_input("Faça uma pergunta", placeholder="Ex.: Quantos dias de férias eu tenho após 1 ano?")

if question.strip():
    with st.spinner("Consultando as políticas..."):
        result = answer(question.strip())
    st.markdown(result["answer"])

    if result["sources"]:
        with st.expander(f"Fontes ({len(result['sources'])})"):
            for i, (score, chunk) in enumerate(result["sources"], 1):
                st.markdown(f"**[{i}] {chunk['titulo']}** · {chunk['secao']} · versão {chunk['versao']} · relevância {score:.2f}")
                st.caption(chunk["texto"])
