"""Question answering: retrieve, filter, ask the LLM, return the answer with its sources.

Used by app.py and, in the next step, by the eval.
"""
import re
from datetime import date

from llm import generate
from retrieval import search

TOP_K = 5
MIN_SCORE = 0.45  # chunks below this cosine similarity count as unrelated (tune with the eval)

NO_ANSWER = ("Não encontrei essa informação nas políticas de RH. "
             "Para um caso específico, fale com o time de RH.")

SYSTEM_PROMPT = (
    "Você é o assistente de RH da DataFlow Brasil. Responda em português, usando somente os trechos numerados. "
    "Seja breve: responda só o que foi perguntado, em no máximo 4 frases. "
    "Cite a fonte de cada afirmação com o número entre colchetes simples, como [1], sem outros símbolos. "
    "Se os trechos não responderem à pergunta, diga que não encontrou essa informação nas políticas. "
    "Não aprove pedidos, não dê aconselhamento jurídico e não informe salários individuais."
)


def is_current(chunk, today):
    """A chunk counts only once its policy is in force. ISO dates compare correctly as text."""
    return chunk["vigente_de"] <= today


# some models write citations as 【1】 or ［1］ instead of [1]; normalize so the sources panel matches
FULLWIDTH_CITATION = re.compile(r"[【［]\s*(\d+)\s*[】］]")


def normalize_citations(text):
    return FULLWIDTH_CITATION.sub(r"[\1]", text)


def answer(question):
    today = date.today().isoformat()
    hits = [(score, chunk) for score, chunk in search(question, k=TOP_K)
            if score >= MIN_SCORE and is_current(chunk, today)]
    if not hits:
        return {"answer": NO_ANSWER, "sources": []}  # nothing relevant: skip the LLM call entirely

    context = "\n\n".join(f"[{i}] {c['titulo']} · {c['secao']}\n{c['texto']}"
                          for i, (_, c) in enumerate(hits, 1))
    text = normalize_citations(generate(SYSTEM_PROMPT, f"Trechos:\n\n{context}\n\nPergunta: {question}"))
    return {"answer": text, "sources": hits}
