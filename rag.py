"""Question answering: retrieve, filter, ask the LLM, return the answer with its sources.

Used by app.py and by the eval. Each stage reports its name through `on_step`,
which the interface uses to light up the pipeline diagram.
"""
import re
from datetime import date

from access import levels_for
from llm import generate
from retrieval import search

TOP_K = 5
MIN_SCORE = 0.45  # chunks below this cosine similarity count as unrelated (tuned on the dev eval)

NO_ANSWER = ("I couldn't find this in the HR policies. "
             "For a specific case, please contact the HR team.")

SYSTEM_PROMPT = (
    "You are the HR assistant of DataFlow Brasil. Answer in English, using only the numbered excerpts. "
    "Be brief: answer only what was asked, in at most 4 sentences. "
    "Cite the source of each statement with the number in square brackets, like [1], and no other symbols. "
    "If the excerpts do not answer the question, say you couldn't find that information in the policies. "
    "Do not approve requests, do not give legal advice, and do not disclose individual salaries."
)

# some models write citations as 【1】 or ［1］ instead of [1]; normalize so the sources panel matches
FULLWIDTH_CITATION = re.compile(r"[【［]\s*(\d+)\s*[】］]")


def normalize_citations(text):
    return FULLWIDTH_CITATION.sub(r"[\1]", text)


def is_current(chunk, today):
    """A chunk counts only once its policy is in force. ISO dates compare correctly as text."""
    return chunk["valid_from"] <= today


def answer(question, role="visitor", on_step=None):
    """Answer one question for a role. Returns the answer, its sources and whether it was refused."""
    report = on_step or (lambda name: None)
    today = date.today().isoformat()

    report("retrieve")
    found = search(question, k=TOP_K, levels=levels_for(role))

    report("filter")
    hits = [(score, chunk) for score, chunk in found if score >= MIN_SCORE and is_current(chunk, today)]
    if not hits:
        report("done")
        return {"answer": NO_ANSWER, "sources": [], "refused": True}

    report("prompt")
    context = "\n\n".join(f"[{i}] {c['title']} · {c['section']}\n{c['text']}"
                          for i, (_, c) in enumerate(hits, 1))
    user_message = f"Excerpts:\n\n{context}\n\nQuestion: {question}"

    report("llm")
    text = normalize_citations(generate(SYSTEM_PROMPT, user_message))
    report("done")
    return {"answer": text, "sources": hits, "refused": False}
