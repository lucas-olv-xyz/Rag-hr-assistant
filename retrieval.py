"""Retrieval: find the chunks that answer a question.

Two rankings are fused:
  - dense: cosine similarity between embeddings (the FAISS index built by ingest.py)
  - sparse: BM25 keyword match, which catches exact terms such as 'férias' or '1/3'
They are combined with Reciprocal Rank Fusion (RRF). Access is checked before ranking,
so a chunk the caller may not read never reaches the ranking or the prompt.

Try it:  python retrieval.py "quantos dias de férias eu tenho?"
"""
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from fastembed import TextEmbedding

INDEX_DIR = Path(__file__).resolve().parent / "data" / "index"
CANDIDATES = 10  # how many results each ranking contributes before fusion
RRF_K = 60       # standard constant for reciprocal rank fusion
BM25_K1, BM25_B = 1.5, 0.75
STOPWORDS = set(
    "a o as os de do da dos das em no na nos nas e que um uma para por com ao aos se eu me meu minha "
    "quais qual quantos quanto quando como".split()
)


def tokens(text):
    """Lowercase and drop accents, so 'férias' and 'ferias' match. Stopwords are removed."""
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return [t for t in re.findall(r"[a-z0-9]+", folded) if t not in STOPWORDS]


def bm25_scores(query_tokens, docs_tokens):
    """BM25 score of every document for the query: the classic keyword ranking."""
    n = len(docs_tokens)
    avg_len = sum(len(d) for d in docs_tokens) / n
    doc_freq = Counter(t for d in docs_tokens for t in set(d))
    scores = []
    for doc in docs_tokens:
        counts = Counter(doc)
        score = 0.0
        for term in query_tokens:
            if counts[term] == 0:
                continue
            idf = math.log(1 + (n - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            norm = counts[term] + BM25_K1 * (1 - BM25_B + BM25_B * len(doc) / avg_len)
            score += idf * counts[term] * (BM25_K1 + 1) / norm
        scores.append(score)
    return scores


@lru_cache(maxsize=1)
def load():
    """Load the embedding model, the FAISS index and the chunks once per process."""
    meta = json.loads((INDEX_DIR / "index_meta.json").read_text(encoding="utf-8"))
    embedder = TextEmbedding(meta["embed_model"])  # same model that built the index
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    chunks = [json.loads(line) for line in (INDEX_DIR / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]
    return embedder, index, chunks


def search(question, k=5, levels=("todos",)):
    """Return up to k (dense score, chunk) pairs, best first. Only chunks whose level is in `levels` are considered."""
    embedder, index, chunks = load()
    allowed = [i for i, c in enumerate(chunks) if c.get("acesso", "rh") in levels]  # missing level = restricted
    if not allowed:
        return []
    allowed_set = set(allowed)

    vector = np.array(list(embedder.embed([question])), dtype="float32")
    faiss.normalize_L2(vector)
    scores, rows = index.search(vector, index.ntotal)  # score every chunk, then keep the allowed ones
    dense = [(float(s), int(r)) for s, r in zip(scores[0], rows[0]) if int(r) in allowed_set]
    dense_score = {r: s for s, r in dense}
    dense_rank = {r: pos for pos, (_, r) in enumerate(dense[:CANDIDATES])}

    query = tokens(question)
    sparse = []
    if query:
        sparse = sorted(zip(bm25_scores(query, [tokens(chunks[i]["texto"]) for i in allowed]), allowed), reverse=True)
    sparse_rank = {r: pos for pos, (s, r) in enumerate(sparse[:CANDIDATES]) if s > 0}

    fused = {}
    for row in set(dense_rank) | set(sparse_rank):
        fused[row] = sum(1 / (RRF_K + rank[row]) for rank in (dense_rank, sparse_rank) if row in rank)
    best = sorted(fused, key=fused.get, reverse=True)[:k]
    return [(dense_score[row], chunks[row]) for row in best]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Quantos dias de férias eu tenho após 1 ano?"
    for score, chunk in search(question, levels=("todos", "rh")):
        print(f"{score:.3f}  {chunk['id']:<12} {chunk['secao']}")
