"""Retrieval: find the chunks closest to a question.

Uses the index written by ingest.py. Run a quick check with:
    python retrieval.py "quantos dias de ferias eu tenho?"
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from fastembed import TextEmbedding

INDEX_DIR = Path(__file__).parent / "data" / "index"


@lru_cache(maxsize=1)
def load():
    """Load the embedding model, the FAISS index and the chunk texts once."""
    meta = json.loads((INDEX_DIR / "index_meta.json").read_text(encoding="utf-8"))
    embedder = TextEmbedding(meta["embed_model"])  # same model that built the index
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    chunks = [json.loads(line) for line in (INDEX_DIR / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]
    return embedder, index, chunks


def search(question, k=5):
    """Return up to k (score, chunk) pairs, best match first."""
    embedder, index, chunks = load()
    vector = np.array(list(embedder.embed([question])), dtype="float32")
    faiss.normalize_L2(vector)
    scores, rows = index.search(vector, min(k, index.ntotal))
    return [(float(score), chunks[row]) for score, row in zip(scores[0], rows[0])]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Quantos dias de férias eu tenho após 1 ano?"
    for score, chunk in search(question):
        print(f"{score:.3f}  {chunk['id']:<12} {chunk['secao']}")
