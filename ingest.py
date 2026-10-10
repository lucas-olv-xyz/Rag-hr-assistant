"""Ingestion: read documents, split them into chunks, embed them, save the vector index.

Run from the project folder:
    python ingest.py

Re-running is safe: unchanged documents are skipped, and the index is rebuilt only when something changed.
"""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from fastembed import TextEmbedding

ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "data" / "docs"
INDEX_DIR = ROOT / "data" / "index"
CHUNKS_FILE = INDEX_DIR / "chunks.jsonl"
MANIFEST_FILE = INDEX_DIR / "manifest.json"
INDEX_FILE = INDEX_DIR / "faiss.index"
INDEX_META_FILE = INDEX_DIR / "index_meta.json"

# Must match the model used at query time (retrieval.py reads it from index_meta.json).
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

MAX_WORDS = 150  # longest chunk, in words
OVERLAP = 30     # words shared between pieces of a long section
DISCLAIMER = "*This is a fictional"  # closing note of every document; it is not useful for answers

HEADING = re.compile(r"^(#{1,2})\s+(.*)$", re.MULTILINE)


def parse_front_matter(text):
    """Split the '---' metadata block from the body."""
    if not text.startswith("---"):
        return {}, text
    _, block, body = text.split("---", 2)
    meta = {}
    for line in block.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta, body


def clean(body):
    """Remove the disclaimer line that every document ends with."""
    lines = [line for line in body.splitlines() if not line.startswith(DISCLAIMER)]
    return "\n".join(lines).strip()


def split_sections(body):
    """Return (heading path, text) for each heading that has text under it."""
    sections, h1 = [], ""
    matches = list(HEADING.finditer(body))
    for i, m in enumerate(matches):
        level, title = len(m.group(1)), m.group(2).strip()
        if level == 1:
            h1 = title
            path = title
        else:
            path = f"{h1} > {title}" if h1 else title
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        content = body[m.end():end].strip()
        if content:
            sections.append((path, content))
    return sections


def window(text):
    """Cut a long section into overlapping pieces. Short sections stay whole."""
    words = text.split()
    if len(words) <= MAX_WORDS:
        return [" ".join(words)]
    step = MAX_WORDS - OVERLAP
    return [" ".join(words[i:i + MAX_WORDS]) for i in range(0, len(words) - OVERLAP, step)]


def chunk_document(path, raw_text):
    meta, body = parse_front_matter(raw_text)
    chunks = []
    for section, content in split_sections(clean(body)):
        for piece in window(content):
            chunks.append({
                "id": f"{meta['document_id']}:{len(chunks)}",
                "doc_id": meta["document_id"],
                "title": meta["title"],
                "version": meta["version"],
                "valid_from": meta["valid_from"],
                "area": meta["area"],
                "legal_basis": meta["legal_basis"],
                "access": meta.get("access", "hr"),  # fail closed: a document without a level is restricted
                "source": path.name,
                "section": section,
                # the section title goes into the text, so each chunk makes sense on its own
                "text": f"{section}\n{piece}",
                "words": len(piece.split()),
            })
    return chunks


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_index(chunks):
    """Embed every chunk and save a FAISS index. Row i of the index is line i of chunks.jsonl."""
    embedder = TextEmbedding(EMBED_MODEL)
    vectors = np.array(list(embedder.embed([c["text"] for c in chunks])), dtype="float32")
    faiss.normalize_L2(vectors)  # unit length, so inner product = cosine similarity
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(INDEX_FILE))
    INDEX_META_FILE.write_text(json.dumps(
        {"embed_model": EMBED_MODEL, "dim": int(vectors.shape[1]), "count": len(chunks)}, indent=2), encoding="utf-8")
    print(f"index  faiss.index: {len(chunks)} vectors of {vectors.shape[1]} dims")


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8")) if MANIFEST_FILE.exists() else {}
    chunks = []
    if CHUNKS_FILE.exists():
        chunks = [json.loads(line) for line in CHUNKS_FILE.read_text(encoding="utf-8").splitlines() if line]

    files = {p.name: p for p in sorted(DOCS_DIR.glob("*.md"))}
    changed = False

    # a file removed from data/docs/ also leaves the index
    for name in set(manifest) - set(files):
        doc_id = manifest.pop(name)["doc_id"]
        chunks = [c for c in chunks if c["doc_id"] != doc_id]
        changed = True
        print(f"remove {name}: its chunks were dropped")

    for name, path in files.items():
        digest = file_hash(path)
        if manifest.get(name, {}).get("hash") == digest:
            print(f"skip   {name} (unchanged)")
            continue

        new_chunks = chunk_document(path, path.read_text(encoding="utf-8"))
        doc_id = new_chunks[0]["doc_id"]
        chunks = [c for c in chunks if c["doc_id"] != doc_id]  # replace the previous version
        chunks += new_chunks
        changed = True

        manifest[name] = {
            "hash": digest,
            "doc_id": doc_id,
            "chunks": len(new_chunks),
            "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        print(f"chunk  {name}: {len(new_chunks)} chunks")
        for c in new_chunks:
            print(f"         {c['id']:<14} {c['words']:>4} words  {c['section']}")

    CHUNKS_FILE.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")
    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done: {len(chunks)} chunks in {CHUNKS_FILE.relative_to(ROOT)}")

    # the index must match chunks.jsonl row by row, so rebuild it whenever the chunks changed
    if chunks and (changed or not INDEX_FILE.exists()):
        build_index(chunks)
    elif not changed:
        print("index  unchanged, skipped")


if __name__ == "__main__":
    main()
