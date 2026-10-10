"""Documents uploaded by the HR team: validate them, save them to data/docs/ and re-index.

Only the metadata block and the access level are checked here. The ingestion pipeline does the rest.
"""
import contextlib
import io
import json
import re

import ingest
import retrieval

MAX_BYTES = 200_000
REQUIRED_FIELDS = {"titulo", "documento_id", "versao", "vigente_de", "area", "acesso", "base_legal"}


def safe_name(filename):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", filename)


def validate(data, filename):
    """Return (True, '') when the upload can be indexed, otherwise (False, reason)."""
    if not filename.lower().endswith(".md"):
        return False, "O arquivo precisa ser .md."
    if len(data) > MAX_BYTES:
        return False, "O arquivo passa de 200 KB."
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False, "O arquivo precisa estar em UTF-8."
    meta, _ = ingest.parse_front_matter(text)
    missing = REQUIRED_FIELDS - set(meta)
    if missing:
        return False, "Faltam campos no bloco de metadados: " + ", ".join(sorted(missing)) + "."
    if meta["acesso"] not in ("todos", "rh"):
        return False, "O campo acesso precisa ser 'todos' ou 'rh'."
    # one document id per file: a second file with the same id would replace the first one's chunks
    if ingest.MANIFEST_FILE.exists():
        manifest = json.loads(ingest.MANIFEST_FILE.read_text(encoding="utf-8"))
        owners = [name for name, info in manifest.items()
                  if info["doc_id"] == meta["documento_id"] and name != safe_name(filename)]
        if owners:
            return False, f"O documento_id {meta['documento_id']} já é usado por {owners[0]}."
    return True, ""


def save_and_index(data, filename):
    """Save the file in data/docs/ and re-index only what changed. Returns the ingestion log."""
    (ingest.DOCS_DIR / safe_name(filename)).write_bytes(data)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        ingest.main()
    retrieval.load.cache_clear()  # the running app must load the new index
    return log.getvalue()
