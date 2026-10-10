# HR Assistant · RAG

An assistant that answers employees' questions about HR policies. It answers only from the company's documents, cites the section it used, refuses when the documents don't cover the question, and only shows restricted documents to the HR team.

Built as an end-to-end RAG pipeline: ingestion with metadata, hybrid retrieval, grounded generation with citations, access control, an audit log, and automated evaluation.

**Live demo:** https://rag-hrs-assistant.streamlit.app/

> The company (DataFlow Brasil) and its policies are fictional. Legal references follow the Brazilian CLT, but nothing here is legal advice.

## How it works

The left side of the app shows the pipeline and lights up each stage while a question is answered. Open [docs/rag_pipeline.html](docs/rag_pipeline.html) for the full diagram.

1. **Ingestion** ([ingest.py](ingest.py)): reads the Markdown policies, parses their metadata (version, validity date, legal basis, access level), and splits each document by section. Each file gets a SHA-256 hash, so only changed documents are reprocessed. Chunks are embedded with `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions, via fastembed) and saved to a FAISS index.
2. **Access** ([access.py](access.py)): a document's `acesso` field is `todos` (everyone) or `rh` (HR only). A role grants a set of levels. Unknown roles and documents without a level get the most restrictive access.
3. **Retrieval** ([retrieval.py](retrieval.py)): chunks the role may read are ranked two ways, by embedding similarity and by BM25 keyword match, and the two rankings are fused with Reciprocal Rank Fusion. Access is checked before ranking, so restricted text never reaches the prompt.
4. **Answer** ([rag.py](rag.py)): drops chunks below a relevance threshold (0.45) and chunks whose policy is not yet in force. If nothing is left, it refuses without calling the LLM. Otherwise it sends the numbered chunks to the LLM with rules: answer only from the context, cite `[n]`, don't approve requests, don't give legal advice, don't disclose individual salaries.
5. **LLM** ([llm.py](llm.py)): Groq (hosted open-weight `openai/gpt-oss-120b`, free tier) or Ollama (local), chosen by the `LLM_PROVIDER` variable. Rate-limit and server errors are retried with backoff.
6. **Audit** ([audit.py](audit.py)): every question is logged with the role (not the person), the answer and the source ids. Rows older than 90 days (`AUDIT_RETENTION_DAYS`) are deleted on each write.
7. **Interface** ([app.py](app.py)): the pipeline diagram on the left, the assistant on the right. Each answer shows its sources. HR users can upload a new `.md` document from the sidebar ([uploads.py](uploads.py)), which validates the metadata, avoids duplicate document ids, and indexes it right away.

## Run it

```bash
pip install -r requirements.txt
python ingest.py                 # builds data/index/ (already committed; re-run after editing data/docs/)
streamlit run app.py
```

Choose the LLM provider:

- **Groq (hosted):** create a free API key at console.groq.com and put it in a `.env` file in the project folder. The file is git-ignored:

  ```
  LLM_PROVIDER=groq
  GROQ_API_KEY=your_key
  ```

  Environment variables work too, and take precedence over `.env`.
- **Ollama (local, default):** `ollama pull qwen3:8b`, then run the app. Optional variables: `OLLAMA_MODEL`, `OLLAMA_URL`.

### Login for HR (optional)

Without login, everyone is a visitor and sees only `todos` documents. To give the HR team access to `rh` documents and to the upload panel:

1. Create a Google OAuth client (Google Cloud Console) with the redirect URI `https://rag-hrs-assistant.streamlit.app/oauth2callback` (and `http://localhost:8501/oauth2callback` for local runs).
2. Add this to the app's secrets (Streamlit Cloud settings, or `.streamlit/secrets.toml` locally, which is git-ignored):

   ```toml
   RH_EMAILS = ["hr.person@example.com"]

   [auth]
   redirect_uri = "https://rag-hrs-assistant.streamlit.app/oauth2callback"
   cookie_secret = "a-long-random-string"
   client_id = "..."
   client_secret = "..."
   server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
   ```

The login flow is wired up but has not been tested end to end against a real Google account yet.

## Evaluation

[evals/run_eval.py](evals/run_eval.py) runs a fixed set of questions and checks three things: the right section was retrieved, the answer contains the expected fact and a citation, and out-of-scope questions are refused. Two sets:

- [evals/questions.json](evals/questions.json) (`--set dev`): used while tuning the system.
- [evals/questions_holdout.json](evals/questions_holdout.json) (`--set holdout`): written before the hybrid retrieval and the access control were built, and not used for tuning.

```bash
LLM_PROVIDER=groq GROQ_API_KEY=your_key python evals/run_eval.py --set holdout
```

| Set | Runs | Answers | Retrieval@5 | Refusals |
|---|---|---|---|---|
| dev (9 questions) | 1 | 6/6 | 6/6 | 3/3 |
| holdout (13 questions) | 2 | 10/10 in each run | 10/10 in each run | 3/3 in each run |

Results are saved in [evals/results/](evals/results/). The earlier runs in that folder are kept on purpose: they show two bugs the eval caught, citations written as `【1】` instead of `[1]` (fixed by normalizing the output) and rate-limit errors on the free tier (fixed with retries and lower reasoning effort).

[.github/workflows/eval.yml](.github/workflows/eval.yml) runs both sets on every push and pull request. It needs a repository secret named `GROQ_API_KEY`.

**Limitation:** 13 held-out questions is a small sample, so these numbers are a sanity check, not a measure of production quality.

## Known limitations

- Small corpus: 4 policy documents, 22 chunks.
- On Streamlit Community Cloud, uploaded documents and the audit log live on the app's disk and disappear when the app restarts. To keep an uploaded document, commit it to the repository.
- The free Groq tier allows about 8,000 tokens per minute for this model, so several users at once will hit rate limits.
- The relevance threshold was tuned on a small set. A valid question phrased in an unusual way may be refused.
- Login and the upload panel are not tested end to end against a real Google account.
- Access control is per document level, not per person or per department.

## Project layout

```
app.py              Streamlit interface (pipeline diagram + assistant)
rag.py              question answering pipeline (retrieve, filter, answer)
retrieval.py        hybrid search: embeddings (FAISS) + BM25, fused with RRF
access.py           roles and the access levels they can read
llm.py              LLM providers (Groq, Ollama) with retries
ingest.py           ingestion: parse, chunk, embed, index
uploads.py          validate and index documents uploaded by HR
audit.py            audit log with a retention limit
data/docs/          source policy documents (Markdown, with access levels)
data/index/         chunks, FAISS index and manifest produced by ingest.py
evals/              evaluation sets, runner and saved results
.github/workflows/  CI: runs the evals on every push
docs/               architecture and pipeline diagrams (open in a browser)
```
