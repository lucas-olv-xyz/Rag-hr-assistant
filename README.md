# HR Assistant · RAG

An assistant that answers employees' questions about HR policies. It answers only from the company's documents, cites the section it used, and refuses when the documents don't cover the question.

Built as an end-to-end RAG pipeline: ingestion with metadata, embeddings, vector search, grounded generation with citations, and an automated evaluation.

> The company (DataFlow Brasil) and its policies are fictional. Legal references follow the Brazilian CLT, but nothing here is legal advice.

## How it works

Open [docs/rag_pipeline.html](docs/rag_pipeline.html) for the diagram.

1. **Ingestion** ([ingest.py](ingest.py)): reads the Markdown policies, parses their metadata (version, validity date, legal basis), and splits each document by section. Long sections are cut into overlapping pieces. Each file gets a SHA-256 hash, so only changed documents are reprocessed. Chunks are embedded with `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions, via fastembed) and saved to a FAISS index.
2. **Retrieval** ([retrieval.py](retrieval.py)): embeds the question and returns the 5 nearest chunks by cosine similarity.
3. **Answer** ([rag.py](rag.py)): drops chunks below a relevance threshold (0.45) and chunks whose policy is not yet in force. If nothing is left, it refuses without calling the LLM. Otherwise it sends the numbered chunks to the LLM with rules: answer only from the context, cite `[n]`, don't approve requests, don't give legal advice, don't disclose individual salaries.
4. **LLM** ([llm.py](llm.py)): Groq (hosted open-weight `openai/gpt-oss-120b`, free tier) or Ollama (local), chosen by the `LLM_PROVIDER` variable. Rate-limit and server errors are retried with backoff.
5. **Interface** ([app.py](app.py)): Streamlit. Each answer shows its sources: section, version and the text used.

## Run it

```bash
pip install -r requirements.txt
python ingest.py                 # builds data/index/ (already committed; re-run after editing data/docs/)
streamlit run app.py
```

Choose the LLM provider:

- **Groq (hosted):** create a free API key at console.groq.com, then set `LLM_PROVIDER=groq` and `GROQ_API_KEY`. On Windows PowerShell: `$env:LLM_PROVIDER="groq"; $env:GROQ_API_KEY="your_key"`.
- **Ollama (local, default):** `ollama pull qwen3:8b`, then run the app. Optional variables: `OLLAMA_MODEL`, `OLLAMA_URL`.

## Evaluation

[evals/run_eval.py](evals/run_eval.py) runs a fixed set of questions ([evals/questions.json](evals/questions.json)) and checks three things: the right section was retrieved, the answer contains the expected fact and a citation, and out-of-scope questions are refused.

```bash
LLM_PROVIDER=groq GROQ_API_KEY=your_key python evals/run_eval.py --runs 3
```

| Run | Answers | Retrieval@5 | Refusals |
|---|---|---|---|
| 1 | 5/5 | 5/5 | 2/2 |
| 2 | 5/5 | 5/5 | 2/2 |
| 3 | 5/5 | 5/5 | 2/2 |

Results are saved in [evals/results/](evals/results/). The earlier runs in that folder are kept on purpose. They show two bugs the eval caught: citations written as `【1】` instead of `[1]` (fixed by normalizing the output), and rate-limit errors on the free tier (fixed with retries and lower reasoning effort).

**Limitation:** the questions were used while tuning the system, so these numbers are optimistic. A held-out question set is the next step.

## Known limitations

- Small corpus: 3 policy documents, 18 chunks.
- The free Groq tier allows about 8,000 tokens per minute for this model, so several users at once will hit rate limits.
- The relevance threshold was tuned on a small set. A valid question phrased in an unusual way may be refused.
- No authentication and no access control per document. Don't use it with real employee data.

## Project layout

```
app.py              Streamlit interface
rag.py              question answering pipeline (retrieve, filter, answer)
retrieval.py        vector search over the FAISS index
llm.py              LLM providers (Groq, Ollama) with retries
ingest.py           ingestion: parse, chunk, embed, index
data/docs/          source policy documents (Markdown)
data/index/         chunks, FAISS index and manifest produced by ingest.py
evals/              evaluation questions, runner and saved results
docs/               architecture and pipeline diagrams (open in a browser)
```
