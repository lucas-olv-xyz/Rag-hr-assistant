"""Answer step: send the prompt to an LLM.

Provider is chosen by the LLM_PROVIDER environment variable, read on each call:
  - "ollama" (default): local model, for development on your own PC.
  - "groq": hosted open-weight models on Groq's free tier, for the deployed app.

Groq needs GROQ_API_KEY, read from the environment or from a local .env file. Never put the key in the code or the repo.
"""
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")  # local development; the deployed app gets its secrets from the host

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"  # open-weight, Apache 2.0; if this 404s, check the model list
RETRY_STATUS = {429, 500, 502, 503, 504}  # rate limit or server hiccup: worth waiting and retrying
MAX_ATTEMPTS = 4


def generate(system, user):
    provider = os.getenv("LLM_PROVIDER", "ollama")
    if provider == "groq":
        return _groq(system, user)
    if provider == "ollama":
        return _ollama(system, user)
    raise ValueError(f"LLM_PROVIDER must be 'groq' or 'ollama', got {provider!r}")


def _groq(system, user):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set")
    model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    body = {
        "model": model,
        "temperature": 0.2,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    if model.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = "low"  # fewer reasoning tokens: the free tier limits tokens per minute
    for attempt in range(MAX_ATTEMPTS):
        r = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {key}"}, timeout=60, json=body)
        if r.status_code not in RETRY_STATUS or attempt == MAX_ATTEMPTS - 1:
            break
        time.sleep(_retry_wait(r, attempt))
    if not r.ok:  # show Groq's own message, e.g. "model not found" or the rate limit details
        raise RuntimeError(f"Groq returned {r.status_code} for model {model!r}: {r.text}")
    return r.json()["choices"][0]["message"]["content"]


def _retry_wait(r, attempt):
    """Seconds to wait before retrying: Groq's own hint when it gives one, otherwise exponential backoff."""
    hint = re.search(r"try again in ([\d.]+)s", r.text)
    if hint:
        return float(hint.group(1)) + 0.5
    return min(2 ** attempt * 2, 30)


def _ollama(system, user):
    url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    r = requests.post(f"{url}/api/chat", timeout=300, json={
        "model": model,
        "stream": False,
        "think": False,
        "options": {"num_ctx": 4096, "temperature": 0.2},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    })
    r.raise_for_status()
    return r.json()["message"]["content"]
