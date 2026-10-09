"""Eval: measure the HR assistant against a fixed set of questions.

Run from the project folder:
    python evals/run_eval.py            # one run
    python evals/run_eval.py --runs 3   # repeat: the LLM is not deterministic, so check the variance

Exit code is 1 if a check fails, so the same script can gate a CI pipeline later.
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm import DEFAULT_GROQ_MODEL  # noqa: E402
from rag import NO_ANSWER, TOP_K, answer  # noqa: E402
from retrieval import search  # noqa: E402

QUESTIONS_FILE = ROOT / "evals" / "questions.json"
RESULTS_DIR = ROOT / "evals" / "results"
CITATION = re.compile(r"\[\d\]")


def model_name():
    if os.getenv("LLM_PROVIDER", "ollama") == "groq":
        return os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    return os.getenv("OLLAMA_MODEL", "qwen3:8b")


def run_case(case):
    """In scope: the right section was retrieved, and the answer has the fact and a citation.
    Out of scope: the assistant must refuse."""
    text = answer(case["question"])["answer"]

    if not case["in_scope"]:
        return {"id": case["id"], "passed": text == NO_ANSWER, "retrieved": None, "answer": text}

    sections = [chunk["secao"] for _, chunk in search(case["question"], k=TOP_K)]
    retrieved = any(expected in section for expected in case["expected_sections"] for section in sections)
    facts = all(re.search(pattern, text, re.IGNORECASE) for pattern in case["must_match"])
    cited = bool(CITATION.search(text))
    return {"id": case["id"], "passed": facts and cited, "retrieved": retrieved,
            "facts": facts, "cited": cited, "answer": text}


def summarize(rows):
    in_scope = [r for r in rows if r["retrieved"] is not None]
    out_of_scope = [r for r in rows if r["retrieved"] is None]
    return {
        "answers_correct": sum(r["passed"] for r in in_scope),
        "answers_total": len(in_scope),
        "retrieval_hits": sum(r["retrieved"] for r in in_scope),
        "refusals_correct": sum(r["passed"] for r in out_of_scope),
        "refusals_total": len(out_of_scope),
    }


def passed_all(s):
    return (s["answers_correct"] == s["answers_total"]
            and s["retrieval_hits"] == s["answers_total"]
            and s["refusals_correct"] == s["refusals_total"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    cases = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    runs, all_passed = [], True

    for run in range(1, args.runs + 1):
        rows = [run_case(case) for case in cases]
        summary = summarize(rows)
        all_passed &= passed_all(summary)
        runs.append({"run": run, "summary": summary, "cases": rows})

        print(f"\nRun {run}")
        for row in rows:
            status = "PASS" if row["passed"] else "FAIL"
            if row["retrieved"] is None:
                detail = "refused" if row["passed"] else "should have refused"
            else:
                detail = f"retrieval {'ok' if row['retrieved'] else 'MISS'}"
            print(f"  {status}  {row['id']:<22} {detail}")
        print(f"  answers {summary['answers_correct']}/{summary['answers_total']} | "
              f"retrieval@{TOP_K} {summary['retrieval_hits']}/{summary['answers_total']} | "
              f"refusals {summary['refusals_correct']}/{summary['refusals_total']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({
        "provider": os.getenv("LLM_PROVIDER", "ollama"),
        "model": model_name(),
        "top_k": TOP_K,
        "runs": runs,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved {out.relative_to(ROOT)}")
    print("RESULT: " + ("PASS" if all_passed else "FAIL"))
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
