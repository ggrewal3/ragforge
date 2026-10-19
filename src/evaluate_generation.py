"""Generation evaluation for M4: dry run (mocked Gemini, no API calls) or an explicitly requested live run.

Usage:
    python -m src.evaluate_generation --dry-run                      # all 37 dev questions, mocked, free
    python -m src.evaluate_generation --live                         # live smoke test: 3 dev questions
    python -m src.evaluate_generation --live --max-questions 8
    python -m src.evaluate_generation --live --all --confirm-all     # all 37 dev questions (costs money)
    python -m src.evaluate_generation --review results/generation/<run>.jsonl            # label answers
    python -m src.evaluate_generation --summarize-review results/generation/<run>.jsonl  # count labels

Only development questions are ever loaded: this module has no option to
evaluate the held-out test split.

Each run writes one JSON line per question as soon as it finishes, so an
interrupted run (Ctrl-C, API failure) keeps everything completed so far.

What the metrics do and do not say:
  * Structured-output validity and citation validity check form and
    existence, not truth. A valid citation can point at a section that does
    not support the claim.
  * Answer correctness is judged by a person with --review. No string
    matching and no LLM judge is used in M4.
  * Citing a gold-labeled section is shown for information only; it is not
    correctness either.
"""

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from src.evaluate import CATEGORIES, filter_split, load_evaluation_questions
from src.ingest import PROJECT_ROOT
from src.llm import PRICING_SOURCE, LLMRateLimitError, LLMResponse, load_env_file, sampling_warnings
from src.rag import DEFAULT_EVIDENCE_K, RAG_METHOD, answer_question, retrieval_options

RESULTS_DIR = PROJECT_ROOT / "results" / "generation"
DEFAULT_LIVE_QUESTIONS = 3
REVIEW_LABELS = {
    "c": "correct",
    "p": "partially_correct",
    "i": "incorrect",
    "a": "correct_abstention",
    "u": "unnecessary_abstention",
}


# ---------------------------------------------------------------------------
# Dry-run model
# ---------------------------------------------------------------------------

class MockGemini:
    """Deterministic stand-in for GeminiClient. Makes no network calls.

    It cycles through scripted behaviors so a dry run exercises every path:
    grounded answers, abstentions, unknown and duplicate citations, missing
    citations, malformed JSON and an API failure. Its outputs are synthetic,
    so dry-run metrics test the plumbing, not Gemini.
    """

    model = "dry-run-mock"
    SCENARIOS = ("grounded", "grounded", "abstain", "unknown_citation", "grounded",
                 "malformed_json", "duplicate_citation", "missing_citation", "api_error")

    def __init__(self):
        self.calls = 0

    def generate(self, system_instruction, user_prompt, response_json_schema):
        scenario = self.SCENARIOS[self.calls % len(self.SCENARIOS)]
        self.calls += 1
        sections = re.findall(r'<evidence ref="E\d+" section_id="([^"]+)">', user_prompt)
        first_text = user_prompt.split("Text:\n", 1)[1].split("\n</evidence>", 1)[0] if sections else ""
        first_sentence = first_text.split(". ")[0].rstrip(".") + "."
        if scenario == "api_error":
            raise LLMRateLimitError("Simulated rate limit (dry run).")
        payload = {
            "grounded": {"answer": f"According to the policy: {first_sentence}", "citations": sections[:1],
                         "abstained": False},
            "abstain": {"answer": "The supplied evidence does not cover this question.", "citations": [],
                        "abstained": True},
            "unknown_citation": {"answer": f"According to the policy: {first_sentence}",
                                 "citations": ["XYZ-999.9"], "abstained": False},
            "duplicate_citation": {"answer": f"According to the policy: {first_sentence}",
                                   "citations": sections[:1] * 2, "abstained": False},
            "missing_citation": {"answer": f"According to the policy: {first_sentence}", "citations": [],
                                 "abstained": False},
        }.get(scenario)
        text = "not valid json {" if scenario == "malformed_json" else json.dumps(payload)
        input_tokens = (len(system_instruction) + len(user_prompt)) // 4  # synthetic, roughly 4 characters per token
        return LLMResponse(text=text, model=self.model, latency_s=0.0, attempts=1, finish_reason="STOP",
                           usage={"input_tokens": input_tokens, "output_tokens": len(text) // 4,
                                  "thinking_tokens": 0, "total_tokens": input_tokens + len(text) // 4})


# ---------------------------------------------------------------------------
# Question selection and running
# ---------------------------------------------------------------------------

def development_questions():
    """The development split only. There is deliberately no way to request test questions here."""
    from src.ingest import load_documents

    questions = filter_split(load_evaluation_questions(documents=load_documents()), "development")
    assert all(q["split"] == "development" for q in questions)
    return questions


def select_questions(questions, max_questions=None):
    """Deterministic spread across categories: round-robin over CATEGORIES, IDs in order within each."""
    if max_questions is None or max_questions >= len(questions):
        return list(questions)
    by_category = {category: sorted((q for q in questions if q["category"] == category),
                                    key=lambda q: q["question_id"]) for category in CATEGORIES}
    selected = []
    while len(selected) < max_questions:
        for category in CATEGORIES:
            if by_category[category] and len(selected) < max_questions:
                selected.append(by_category[category].pop(0))
    return selected


def run_generation_eval(questions, index, llm, out_path, method=RAG_METHOD, evidence_k=DEFAULT_EVIDENCE_K,
                        dense_weight=None, bm25_weight=None, progress=print):
    """Answer each question, appending one JSON line per result. Returns (rows, interrupted)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows, interrupted = [], False
    with out_path.open("w", encoding="utf-8") as handle:
        try:
            for number, question in enumerate(questions, start=1):
                result = answer_question(question["question"], index, llm, method, evidence_k, dense_weight,
                                         bm25_weight)
                gold = set(question["relevant_section_ids"])
                row = {
                    "question_id": question["question_id"],
                    "category": question["category"],
                    "gold_section_ids": question["relevant_section_ids"],
                    "cites_gold_section": bool(gold & set(result["citations"])) if result["citations"] else False,
                    **result,
                    "manual_review": {"label": None, "notes": ""},
                }
                rows.append(row)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                progress(f"  [{number}/{len(questions)}] {question['question_id']} {result['status']}")
        except KeyboardInterrupt:
            interrupted = True
            progress(f"\nInterrupted after {len(rows)} of {len(questions)} questions; partial results kept.")
    return rows, interrupted


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _rate(numerator, denominator):
    return None if denominator == 0 else numerator / denominator


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def summarize_generation(rows):
    """Aggregate metrics over result rows. Rates are None when their denominator is zero."""
    attempted = [r for r in rows if r["status"] != "retrieval_error"]
    generated = [r for r in attempted if r["status"] != "llm_error"]
    valid = [r for r in generated if r["status"] in ("answered", "abstained", "citation_failure")]
    non_abstained = [r for r in valid if not r["abstained"]]
    cited = [c for r in valid for c in dict.fromkeys(r["citations"])]
    unknown = [c for r in valid for c in r["citation_validation"]["unknown"]]
    answered = [r for r in valid if r["status"] == "answered"]

    input_tokens = [r["usage"].get("input_tokens") for r in generated]
    output_tokens = [r["usage"].get("output_tokens") for r in generated]
    costs = [r["estimated_cost_usd"] for r in generated]
    return {
        "questions": len(rows),
        "status_counts": {status: sum(r["status"] == status for r in rows)
                          for status in ("answered", "abstained", "citation_failure", "invalid_output",
                                         "llm_error", "retrieval_error")},
        "generation_success_rate": _rate(len(generated), len(attempted)),
        "structured_output_validity": _rate(len(valid), len(generated)),
        "citation_id_validity": _rate(len(cited) - len(unknown), len(cited)),
        "responses_with_all_citations_valid": _rate(sum(not r["citation_validation"]["unknown"] for r in valid),
                                                    len(valid)),
        "missing_citation_rate": _rate(sum(r["citation_validation"]["missing_citation"] for r in non_abstained),
                                       len(non_abstained)),
        "abstention_rate": _rate(sum(bool(r["abstained"]) for r in valid), len(valid)),
        "cites_gold_section_rate": _rate(sum(r["cites_gold_section"] for r in answered), len(answered)),
        "mean_retrieval_s": _mean(r["latency_s"]["retrieval"] for r in rows),
        "mean_generation_s": _mean(r["latency_s"]["generation"] for r in generated),
        "mean_total_s": _mean(r["latency_s"]["total"] for r in generated),
        "input_tokens": sum(t for t in input_tokens if t is not None) if any(t is not None for t in input_tokens) else None,
        "output_tokens": sum(t for t in output_tokens if t is not None) if any(t is not None for t in output_tokens) else None,
        "estimated_cost_usd": sum(costs) if costs and all(c is not None for c in costs) else None,
        "manual_review": review_counts(rows),
        "semantic_citation_support": "not evaluated (M5)",
    }


def review_counts(rows):
    counts = {label: 0 for label in REVIEW_LABELS.values()}
    unlabeled = 0
    for row in rows:
        label = row.get("manual_review", {}).get("label")
        if label in counts:
            counts[label] += 1
        else:
            unlabeled += 1
    return {**counts, "unlabeled": unlabeled}


def _fmt(value, kind="rate"):
    if value is None:
        return "n/a"
    if kind == "rate":
        return f"{value:.3f}"
    if kind == "s":
        return f"{value:.3f} s"
    if kind == "usd":
        return f"${value:.6f}"
    return str(value)


def print_generation_summary(summary, mode, model, out_path, interrupted=False):
    print(f"\nRAGForge — Generation Evaluation ({mode})")
    print("Split: development")
    print(f"Model: {model}")
    print(f"Questions: {summary['questions']}" + ("  (interrupted: partial run)" if interrupted else ""))
    print("Status counts: " + ", ".join(f"{k} {v}" for k, v in summary["status_counts"].items()))
    rows = (
        ("Generation success rate", "generation_success_rate", "rate"),
        ("Structured-output validity", "structured_output_validity", "rate"),
        ("Citation ID validity", "citation_id_validity", "rate"),
        ("Responses, all IDs valid", "responses_with_all_citations_valid", "rate"),
        ("Missing-citation rate", "missing_citation_rate", "rate"),
        ("Abstention rate", "abstention_rate", "rate"),
        ("Cites a gold section*", "cites_gold_section_rate", "rate"),
        ("Mean retrieval latency", "mean_retrieval_s", "s"),
        ("Mean generation latency", "mean_generation_s", "s"),
        ("Mean total latency", "mean_total_s", "s"),
        ("Input tokens (sum)", "input_tokens", "int"),
        ("Output tokens (sum)", "output_tokens", "int"),
        ("Estimated cost", "estimated_cost_usd", "usd"),
    )
    for label, key, kind in rows:
        print(f"  {label:<28}{_fmt(summary[key], kind)}")
    print(f"  {'Semantic citation support':<28}{summary['semantic_citation_support']}")
    review = summary["manual_review"]
    print(f"  {'Manual review':<28}" + ", ".join(f"{k} {v}" for k, v in review.items()))
    print("\n* Informational only: citing a gold-labeled section is not the same as answering correctly.")
    print("Citation validity means the cited IDs were supplied to the model, not that they support the answer.")
    if summary["estimated_cost_usd"] is not None:
        print(f"Cost is a paid-tier estimate from {PRICING_SOURCE}; free-tier usage costs nothing.")
    print(f"Results: {out_path}")


# ---------------------------------------------------------------------------
# Manual review
# ---------------------------------------------------------------------------

def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def save_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def review(path, input_fn=input, output_fn=print):
    """Label unlabeled answers interactively; saves after every label. Returns the number labeled."""
    rows = load_rows(path)
    keys = ", ".join(f"{k}={v}" for k, v in REVIEW_LABELS.items())
    labeled = 0
    for row in rows:
        if row.get("manual_review", {}).get("label") in REVIEW_LABELS.values() or row["answer"] is None:
            continue
        output_fn(f"\n{row['question_id']} [{row['category']}] {row['question']}")
        output_fn(f"Status: {row['status']}  Abstained: {row['abstained']}  Citations: {row['citations']}")
        output_fn(f"Answer: {row['answer']}")
        output_fn(f"Gold sections (reference only): {row['gold_section_ids']}")
        for source in row["sources"]:
            if source["section_id"] in row["citations"] or source["section_id"] in row["gold_section_ids"]:
                output_fn(f"  [{source['section_id']}] {source['text']}")
        while True:
            choice = input_fn(f"Label ({keys}, s=skip, q=quit): ").strip().lower()
            if choice in REVIEW_LABELS or choice in ("s", "q"):
                break
        if choice == "q":
            break
        if choice == "s":
            continue
        row["manual_review"] = {"label": REVIEW_LABELS[choice], "notes": input_fn("Notes (optional): ").strip()}
        save_rows(path, rows)
        labeled += 1
    return labeled


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description="RAGForge M4 generation evaluation (development split only).")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="mocked Gemini responses; no API calls, no cost")
    mode.add_argument("--live", action="store_true", help="real Gemini API calls (costs money on the paid tier)")
    mode.add_argument("--review", metavar="RESULTS_JSONL", help="manually label answers in a results file")
    mode.add_argument("--summarize-review", metavar="RESULTS_JSONL", help="print metrics and labels for a results file")
    parser.add_argument("--max-questions", type=int, default=None,
                        help=f"limit the number of dev questions (live default {DEFAULT_LIVE_QUESTIONS}; dry run: all)")
    parser.add_argument("--all", action="store_true", help="live: evaluate all development questions")
    parser.add_argument("--confirm-all", action="store_true",
                        help="required with --live when evaluating all development questions")
    parser.add_argument("--method", default=RAG_METHOD, help=f"retrieval method (default {RAG_METHOD})")
    parser.add_argument("--evidence-k", type=int, default=DEFAULT_EVIDENCE_K)
    parser.add_argument("--model", default=None, help="Gemini model ID (live only)")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--out", default=None, help="output JSONL path (default: results/generation/<timestamp>.jsonl)")
    return parser


def main(argv=None, llm=None, index=None):
    from src.llm import GeminiClient, LLMError
    from src.retrieve import ALL_METHODS, build_search_index

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.review:
        labeled = review(args.review)
        print(f"Labeled {labeled} answers in {args.review}.")
        return 0
    if args.summarize_review:
        rows = load_rows(args.summarize_review)
        model = rows[0]["model"] if rows else "n/a"
        print_generation_summary(summarize_generation(rows), "saved results", model, args.summarize_review)
        return 0

    if args.method not in ALL_METHODS:
        parser.error(f"--method must be one of {ALL_METHODS}.")
    try:
        retrieval_options(args.method)
    except ValueError as error:
        parser.error(str(error))
    if args.max_questions is not None and args.max_questions < 1:
        parser.error("--max-questions must be at least 1.")
    if args.all and args.max_questions is not None:
        parser.error("use either --all or --max-questions, not both.")

    questions = development_questions()
    if args.live:
        limit = len(questions) if args.all else (args.max_questions or DEFAULT_LIVE_QUESTIONS)
        if limit >= len(questions) and not args.confirm_all:
            parser.error(f"A live run over all {len(questions)} development questions makes {len(questions)} paid "
                         "API calls; add --confirm-all to proceed, or use --max-questions for a smaller run.")
    else:
        limit = args.max_questions
    selected = select_questions(questions, limit)

    if llm is None:
        if args.dry_run:
            llm = MockGemini()
        else:
            load_env_file(PROJECT_ROOT / ".env")
            try:
                llm = GeminiClient(model=args.model, temperature=args.temperature, top_p=args.top_p)
            except LLMError as error:
                parser.error(str(error))
            for warning in sampling_warnings(llm.model, args.temperature, args.top_p):
                print(f"Warning: {warning}", file=sys.stderr)

    mode = "dry run, mocked Gemini" if args.dry_run else "live Gemini API"
    print(f"About to evaluate {len(selected)} development questions ({mode}, model {llm.model}, "
          f"retrieval {args.method}, top {args.evidence_k}).")
    if args.live:
        print(f"This makes up to {len(selected)} API requests (plus bounded retries on transient errors).")

    if index is None:
        index = build_search_index(methods=(args.method,))
    out_path = Path(args.out) if args.out else RESULTS_DIR / (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{'dry-run' if args.dry_run else 'live'}_{len(selected)}q.jsonl")
    start = time.perf_counter()
    rows, interrupted = run_generation_eval(selected, index, llm, out_path, args.method, args.evidence_k)
    print(f"Finished in {time.perf_counter() - start:.1f} s.")
    print_generation_summary(summarize_generation(rows), mode, llm.model, out_path, interrupted)
    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
