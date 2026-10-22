"""Generation evaluation for M4: dry run (mocked Gemini, no API calls) or an explicitly requested live run.

Usage:
    python -m src.evaluate_generation --dry-run                      # all 37 dev questions, mocked, free
    python -m src.evaluate_generation --live                         # live smoke test: 3 dev questions
    python -m src.evaluate_generation --live --max-questions 8
    python -m src.evaluate_generation --live --all --confirm-all     # all 37 dev questions (costs money)
    python -m src.evaluate_generation --review results/generation/<run>.jsonl            # label answers
    python -m src.evaluate_generation --summarize-review results/generation/<run>.jsonl  # count labels
    python -m src.evaluate_generation --dry-run --verify             # M5 pipeline, all offline mocks
    python -m src.evaluate_generation --live --verify                # M5 live smoke test: 3 questions

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

--verify (M5) runs each answer through claim verification and at most one
revision. The M4 fields of every row stay the baseline result, so the M4
metrics are unchanged and directly comparable; the verified result is added
under "verified". Verification metrics keep API availability failures apart
from invalid outputs and verdicts. In a dry run every component is a mock
(sentence-split extractor, lexical verifier, scripted reviser), so the numbers
test the plumbing only. Claim-level verifier accuracy comes from
src/evaluate_verification.py, not from this module.
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
from src.self_correction import OUTCOMES, answer_question_verified

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

    With inject_unsupported=True (used by --dry-run --verify), every second
    grounded answer gets an extra sentence that no evidence supports, so the
    M5 revision path is exercised. Statuses and citations are unchanged.
    """

    model = "dry-run-mock"
    SCENARIOS = ("grounded", "grounded", "abstain", "unknown_citation", "grounded",
                 "malformed_json", "duplicate_citation", "missing_citation", "api_error")
    UNSUPPORTED_SENTENCE = "Exceptions are granted automatically by the regional ombudsman."

    def __init__(self, inject_unsupported=False):
        self.calls = 0
        self.grounded = 0
        self.inject_unsupported = inject_unsupported

    def generate(self, system_instruction, user_prompt, response_json_schema):
        scenario = self.SCENARIOS[self.calls % len(self.SCENARIOS)]
        self.calls += 1
        sections = re.findall(r'<evidence ref="E\d+" section_id="([^"]+)">', user_prompt)
        first_text = user_prompt.split("Text:\n", 1)[1].split("\n</evidence>", 1)[0] if sections else ""
        first_sentence = first_text.split(". ")[0].rstrip(".") + "."
        if scenario == "api_error":
            raise LLMRateLimitError("Simulated rate limit (dry run).")
        if scenario == "grounded":
            self.grounded += 1
            if self.inject_unsupported and self.grounded % 2 == 0:
                first_sentence += " " + self.UNSUPPORTED_SENTENCE
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


class MockReviser:
    """Deterministic stand-in for the revision request (dry runs only). Makes no network calls.

    It keeps the claims the verifier marked SUPPORTED, cites the sections that
    supported them, and abstains when none were supported. Its output is
    re-verified like any revision.
    """

    model = "dry-run-mock-reviser"

    def generate(self, system_instruction, user_prompt, response_json_schema):
        checks = re.findall(r'<claim_check claim_id="C\d+" verdict="(\w+)" evidence="([^"]*)">\nClaim: (.*)\n',
                            user_prompt)
        kept = [(evidence, claim) for verdict, evidence, claim in checks if verdict == "SUPPORTED"]
        if kept:
            citations = list(dict.fromkeys(s for evidence, _ in kept for s in evidence.split(",") if s))
            payload = {"answer": " ".join(claim for _, claim in kept), "citations": citations, "abstained": False}
        else:
            payload = {"answer": "The supplied evidence does not support an answer to this question.",
                       "citations": [], "abstained": True}
        text = json.dumps(payload)
        return LLMResponse(text=text, model=self.model, latency_s=0.0, attempts=1, finish_reason="STOP",
                           usage={"input_tokens": len(user_prompt) // 4, "output_tokens": len(text) // 4,
                                  "thinking_tokens": 0, "total_tokens": (len(user_prompt) + len(text)) // 4})


def dry_run_verification():
    """Offline extractor, verifier and reviser for --dry-run --verify."""
    from src.claims import SentenceClaimExtractor
    from src.verification import MockVerifier

    return {"extractor": SentenceClaimExtractor(), "verifier": MockVerifier(), "reviser": MockReviser()}


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
                        dense_weight=None, bm25_weight=None, progress=print, verify=None):
    """Answer each question, appending one JSON line per result. Returns (rows, interrupted).

    `verify` (M5) is None for the M4 baseline, or a dict with optional
    "extractor", "verifier" and "reviser" (missing ones default to Gemini via
    `llm`). Verified rows keep the M4 baseline fields and add "verified".
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows, interrupted = [], False
    with out_path.open("w", encoding="utf-8") as handle:
        try:
            for number, question in enumerate(questions, start=1):
                verified = None
                if verify is None:
                    result = answer_question(question["question"], index, llm, method, evidence_k, dense_weight,
                                             bm25_weight)
                else:
                    verified = answer_question_verified(question["question"], index, llm, method=method,
                                                        evidence_k=evidence_k, dense_weight=dense_weight,
                                                        bm25_weight=bm25_weight, **verify)
                    result = verified.pop("baseline")
                gold = set(question["relevant_section_ids"])
                row = {
                    "question_id": question["question_id"],
                    "category": question["category"],
                    "gold_section_ids": question["relevant_section_ids"],
                    "cites_gold_section": bool(gold & set(result["citations"])) if result["citations"] else False,
                    **result,
                    "manual_review": {"label": None, "notes": ""},
                }
                if verified is not None:
                    row["verified"] = verified
                rows.append(row)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                progress(f"  [{number}/{len(questions)}] {question['question_id']} {result['status']}"
                         + (f" -> {verified['verification']['outcome']}" if verified else ""))
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


def summarize_verification(rows):
    """M5 metrics over rows that carry a "verified" result. Rates are None when their denominator is zero."""
    verified = [r["verified"] for r in rows if "verified" in r]
    trace = [v["verification"] for v in verified]
    passes = [p for t in trace for p in t["passes"]]
    extracted = [p for p in passes if p["extraction"]["status"] == "ok"]
    completed = [p for p in extracted if p["verification"]["status"] == "complete"]

    def verdicts(stage):
        counts = {"SUPPORTED": 0, "CONTRADICTED": 0, "INSUFFICIENT_EVIDENCE": 0}
        for record in completed:
            if record["stage"] == stage:
                for verdict, count in record["verification"]["verdict_counts"].items():
                    counts[verdict] += count
        return counts

    def errors(api):
        return sum(1 for t in trace if t["error"] and t["error"].get("step") != "generation"
                   and bool(t["error"].get("api_error")) == api)

    ran = [t for t in trace if t["passes"]]          # verification actually started
    revised = [t for t in trace if t["revision_attempted"]]
    outcomes = {outcome: sum(t["outcome"] == outcome for t in trace) for outcome in OUTCOMES}
    usage = [t["usage"] for t in ran]
    costs = [t["estimated_cost_usd"] for t in ran]
    return {
        "questions": len(verified),
        "extractor": trace[0].get("extractor") if trace else None,
        "verifier": trace[0].get("verifier") if trace else None,
        "final_status_counts": {s: sum(v["status"] == s for v in verified)
                                for s in dict.fromkeys(v["status"] for v in verified)},
        "outcome_counts": outcomes,
        "answers_verified": len(ran),
        "extraction_success_rate": _rate(len(extracted), len(passes)),
        "verification_completion_rate": _rate(len(completed), len(extracted)),
        "claims_original": verdicts("original"),
        "claims_revision": verdicts("revision"),
        "original_accepted": outcomes["accepted_original"],
        "revision_attempted": len(revised),
        "revised_successfully": outcomes["accepted_revision"],
        "revision_success_rate": _rate(outcomes["accepted_revision"], len(revised)),
        "abstained": outcomes["abstained"] + outcomes["revision_unsupported"],
        "verification_api_failures": errors(api=True),
        "verification_invalid_outputs": errors(api=False),
        "supported_claims_citing_support": _rate(
            sum(1 for p in completed for c in p["verification"]["results"] if c["support_cited"]),
            sum(1 for p in completed for c in p["verification"]["results"] if c["verdict"] == "SUPPORTED")),
        "mean_verification_s": _mean(t["latency_s"] for t in ran),
        "verification_llm_calls": sum(t.get("llm_calls", 0) for t in trace),
        "verification_input_tokens": sum(u.get("input_tokens", 0) for u in usage) if usage else None,
        "verification_output_tokens": sum(u.get("output_tokens", 0) for u in usage) if usage else None,
        "verification_cost_usd": sum(costs) if costs and all(c is not None for c in costs) else None,
        "claim_accuracy": "measured separately on labeled claims: python -m src.evaluate_verification",
    }


def print_verification_summary(summary):
    mock = (summary["verifier"] or "").startswith("mock")
    print("\nM5 verification (same questions; baseline metrics above are unchanged M4 results)")
    print(f"Extractor: {summary['extractor']}   Verifier: {summary['verifier']}")
    if mock:
        print("*** Mock components: these numbers test the pipeline, not verification quality. ***")
    print("Final statuses: " + ", ".join(f"{k} {v}" for k, v in summary["final_status_counts"].items()))
    print("Outcomes: " + ", ".join(f"{k} {v}" for k, v in summary["outcome_counts"].items() if v))
    rows = (
        ("Answers sent to verification", "answers_verified", "int"),
        ("Extraction success rate", "extraction_success_rate", "rate"),
        ("Verification completion rate", "verification_completion_rate", "rate"),
        ("Original answers accepted", "original_accepted", "int"),
        ("Revisions attempted", "revision_attempted", "int"),
        ("Revised successfully", "revised_successfully", "int"),
        ("Abstained (all reasons)", "abstained", "int"),
        ("API failures (availability)", "verification_api_failures", "int"),
        ("Invalid outputs (validation)", "verification_invalid_outputs", "int"),
        ("SUPPORTED claims citing support", "supported_claims_citing_support", "rate"),
        ("Mean added latency", "mean_verification_s", "s"),
        ("Added LLM calls", "verification_llm_calls", "int"),
        ("Added input tokens", "verification_input_tokens", "int"),
        ("Added output tokens", "verification_output_tokens", "int"),
        ("Added estimated cost", "verification_cost_usd", "usd"),
    )
    for label, key, kind in rows:
        print(f"  {label:<32}{_fmt(summary[key], kind)}")
    for stage in ("original", "revision"):
        counts = summary[f"claims_{stage}"]
        print(f"  {'Claim verdicts, ' + stage:<32}" + ", ".join(f"{k} {v}" for k, v in counts.items()))
    print(f"  {'Claim-level accuracy':<32}{summary['claim_accuracy']}")
    print("A verdict is the verifier's judgment, not ground truth; it has not been shown to improve accuracy.")


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
    parser.add_argument("--verify", action="store_true",
                        help="M5: add claim verification and at most one revision (dry run: offline mocks)")
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
        if any("verified" in row for row in rows):
            print_verification_summary(summarize_verification(rows))
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
            llm = MockGemini(inject_unsupported=args.verify)
        else:
            load_env_file(PROJECT_ROOT / ".env")
            try:
                llm = GeminiClient(model=args.model, temperature=args.temperature, top_p=args.top_p)
            except LLMError as error:
                parser.error(str(error))
            for warning in sampling_warnings(llm.model, args.temperature, args.top_p):
                print(f"Warning: {warning}", file=sys.stderr)

    mode = "dry run, mocked Gemini" if args.dry_run else "live Gemini API"
    verify = (dry_run_verification() if args.dry_run else {}) if args.verify else None
    print(f"About to evaluate {len(selected)} development questions ({mode}, model {llm.model}, "
          f"retrieval {args.method}, top {args.evidence_k})" + (", with M5 verification." if args.verify else "."))
    if args.live:
        per_question = 6 if args.verify else 1
        print(f"This makes up to {per_question * len(selected)} API requests (plus bounded retries on transient "
              "errors)" + (": generation, extraction, verification and at most one revision with re-check per "
                           "question." if args.verify else "."))

    if index is None:
        index = build_search_index(methods=(args.method,))
    out_path = Path(args.out) if args.out else RESULTS_DIR / (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{'dry-run' if args.dry_run else 'live'}"
        f"{'_verify' if args.verify else ''}_{len(selected)}q.jsonl")
    start = time.perf_counter()
    rows, interrupted = run_generation_eval(selected, index, llm, out_path, args.method, args.evidence_k,
                                            verify=verify)
    print(f"Finished in {time.perf_counter() - start:.1f} s.")
    print_generation_summary(summarize_generation(rows), mode, llm.model, out_path, interrupted)
    if args.verify:
        print_verification_summary(summarize_verification(rows))
    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
