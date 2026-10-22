"""Claim-verification benchmark (M5): score a verifier against hand-labeled claim-evidence examples.

Usage:
    python -m src.evaluate_verification --dry-run                        # MockVerifier, all examples, free
    python -m src.evaluate_verification --live                           # Gemini verifier, 6 examples
    python -m src.evaluate_verification --live --max-examples 12
    python -m src.evaluate_verification --live --all --confirm-all       # every example (one API call each)
    python -m src.evaluate_verification --summarize results/verification/<run>.jsonl

Labels live in data/evaluations/verification_claims.json. Predictions are
written to results/verification/*.jsonl and contain no labels; scoring joins
the two by example_id. Each example is one claim verified against its listed
evidence sections, in one verifier call.

Reported separately, so that an unavailable API is never mistaken for a wrong verdict:
  * completion: predictions that passed validation;
  * API failures (LLMError / VerifierUnavailable) and invalid outputs;
  * accuracy over completed predictions, and strict accuracy that counts
    every failure as wrong;
  * a confusion matrix, per-verdict precision and recall, per-phenomenon accuracy;
  * false SUPPORTED: a CONTRADICTED or INSUFFICIENT_EVIDENCE claim accepted
    as SUPPORTED, the error that lets a hallucination through.

The MockVerifier is a lexical heuristic. Its scores describe the heuristic,
never a model, and every report on it says so. The benchmark is meant for
later comparison with the M6 fine-tuned local verifier on the same examples.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from src.claims import Claim
from src.ingest import PROJECT_ROOT
from src.llm import PRICING_SOURCE, LLMError, estimate_cost_usd, load_env_file, sampling_warnings
from src.prompts import VERDICTS
from src.verification import GeminiVerifier, InvalidVerifierOutput, MockVerifier, VerifierUnavailable, parse_verdicts

BENCHMARK_PATH = PROJECT_ROOT / "data" / "evaluations" / "verification_claims.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "verification"
DEFAULT_LIVE_EXAMPLES = 6
EXAMPLE_FIELDS = {"example_id", "claim", "evidence_section_ids", "label", "label_evidence_section_ids", "phenomena",
                  "decisive_text", "rationale"}


class BenchmarkValidationError(ValueError):
    pass


def section_index(documents):
    """section_id -> chunk-like evidence dict (the shape verifiers receive)."""
    sections = {}
    for document in documents:
        for section in document["sections"]:
            sections[section["section_id"]] = {
                "chunk_id": f"{section['section_id']}#0", "document_id": document["document_id"],
                "section_id": section["section_id"], "title": document["title"],
                "section_title": section["section_title"], "text": section["text"]}
    return sections


def validate_benchmark(benchmark, sections):
    """Structural and label-integrity checks. Raises BenchmarkValidationError."""
    examples = benchmark.get("examples")
    if benchmark.get("split") != "development" or not isinstance(examples, list) or not examples:
        raise BenchmarkValidationError("Benchmark must be a development split with a non-empty example list.")
    seen = set()
    for example in examples:
        example_id = example.get("example_id")
        if set(example) != EXAMPLE_FIELDS:
            raise BenchmarkValidationError(f"{example_id}: fields must be exactly {sorted(EXAMPLE_FIELDS)}.")
        if example_id in seen:
            raise BenchmarkValidationError(f"Duplicate example_id {example_id}.")
        seen.add(example_id)
        if example["label"] not in VERDICTS:
            raise BenchmarkValidationError(f"{example_id}: unknown label {example['label']!r}.")
        if not example["claim"].strip() or not example["evidence_section_ids"]:
            raise BenchmarkValidationError(f"{example_id}: claim and evidence_section_ids must be non-empty.")
        for section_id in example["evidence_section_ids"]:
            if section_id not in sections:
                raise BenchmarkValidationError(f"{example_id}: unknown section {section_id}.")
        if not set(example["label_evidence_section_ids"]) <= set(example["evidence_section_ids"]):
            raise BenchmarkValidationError(f"{example_id}: label evidence must be among the supplied evidence.")
        if example["label"] == "INSUFFICIENT_EVIDENCE":
            if example["decisive_text"] is not None or example["label_evidence_section_ids"]:
                raise BenchmarkValidationError(f"{example_id}: INSUFFICIENT_EVIDENCE has no decisive text or evidence.")
        else:
            if not example["label_evidence_section_ids"]:
                raise BenchmarkValidationError(f"{example_id}: {example['label']} needs label evidence.")
            texts = [sections[s]["text"] for s in example["label_evidence_section_ids"]]
            if not example["decisive_text"] or not any(example["decisive_text"] in text for text in texts):
                raise BenchmarkValidationError(f"{example_id}: decisive_text is not an exact quote of its evidence.")
    return examples


def load_benchmark(path=BENCHMARK_PATH, documents=None):
    """(examples, sections). Loads documents if not given."""
    from src.ingest import load_documents

    benchmark = json.loads(Path(path).read_text(encoding="utf-8"))
    sections = section_index(documents if documents is not None else load_documents())
    return validate_benchmark(benchmark, sections), sections


def select_examples(examples, max_examples=None):
    """Deterministic spread over labels: round-robin over VERDICTS, benchmark order within each."""
    if max_examples is None or max_examples >= len(examples):
        return list(examples)
    by_label = {label: [e for e in examples if e["label"] == label] for label in VERDICTS}
    selected = []
    while len(selected) < max_examples:
        for label in VERDICTS:
            if by_label[label] and len(selected) < max_examples:
                selected.append(by_label[label].pop(0))
    return selected


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def predict(example, sections, verifier):
    """One prediction row (no label). Failures are recorded, never raised."""
    claim = Claim("C1", example["claim"])
    evidence = [sections[s] for s in example["evidence_section_ids"]]
    row = {"example_id": example["example_id"], "verifier": verifier.name, "status": None, "verdict": None,
           "evidence_section_ids": [], "explanation": None, "error": None, "model": None,
           "latency_s": None, "usage": {}, "estimated_cost_usd": None}
    start = time.perf_counter()
    try:
        output = verifier.verify([claim], evidence)
    except (LLMError, VerifierUnavailable) as error:
        row.update(status="api_failure", latency_s=time.perf_counter() - start,
                   error={"kind": error.kind, "message": str(error)})
        return row
    row.update(latency_s=time.perf_counter() - start, model=output.model, usage=dict(output.usage))
    if output.llm_call:
        row["estimated_cost_usd"] = estimate_cost_usd(output.model, output.usage.get("input_tokens"),
                                                      output.usage.get("output_tokens"))
    try:
        result = parse_verdicts(output.payload, [claim], example["evidence_section_ids"])[0]
    except InvalidVerifierOutput as error:
        raw = output.payload if isinstance(output.payload, str) else json.dumps(output.payload)
        row.update(status="invalid_output", error={"kind": error.kind, "message": str(error)}, raw_output=raw[:2000])
        return row
    row.update(status="complete", verdict=result.verdict, evidence_section_ids=result.evidence_section_ids,
               explanation=result.explanation)
    return row


def run_benchmark(examples, sections, verifier, out_path, progress=print):
    """Append one prediction per line as it finishes. Returns (rows, interrupted)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows, interrupted = [], False
    with out_path.open("w", encoding="utf-8") as handle:
        try:
            for number, example in enumerate(examples, start=1):
                row = predict(example, sections, verifier)
                rows.append(row)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                progress(f"  [{number}/{len(examples)}] {example['example_id']} {row['status']} {row['verdict'] or ''}")
        except KeyboardInterrupt:
            interrupted = True
            progress(f"\nInterrupted after {len(rows)} of {len(examples)} examples; partial results kept.")
    return rows, interrupted


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _rate(numerator, denominator):
    return None if denominator == 0 else numerator / denominator


def score(rows, examples):
    """Join predictions with labels by example_id and compute metrics."""
    labels = {e["example_id"]: e for e in examples}
    rows = [r for r in rows if r["example_id"] in labels]
    completed = [r for r in rows if r["status"] == "complete"]
    correct = [r for r in completed if r["verdict"] == labels[r["example_id"]]["label"]]
    confusion = {gold: {pred: 0 for pred in VERDICTS} for gold in VERDICTS}
    for row in completed:
        confusion[labels[row["example_id"]]["label"]][row["verdict"]] += 1
    per_verdict = {}
    for verdict in VERDICTS:
        predicted = sum(confusion[gold][verdict] for gold in VERDICTS)
        actual = sum(confusion[verdict].values())
        per_verdict[verdict] = {"precision": _rate(confusion[verdict][verdict], predicted),
                                "recall": _rate(confusion[verdict][verdict], actual), "support": actual}
    phenomena = sorted({p for e in examples for p in e["phenomena"]})
    per_phenomenon = {}
    for phenomenon in phenomena:
        subset = [r for r in completed if phenomenon in labels[r["example_id"]]["phenomena"]]
        per_phenomenon[phenomenon] = {"n": len(subset), "accuracy": _rate(
            sum(r["verdict"] == labels[r["example_id"]]["label"] for r in subset), len(subset))}
    # Correct verdict but a decisive section the label names is missing from the prediction's evidence.
    evidence_misses = [r["example_id"] for r in correct if labels[r["example_id"]]["label_evidence_section_ids"]
                       and not set(labels[r["example_id"]]["label_evidence_section_ids"]) & set(r["evidence_section_ids"])]
    costs = [r["estimated_cost_usd"] for r in rows if r.get("usage")]
    usage_rows = [r["usage"] for r in rows if r.get("usage")]
    return {
        "examples": len(rows),
        "verifier": rows[0]["verifier"] if rows else None,
        "status_counts": {s: sum(r["status"] == s for r in rows) for s in ("complete", "api_failure", "invalid_output")},
        "completion_rate": _rate(len(completed), len(rows)),
        "accuracy_completed": _rate(len(correct), len(completed)),
        "accuracy_strict": _rate(len(correct), len(rows)),
        "label_counts": {v: sum(labels[r["example_id"]]["label"] == v for r in rows) for v in VERDICTS},
        "prediction_counts": {v: sum(r["verdict"] == v for r in completed) for v in VERDICTS},
        "confusion": confusion,
        "per_verdict": per_verdict,
        "per_phenomenon": per_phenomenon,
        "false_supported": [r["example_id"] for r in completed
                            if r["verdict"] == "SUPPORTED" and labels[r["example_id"]]["label"] != "SUPPORTED"],
        "missing_contradictions_as_insufficient": [
            r["example_id"] for r in completed
            if labels[r["example_id"]]["label"] == "CONTRADICTED" and r["verdict"] == "INSUFFICIENT_EVIDENCE"],
        "insufficient_called_contradicted": [
            r["example_id"] for r in completed
            if labels[r["example_id"]]["label"] == "INSUFFICIENT_EVIDENCE" and r["verdict"] == "CONTRADICTED"],
        "correct_verdict_wrong_evidence": evidence_misses,
        "errors": [r["example_id"] for r in completed if r["verdict"] != labels[r["example_id"]]["label"]],
        "mean_latency_s": _rate(sum(r["latency_s"] or 0 for r in rows), len(rows)),
        "input_tokens": sum(u.get("input_tokens") or 0 for u in usage_rows) if usage_rows else None,
        "output_tokens": sum(u.get("output_tokens") or 0 for u in usage_rows) if usage_rows else None,
        "estimated_cost_usd": sum(costs) if costs and all(c is not None for c in costs) else None,
    }


def _fmt(value):
    return "n/a" if value is None else f"{value:.3f}"


def print_report(summary, out_path, interrupted=False):
    mock = (summary["verifier"] or "").startswith("mock")
    print("\nRAGForge — Claim Verification Benchmark (development)")
    print(f"Verifier: {summary['verifier']}")
    if mock:
        print("*** MOCK VERIFIER: lexical heuristic for offline plumbing checks. These are NOT model accuracy. ***")
    print(f"Examples: {summary['examples']}" + ("  (interrupted: partial run)" if interrupted else ""))
    print("Labels (draft, hand-written): " + ", ".join(f"{k} {v}" for k, v in summary["label_counts"].items()))
    print("Status: " + ", ".join(f"{k} {v}" for k, v in summary["status_counts"].items())
          + "   (api_failure = availability, not a verdict error)")
    print(f"  Completion rate                {_fmt(summary['completion_rate'])}")
    print(f"  Accuracy (completed only)      {_fmt(summary['accuracy_completed'])}")
    print(f"  Accuracy (failures count wrong){_fmt(summary['accuracy_strict']):>6}")
    print("\n  Confusion (rows = label, columns = prediction)")
    print("  " + " " * 24 + "".join(f"{v[:12]:>14}" for v in VERDICTS))
    for gold in VERDICTS:
        print(f"  {gold:<24}" + "".join(f"{summary['confusion'][gold][p]:>14}" for p in VERDICTS))
    print("\n  Per verdict: precision / recall (support)")
    for verdict, stats in summary["per_verdict"].items():
        print(f"    {verdict:<24}{_fmt(stats['precision'])} / {_fmt(stats['recall'])} ({stats['support']})")
    print("\n  Per phenomenon: accuracy (n)")
    for phenomenon, stats in summary["per_phenomenon"].items():
        print(f"    {phenomenon:<24}{_fmt(stats['accuracy'])} ({stats['n']})")
    print(f"\n  False SUPPORTED (most harmful): {summary['false_supported'] or 'none'}")
    print(f"  CONTRADICTED predicted as INSUFFICIENT: {summary['missing_contradictions_as_insufficient'] or 'none'}")
    print(f"  INSUFFICIENT predicted as CONTRADICTED: {summary['insufficient_called_contradicted'] or 'none'}")
    print(f"  Correct verdict, decisive section not cited: {summary['correct_verdict_wrong_evidence'] or 'none'}")
    print(f"  Mean latency per example: {_fmt(summary['mean_latency_s'])} s")
    if summary["input_tokens"] is not None:
        print(f"  Tokens: input {summary['input_tokens']}, output {summary['output_tokens']}")
    if summary["estimated_cost_usd"] is not None:
        print(f"  Estimated cost: ${summary['estimated_cost_usd']:.6f} ({PRICING_SOURCE})")
    print(f"\n{summary['examples']} examples: one example moves accuracy by {_fmt(_rate(1, summary['examples']))}.")
    print(f"Predictions: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description="RAGForge M5 claim-verification benchmark (development only).")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="MockVerifier (lexical heuristic); no API calls")
    mode.add_argument("--live", action="store_true", help="Gemini verifier; real API calls (costs money on paid tier)")
    mode.add_argument("--summarize", metavar="PREDICTIONS_JSONL", help="score a saved predictions file")
    parser.add_argument("--max-examples", type=int, default=None,
                        help=f"limit examples (live default {DEFAULT_LIVE_EXAMPLES}; dry run: all)")
    parser.add_argument("--all", action="store_true", help="live: run every example")
    parser.add_argument("--confirm-all", action="store_true", help="required with --live for every example")
    parser.add_argument("--model", default=None, help="Gemini model ID (live only)")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--out", default=None, help="predictions path (default results/verification/<timestamp>.jsonl)")
    return parser


def main(argv=None, verifier=None):
    from src.llm import GeminiClient

    parser = build_parser()
    args = parser.parse_args(argv)
    examples, sections = load_benchmark()

    if args.summarize:
        rows = [json.loads(line) for line in Path(args.summarize).read_text(encoding="utf-8").splitlines()
                if line.strip()]
        print_report(score(rows, examples), args.summarize)
        return 0
    if args.max_examples is not None and args.max_examples < 1:
        parser.error("--max-examples must be at least 1.")
    if args.all and args.max_examples is not None:
        parser.error("use either --all or --max-examples, not both.")
    if args.live:
        limit = len(examples) if args.all else (args.max_examples or DEFAULT_LIVE_EXAMPLES)
        if limit >= len(examples) and not args.confirm_all:
            parser.error(f"A live run over all {len(examples)} examples makes {len(examples)} paid API calls; "
                         "add --confirm-all to proceed, or use --max-examples for a smaller run.")
    else:
        limit = args.max_examples
    selected = select_examples(examples, limit)

    if verifier is None:
        if args.dry_run:
            verifier = MockVerifier()
        else:
            load_env_file(PROJECT_ROOT / ".env")
            try:
                llm = GeminiClient(model=args.model, temperature=args.temperature, top_p=args.top_p)
            except LLMError as error:
                parser.error(str(error))
            for warning in sampling_warnings(llm.model, args.temperature, args.top_p):
                print(f"Warning: {warning}", file=sys.stderr)
            verifier = GeminiVerifier(llm)

    print(f"About to verify {len(selected)} labeled claims with {verifier.name}.")
    if args.live:
        print(f"This makes up to {len(selected)} API requests (plus bounded retries on transient errors).")
    out_path = Path(args.out) if args.out else RESULTS_DIR / (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{'dry-run' if args.dry_run else 'live'}_{len(selected)}ex.jsonl")
    rows, interrupted = run_benchmark(selected, sections, verifier, out_path)
    print_report(score(rows, examples), out_path, interrupted)
    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
