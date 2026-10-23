"""Zero-shot local NLI baseline (M6.1) on the M5 claim-verification benchmark. No API calls.

Usage:
    python -m src.evaluate_local_nli                                   # default model, all 40 examples
    python -m src.evaluate_local_nli --model cross-encoder/nli-MiniLM2-L6-H768 --device cpu --batch-size 8
    python -m src.evaluate_local_nli --example V005                    # one benchmark example, per section
    python -m src.evaluate_local_nli --premise "..." --hypothesis "..."  # one evidence-claim pair
    python -m src.evaluate_local_nli --interactive                     # type pairs until 'exit'
    python -m src.evaluate_local_nli --summarize results/nli/<run>.jsonl

The first run downloads the checkpoint (about 330 MB) to the Hugging Face
cache; later runs are offline. --local-files-only forbids downloads.

Each example's claim is checked against its listed evidence sections, one NLI
pass per section, aggregated as described in src/nli.py (contradiction first).
Predictions, with per-section logits and probabilities, go to
results/nli/*.jsonl and carry no labels; scoring joins them with the draft
labels in data/evaluations/verification_claims.json. Labels are never edited
to fit a model.

Latency excludes model loading and one untimed warm-up batch. It is wall time
on whatever device ran, from a single run.
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from src import evaluate_verification as evalver
from src.ingest import PROJECT_ROOT
from src.nli import (
    DEFAULT_NLI_BATCH_SIZE,
    DEFAULT_NLI_MAX_LENGTH,
    DEFAULT_NLI_MODEL,
    DEFAULT_NLI_REVISION,
    LocalNLI,
)
from src.prompts import VERDICTS

RESULTS_DIR = PROJECT_ROOT / "results" / "nli"
LABEL_STATUS = "draft: hand-written labels, not independently reviewed"


def evaluate_example(nli, example, sections):
    """One prediction row (no label)."""
    evidence = [sections[s] for s in example["evidence_section_ids"]]
    start = time.perf_counter()
    verdict, ids, explanation, predictions = nli.classify_claim(example["claim"], evidence)
    return {"example_id": example["example_id"], "verifier": nli.name, "model": nli.model_name,
            "revision": nli.revision, "device": nli.device, "status": "complete", "verdict": verdict,
            "evidence_section_ids": ids, "explanation": explanation, "error": None,
            "latency_s": time.perf_counter() - start, "usage": {}, "estimated_cost_usd": None,
            "sections": [p.to_dict() for p in predictions]}


def run(nli, examples, sections, out_path, progress=print):
    """Warm up once, then predict every example, writing one JSON line each. Returns rows."""
    nli.predict("Warm-up premise.", "Warm-up hypothesis.")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out_path.open("w", encoding="utf-8") as handle:
        for number, example in enumerate(examples, start=1):
            row = evaluate_example(nli, example, sections)
            rows.append(row)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            progress(f"  [{number}/{len(examples)}] {example['example_id']} {row['verdict']}")
    return rows


def _f1(precision, recall):
    if precision is None or recall is None or precision + recall == 0:
        return 0.0 if precision is not None and recall is not None else None
    return 2 * precision * recall / (precision + recall)


def score(rows, examples):
    """M5 benchmark metrics plus macro-F1, false-SUPPORTED rate and latency."""
    summary = evalver.score(rows, examples)
    labels = {e["example_id"]: e for e in examples}
    completed = [r for r in rows if r["status"] == "complete" and r["example_id"] in labels]
    f1s = {}
    for verdict, stats in summary["per_verdict"].items():
        precision = stats["precision"] if stats["precision"] is not None else 0.0   # never predicted -> 0
        f1s[verdict] = _f1(precision, stats["recall"]) if stats["support"] else None
        stats["f1"] = f1s[verdict]
    present = [value for value in f1s.values() if value is not None]
    not_supported = [r for r in completed if labels[r["example_id"]]["label"] != "SUPPORTED"]
    multi = [r for r in completed if len(labels[r["example_id"]]["evidence_section_ids"]) > 1]
    single = [r for r in completed if len(labels[r["example_id"]]["evidence_section_ids"]) == 1]
    pair_latencies = [s["latency_s"] for r in completed for s in r.get("sections", [])]
    example_latencies = [r["latency_s"] for r in completed]
    sections = [s for r in completed for s in r.get("sections", [])]
    summary.update({
        "model": completed[0]["model"] if completed else None,
        "revision": completed[0].get("revision") if completed else None,
        "device": completed[0].get("device") if completed else None,
        "macro_f1": statistics.fmean(present) if present else None,
        "false_supported_count": len(summary["false_supported"]),
        "false_supported_rate": len(summary["false_supported"]) / len(not_supported) if not_supported else None,
        "not_supported_examples": len(not_supported),
        "accuracy_single_section": _accuracy(single, labels),
        "accuracy_multi_section": _accuracy(multi, labels),
        "single_section_examples": len(single),
        "multi_section_examples": len(multi),
        "pairs": len(pair_latencies),
        "truncated_pairs": sum(s["truncated"] for s in sections),
        "max_input_tokens": max((s["input_tokens"] for s in sections), default=None),
        "mean_pair_latency_ms": 1000 * statistics.fmean(pair_latencies) if pair_latencies else None,
        "mean_example_latency_ms": 1000 * statistics.fmean(example_latencies) if example_latencies else None,
        "median_example_latency_ms": 1000 * statistics.median(example_latencies) if example_latencies else None,
        "total_inference_s": sum(example_latencies),
        "label_status": LABEL_STATUS,
    })
    return summary


def _accuracy(rows, labels):
    return sum(r["verdict"] == labels[r["example_id"]]["label"] for r in rows) / len(rows) if rows else None


def _fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def print_report(summary, out_path, load_s=None, device_note=None):
    print("\nRAGForge — Local NLI Baseline (M6.1, zero-shot, not fine-tuned)")
    print(f"Model: {summary['model']} @ {summary['revision']}")
    print(f"Device: {summary['device']}" + (f"  ({device_note})" if device_note else ""))
    print(f"Labels: {summary['label_status']}")
    print("Aggregation: per-section NLI; any contradiction -> CONTRADICTED, else any entailment -> SUPPORTED")
    print(f"Examples: {summary['examples']} ({summary['single_section_examples']} single-section, "
          f"{summary['multi_section_examples']} multi-section); NLI pairs: {summary['pairs']}, "
          f"truncated: {summary['truncated_pairs']}, max tokens: {summary['max_input_tokens']}")
    print("Labels: " + ", ".join(f"{k} {v}" for k, v in summary["label_counts"].items()))
    print(f"\n  Accuracy                       {_fmt(summary['accuracy_completed'])}")
    print(f"  Macro-F1                       {_fmt(summary['macro_f1'])}")
    print(f"  Accuracy, single-section       {_fmt(summary['accuracy_single_section'])}")
    print(f"  Accuracy, multi-section        {_fmt(summary['accuracy_multi_section'])}")
    print(f"  False SUPPORTED                {summary['false_supported_count']} of "
          f"{summary['not_supported_examples']} non-SUPPORTED claims (rate {_fmt(summary['false_supported_rate'])})")
    print("\n  Confusion (rows = label, columns = prediction)")
    print("  " + " " * 24 + "".join(f"{v[:12]:>14}" for v in VERDICTS))
    for gold in VERDICTS:
        print(f"  {gold:<24}" + "".join(f"{summary['confusion'][gold][p]:>14}" for p in VERDICTS))
    print("\n  Per verdict: precision / recall / F1 (support)")
    for verdict, stats in summary["per_verdict"].items():
        print(f"    {verdict:<24}{_fmt(stats['precision'])} / {_fmt(stats['recall'])} / {_fmt(stats['f1'])} "
              f"({stats['support']})")
    print("\n  Per phenomenon: accuracy (n)")
    for phenomenon, stats in summary["per_phenomenon"].items():
        print(f"    {phenomenon:<24}{_fmt(stats['accuracy'])} ({stats['n']})")
    print(f"\n  False SUPPORTED: {summary['false_supported'] or 'none'}")
    print(f"  CONTRADICTED predicted as INSUFFICIENT: {summary['missing_contradictions_as_insufficient'] or 'none'}")
    print(f"  INSUFFICIENT predicted as CONTRADICTED: {summary['insufficient_called_contradicted'] or 'none'}")
    print(f"  All errors: {summary['errors'] or 'none'}")
    print(f"\n  Latency: mean {_fmt(summary['mean_pair_latency_ms'], 1)} ms per NLI pair, "
          f"mean {_fmt(summary['mean_example_latency_ms'], 1)} / median "
          f"{_fmt(summary['median_example_latency_ms'], 1)} ms per example, total "
          f"{_fmt(summary['total_inference_s'], 2)} s" + (f"; model load {load_s:.1f} s" if load_s else ""))
    print(f"\n{summary['examples']} examples: one example moves accuracy by "
          f"{_fmt(1 / summary['examples'] if summary['examples'] else None)}. Labels are a draft.")
    print(f"Predictions: {out_path}")


def format_prediction(prediction):
    probs = ", ".join(f"{label} {prediction.probabilities[label]:.3f}" for label in prediction.labels)
    logits = ", ".join(f"{label} {value:+.3f}" for label, value in zip(prediction.labels, prediction.logits))
    where = f"[{prediction.premise_id}] " if prediction.premise_id else ""
    return (f"  {where}{prediction.nli_label} -> {prediction.verdict}\n"
            f"    logits: {logits}\n    probabilities: {probs}\n"
            f"    tokens: {prediction.input_tokens}{' (truncated)' if prediction.truncated else ''}, "
            f"{prediction.latency_s * 1000:.1f} ms on {prediction.device}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description="RAGForge M6.1 zero-shot local NLI baseline (no API calls).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--example", metavar="EXAMPLE_ID", help="inspect one benchmark example, section by section")
    mode.add_argument("--premise", help="evidence text for a single pair (use with --hypothesis)")
    mode.add_argument("--interactive", action="store_true", help="enter evidence-claim pairs until 'exit'")
    mode.add_argument("--summarize", metavar="PREDICTIONS_JSONL", help="score a saved predictions file")
    parser.add_argument("--hypothesis", help="claim text for --premise")
    parser.add_argument("--model", default=DEFAULT_NLI_MODEL, help=f"checkpoint (default {DEFAULT_NLI_MODEL})")
    parser.add_argument("--revision", default=None,
                        help="checkpoint revision (default: pinned commit for the default model, else main)")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_NLI_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_NLI_MAX_LENGTH)
    parser.add_argument("--local-files-only", action="store_true", help="never download; use the local HF cache")
    parser.add_argument("--out", default=None, help="predictions path (default results/nli/<timestamp>.jsonl)")
    return parser


def main(argv=None, nli=None, input_fn=input):
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.premise is None) != (args.hypothesis is None):
        parser.error("--premise and --hypothesis must be given together.")
    examples, sections = evalver.load_benchmark()

    if args.summarize:
        rows = [json.loads(line) for line in Path(args.summarize).read_text(encoding="utf-8").splitlines() if line.strip()]
        print_report(score(rows, examples), args.summarize)
        return 0
    if args.example and args.example not in {e["example_id"] for e in examples}:
        parser.error(f"Unknown example {args.example!r}.")

    if nli is None:
        revision = args.revision or (DEFAULT_NLI_REVISION if args.model == DEFAULT_NLI_MODEL else "main")
        try:
            nli = LocalNLI(args.model, revision=revision, device=args.device, batch_size=args.batch_size,
                           max_length=args.max_length, local_files_only=args.local_files_only)
        except ValueError as error:
            parser.error(str(error))
        except OSError as error:   # missing checkpoint, no network with --local-files-only, bad revision
            parser.error(f"Could not load {args.model!r}: {error}")
    print(f"Model: {nli.model_name} @ {nli.revision} on {nli.device}"
          + (f" ({nli.device_note})" if nli.device_note else "") + f"; labels {nli.labels}")

    if args.premise is not None:
        try:
            print(format_prediction(nli.predict(args.premise, args.hypothesis)))
        except ValueError as error:
            parser.error(str(error))
        return 0
    if args.example:
        example = next(e for e in examples if e["example_id"] == args.example)
        verdict, ids, explanation, predictions = nli.classify_claim(
            example["claim"], [sections[s] for s in example["evidence_section_ids"]])
        print(f"\n{example['example_id']} claim: {example['claim']}")
        for prediction in predictions:
            print(format_prediction(prediction))
        print(f"Aggregated verdict: {verdict} {ids} - {explanation}")
        print(f"Draft label: {example['label']} {example['label_evidence_section_ids']}")
        return 0
    if args.interactive:
        while True:
            premise = input_fn("Evidence (or 'exit'): ").strip()
            if premise.lower() in ("exit", "quit"):
                return 0
            hypothesis = input_fn("Claim: ").strip()
            try:
                print(format_prediction(nli.predict(premise, hypothesis)))
            except ValueError as error:
                print(f"Error: {error}")

    out_path = Path(args.out) if args.out else RESULTS_DIR / (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{args.model.replace('/', '__')}_{len(examples)}ex.jsonl")
    print(f"Evaluating {len(examples)} benchmark examples locally (no API calls).")
    rows = run(nli, examples, sections, out_path)
    print_report(score(rows, examples), out_path, nli.load_s, nli.device_note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
