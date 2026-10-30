"""M6.4A offline verification evaluation (exploratory; evaluation only).

    python scripts/m64a_offline_eval.py [--device auto|mps|cpu]

A. The M5 40-claim benchmark (data/evaluations/verification_claims.json) through the
   production LocalNLIVerifier, one claim per call with that example's own evidence set
   (evidence_section_ids, full section texts via evaluate_verification.section_index) -
   the benchmark's per-claim evidence, not a shared retrieval snapshot.
B. Aggregation: for examples with label evidence, the claim against its label evidence
   only vs the full evidence set; per-section model labels separate classification
   errors from deterministic aggregation effects.
C. Title sensitivity on the frozen v0.1-exp VALIDATION split only (src.nli_train.load_split
   refuses the test split): raw premise vs "{document_title}, {section_title}: {evidence}",
   both through src.nli_train.evaluate with the same collate, checkpoint and label mapping.

No Gemini calls, no training, no threshold changes, no production code changes. Outputs
go to a new directory artifacts/nli/m6.4a-evaluation/<timestamp>/ (never overwritten).
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluate_verification as evalver  # noqa: E402
from src import local_verifier as lv  # noqa: E402
from src import nli_dataset as nd  # noqa: E402
from src import nli_review as nr  # noqa: E402
from src import nli_train as nt  # noqa: E402
from src.claims import Claim  # noqa: E402
from src.ingest import PROJECT_ROOT  # noqa: E402
from src.verification import parse_verdicts  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "artifacts" / "nli" / "m6.4a-evaluation"
LABELS = nd.LABEL_VALUES
S, C, I = "SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state():
    def run(*args):
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "uncommitted_changes": bool(run("status", "--porcelain"))}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with open(path, "x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def false_rate(gold, predicted, label):
    """Gold non-`label` examples predicted `label`, over all gold non-`label` examples."""
    others = [p for g, p in zip(gold, predicted) if g != label]
    return {"count": sum(p == label for p in others), "denominator": len(others),
            "rate": sum(p == label for p in others) / len(others) if others else None}


def metrics(gold, predicted):
    m = nt.classification_metrics(gold, predicted)
    m["false_contradicted"] = false_rate(gold, predicted, C)
    return m


# ---------------------------------------------------------------------------
# Training overlap (train and validation splits only)
# ---------------------------------------------------------------------------

def training_overlap(examples, sections, train, validation):
    train_docs = {r["document_id"] for r in train}
    val_docs = {r["document_id"] for r in validation}
    train_sections = {r["section_id"] for r in train}
    val_sections = {r["section_id"] for r in validation}
    seen_claims = [(r["example_id"], r["claim"]) for r in train + validation]
    out = {}
    for e in examples:
        docs = sorted({sections[s]["document_id"] for s in e["evidence_section_ids"]})
        in_train = [d for d in docs if d in train_docs]
        in_val = [d for d in docs if d in val_docs]
        if len(in_train) == len(docs):
            group = "train_documents"
        elif len(in_val) == len(docs):
            group = "validation_documents"
        elif not in_train and not in_val:
            group = "unseen_documents"
        else:
            group = "mixed"
        best_id, best = max(((i, nd.jaccard(e["claim"], c)) for i, c in seen_claims), key=lambda x: x[1])
        out[e["example_id"]] = {
            "documents": docs, "group": group,
            "evidence_sections_in_train": [s for s in e["evidence_section_ids"] if s in train_sections],
            "evidence_sections_in_validation": [s for s in e["evidence_section_ids"] if s in val_sections],
            "max_claim_jaccard_vs_train_validation": round(best, 3), "closest_training_example": best_id,
            "near_duplicate_claim": best >= nd.NEAR_DUPLICATE_JACCARD}
    return out


# ---------------------------------------------------------------------------
# A + B: benchmark through the production verifier
# ---------------------------------------------------------------------------

def run_claim(verifier, claim_text, evidence):
    claim = Claim("C1", claim_text)
    start = time.perf_counter()
    output = verifier.verify([claim], evidence)
    elapsed = time.perf_counter() - start
    result = parse_verdicts(output.payload, [claim], [c["section_id"] for c in evidence])[0]
    return result, output.trace["claims"][0], elapsed


def benchmark(verifier, examples, sections, overlap):
    run_claim(verifier, examples[0]["claim"], [sections[s] for s in examples[0]["evidence_section_ids"]])  # warm-up
    rows = []
    for e in examples:
        result, trace, elapsed = run_claim(verifier, e["claim"], [sections[s] for s in e["evidence_section_ids"]])
        row = {"example_id": e["example_id"], "claim": e["claim"], "gold": e["label"], "predicted": result.verdict,
               "correct": result.verdict == e["label"], "reason": trace["reason"],
               "explanation": result.explanation, "evidence_section_ids": e["evidence_section_ids"],
               "label_evidence_section_ids": e["label_evidence_section_ids"],
               "predicted_evidence_section_ids": result.evidence_section_ids, "phenomena": e["phenomena"],
               "latency_s": elapsed, "pairs": len(trace["premises"]), "overlap": overlap[e["example_id"]],
               "section_labels": {p["section_id"]: p["nli_label"] for p in trace["premises"]}, "trace": trace}
        if e["label_evidence_section_ids"] and set(e["label_evidence_section_ids"]) != set(e["evidence_section_ids"]):
            only, only_trace, _ = run_claim(verifier, e["claim"],
                                            [sections[s] for s in e["label_evidence_section_ids"]])
            row["label_evidence_only"] = {"predicted": only.verdict, "reason": only_trace["reason"],
                                          "section_labels": {p["section_id"]: p["nli_label"]
                                                             for p in only_trace["premises"]}}
        rows.append(row)
    return rows


def benchmark_summary(rows):
    gold, pred = [r["gold"] for r in rows], [r["predicted"] for r in rows]
    latencies = [r["latency_s"] for r in rows]
    by_group = {}
    for group in sorted({r["overlap"]["group"] for r in rows}):
        sub = [r for r in rows if r["overlap"]["group"] == group]
        by_group[group] = {"n": len(sub), "documents": sorted({d for r in sub for d in r["overlap"]["documents"]}),
                           **{k: v for k, v in metrics([r["gold"] for r in sub], [r["predicted"] for r in sub]).items()
                              if k in ("accuracy", "macro_f1", "false_supported", "false_supported_rate",
                                       "false_contradicted", "confusion")}}
    errors_by_document = defaultdict(lambda: {"n": 0, "errors": 0, "error_ids": []})
    for r in rows:
        decisive = r["label_evidence_section_ids"] or r["evidence_section_ids"]
        for doc in sorted({s.rsplit(".", 1)[0] for s in decisive}):
            errors_by_document[doc]["n"] += 1
            if not r["correct"]:
                errors_by_document[doc]["errors"] += 1
                errors_by_document[doc]["error_ids"].append(r["example_id"])
    return {"n": len(rows), "gold_counts": dict(Counter(gold)), "metrics": metrics(gold, pred),
            "latency_ms_per_claim": {"mean": 1000 * statistics.fmean(latencies),
                                     "median": 1000 * statistics.median(latencies), "max": 1000 * max(latencies),
                                     "pairs_per_claim": dict(Counter(r["pairs"] for r in rows)),
                                     "note": "one verify() call per claim after one untimed warm-up; includes "
                                             "tokenization, grouping and decision rules"},
            "reasons": dict(Counter(r["reason"] for r in rows)),
            "by_training_overlap": by_group, "errors_by_document": dict(errors_by_document),
            "near_duplicate_claims": [r["example_id"] for r in rows if r["overlap"]["near_duplicate_claim"]]}


def aggregation_analysis(rows):
    """B: classify every multi-section example and every error as classification vs aggregation."""
    cases = []
    for r in rows:
        labels, label_ids = r["section_labels"], set(r["label_evidence_section_ids"])
        unrelated = [s for s in r["evidence_section_ids"] if s not in label_ids]
        case = {"example_id": r["example_id"], "gold": r["gold"], "full_set": r["predicted"], "reason": r["reason"],
                "label_sections": {s: labels[s] for s in r["label_evidence_section_ids"]},
                "unrelated_sections": {s: labels[s] for s in unrelated},
                "label_evidence_only": (r.get("label_evidence_only") or {}).get("predicted")}
        case["unrelated_contradiction"] = any(labels[s] == "contradiction" for s in unrelated) and bool(label_ids)
        case["unrelated_entailment"] = any(labels[s] == "entailment" for s in unrelated)
        case["false_conflict"] = r["reason"] == "conflicting_evidence" and r["gold"] != C
        case["support_overridden"] = (case["label_evidence_only"] == S and r["predicted"] != S)
        if r["correct"]:
            case["error_type"] = None
        elif case["label_evidence_only"] == r["gold"]:
            case["error_type"] = "aggregation_effect (correct on label evidence alone, wrong with the full set)"
        elif not label_ids and any(labels[s] != "neutral" for s in unrelated):
            case["error_type"] = "classification_error_on_unrelated_section (no label evidence; gold insufficient)"
        else:
            case["error_type"] = "classification_error_on_label_evidence"
        cases.append(case)
    multi = [c for c in cases if len(c["label_sections"]) + len(c["unrelated_sections"]) > 1]
    comparable = [c for c in cases if c["label_evidence_only"] is not None]
    return {
        "evidence_structure": "per-claim curated evidence sets from the benchmark (1-2 sections); no retrieval "
                              "snapshots exist, so production-style 5-section sets are not evaluated",
        "examples": len(cases), "multi_section_examples": len(multi),
        "comparable_examples (label evidence is a strict subset of the evidence set)": len(comparable),
        "verdict_changed_label_only_vs_full": [c["example_id"] for c in comparable
                                               if c["label_evidence_only"] != c["full_set"]],
        "unrelated_section_contradictions": [c["example_id"] for c in cases if c["unrelated_contradiction"]],
        "unrelated_section_entailments": [c["example_id"] for c in cases if c["unrelated_entailment"]],
        "false_conflicts": [c["example_id"] for c in cases if c["false_conflict"]],
        "support_overridden": [c["example_id"] for c in cases if c["support_overridden"]],
        "error_types": dict(Counter(c["error_type"] for c in cases if c["error_type"])),
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# C: title sensitivity on validation
# ---------------------------------------------------------------------------

def titled(row):
    return {**row, "evidence": f"{row['document_title']}, {row['section_title']}: {row['evidence']}"}


def title_sensitivity(loaded, validation):
    nli = loaded.nli
    label_to_id = lv.checkpoint_label_ids(nli.model.config.id2label)
    id_to_label = {i: label for label, i in label_to_id.items()}
    collate = nt.make_collate(nli.tokenizer, label_to_id, nli.max_length)
    missing = [r["example_id"] for r in validation if not (r.get("document_title") and r.get("section_title"))]
    rows = [r for r in validation if r["example_id"] not in missing]
    titled_rows = [titled(r) for r in rows]
    lengths = nli.pair_token_counts([(r["evidence"], r["claim"]) for r in titled_rows])
    if max(lengths) > nli.max_length:
        raise SystemExit("A titled pair exceeds the token limit; it would be truncated.")
    raw_m, raw_p = nt.evaluate(nli.model, rows, collate, nli.device, id_to_label)
    tit_m, tit_p = nt.evaluate(nli.model, titled_rows, collate, nli.device, id_to_label)
    m63 = json.loads((lv.DEFAULT_V3_RUN / "validation_predictions.json").read_text(encoding="utf-8"))
    m63_by_id = {p["example_id"]: p["predicted"] for p in m63}
    changes, by_gold = [], defaultdict(Counter)
    for r, a, b in zip(rows, raw_p, tit_p):
        if a["predicted"] != b["predicted"]:
            kind = ("improvement" if b["predicted"] == r["label"] else
                    "regression" if a["predicted"] == r["label"] else "wrong_to_wrong")
            changes.append({"example_id": r["example_id"], "gold": r["label"], "raw": a["predicted"],
                            "titled": b["predicted"], "kind": kind, "evidence_format": r["evidence_format"],
                            "family_id": r["family_id"], "phenomenon": r["phenomenon"], "claim": r["claim"],
                            "document_title": r["document_title"], "section_title": r["section_title"]})
            by_gold[r["label"]][kind] += 1

    def keep(m):
        return {k: m[k] for k in ("n", "accuracy", "macro_f1", "per_class", "confusion", "false_supported",
                                  "false_supported_rate", "loss", "by_format", "by_family", "by_label_source")}

    return {
        "split": "validation (frozen v0.1-exp)", "examples": len(rows), "excluded_missing_titles": missing,
        "titled_format": "{document_title}, {section_title}: {evidence} (the M6.1 premise_text format)",
        "max_titled_pair_tokens": max(lengths),
        "raw": {**keep(raw_m), "false_contradicted": false_rate([r["label"] for r in rows],
                                                                [p["predicted"] for p in raw_p], C)},
        "titled": {**keep(tit_m), "false_contradicted": false_rate([r["label"] for r in rows],
                                                                   [p["predicted"] for p in tit_p], C)},
        "raw_vs_m6.3_saved_validation_predictions_disagreements": sum(
            m63_by_id.get(p["example_id"]) != p["predicted"] for p in raw_p),
        "changed_predictions": len(changes), "changes_by_gold_label": {k: dict(v) for k, v in by_gold.items()},
        "changes_by_kind": dict(Counter(c["kind"] for c in changes)), "changes": changes,
    }, raw_p, tit_p


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    out = OUT_ROOT / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=False)

    loaded = lv.load_v3(lv.DEFAULT_V3_RUN, args.device)
    verifier = lv.LocalNLIVerifier(lv.DEFAULT_V3_RUN, args.device, loaded=loaded)
    examples, sections = evalver.load_benchmark()
    train, train_manifest = nt.load_split("train")
    validation, _ = nt.load_split("validation")
    overlap = training_overlap(examples, sections, train, validation)

    rows = benchmark(verifier, examples, sections, overlap)
    summary = benchmark_summary(rows)
    aggregation = aggregation_analysis(rows)
    titles, raw_p, tit_p = title_sensitivity(loaded, validation)

    import torch
    import transformers
    write_json(out / "manifest.json", {
        "purpose": "M6.4A exploratory offline evaluation (not a held-out test; no model selection)",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": git_state(),
        "model": loaded.info, "torch": torch.__version__, "transformers": transformers.__version__,
        "benchmark": {"path": str(evalver.BENCHMARK_PATH.relative_to(PROJECT_ROOT)),
                      "sha256": sha256(evalver.BENCHMARK_PATH), "examples": len(examples),
                      "evidence": "per-claim evidence_section_ids, full section text (section_index), raw premise"},
        "validation": {"version": nt.DATASET_VERSION, "file_sha256": train_manifest["file_sha256"]["validation.jsonl"],
                       "examples": len(validation)},
        "test_split": "not loaded (load_split refuses it)", "gemini_calls": 0})
    write_jsonl(out / "benchmark_predictions.jsonl", rows)
    write_json(out / "benchmark_metrics.json", summary)
    write_json(out / "aggregation.json", aggregation)
    write_json(out / "title_sensitivity.json", titles)
    write_jsonl(out / "validation_predictions_raw.jsonl", raw_p)
    write_jsonl(out / "validation_predictions_titled.jsonl", tit_p)
    print(json.dumps({"out": str(out), "benchmark": {k: summary["metrics"][k] for k in
                                                     ("accuracy", "macro_f1", "false_supported_rate")}}, indent=1))


if __name__ == "__main__":
    main()
