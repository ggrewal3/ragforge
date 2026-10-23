"""Human review, shortcut audit and freeze for the policy NLI dataset (M6.2 quality gate). No training, no APIs.

Usage:
    python -m src.nli_review --plan                                   # what must be reviewed, and in what order
    python -m src.nli_review --review --reviewer "Your Name"          # start or resume the review
    python -m src.nli_review --status                                 # progress and freeze readiness
    python -m src.nli_review --history nli-0123456789ab               # every decision for one example
    python -m src.nli_review --redo nli-0123456789ab --reviewer "..." # replace a decision (kept in history)
    python -m src.nli_review --shortcut-audit --out data/nli/shortcut_audit_v1.json
    python -m src.nli_review --freeze --version v1.0                  # fails until required review is complete

    Random quality audit and experimental freeze (for exploratory M6.3 training):
    python -m src.nli_review --sample                                 # write the 50-example stratified sample
    python -m src.nli_review --review-sample --reviewer "Your Name"   # review it (resumable)
    python -m src.nli_review --sample-report                          # agreement estimate
    python -m src.nli_review --freeze-experimental --version v0.1-exp # needs the whole sample decided

The reviewer name can also come from the RAGFORGE_REVIEWER environment variable.

Integrity model
---------------
* data/nli/policy_nli_v1.jsonl holds the AI-authored drafts and is never edited
  by review. It must regenerate byte-for-byte from data/nli/policy_facts.json.
* Every human decision is appended to data/nli/reviews.jsonl. A record keeps the
  AI label, the human label (None for a rejection), the reviewer, the timestamp
  and the reason, plus the example's content fingerprint. Records are hash-chained
  (prev_hash -> record_hash), so editing or deleting a past line is detected.
* Only an explicit accept / correct / reject writes a record. Skipping writes
  nothing, automated checks write nothing, and nothing is ever recorded on the
  reviewer's behalf. An example already decided cannot be decided again except
  through --redo, which needs a reason and keeps the earlier decision in history.
* --freeze applies the latest decision per example to a copy of the drafts and
  writes an immutable, versioned snapshot. It refuses to run while any
  validation or test example lacks an explicit human decision.
"""

import argparse
import copy
import hashlib
import json
import os
import random
import re
import shutil
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src import nli_dataset as nd
from src.ingest import PROJECT_ROOT

REVIEWS_PATH = nd.REVIEWS_PATH
FROZEN_DIR = PROJECT_ROOT / "data" / "nli" / "frozen"
SHORTCUT_AUDIT_PATH = PROJECT_ROOT / "data" / "nli" / "shortcut_audit_v1.json"
REVIEW_TOOL_VERSION = "m6.2-review.1"
HIGH_RISK_COUNT = 70
GENESIS = "genesis"
DECISIONS = ("reviewed", "corrected", "rejected")
EVALUATION_SPLITS = ("validation", "test")
REVIEWER_ENV = "RAGFORGE_REVIEWER"


class ReviewError(ValueError):
    """An invalid or disallowed review decision."""


class DuplicateReviewError(ReviewError):
    """The example already has a decision; use --redo to replace it."""


class ReviewIntegrityError(ValueError):
    """The decision log was edited, reordered or truncated outside this tool."""


class FreezeError(ValueError):
    """The dataset cannot be frozen; `problems` lists every reason."""

    def __init__(self, problems):
        super().__init__("Freeze refused:\n  - " + "\n  - ".join(problems))
        self.problems = problems


def _canonical(record):
    return json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Decision log
# ---------------------------------------------------------------------------

def load_decisions(path=REVIEWS_PATH):
    """All decisions in order, after verifying the hash chain. Missing file -> []."""
    path = Path(path)
    if not path.is_file():
        return []
    decisions, previous = [], GENESIS
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        claimed = record.get("record_hash")
        body = {k: v for k, v in record.items() if k != "record_hash"}
        if record.get("prev_hash") != previous or claimed != _sha256(_canonical(body)):
            raise ReviewIntegrityError(f"{path}:{number}: the review log hash chain is broken; the file was edited "
                                       "outside the review tool. Restore it from backup or version control.")
        if record.get("sequence") != len(decisions) + 1:
            raise ReviewIntegrityError(f"{path}:{number}: unexpected sequence number {record.get('sequence')}.")
        decisions.append(record)
        previous = claimed
    nd.load_reviews(path)   # the same schema checks the dataset tools apply
    return decisions


def effective_decisions(decisions):
    """Latest decision per example ID."""
    latest = {}
    for decision in decisions:
        latest[decision["example_id"]] = decision
    return latest


def record_decision(path, example, status, reviewer, human_label=None, reason=None, now=None, redo=False):
    """Append one explicit human decision. Returns the record. Raises ReviewError on invalid input."""
    if status not in DECISIONS:
        raise ReviewError(f"Unknown decision {status!r}; expected one of {DECISIONS}.")
    reviewer = (reviewer or "").strip()
    if not reviewer:
        raise ReviewError("A reviewer identifier is required.")
    reason = (reason or "").strip() or None
    ai_label = example["review"]["ai_label"]
    if status == "reviewed":
        human_label = ai_label
    elif status == "corrected":
        if human_label not in nd.LABEL_VALUES or human_label == ai_label:
            raise ReviewError("A correction needs a label different from the proposed one.")
        if not reason:
            raise ReviewError("A correction needs a reason.")
    else:
        human_label = None
        if not reason:
            raise ReviewError("A rejection needs a reason.")
    decisions = load_decisions(path)
    previous = effective_decisions(decisions).get(example["example_id"])
    if previous and not redo:
        raise DuplicateReviewError(f"{example['example_id']} already has a decision ({previous['status']} by "
                                   f"{previous['reviewer']}); use --redo to replace it.")
    if redo and not previous:
        raise ReviewError(f"{example['example_id']} has no decision to redo.")
    if redo and not reason:
        raise ReviewError("Replacing a decision needs a reason.")
    record = {
        "sequence": len(decisions) + 1,
        "example_id": example["example_id"],
        "fingerprint": example["review"]["fingerprint"],
        "split": example["split"],
        "family_id": example["family_id"],
        "status": status,
        "ai_label": ai_label,
        "human_label": human_label,
        "corrected_label": human_label if status == "corrected" else None,
        "reviewer": reviewer,
        "reason": reason,
        "note": reason,
        "reviewed_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "supersedes": previous["decision_id"] if previous else None,
        "tool_version": REVIEW_TOOL_VERSION,
        "prev_hash": decisions[-1]["record_hash"] if decisions else GENESIS,
    }
    record["decision_id"] = "rev-" + _sha256(_canonical(record))[:12]
    record["record_hash"] = _sha256(_canonical(record))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record


def history(path, example_id):
    return [d for d in load_decisions(path) if d["example_id"] == example_id]


# ---------------------------------------------------------------------------
# Review plan and progress
# ---------------------------------------------------------------------------

def review_plan(examples, documents=None):
    """Ordered, de-duplicated review plan: the 70 highest-risk examples, then all validation, then all test.

    Within validation and test the order follows the risk queue too. Returns
    (items, report); each item is a queue entry plus the groups it belongs to.
    """
    queue = nd.review_queue(examples, documents)
    by_id = {item["example_id"]: item for item in queue}
    groups = defaultdict(list)
    for item in queue[:HIGH_RISK_COUNT]:
        groups[item["example_id"]].append("high_risk")
    for split in EVALUATION_SPLITS:
        for item in queue:
            if item["split"] == split:
                groups[item["example_id"]].append(split)
    order, seen = [], set()
    for group in ("high_risk",) + EVALUATION_SPLITS:
        for item in queue:
            if group in groups[item["example_id"]] and item["example_id"] not in seen:
                seen.add(item["example_id"])
                order.append({**by_id[item["example_id"]], "groups": groups[item["example_id"]]})
    flagged_train = [item for item in queue if item["split"] == "train" and item["example_id"] not in seen
                     and any(r.startswith("flag:") for r in item["reasons"])]
    high_risk = queue[:HIGH_RISK_COUNT]
    report = {
        "unique_examples": len(order),
        "high_risk": len(high_risk),
        "high_risk_min_priority": high_risk[-1]["priority"] if high_risk else None,
        "high_risk_by_split": dict(Counter(item["split"] for item in high_risk)),
        "validation": sum(item["split"] == "validation" for item in queue),
        "test": sum(item["split"] == "test" for item in queue),
        "overlap_removed": HIGH_RISK_COUNT + sum(item["split"] in EVALUATION_SPLITS for item in queue) - len(order),
        "additional_flagged_training_examples": len(flagged_train),
        "additional_flagged_training_ids": [item["example_id"] for item in flagged_train],
    }
    return order, report


def progress(plan, decisions_by_id):
    """Decided counts per plan group (in first-seen order), plus "all"."""
    result = {}
    groups = list(dict.fromkeys(group for item in plan for group in item["groups"]))
    for group in groups + ["all"]:
        items = [item for item in plan if group == "all" or group in item["groups"]]
        decided = [item for item in items if item["example_id"] in decisions_by_id]
        result[group] = {"decided": len(decided), "total": len(items)}
    return result


def stale_decisions(examples, decisions_by_id):
    """Decisions whose example no longer exists or whose content fingerprint changed."""
    by_id = {e["example_id"]: e for e in examples}
    return sorted(example_id for example_id, decision in decisions_by_id.items()
                  if example_id not in by_id or by_id[example_id]["review"]["fingerprint"] != decision["fingerprint"])


def _progress_line(stats):
    groups = ", ".join(f"{group.replace('_', '-')} {s['decided']}/{s['total']}" for group, s in stats.items()
                       if group != "all")
    return f"decided {stats['all']['decided']}/{stats['all']['total']}" + (f"  |  {groups}" if groups else "")


LABEL_KEYS = {"1": "SUPPORTED", "2": "CONTRADICTED", "3": "INSUFFICIENT_EVIDENCE"}


def run_session(examples, plan, path, reviewer, input_fn=input, output_fn=print, now=None):
    """Interactive review of undecided plan items, resuming where the log leaves off.

    Keys: a = accept, c = correct, r = reject, s = skip (nothing recorded), q = save and exit.
    Every decision is written to the log immediately, so an interruption loses nothing.
    """
    by_id = {e["example_id"]: e for e in examples}
    decided = effective_decisions(load_decisions(path))
    pending = [item for item in plan if item["example_id"] not in decided]
    summary = {"recorded": 0, "skipped": 0, "stopped": False}
    output_fn(f"Reviewer: {reviewer}   Log: {path}")
    output_fn(f"Resuming: {_progress_line(progress(plan, decided))}; {len(pending)} to go.")
    try:
        for position, item in enumerate(pending, start=1):
            example = by_id[item["example_id"]]
            warnings = [r[5:] for r in item["reasons"] if r.startswith("flag:")]
            output_fn("\n" + "=" * 78)
            output_fn(f"[{position}/{len(pending)} this session]  {_progress_line(progress(plan, decided))}")
            output_fn(f"Example: {item['example_id']}   in: {', '.join(item['groups'])}   split: {item['split']}")
            output_fn(f"Family: {example['family_id']}   Section: {item['section_id']}   "
                      f"Format: {item['evidence_format']}   Phenomenon: {example['phenomenon']}")
            output_fn(f"Warnings: {', '.join(warnings) if warnings else 'none'}")
            output_fn(f"\nEVIDENCE:\n  {item['evidence']}")
            output_fn(f"\nCLAIM:\n  {item['claim']}")
            output_fn(f"\nPROPOSED LABEL (AI draft): {item['proposed_label']}")
            output_fn(f"Rationale: {item['rationale']}")
            output_fn(f"\nOriginal section {item['section_id']}:\n  {item['original_section_text']}")
            while True:
                choice = input_fn("\n[a]ccept  [c]orrect  [r]eject  [s]kip  [q]uit > ").strip().lower()
                if choice in ("a", "c", "r", "s", "q"):
                    break
            if choice == "q":
                summary["stopped"] = True
                break
            if choice == "s":
                summary["skipped"] += 1
                continue
            label, reason = None, None
            if choice == "c":
                options = {k: v for k, v in LABEL_KEYS.items() if v != item["proposed_label"]}
                while label is None:
                    key = input_fn("Correct label: " + "  ".join(f"[{k}] {v}" for k, v in options.items()) + " > ").strip()
                    label = options.get(key)
            if choice in ("c", "r"):
                while not reason:
                    reason = input_fn("Reason (required): ").strip()
            else:
                reason = input_fn("Note (optional, Enter to skip): ").strip() or None
            status = {"a": "reviewed", "c": "corrected", "r": "rejected"}[choice]
            decided[item["example_id"]] = record_decision(path, example, status, reviewer, label, reason, now=now)
            summary["recorded"] += 1
    except (KeyboardInterrupt, EOFError):
        summary["stopped"] = True
        output_fn("\nInterrupted. Every decision made so far is saved.")
    summary["remaining"] = len([item for item in plan if item["example_id"] not in decided])
    output_fn(f"\nSession: {summary['recorded']} recorded, {summary['skipped']} skipped (not recorded). "
              f"{summary['remaining']} plan examples still undecided. Run the same command to resume.")
    return summary


def redo_decision(examples, path, example_id, reviewer, input_fn=input, output_fn=print, now=None):
    """Show one decided example and its history, then record a replacement decision (reason required)."""
    example = next((e for e in examples if e["example_id"] == example_id), None)
    if example is None:
        raise ReviewError(f"Unknown example {example_id}.")
    past = history(path, example_id)
    if not past:
        raise ReviewError(f"{example_id} has no decision to redo; review it normally.")
    for record in past:
        output_fn(f"  #{record['sequence']} {record['status']} -> {record['human_label']} by {record['reviewer']} "
                  f"at {record['reviewed_at']}: {record['reason'] or ''}")
    output_fn(f"Evidence: {example['evidence']}\nClaim: {example['claim']}\nAI label: {example['review']['ai_label']}")
    choice = ""
    while choice not in ("a", "c", "r", "q"):
        choice = input_fn("New decision: [a]ccept  [c]orrect  [r]eject  [q]uit > ").strip().lower()
    if choice == "q":
        return None
    label = None
    if choice == "c":
        options = {k: v for k, v in LABEL_KEYS.items() if v != example["review"]["ai_label"]}
        while label is None:
            label = options.get(input_fn("Label: " + "  ".join(f"[{k}] {v}" for k, v in options.items()) + " > ").strip())
    reason = ""
    while not reason:
        reason = input_fn("Reason for replacing the earlier decision (required): ").strip()
    status = {"a": "reviewed", "c": "corrected", "r": "rejected"}[choice]
    return record_decision(path, example, status, reviewer, label, reason, now=now, redo=True)


# ---------------------------------------------------------------------------
# Shortcut-learning audit
# ---------------------------------------------------------------------------

PURITY_THRESHOLD = 0.85
MIN_SUPPORT = 30


def text_cues(example):
    """Cues a model can see in the text itself (not metadata)."""
    claim, evidence = example["claim"], example["evidence"]
    words = len(claim.split())
    overlap = nd.coverage(claim, evidence)
    stripped = claim.rstrip(".").lower()
    return {
        "numbers_not_in_evidence": bool(nd.numbers_in(claim) - nd.numbers_in(evidence)),
        "claim_has_numbers": bool(nd.numbers_in(claim)),
        "direct_quotation": stripped in evidence.lower() or overlap >= 0.95,
        "claim_has_negation": nd.has_negation(claim),
        "negation_differs_from_evidence": nd.has_negation(claim) != nd.has_negation(evidence),
        "scope_terms_not_in_evidence": bool(nd.scope_terms(claim) - nd.scope_terms(evidence)),
        "evidence_format": example["evidence_format"],
        "claim_length": "<=8 words" if words <= 8 else "9-14 words" if words <= 14 else "15-20 words" if words <= 20
        else ">20 words",
        "content_overlap": "<0.4" if overlap < 0.4 else "0.4-0.7" if overlap < 0.7 else "0.7-0.95" if overlap < 0.95
        else ">=0.95",
    }


def _distribution(rows):
    counts = Counter(e["label"] for e in rows)
    total = sum(counts.values())
    dominant, share = (counts.most_common(1)[0][0], counts.most_common(1)[0][1] / total) if total else (None, None)
    return {"n": total, **{label: counts.get(label, 0) for label in nd.LABEL_VALUES},
            "dominant": dominant, "dominant_share": round(share, 3) if share is not None else None}


RECOMMENDATIONS = {
    "numerical": "Add SUPPORTED claims that restate or correctly instantiate numbers ('a $3,000 purchase' under "
                 "'up to $5,000') and INSUFFICIENT_EVIDENCE claims with numbers the evidence never mentions, so "
                 "'number changed' is not a perfect contradiction cue.",
    "direct": "Add CONTRADICTED and INSUFFICIENT_EVIDENCE claims that copy most of the evidence wording but change "
              "one decisive detail, so near-verbatim wording is not a perfect support cue.",
    "paraphrase": "Add paraphrased claims that are CONTRADICTED or unsupported, not only supported ones.",
    "condition": "Add SUPPORTED claims that restate conditions correctly in new words, so 'condition mentioned' "
                 "is not a contradiction cue.",
    "missing_information": "Add INSUFFICIENT_EVIDENCE claims with high word overlap with the evidence (same topic, "
                           "unstated detail), so low overlap is not the only signal for missing information.",
    "distractor_sentence": "Distractors are INSUFFICIENT_EVIDENCE by construction; keep them, but report "
                           "per-phenomenon accuracy so they cannot inflate the headline score.",
    "multi_sentence_partial": "Partial-evidence examples are INSUFFICIENT_EVIDENCE by construction; add more "
                              "multi-sentence claims so this group is not tiny.",
}


def shortcut_audit(examples):
    """Label associations a model could exploit without verifying anything. Reports; never rewrites data."""
    phenomenon_table = {}
    for phenomenon in sorted({e["phenomenon"] for e in examples}):
        rows = [e for e in examples if e["phenomenon"] == phenomenon]
        dist = _distribution(rows)
        dist["single_label"] = sum(dist[label] > 0 for label in nd.LABEL_VALUES) == 1
        phenomenon_table[phenomenon] = dist

    cue_rows = [(e, text_cues(e)) for e in examples]
    cue_tables, risky = {}, []
    for cue in cue_rows[0][1] if cue_rows else []:
        values = sorted({str(cues[cue]) for _, cues in cue_rows})
        cue_tables[cue] = {}
        for value in values:
            dist = _distribution([e for e, cues in cue_rows if str(cues[cue]) == value])
            cue_tables[cue][value] = dist
            if dist["n"] >= MIN_SUPPORT and dist["dominant_share"] >= PURITY_THRESHOLD:
                risky.append({"cue": cue, "value": value, **dist})

    length_by_label = {label: round(statistics.fmean(len(e["claim"].split()) for e in examples if e["label"] == label), 1)
                       for label in nd.LABEL_VALUES if any(e["label"] == label for e in examples)}

    # Cue-only baseline: majority label per combination of text cues, fit on train, scored on validation/test.
    def key(cues):
        return (cues["numbers_not_in_evidence"], cues["direct_quotation"], cues["claim_has_negation"],
                cues["content_overlap"], cues["evidence_format"])

    train = [(e, c) for e, c in cue_rows if e["split"] == "train"]
    majority_label = Counter(e["label"] for e, _ in train).most_common(1)[0][0] if train else None
    table = defaultdict(Counter)
    for e, cues in train:
        table[key(cues)][e["label"]] += 1

    def predict(cues):
        counts = table.get(key(cues))
        if not counts:
            return majority_label
        best = max(counts.values())
        return next(label for label in nd.LABEL_VALUES if counts[label] == best)

    baseline = {}
    for split in nd.SPLITS:
        rows = [(e, c) for e, c in cue_rows if e["split"] == split]
        if rows:
            baseline[split] = {
                "n": len(rows),
                "cue_rule_accuracy": round(sum(predict(c) == e["label"] for e, c in rows) / len(rows), 3),
                "majority_class_accuracy": round(sum(e["label"] == majority_label for e, _ in rows) / len(rows), 3),
            }

    recommendations = [f"{p}: {RECOMMENDATIONS[p]}" for p, d in phenomenon_table.items()
                       if d["single_label"] and p in RECOMMENDATIONS]
    for item in risky:
        recommendations.append(
            f"Text cue {item['cue']}={item['value']} predicts {item['dominant']} for {item['dominant_share']:.0%} of "
            f"{item['n']} examples; add counter-examples with the same cue and a different label.")
    if baseline.get("test", {}).get("cue_rule_accuracy", 0) > baseline.get("test", {}).get("majority_class_accuracy", 1):
        recommendations.append(
            "A rule using only surface cues (number mismatch, verbatim overlap, negation, overlap, format) beats the "
            "majority class on held-out splits. In M6.3, report this cue-only baseline next to every model and break "
            "results down by phenomenon; a model that merely matches it has learned shortcuts.")
    return {
        "examples": len(examples),
        "note": ("Phenomenon tags are partly defined by label (for example 'direct' means a supported restatement), "
                 "so single-label phenomena are expected by construction. A model never sees these tags; the text "
                 "cues below are what it can actually exploit. Nothing was rewritten."),
        "phenomenon_by_label": phenomenon_table,
        "single_label_phenomena": [p for p, d in phenomenon_table.items() if d["single_label"]],
        "cue_by_label": cue_tables,
        "high_purity_cues": risky,
        "mean_claim_words_by_label": length_by_label,
        "label_by_format": cue_tables.get("evidence_format", {}),
        "cue_only_baseline": baseline,
        "recommendations": recommendations,
    }


def print_shortcut_audit(report):
    print("\nShortcut-learning audit (AI-authored draft labels unless reviews say otherwise)")
    print(f"\n  {'phenomenon':<24}{'n':>5}{'SUPP':>6}{'CONTR':>7}{'INSUF':>7}   dominant")
    for phenomenon, d in report["phenomenon_by_label"].items():
        print(f"  {phenomenon:<24}{d['n']:>5}{d['SUPPORTED']:>6}{d['CONTRADICTED']:>7}{d['INSUFFICIENT_EVIDENCE']:>7}"
              f"   {d['dominant']} {d['dominant_share']:.0%}" + ("  <- single label" if d["single_label"] else ""))
    print(f"\n  {'text cue = value':<46}{'n':>5}{'SUPP':>6}{'CONTR':>7}{'INSUF':>7}")
    for cue, values in report["cue_by_label"].items():
        for value, d in values.items():
            mark = "  <- high purity" if d["n"] >= MIN_SUPPORT and d["dominant_share"] >= PURITY_THRESHOLD else ""
            print(f"  {cue + ' = ' + value:<46}{d['n']:>5}{d['SUPPORTED']:>6}{d['CONTRADICTED']:>7}"
                  f"{d['INSUFFICIENT_EVIDENCE']:>7}{mark}")
    print("\n  Mean claim length (words): " + ", ".join(f"{k} {v}" for k, v in report["mean_claim_words_by_label"].items()))
    print("  Cue-only baseline (majority label per surface-cue combination, fit on train):")
    for split, b in report["cue_only_baseline"].items():
        print(f"    {split:<11} cue rule {b['cue_rule_accuracy']:.3f}   majority class {b['majority_class_accuracy']:.3f}"
              f"   (n={b['n']})")
    print("\n  " + report["note"])
    print("\n  Recommendations (not applied):")
    for recommendation in report["recommendations"]:
        print(f"    - {recommendation}")


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------

_VERSION = re.compile(r"^v\d+(\.\d+){0,2}$")
LEAKAGE_KEYS = ("exact_duplicate_pairs", "conflicting_label_pairs", "facts_crossing_splits", "claims_crossing_splits",
                "families_crossing_splits", "sections_crossing_splits", "evidence_texts_crossing_splits",
                "evidence_sentences_crossing_splits")


def decisions_hash(decisions_by_id):
    """Hash of the effective decisions, independent of log formatting."""
    rows = [[d["example_id"], d["fingerprint"], d["status"], d["human_label"], d["reviewer"], d["reviewed_at"],
             d["decision_id"]] for _, d in sorted(decisions_by_id.items())]
    return _sha256(_canonical(rows))


def _counts(rows):
    table = defaultdict(lambda: defaultdict(Counter))
    for e in rows:
        table[e["split"]][e["evidence_format"]][e["label"]] += 1
    return {split: {fmt: dict(sorted(c.items())) for fmt, c in sorted(fmts.items())} for split, fmts in sorted(table.items())}


def freeze(version, dataset_path=nd.DATASET_PATH, reviews_path=REVIEWS_PATH, out_root=FROZEN_DIR,
           facts_path=nd.FACTS_PATH, documents=None, now=None):
    """Write an immutable reviewed snapshot. Raises FreezeError listing every unmet requirement."""
    from src.ingest import DOCUMENTS_DIR, load_documents

    if not _VERSION.match(version or ""):
        raise FreezeError([f"Version {version!r} must look like v1, v1.0 or v1.0.0."])
    target = Path(out_root) / version
    if target.exists():
        raise FreezeError([f"{target} already exists. Frozen versions are never overwritten; choose a new version."])

    problems = []
    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise FreezeError([f"Dataset {dataset_path} not found."])
    drafts_text = dataset_path.read_text(encoding="utf-8")
    drafts = nd.read_jsonl(dataset_path)
    try:
        rebuilt, manifest = nd.build(facts_path, documents=documents)
    except nd.DatasetError as error:
        raise FreezeError([f"The facts file does not load against the documents: {error}"]) from None
    if "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in rebuilt) != drafts_text:
        problems.append(f"{dataset_path} does not regenerate from {facts_path}; run "
                        "`python -m src.nli_dataset --generate --force` and re-check reviews before freezing.")
    try:
        decisions = load_decisions(reviews_path)
    except ReviewIntegrityError as error:
        raise FreezeError([str(error)]) from None
    effective = effective_decisions(decisions)
    stale = stale_decisions(drafts, effective)
    if stale:
        problems.append(f"{len(stale)} decisions refer to examples that changed or no longer exist (e.g. {stale[:3]}); "
                        "review them again with --redo or remove the stale examples' decisions from a backup.")

    unresolved = [e for e in drafts if e["split"] in EVALUATION_SPLITS and e["example_id"] not in effective]
    if unresolved:
        by_split = Counter(e["split"] for e in unresolved)
        problems.append(f"{len(unresolved)} evaluation examples have no human decision "
                        f"({', '.join(f'{k} {v}' for k, v in sorted(by_split.items()))}). Every validation and test "
                        "example must be accepted, corrected or explicitly rejected: "
                        "python -m src.nli_review --review --reviewer \"Your Name\"")
    if problems:
        raise FreezeError(problems)

    reviewed = copy.deepcopy(drafts)
    applied = nd.apply_reviews(reviewed, list(effective.values()))
    retained = [e for e in reviewed if e["review"]["status"] != "rejected"]
    excluded = [e for e in reviewed if e["review"]["status"] == "rejected"]
    for e in retained:
        if e["split"] in EVALUATION_SPLITS and e["review"]["status"] not in ("reviewed", "corrected"):
            problems.append(f"{e['example_id']} ({e['split']}) is not human-reviewed.")
    errors = nd.validate(retained, documents, family_splits=manifest["family_splits"])
    problems += [f"validation: {error}" for error in errors[:20]]
    leakage = nd.audit(retained, documents, benchmark_claims=[], manifest=manifest)
    for key in LEAKAGE_KEYS:
        if leakage[key]:
            problems.append(f"leakage check failed: {key} = {leakage[key][:3]}")
    if problems:
        raise FreezeError(problems)

    documents = documents if documents is not None else load_documents()
    document_hashes = {d["document_id"]: _sha256(_canonical(d)) for d in sorted(documents, key=lambda d: d["document_id"])}
    status_by_split = {split: dict(Counter(e["review"]["status"] for e in reviewed if e["split"] == split))
                       for split in nd.SPLITS}
    out_files = {}
    Path(out_root).mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".staging-{version}-", dir=out_root))   # same filesystem as the target
    try:
        for split in nd.SPLITS:
            rows = [e for e in retained if e["split"] == split]
            nd.write_jsonl(staging / f"{split}.jsonl", rows)
        nd.write_jsonl(staging / "excluded.jsonl", [
            {"example_id": e["example_id"], "split": e["split"], "reason": e["review"]["note"],
             "reviewer": e["review"]["reviewer"], "decision_id": e["review"].get("decision_id")} for e in excluded])
        if Path(reviews_path).is_file():
            shutil.copyfile(reviews_path, staging / "reviews.snapshot.jsonl")
        else:
            (staging / "reviews.snapshot.jsonl").write_text("", encoding="utf-8")
        for name in sorted(p.name for p in staging.iterdir()):
            out_files[name] = _sha256((staging / name).read_bytes())
        frozen_manifest = {
            "dataset": "RAGForge policy NLI (reviewed, frozen)",
            "version": version,
            "frozen_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
            "review_tool_version": REVIEW_TOOL_VERSION,
            "generator_version": nd.GENERATOR_VERSION,
            "requirements": "Every retained validation and test example has an explicit human decision; rejected "
                            "examples are excluded and listed in excluded.jsonl. Training examples may still be "
                            "AI-authored drafts (see review_status_by_split).",
            "source_hashes": {
                "facts_sha256": _sha256(Path(facts_path).read_bytes()),
                "draft_dataset_sha256": _sha256(drafts_text),
                "documents_dir": str(DOCUMENTS_DIR.relative_to(PROJECT_ROOT)),
                "documents_sha256": document_hashes,
                "documents_combined_sha256": _sha256(_canonical(document_hashes)),
            },
            "review_hashes": {
                "review_log_sha256": _sha256(Path(reviews_path).read_bytes()) if Path(reviews_path).is_file() else None,
                "review_log_head": decisions[-1]["record_hash"] if decisions else None,
                "decisions_logged": len(decisions),
                "effective_decisions": len(effective),
                "effective_decisions_sha256": decisions_hash(effective),
            },
            "file_sha256": out_files,
            "counts": {
                "retained": len(retained),
                "excluded_rejected": len(excluded),
                "by_split": dict(Counter(e["split"] for e in retained)),
                "by_split_format_label": _counts(retained),
                "by_label": dict(Counter(e["label"] for e in retained)),
                "by_format": dict(Counter(e["evidence_format"] for e in retained)),
                "corrected_labels": sum(e["review"]["status"] == "corrected" for e in retained),
            },
            "review_status_by_split": status_by_split,
            "reviewers": sorted({d["reviewer"] for d in effective.values()}),
            "applied_reviews": applied["applied"],
            "family_splits": manifest["family_splits"],
            "leakage_checks": {key: len(leakage[key]) for key in LEAKAGE_KEYS},
            "leakage_limitations": leakage["limitations"],
        }
        (staging / "manifest.json").write_text(json.dumps(frozen_manifest, indent=2, ensure_ascii=False) + "\n",
                                               encoding="utf-8")
        if target.exists():
            raise FreezeError([f"{target} appeared while freezing; nothing was overwritten."])
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    for path in target.iterdir():
        path.chmod(0o444)
    return frozen_manifest


def verify_frozen(directory):
    """Re-hash a frozen version's files against its manifest. Returns a list of mismatches."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    problems = []
    for name, digest in manifest["file_sha256"].items():
        path = directory / name
        if not path.is_file() or _sha256(path.read_bytes()) != digest:
            problems.append(f"{name} is missing or was modified.")
    return problems


# ---------------------------------------------------------------------------
# Random quality audit (stratified sample) and experimental freeze
# ---------------------------------------------------------------------------

AUDIT_SAMPLE_SIZE = 50
AUDIT_SAMPLE_SEED = 6203
AUDIT_SAMPLE_PATH = PROJECT_ROOT / "data" / "nli" / "audit_sample_v1.json"
EXPERIMENTAL_TYPE = "experimental_ai_labeled_sample_audited"
_EXPERIMENTAL_VERSION = re.compile(r"^v\d+(\.\d+){0,2}-exp$")
AGREEMENT_WARNING = 0.90       # below this observed label agreement, recommend targeted review before training


def _stratum(example):
    return (example["split"], example["review"]["ai_label"], example["evidence_format"])


def stratified_sample(examples, size=AUDIT_SAMPLE_SIZE, seed=AUDIT_SAMPLE_SEED):
    """Deterministic proportional sample over split x AI label x evidence format.

    Allocation is proportional with largest remainders (ties broken by stratum
    name), then adjusted so every split, every label and both formats appear. Examples are
    drawn uniformly at random inside each stratum (review priority, flags and
    existing decisions play no part), and the review order is shuffled so
    strata are not reviewed in blocks.
    """
    if not isinstance(size, int) or size < 1 or size > len(examples):
        raise ValueError(f"Sample size must be between 1 and {len(examples)}, got {size!r}.")
    strata = defaultdict(list)
    for example in examples:
        strata[_stratum(example)].append(example["example_id"])
    keys = sorted(strata)
    total = len(examples)
    quotas = {k: size * len(strata[k]) / total for k in keys}
    allocation = {k: int(quotas[k]) for k in keys}
    for k in sorted(keys, key=lambda k: (-(quotas[k] - allocation[k]), k))[:size - sum(allocation.values())]:
        allocation[k] += 1

    def ensure(position, value):
        if any(allocation[k] for k in keys if k[position] == value):
            return
        candidates = [k for k in keys if k[position] == value and strata[k]]
        donors = [k for k in keys if allocation[k] > 1]
        if candidates and donors:
            allocation[max(candidates, key=lambda k: (quotas[k], k))] += 1
            allocation[max(donors, key=lambda k: (allocation[k], k))] -= 1

    for split in nd.SPLITS:
        ensure(0, split)
    for label in nd.LABEL_VALUES:
        ensure(1, label)
    for evidence_format in nd.FORMATS:
        ensure(2, evidence_format)
    rng = random.Random(seed)
    chosen = []
    for k in keys:
        chosen += rng.sample(sorted(strata[k]), allocation[k])
    random.Random(f"{seed}:order").shuffle(chosen)
    return {
        "seed": seed,
        "size": size,
        "population": total,
        "method": "proportional stratified random sample over split x AI label x evidence format",
        "allocation": {"|".join(k): allocation[k] for k in keys if allocation[k]},
        "population_by_stratum": {"|".join(k): len(strata[k]) for k in keys},
        "example_ids": chosen,
    }


def sample_distribution(examples, sample):
    by_id = {e["example_id"]: e for e in examples}
    rows = [by_id[i] for i in sample["example_ids"]]
    return {"split": dict(Counter(e["split"] for e in rows)),
            "label": dict(Counter(e["review"]["ai_label"] for e in rows)),
            "format": dict(Counter(e["evidence_format"] for e in rows))}


def sample_plan(examples, sample, documents=None):
    """Review-plan items for the sample, in the sample's shuffled order."""
    queue = {item["example_id"]: item for item in nd.review_queue(examples, documents)}
    missing = [i for i in sample["example_ids"] if i not in queue]
    if missing:
        raise ReviewError(f"Sampled examples are not drafts in this dataset: {missing[:3]}.")
    return [{**queue[i], "groups": ["audit_sample", queue[i]["split"]]} for i in sample["example_ids"]]


def wilson_interval(successes, n, z=1.96):
    """95% Wilson score interval for a proportion; (None, None) when n is 0."""
    if n == 0:
        return None, None
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)


def _agreement(rows):
    accepted = sum(d["status"] == "reviewed" for _, d in rows)
    corrected = sum(d["status"] == "corrected" for _, d in rows)
    rejected = sum(d["status"] == "rejected" for _, d in rows)
    labelled = accepted + corrected
    low, high = wilson_interval(accepted, labelled)
    return {"reviewed": len(rows), "accepted": accepted, "corrected": corrected, "rejected": rejected,
            "label_agreement": round(accepted / labelled, 3) if labelled else None, "ci95": [low, high],
            "acceptance_rate": round(accepted / len(rows), 3) if rows else None}


def sample_report(examples, sample, decisions_by_id):
    """Agreement between AI labels and human decisions on the random sample (an estimate, not proof)."""
    by_id = {e["example_id"]: e for e in examples}
    decided = [(by_id[i], decisions_by_id[i]) for i in sample["example_ids"] if i in decisions_by_id]
    overall = _agreement(decided)

    def by(key, values):
        return {value: _agreement([(e, d) for e, d in decided if key(e) == value]) for value in values}

    report = {
        "sample_size": sample["size"],
        "seed": sample["seed"],
        **overall,
        "remaining_in_sample": sample["size"] - len(decided),
        "by_class": by(lambda e: e["review"]["ai_label"], nd.LABEL_VALUES),
        "by_format": by(lambda e: e["evidence_format"], nd.FORMATS),
        "by_split": by(lambda e: e["split"], nd.SPLITS),
        "corrections": [{"example_id": e["example_id"], "split": e["split"], "format": e["evidence_format"],
                         "claim": e["claim"], "ai_label": d["ai_label"], "human_label": d["human_label"],
                         "reason": d["reason"]} for e, d in decided if d["status"] == "corrected"],
        "rejections": [{"example_id": e["example_id"], "split": e["split"], "claim": e["claim"],
                        "ai_label": d["ai_label"], "reason": d["reason"]} for e, d in decided if d["status"] == "rejected"],
        "decisions_outside_sample": sum(i not in set(sample["example_ids"]) for i in decisions_by_id),
        "unreviewed_overall": sum(e["example_id"] not in decisions_by_id for e in examples),
        "warnings": [],
        "interpretation": "Observed agreement on a 50-example random sample estimates label quality; it is not proof "
                          "that every label is correct. Decisions made outside the sample (for example on the "
                          "risk-ordered queue) are not part of this estimate.",
    }
    if report["remaining_in_sample"]:
        report["warnings"].append(f"The sample is incomplete: {report['remaining_in_sample']} of {sample['size']} "
                                  "examples still need a decision.")
    if overall["rejected"]:
        report["warnings"].append(f"{overall['rejected']} sampled examples were rejected as defective. Recommend "
                                  "targeted review of similar examples before training.")
    if overall["label_agreement"] is not None and overall["label_agreement"] < AGREEMENT_WARNING:
        report["warnings"].append(
            f"Label agreement {overall['label_agreement']:.0%} is below {AGREEMENT_WARNING:.0%}: the draft labels have "
            "a material error rate. Recommend targeted review (at least the corrected examples' phenomena and "
            "families, and the validation split) before training.")
    for name, groups in (("class", report["by_class"]), ("format", report["by_format"]),
                         ("split", report["by_split"])):
        for value, stats in groups.items():
            if stats["corrected"] + stats["rejected"] >= 2:
                report["warnings"].append(f"{stats['corrected'] + stats['rejected']} errors concentrated in {name} "
                                          f"{value}; review more of that group.")
    return report


def print_sample_report(report):
    print(f"\nRandom quality audit: {report['reviewed']} of {report['sample_size']} sampled examples reviewed "
          f"(seed {report['seed']})")
    print(f"  accepted {report['accepted']}, corrected {report['corrected']}, rejected {report['rejected']}")
    if report["label_agreement"] is not None:
        print(f"  Observed label agreement: {report['label_agreement']:.1%} (95% CI {report['ci95'][0]:.1%}-"
              f"{report['ci95'][1]:.1%}; rejected examples excluded) - acceptance rate {report['acceptance_rate']:.1%}")
    for name in ("by_class", "by_format", "by_split"):
        parts = [f"{k} {v['accepted']}/{v['accepted'] + v['corrected']}" + (f" (+{v['rejected']} rejected)" if v["rejected"]
                 else "") for k, v in report[name].items() if v["reviewed"]]
        print(f"  Agreement {name.replace('by_', 'by ')}: " + (", ".join(parts) or "n/a"))
    for item in report["corrections"]:
        print(f"  CORRECTED {item['example_id']} [{item['split']}/{item['format']}] {item['ai_label']} -> "
              f"{item['human_label']}: {item['claim']} ({item['reason']})")
    for item in report["rejections"]:
        print(f"  REJECTED  {item['example_id']} [{item['split']}] {item['claim']} ({item['reason']})")
    print(f"  Remaining in sample: {report['remaining_in_sample']}; unreviewed examples overall: "
          f"{report['unreviewed_overall']}; decisions outside the sample: {report['decisions_outside_sample']}")
    for warning in report["warnings"]:
        print(f"  WARNING: {warning}")
    print(f"  {report['interpretation']}")


EXPERIMENTAL_LIMITATIONS = [
    "Labels are AI-authored drafts except where a human decision is recorded; most examples were never reviewed.",
    "Label quality is estimated from a 50-example random sample; the confidence interval is wide and errors outside "
    "the sample remain.",
    "Surface cues alone predict draft labels with 0.758 (validation) and 0.730 (test) accuracy; models must be "
    "compared against that cue-only baseline and by phenomenon.",
    "Validation and test each contain only 3 of 15 policy families.",
    "The M5 benchmark shares documents and many facts with this dataset and is not an independent test.",
    "For exploratory M6.3 fine-tuning only; not a verified benchmark. Use the strict freeze for verified evaluation.",
]


def _write_version(out_root, target, version, files, manifest):
    """Stage files, hash them into the manifest, move into place atomically, make read-only."""
    Path(out_root).mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".staging-{version}-", dir=out_root))
    try:
        for name, content in files.items():
            (staging / name).write_bytes(content)
        manifest["file_sha256"] = {name: _sha256((staging / name).read_bytes()) for name in sorted(files)}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                               encoding="utf-8")
        if target.exists():
            raise FreezeError([f"{target} appeared while freezing; nothing was overwritten."])
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    for path in target.iterdir():
        path.chmod(0o444)
    return manifest


def _jsonl_bytes(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def freeze_experimental(version, dataset_path=nd.DATASET_PATH, reviews_path=REVIEWS_PATH, out_root=FROZEN_DIR,
                        facts_path=nd.FACTS_PATH, documents=None, now=None, sample_size=AUDIT_SAMPLE_SIZE,
                        seed=AUDIT_SAMPLE_SEED):
    """Freeze AI-labeled data after a completed random sample audit. Raises FreezeError listing every problem.

    Separate from the strict freeze(), which is unchanged: here only the sampled
    examples must have human decisions. Unreviewed examples stay "draft".
    """
    from src.ingest import DOCUMENTS_DIR, load_documents

    if not _EXPERIMENTAL_VERSION.match(version or ""):
        raise FreezeError([f"Experimental version {version!r} must end in -exp, e.g. v0.1-exp, so it is never "
                           "mistaken for a verified freeze."])
    target = Path(out_root) / version
    if target.exists():
        raise FreezeError([f"{target} already exists. Frozen versions are never overwritten; choose a new version."])
    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise FreezeError([f"Dataset {dataset_path} not found."])
    problems = []
    drafts_text = dataset_path.read_text(encoding="utf-8")
    drafts = nd.read_jsonl(dataset_path)
    try:
        rebuilt, manifest = nd.build(facts_path, documents=documents)
    except nd.DatasetError as error:
        raise FreezeError([f"The facts file does not load against the documents: {error}"]) from None
    if _jsonl_bytes(rebuilt).decode("utf-8") != drafts_text:
        problems.append(f"{dataset_path} does not regenerate deterministically from {facts_path}.")
    schema_errors = nd.validate(drafts, documents, family_splits=manifest["family_splits"])
    problems += [f"schema: {error}" for error in schema_errors[:20]]
    try:
        decisions = load_decisions(reviews_path)
    except ReviewIntegrityError as error:
        raise FreezeError([str(error)]) from None
    effective = effective_decisions(decisions)
    stale = stale_decisions(drafts, effective)
    if stale:
        problems.append(f"{len(stale)} stale decisions (example changed or missing), e.g. {stale[:3]}.")
    sample = stratified_sample(drafts, sample_size, seed)
    undecided = [i for i in sample["example_ids"] if i not in effective]
    if undecided:
        problems.append(f"{len(undecided)} of {sample_size} sampled examples have no human decision. Finish the "
                        "audit: python -m src.nli_review --review-sample")
    rejected = [i for i in sample["example_ids"] if i in effective and effective[i]["status"] == "rejected"]
    if rejected:
        problems.append(f"{len(rejected)} sampled examples were rejected ({rejected[:3]}). A rejected sample means "
                        "a defective example: fix its fact in data/nli/policy_facts.json and regenerate, or, if only "
                        "the label is wrong, replace the decision with a correction (--redo).")
    if problems:
        raise FreezeError(problems)

    reviewed = copy.deepcopy(drafts)
    applied = nd.apply_reviews(reviewed, list(effective.values()))
    retained = [e for e in reviewed if e["review"]["status"] != "rejected"]
    excluded = [e for e in reviewed if e["review"]["status"] == "rejected"]
    errors = nd.validate(retained, documents, family_splits=manifest["family_splits"])
    problems += [f"validation: {error}" for error in errors[:20]]
    leakage = nd.audit(retained, documents, benchmark_claims=[], manifest=manifest)
    for key in LEAKAGE_KEYS:
        if leakage[key]:
            problems.append(f"leakage check failed: {key} = {leakage[key][:3]}")
    if problems:
        raise FreezeError(problems)

    documents = documents if documents is not None else load_documents()
    document_hashes = {d["document_id"]: _sha256(_canonical(d)) for d in sorted(documents, key=lambda d: d["document_id"])}
    report = sample_report(drafts, sample, effective)
    status = lambda rows: dict(Counter(e["review"]["status"] for e in rows))   # noqa: E731
    files = {f"{split}.jsonl": _jsonl_bytes([e for e in retained if e["split"] == split]) for split in nd.SPLITS}
    files["excluded.jsonl"] = _jsonl_bytes([
        {"example_id": e["example_id"], "split": e["split"], "reason": e["review"]["note"],
         "reviewer": e["review"]["reviewer"], "decision_id": e["review"].get("decision_id")} for e in excluded])
    files["reviews.snapshot.jsonl"] = Path(reviews_path).read_bytes() if Path(reviews_path).is_file() else b""
    files["audit_sample.json"] = (json.dumps({**sample, "report": report}, indent=2, ensure_ascii=False) + "\n").encode()
    frozen_manifest = {
        "dataset": "RAGForge policy NLI",
        "dataset_type": EXPERIMENTAL_TYPE,
        "version": version,
        "frozen_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "review_tool_version": REVIEW_TOOL_VERSION,
        "generator_version": nd.GENERATOR_VERSION,
        "intended_use": "Exploratory M6.3 fine-tuning: train on train.jsonl, select on validation.jsonl. test.jsonl "
                        "is reserved for one final evaluation and must not be used for model or threshold selection.",
        "audit_sample": {
            "size": sample["size"], "seed": sample["seed"], "method": sample["method"],
            "allocation": sample["allocation"], "example_ids_sha256": _sha256(_canonical(sample["example_ids"])),
            "reviewed": report["reviewed"], "accepted": report["accepted"], "corrected": report["corrected"],
            "rejected": report["rejected"], "label_agreement": report["label_agreement"], "ci95": report["ci95"],
            "warnings": report["warnings"],
        },
        "labels": {
            "human_reviewed": sum(e["review"]["status"] in ("reviewed", "corrected") for e in retained),
            "human_corrected": sum(e["review"]["status"] == "corrected" for e in retained),
            "rejected_excluded": len(excluded),
            "remaining_draft": sum(e["review"]["status"] == "draft" for e in retained),
            "remaining_draft_by_split": {s: sum(e["split"] == s and e["review"]["status"] == "draft" for e in retained)
                                         for s in nd.SPLITS},
            "review_status_by_split": {s: status([e for e in reviewed if e["split"] == s]) for s in nd.SPLITS},
            "note": "review.ai_label keeps the AI-authored label and review.human_label the human decision; "
                    "unreviewed examples keep status 'draft' and human_label null.",
        },
        "source_hashes": {
            "facts_sha256": _sha256(Path(facts_path).read_bytes()),
            "draft_dataset_sha256": _sha256(drafts_text),
            "documents_dir": str(DOCUMENTS_DIR.relative_to(PROJECT_ROOT)),
            "documents_sha256": document_hashes,
            "documents_combined_sha256": _sha256(_canonical(document_hashes)),
        },
        "review_hashes": {
            "review_log_sha256": _sha256(Path(reviews_path).read_bytes()) if Path(reviews_path).is_file() else None,
            "review_log_head": decisions[-1]["record_hash"] if decisions else None,
            "decisions_logged": len(decisions),
            "effective_decisions": len(effective),
            "effective_decisions_sha256": decisions_hash(effective),
        },
        "counts": {
            "retained": len(retained),
            "by_split": {s: sum(e["split"] == s for e in retained) for s in nd.SPLITS},
            "by_split_label": {s: dict(Counter(e["label"] for e in retained if e["split"] == s)) for s in nd.SPLITS},
            "by_split_format_label": _counts(retained),
            "by_label": dict(Counter(e["label"] for e in retained)),
            "by_format": dict(Counter(e["evidence_format"] for e in retained)),
        },
        "applied_reviews": applied["applied"],
        "family_splits": manifest["family_splits"],
        "leakage_checks": {key: len(leakage[key]) for key in LEAKAGE_KEYS},
        "known_limitations": EXPERIMENTAL_LIMITATIONS + leakage["limitations"],
    }
    return _write_version(out_root, target, version, files, frozen_manifest)


def load_frozen(version, split, frozen_dir=FROZEN_DIR, allow_test=False):
    """Rows of one split of a frozen version, after verifying every file hash. For M6.3.

    The test split is refused unless allow_test=True, so it cannot leak into
    training or model selection by accident. Rejected examples are already excluded.
    """
    if split not in nd.SPLITS:
        raise ValueError(f"split must be one of {nd.SPLITS}.")
    if split == "test" and not allow_test:
        raise ValueError("The test split is reserved for the final evaluation; pass allow_test=True only for that run.")
    directory = Path(frozen_dir) / version
    if not (directory / "manifest.json").is_file():
        raise FileNotFoundError(f"No frozen version at {directory}.")
    problems = verify_frozen(directory)
    if problems:
        raise ValueError(f"Frozen version {version} failed verification: {problems}")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    return nd.read_jsonl(directory / f"{split}.jsonl"), manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_plan(report):
    print("\nReview plan (de-duplicated by example ID, in this order):")
    print(f"  1. {report['high_risk']} highest-risk examples (priority >= {report['high_risk_min_priority']}; "
          f"by split: {report['high_risk_by_split']})")
    print(f"  2. all {report['validation']} validation examples")
    print(f"  3. all {report['test']} test examples")
    print(f"  Unique examples to review: {report['unique_examples']} ({report['overlap_removed']} overlaps removed)")
    print(f"  Additional flagged training examples outside the plan: {report['additional_flagged_training_examples']} "
          "(optional review; not required for freezing)")


def build_parser():
    parser = argparse.ArgumentParser(description="RAGForge NLI dataset review, shortcut audit and freeze (no training).",
                                     allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="show the review plan")
    mode.add_argument("--review", action="store_true", help="start or resume reviewing")
    mode.add_argument("--status", action="store_true", help="review progress and freeze readiness")
    mode.add_argument("--history", metavar="EXAMPLE_ID", help="every recorded decision for one example")
    mode.add_argument("--redo", metavar="EXAMPLE_ID", help="replace an existing decision (reason required)")
    mode.add_argument("--shortcut-audit", action="store_true", help="label-shortcut analysis (no data changes)")
    mode.add_argument("--freeze", action="store_true", help="write an immutable reviewed version")
    mode.add_argument("--verify-frozen", metavar="VERSION", help="re-check a frozen version's file hashes")
    mode.add_argument("--sample", action="store_true", help="write the 50-example stratified audit sample")
    mode.add_argument("--review-sample", action="store_true", help="review the audit sample (start or resume)")
    mode.add_argument("--sample-report", action="store_true", help="agreement estimate from the audit sample")
    mode.add_argument("--freeze-experimental", action="store_true",
                      help="freeze AI-labeled data after the sample audit (version must end in -exp)")
    parser.add_argument("--dataset", default=str(nd.DATASET_PATH), help="draft dataset (default %(default)s)")
    parser.add_argument("--reviews", default=str(REVIEWS_PATH), help="decision log (default %(default)s)")
    parser.add_argument("--reviewer", default=os.environ.get(REVIEWER_ENV), help=f"reviewer ID (or ${REVIEWER_ENV})")
    parser.add_argument("--version", default=None, help="frozen version name, e.g. v1.0")
    parser.add_argument("--frozen-dir", default=str(FROZEN_DIR), help="where frozen versions live")
    parser.add_argument("--facts", default=str(nd.FACTS_PATH), help="fact specification the drafts must regenerate from")
    parser.add_argument("--out", default=None, help="output path for --shortcut-audit")
    return parser


def main(argv=None, input_fn=input):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.verify_frozen:
        problems = verify_frozen(Path(args.frozen_dir) / args.verify_frozen)
        print("\n".join(problems) or f"{args.verify_frozen}: all file hashes match the manifest.")
        return 1 if problems else 0
    if args.freeze_experimental:
        if not args.version:
            parser.error("--freeze-experimental requires --version (e.g. v0.1-exp).")
        try:
            manifest = freeze_experimental(args.version, args.dataset, args.reviews, args.frozen_dir, args.facts,
                                           sample_size=AUDIT_SAMPLE_SIZE, seed=AUDIT_SAMPLE_SEED)
        except FreezeError as error:
            print(error, file=sys.stderr)
            return 1
        labels = manifest["labels"]
        print(f"Frozen {args.version} ({EXPERIMENTAL_TYPE}): {manifest['counts']['retained']} examples; "
              f"{labels['human_reviewed']} human-reviewed ({labels['human_corrected']} corrected), "
              f"{labels['remaining_draft']} still AI drafts, {labels['rejected_excluded']} rejected and excluded.")
        print(f"Read-only files: {Path(args.frozen_dir) / args.version}")
        return 0
    if args.freeze:
        if not args.version:
            parser.error("--freeze requires --version (e.g. v1.0).")
        try:
            manifest = freeze(args.version, args.dataset, args.reviews, args.frozen_dir, args.facts)
        except FreezeError as error:
            print(error, file=sys.stderr)
            return 1
        print(f"Frozen {args.version}: {manifest['counts']['retained']} examples retained, "
              f"{manifest['counts']['excluded_rejected']} rejected. Files are read-only: "
              f"{Path(args.frozen_dir) / args.version}")
        return 0

    examples = nd.read_jsonl(args.dataset)
    try:
        decisions = load_decisions(args.reviews)
    except ReviewIntegrityError as error:
        print(error, file=sys.stderr)
        return 1
    plan, report = review_plan(examples)
    sample = (stratified_sample(examples, AUDIT_SAMPLE_SIZE, AUDIT_SAMPLE_SEED)
              if len(examples) >= AUDIT_SAMPLE_SIZE else None)
    if sample is None and (args.sample or args.review_sample or args.sample_report):
        parser.error(f"The dataset has fewer than {AUDIT_SAMPLE_SIZE} examples; no audit sample can be drawn.")

    if args.sample:
        path = AUDIT_SAMPLE_PATH
        record = {**sample, "dataset_sha256": _sha256(Path(args.dataset).read_bytes())}
        if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != record:
            parser.error(f"{path} exists with a different sample (the dataset changed?). Remove it deliberately "
                         "if you want a new sample; existing decisions are kept either way.")
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        dist = sample_distribution(examples, sample)
        decided = effective_decisions(decisions)
        already = sum(i in decided for i in sample["example_ids"])
        print(f"Audit sample: {sample['size']} unique examples (seed {sample['seed']}) -> {path}")
        for key in ("split", "label", "format"):
            print(f"  by {key}: " + ", ".join(f"{k} {v}" for k, v in sorted(dist[key].items())))
        print(f"  already decided earlier: {already}; to review: {sample['size'] - already}")
        return 0
    if args.sample_report:
        print_sample_report(sample_report(examples, sample, effective_decisions(decisions)))
        return 0

    if args.plan:
        print_plan(report)
        return 0
    if args.history:
        records = history(args.reviews, args.history)
        if not records:
            print(f"No decisions recorded for {args.history}.")
        for r in records:
            print(f"#{r['sequence']} {r['decision_id']} {r['status']}: AI {r['ai_label']} -> human {r['human_label']} "
                  f"by {r['reviewer']} at {r['reviewed_at']}" + (f" (supersedes {r['supersedes']})" if r["supersedes"]
                                                                 else "") + (f" - {r['reason']}" if r["reason"] else ""))
        return 0
    if args.status:
        effective = effective_decisions(decisions)
        stats = progress(plan, effective)
        print_plan(report)
        print(f"\nProgress: {_progress_line(stats)}")
        print("Decisions: " + (", ".join(f"{k} {v}" for k, v in sorted(Counter(d['status'] for d in effective.values()).items()))
                               or "none yet") + f" ({len(decisions)} log records, hash chain OK)")
        if sample is not None:
            sample_done = sum(i in effective for i in sample["example_ids"])
            print(f"Audit sample: {sample_done}/{sample['size']} decided - experimental freeze "
                  + ("ready to attempt" if sample_done == sample["size"] else "NOT ready"))
        stale = stale_decisions(examples, effective)
        if stale:
            print(f"Stale decisions: {len(stale)}")
        undecided = sum(e["split"] in EVALUATION_SPLITS and e["example_id"] not in effective for e in examples)
        print(f"Unreviewed examples overall: {sum(e['example_id'] not in effective for e in examples)} of {len(examples)}")
        print("Freeze readiness: " + ("ready" if not undecided and not stale else
                                      f"NOT ready - {undecided} validation/test examples undecided"))
        return 0
    if args.shortcut_audit:
        reviewed = copy.deepcopy(examples)
        nd.apply_reviews(reviewed, list(effective_decisions(decisions).values()))
        audit_report = shortcut_audit([e for e in reviewed if e["review"]["status"] != "rejected"])
        print_shortcut_audit(audit_report)
        out = Path(args.out or SHORTCUT_AUDIT_PATH)
        out.write_text(json.dumps(audit_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nReport: {out}")
        return 0
    if not args.reviewer or not args.reviewer.strip():
        parser.error(f"--reviewer is required (or set {REVIEWER_ENV}).")
    if args.redo:
        try:
            record = redo_decision(examples, args.reviews, args.redo, args.reviewer, input_fn=input_fn)
        except ReviewError as error:
            parser.error(str(error))
        print("No change." if record is None else f"Recorded {record['decision_id']} (supersedes {record['supersedes']}).")
        return 0
    if args.review_sample:
        run_session(examples, sample_plan(examples, sample), args.reviews, args.reviewer, input_fn=input_fn)
        return 0
    run_session(examples, plan, args.reviews, args.reviewer, input_fn=input_fn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
