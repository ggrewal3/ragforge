"""M6.4C controlled self-correction evaluation (evaluation only; no production changes).

    python scripts/m64c_eval.py --plan                              # validate cases, show the API budget; no calls
    python scripts/m64c_eval.py --dry-run                           # offline: fake Gemini + local V3, new dryrun-* run
    python scripts/m64c_eval.py --live --run-id RUN                 # prints the plan; add --confirm-live to call Gemini
    python scripts/m64c_eval.py --live --run-id RUN --confirm-live  # runs or resumes RUN
    python scripts/m64c_eval.py --report RUN                        # metrics.json, manual_review.jsonl, report.md

Design
  * Cases (scripts/m64c_cases.json) supply the question, original answer and citations;
    every original answer is evaluation-supplied (seeded or adapted), never regenerated.
  * Freeze (once per case): the evidence snapshot comes from the production retrieval
    (hybrid 0.6/0.4, top 5) and is stored with its fingerprint; citations are checked with
    validate_citations(); the initial claims are extracted ONCE with the production Gemini
    extractor and validated with parse_claims(). Expected-fact excerpts must be exact quotes
    of a snapshot section.
  * Condition A: verify_and_correct(..., revise=False) on the frozen inputs.
    Condition B: verify_and_correct(..., revise=True) on the same frozen inputs. Both replay
    the frozen initial claims for pass 1 (ReplayExtractor); pass 2 (after a correction) uses
    the live Gemini extractor. Verification, citation checks, claim validation, the
    one-revision limit and fail-closed rules are the production code, unchanged.
  * Gemini responses are cached per prompt in the run directory (CachingLLM), and every
    finished case is recorded, so an interrupted run resumes without repeating calls.
    API errors are recorded, never retried to get a better result.
  * Factual labels come from the case definitions (original answers) and from a
    provisional annotation file (revised answers). V3 verdicts are never used as truth.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.citations import FAIL, GroundedAnswer, validate_citations  # noqa: E402
from src.claims import GeminiClaimExtractor, InvalidClaimsOutput, StageOutput, parse_claims  # noqa: E402
from src.ingest import PROJECT_ROOT  # noqa: E402
from src.llm import API_KEY_ENV, PRICING_SOURCE, LLMError, LLMResponse, estimate_cost_usd  # noqa: E402
from src.rag import DEFAULT_EVIDENCE_K, RAG_METHOD, retrieval_options, source_metadata  # noqa: E402
from src.self_correction import evidence_fingerprint, evidence_snapshot, final_status, verify_and_correct  # noqa: E402

CASES_PATH = Path(__file__).with_name("m64c_cases.json")
OUT_ROOT = PROJECT_ROOT / "artifacts" / "nli" / "m6.4c-evaluation"
CATEGORIES = {"genuine_contradiction": 2, "insufficient_evidence": 2, "false_contradiction": 2, "compound_claim": 2,
              "already_supported": 2, "entity_number_confusion": 2}
FACT_LABELS = ("factually_supported", "factually_incorrect", "partially_supported", "insufficient_evidence",
               "unable_to_determine")
EFFECTS = ("fixed_incorrect_claim", "removed_unsupported_claim", "preserved_correct_claim",
           "introduced_incorrect_claim", "removed_useful_correct_information", "changed_claim_meaning",
           "improved_atomicity", "produced_abstention")
FAILURE_CAUSES = ("false_contradiction", "false_support", "missing_evidence", "entity_number_confusion",
                  "compound_claim", "claim_extraction_failure", "correction_failure", "citation_failure",
                  "infrastructure_api_failure")
RETURNED = ("accepted_original", "accepted_revision")
FILES = {"cases": "cases.jsonl", "A": "baseline_predictions.jsonl", "B": "correction_predictions.jsonl"}


class EvaluationError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Small I/O helpers
# ---------------------------------------------------------------------------

def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_state():
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
        except OSError:
            return None
    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "uncommitted_changes": bool(status),
            "changed_paths": status.splitlines() if status else []}


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------

def load_cases(path=CASES_PATH):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data["cases"]
    ids = [c["case_id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise EvaluationError("Duplicate case IDs.")
    for case in cases:
        for key in ("question", "original_answer", "citations", "category", "answer_source", "seeded_error", "expected"):
            if key not in case:
                raise EvaluationError(f"{case['case_id']}: missing {key}.")
        if case["expected"]["original_answer_label"] not in FACT_LABELS:
            raise EvaluationError(f"{case['case_id']}: unknown label {case['expected']['original_answer_label']!r}.")
        for statement in case["expected"]["statements"]:
            if statement["label"] not in FACT_LABELS:
                raise EvaluationError(f"{case['case_id']}: unknown statement label {statement['label']!r}.")
    return data, cases


def check_category_counts(cases, expected=CATEGORIES):
    counts = Counter(c["category"] for c in cases)
    if dict(counts) != expected:
        raise EvaluationError(f"Category counts {dict(counts)} differ from the plan {expected}.")


def check_excerpts(case, snapshot):
    """Every expected-fact excerpt must be an exact quote of a section in the frozen snapshot."""
    texts = {}
    for chunk in snapshot:
        texts[chunk["section_id"]] = texts.get(chunk["section_id"], "") + " " + chunk["text"]
    problems = []
    for statement in case["expected"]["statements"]:
        if statement["section_id"] not in texts:
            problems.append(f"{statement['section_id']} is not in the snapshot")
        elif statement["excerpt"] not in texts[statement["section_id"]]:
            problems.append(f"excerpt not found in {statement['section_id']}: {statement['excerpt']!r}")
    return problems


# ---------------------------------------------------------------------------
# Evaluation-only adapters
# ---------------------------------------------------------------------------

class CachingLLM:
    """Wraps an llm (src/llm.py interface). Each distinct prompt is sent once per namespace; stored per run.

    The namespace is "<case_id>:<stage>" ("freeze" for the initial extraction, "B" for the
    correction pass): responses are never shared between cases, and the revised answer's
    extraction is always a fresh request even when the revised text equals the original. Errors are not cached and propagate to the
    production code, which records them.
    """

    def __init__(self, llm, path, namespace="", pacer=None):
        self.llm, self.path, self.namespace = llm, Path(path), namespace
        self.pacer = pacer if pacer is not None else {"interval_s": 0.0, "last": None}
        self.cache = {row["key"]: row for row in read_jsonl(self.path)}
        self.calls = self.hits = 0

    @property
    def model(self):
        return self.llm.model

    def key(self, system_instruction, user_prompt, schema, namespace=None):
        namespace = self.namespace if namespace is None else namespace
        return sha256_text(json.dumps([namespace, system_instruction, user_prompt, schema], sort_keys=True))

    def generate(self, system_instruction, user_prompt, response_json_schema):
        key = self.key(system_instruction, user_prompt, response_json_schema)
        # Rows written before keys included the case ID used the stage alone ("freeze" / "B"). The
        # prompt embeds the case's own question, answer and evidence, so such a row belongs to this case.
        legacy = self.key(system_instruction, user_prompt, response_json_schema, self.namespace.split(":")[-1])
        for candidate in (key, legacy):
            if candidate in self.cache:
                self.hits += 1
                return LLMResponse(**self.cache[candidate]["response"])
        if self.pacer["interval_s"] and self.pacer["last"] is not None:
            wait = self.pacer["interval_s"] - (time.monotonic() - self.pacer["last"])
            if wait > 0:
                time.sleep(wait)
        self.calls += 1
        self.pacer["last"] = time.monotonic()
        response = self.llm.generate(system_instruction, user_prompt, response_json_schema)
        row = {"key": key, "namespace": self.namespace, "created_at": now(), "system_sha256": sha256_text(system_instruction),
               "response": {"text": response.text, "model": response.model, "latency_s": response.latency_s,
                            "attempts": response.attempts, "finish_reason": response.finish_reason,
                            "usage": dict(response.usage)}}
        append_jsonl(self.path, row)
        self.cache[key] = row
        return response


class ReplayExtractor:
    """Pass 1 replays the frozen initial claims (still validated by parse_claims in the pipeline).

    Any later extraction (the revised answer) goes to the live extractor. The frozen claims
    are only ever returned for the frozen original answer.
    """

    def __init__(self, frozen_claims, original_answer, live=None):
        self.frozen, self.original, self.live, self.calls = frozen_claims, original_answer, live, 0
        self.name = f"frozen initial claims + {live.name}" if live else "frozen initial claims"

    def extract(self, question, answer):
        self.calls += 1
        if self.calls == 1:
            if answer != self.original:
                raise EvaluationError("Pass 1 must extract the frozen original answer.")
            return StageOutput(payload={"claims": self.frozen}, model="frozen", latency_s=0.0)
        if self.live is None:
            raise EvaluationError("A second extraction needs a live extractor.")
        return self.live.extract(question, answer)


class OfflineGemini:
    """Dry-run stand-in for Gemini: sentence-split extraction, and a 'revision' that returns the original answer.

    It makes no network call. Results from it describe the harness, not Gemini.
    """

    model = "offline-dry-run"

    def generate(self, system_instruction, user_prompt, response_json_schema):
        from src.claims import SentenceClaimExtractor
        from src.prompts import CLAIM_EXTRACTION_SYSTEM_INSTRUCTION

        if system_instruction == CLAIM_EXTRACTION_SYSTEM_INSTRUCTION:
            answer = user_prompt.split("<answer>\n", 1)[1].rsplit("\n</answer>", 1)[0]
            text = json.dumps(SentenceClaimExtractor().extract("", answer).payload)
        else:
            answer = user_prompt.split("<original_answer>\n", 1)[1].split("\n</original_answer>", 1)[0]
            section = user_prompt.split('section_id="', 1)[1].split('"', 1)[0]
            text = json.dumps({"answer": answer, "citations": [section], "abstained": False})
        return LLMResponse(text=text, model=self.model, latency_s=0.0, attempts=1, finish_reason="STOP", usage={})


# ---------------------------------------------------------------------------
# Freeze and run
# ---------------------------------------------------------------------------

def freeze_case(case, index, extractor_llm):
    """Frozen inputs for one case: snapshot + fingerprint, citation check, initial claims (one extraction)."""
    from src.retrieve import search

    options = retrieval_options(RAG_METHOD)
    chunks = search(case["question"], index, RAG_METHOD, top_k=DEFAULT_EVIDENCE_K, **options)
    sources = source_metadata(chunks)
    snapshot = [dict(chunk) for chunk in evidence_snapshot(sources)]
    problems = check_excerpts(case, snapshot)
    if problems:
        raise EvaluationError(f"{case['case_id']}: {problems}")
    answer = GroundedAnswer(answer=case["original_answer"], citations=case["citations"], abstained=False)
    report = validate_citations(answer, sources)
    if report.status == FAIL:
        raise EvaluationError(f"{case['case_id']}: original citations fail validation: {report.issues}")
    frozen = {**case, "sources": sources, "evidence_section_ids": [s["section_id"] for s in sources],
              "evidence_fingerprint": evidence_fingerprint(snapshot), "citation_validation": report.to_dict(),
              "retrieval": {"method": RAG_METHOD, "evidence_k": DEFAULT_EVIDENCE_K, **options},
              "frozen_at": now()}
    extractor = GeminiClaimExtractor(extractor_llm)
    start = time.perf_counter()
    try:
        output = extractor.extract(case["question"], case["original_answer"])
        claims = parse_claims(output.payload, case["original_answer"], case["question"])
        frozen["initial_extraction"] = {"status": "ok", "claims": [c.to_dict() for c in claims],
                                        "model": output.model, "usage": dict(output.usage),
                                        "latency_s": time.perf_counter() - start, "llm_call": output.llm_call}
    except (LLMError, InvalidClaimsOutput) as error:
        frozen["initial_extraction"] = {"status": "failed", "claims": [], "latency_s": time.perf_counter() - start,
                                        "error": {"kind": getattr(error, "kind", type(error).__name__),
                                                  "message": str(error), "api_error": isinstance(error, LLMError)}}
    return frozen


def run_condition(frozen, condition, verifier, llm):
    """One condition on frozen inputs through the production verify_and_correct()."""
    extraction = frozen["initial_extraction"]
    row = {"case_id": frozen["case_id"], "condition": condition, "started_at": now()}
    if extraction["status"] != "ok":
        row.update(outcome="extraction_failure", status="verification_failed", returned=False, final_answer=None,
                   final_citations=[], error=extraction["error"], verification=None, wall_s=0.0)
        return row
    revise = condition == "B"
    live = GeminiClaimExtractor(llm) if revise else None
    extractor = ReplayExtractor(extraction["claims"], frozen["original_answer"], live)
    answer = GroundedAnswer(answer=frozen["original_answer"], citations=frozen["citations"], abstained=False)
    start = time.perf_counter()
    verification = verify_and_correct(frozen["question"], frozen["sources"], answer, extractor, verifier,
                                      llm if revise else None, revise=revise)
    wall = time.perf_counter() - start
    if verification["evidence_fingerprint"] != frozen["evidence_fingerprint"]:
        raise EvaluationError(f"{frozen['case_id']}: evidence fingerprint changed.")
    row.update(outcome=verification["outcome"], status=final_status(verification["outcome"]),
               returned=verification["outcome"] in RETURNED, final_answer=verification["final_answer"],
               final_citations=verification["final_citations"], error=verification["error"],
               verification=verification, wall_s=wall, finished_at=now())
    return row


def plan(cases, run_dir=None):
    done = {k: {r["case_id"] for r in read_jsonl(Path(run_dir) / f)} if run_dir else set() for k, f in FILES.items()}
    cache = len(read_jsonl(Path(run_dir) / "gemini_cache.jsonl")) if run_dir else 0
    to_freeze = [c["case_id"] for c in cases if c["case_id"] not in done["cases"]]
    to_correct = [c["case_id"] for c in cases if c["case_id"] not in done["B"]]
    return {"cases": len(cases), "categories": dict(Counter(c["category"] for c in cases)),
            "max_gemini_requests": {"initial_extraction": len(to_freeze), "correction": len(to_correct),
                                    "revised_extraction": len(to_correct),
                                    "total": len(to_freeze) + 2 * len(to_correct),
                                    "note": "upper bound; corrections run only for answers that fail V3. Each "
                                            "request may be retried at most 2 times by GeminiClient on transient "
                                            "errors."},
            "already_done": {k: sorted(v) for k, v in done.items()}, "cached_gemini_responses": cache,
            "condition_A_gemini_requests": 0}


def _api_error(error):
    return bool((error or {}).get("api_error"))


def retry_api_failures(run_dir, progress=print):
    """Re-queue cases whose recorded result is an API/infrastructure failure (never a content failure).

    The superseded rows are moved, with a timestamp, to superseded_api_failures.jsonl.
    Cached successful Gemini responses stay and are replayed, not requested again.
    """
    run_dir = Path(run_dir)
    frozen = read_jsonl(run_dir / FILES["cases"])
    failed_freeze = {r["case_id"] for r in frozen if r["initial_extraction"]["status"] == "failed"
                     and _api_error(r["initial_extraction"].get("error"))}
    rows = {k: read_jsonl(run_dir / FILES[k]) for k in ("A", "B")}
    drop = {"cases": failed_freeze, "A": set(failed_freeze),
            "B": failed_freeze | {r["case_id"] for r in rows["B"] if _api_error(r.get("error"))}}
    for name, ids in drop.items():
        kept, moved = [], []
        for row in (frozen if name == "cases" else rows[name]):
            (moved if row["case_id"] in ids else kept).append(row)
        for row in moved:
            append_jsonl(run_dir / "superseded_api_failures.jsonl", {"file": FILES[name], "superseded_at": now(),
                                                                     "row": row})
        tmp = run_dir / (FILES[name] + ".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in kept), encoding="utf-8")
        tmp.replace(run_dir / FILES[name])
    progress(f"Re-queued after API failures: freeze {sorted(drop['cases'])}, B {sorted(drop['B'])}")
    return drop


def run(cases, run_dir, llm, verifier, index_factory, dry_run, progress=print, min_interval_s=0.0,
        record_api_failures=False):
    """Freeze and run every pending case. By default the run stops at the first API failure, logs it to
    api_failures.jsonl and leaves the case pending (resume later); record_api_failures=True records such
    failures as final outcomes instead."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    pacer = {"interval_s": min_interval_s, "last": None}
    clients = []

    def client(case_id, stage):
        clients.append(CachingLLM(llm, run_dir / "gemini_cache.jsonl", f"{case_id}:{stage}", pacer))
        return clients[-1]

    def summary(stopped=None):
        return {"gemini_requests_made": sum(c.calls for c in clients),
                "cache_hits": sum(c.hits for c in clients), "dry_run": dry_run, "stopped_on_api_failure": stopped}

    def stop(case_id, stage, error):
        append_jsonl(run_dir / "api_failures.jsonl", {"case_id": case_id, "stage": stage, "error": error, "at": now()})
        progress(f"  {case_id} {stage}: API failure ({error.get('kind')}); stopping. The case stays pending; "
                 "resume later with the same --run-id.")
        return summary({"case_id": case_id, "stage": stage, "kind": error.get("kind")})

    frozen_by_id = {r["case_id"]: r for r in read_jsonl(run_dir / FILES["cases"])}
    done = {k: {r["case_id"] for r in read_jsonl(run_dir / FILES[k])} for k in ("A", "B")}
    index = None
    for case in cases:
        case_id = case["case_id"]
        if case_id not in frozen_by_id:
            index = index or index_factory()
            frozen = freeze_case(case, index, client(case_id, "freeze"))
            if _api_error(frozen["initial_extraction"].get("error")) and not record_api_failures:
                return stop(case_id, "freeze", frozen["initial_extraction"]["error"])
            append_jsonl(run_dir / FILES["cases"], frozen)
            frozen_by_id[case_id] = frozen
            progress(f"  {case_id}: frozen ({len(frozen['initial_extraction']['claims'])} claims, "
                     f"extraction {frozen['initial_extraction']['status']})")
        frozen = frozen_by_id[case_id]
        if frozen["category"] != case["category"] or frozen["original_answer"] != case["original_answer"]:
            raise EvaluationError(f"{case_id}: case definition changed after freezing; start a new run.")
        for condition in ("A", "B"):
            if case_id in done[condition]:
                continue
            row = run_condition(frozen, condition, verifier, client(case_id, "B") if condition == "B" else None)
            if _api_error(row.get("error")) and not record_api_failures:
                return stop(case_id, condition, row["error"])
            append_jsonl(run_dir / FILES[condition], row)
            progress(f"  {case_id} {condition}: {row['outcome']}")
    return summary()


# ---------------------------------------------------------------------------
# Metrics and report
# ---------------------------------------------------------------------------

def _rate(numerator, denominator):
    return {"count": numerator, "of": denominator, "rate": numerator / denominator if denominator else None}


def _sum_usage(usages):
    total = {}
    for usage in usages:
        for key in ("input_tokens", "output_tokens", "thinking_tokens", "total_tokens"):
            if usage.get(key) is not None:
                total[key] = total.get(key, 0) + usage[key]
    return total


def _latencies(row):
    v = row["verification"] or {}
    passes = v.get("passes", [])
    out = {}
    for record in passes:
        stage = record["stage"]
        out[f"{stage}_extraction_s"] = record["extraction"].get("latency_s")
        out[f"{stage}_verification_s"] = record["verification"].get("latency_s")
    if v.get("revision"):
        out["correction_s"] = v["revision"].get("latency_s")
    out["pipeline_s"] = v.get("latency_s")
    out["wall_s"] = row.get("wall_s")
    return out


def _stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"n": len(values), "mean_s": statistics.fmean(values), "median_s": statistics.median(values),
            "max_s": max(values)}


def candidate_label(case_id, condition, row, frozen, annotations):
    """Factual label of the answer that condition would show (returned or withheld)."""
    ann = annotations.get(case_id, {})
    revision = (row["verification"] or {}).get("revision") or {}
    if condition == "B" and revision.get("answer") is not None:
        return (ann.get("revised") or {}).get("label"), "revised"
    return (ann.get("original") or {}).get("label") or frozen["expected"]["original_answer_label"], "original"


def compute_metrics(frozen_cases, rows_a, rows_b, annotations):
    a = {r["case_id"]: r for r in rows_a}
    b = {r["case_id"]: r for r in rows_b}
    ids = [c["case_id"] for c in frozen_cases if c["case_id"] in a and c["case_id"] in b]
    by_id = {c["case_id"]: c for c in frozen_cases}
    n = len(ids)
    failed_initial = [i for i in ids if a[i]["outcome"] != "accepted_original"]
    attempts = [i for i in ids if (b[i]["verification"] or {}).get("revision_attempted")]
    api_failures = [i for i in ids if (b[i].get("error") or {}).get("api_error")]
    verification = {
        "initial_v3_acceptance": _rate(sum(a[i]["outcome"] == "accepted_original" for i in ids), n),
        "initial_withheld": _rate(sum(not a[i]["returned"] for i in ids), n),
        "correction_attempts": _rate(len(attempts), n),
        "correction_attempts_of_failed_initial": _rate(len(attempts), len(failed_initial)),
        "post_correction_v3_acceptance": _rate(sum(b[i]["returned"] for i in ids), n),
        "post_correction_withheld": _rate(sum(not b[i]["returned"] for i in ids), n),
        "revisions_accepted_by_v3": _rate(sum(b[i]["outcome"] == "accepted_revision" for i in ids), len(attempts)),
        "correction_api_failures": _rate(sum(b[i]["outcome"] == "revision_api_failure" for i in ids), len(attempts)),
        "any_api_failure_cases": api_failures,
        "outcomes_A": dict(Counter(a[i]["outcome"] for i in ids)),
        "outcomes_B": dict(Counter(b[i]["outcome"] for i in ids)),
        "initial_verdicts_identical_A_vs_B": all(
            [r["verdict"] for r in a[i]["verification"]["passes"][0]["verification"]["results"]]
            == [r["verdict"] for r in b[i]["verification"]["passes"][0]["verification"]["results"]]
            for i in ids if a[i]["verification"] and b[i]["verification"]),
    }

    def factual(condition, rows):
        counts = Counter()
        unlabeled = []
        for i in ids:
            label, which = candidate_label(i, condition, rows[i], by_id[i], annotations)
            if label is None:
                unlabeled.append(i)
                continue
            correct = label == "factually_supported"
            counts[("correct" if correct else "not_correct") + ("_returned" if rows[i]["returned"] else "_withheld")] += 1
        return {"correct_returned": counts["correct_returned"], "incorrect_or_unsupported_returned":
                counts["not_correct_returned"], "correct_withheld": counts["correct_withheld"],
                "incorrect_or_unsupported_withheld": counts["not_correct_withheld"], "unlabeled": unlabeled, "of": n}

    effects = Counter()
    quality = Counter()
    rank = {"factually_supported": 3, "partially_supported": 2, "insufficient_evidence": 1, "factually_incorrect": 0}
    for i in attempts:
        ann = annotations.get(i, {})
        for effect, value in (ann.get("effects") or {}).items():
            if value:
                effects[effect] += 1
        before = (ann.get("original") or {}).get("label") or by_id[i]["expected"]["original_answer_label"]
        after = (ann.get("revised") or {}).get("label")
        if after is None or after not in rank or before not in rank:
            quality["unlabeled_or_undetermined"] += 1
        else:
            quality["improved" if rank[after] > rank[before] else "worsened" if rank[after] < rank[before]
                    else "unchanged"] += 1
    reviewed = [i for i in ids if (annotations.get(i, {}).get("human_review") or {}).get("reviewer")]
    causes = Counter(cause for i in ids for cause in (annotations.get(i, {}).get("failure_causes") or []))

    def performance(rows, condition):
        lat = [_latencies(rows[i]) for i in ids]
        keys = sorted({k for item in lat for k in item})
        usage = [(rows[i]["verification"] or {}).get("usage", {}) for i in ids]
        calls = sum((rows[i]["verification"] or {}).get("llm_calls", 0) for i in ids)
        return {"latency": {k: _stats([item.get(k) for item in lat]) for k in keys},
                "gemini_requests_in_pipeline": calls, "tokens": _sum_usage(usage)}

    freeze_usage = _sum_usage([by_id[i]["initial_extraction"].get("usage", {}) for i in ids])
    freeze_latency = _stats([by_id[i]["initial_extraction"].get("latency_s") for i in ids])
    model = next((by_id[i]["initial_extraction"].get("model") for i in ids
                  if by_id[i]["initial_extraction"].get("model")), None)
    perf_b = performance(b, "B")
    tokens_all = _sum_usage([freeze_usage, perf_b["tokens"]])
    cost = estimate_cost_usd(model, tokens_all.get("input_tokens"), tokens_all.get("output_tokens"))
    return {
        "cases_evaluated": n, "categories": dict(Counter(by_id[i]["category"] for i in ids)),
        "seeded_error_cases": sorted(i for i in ids if by_id[i]["seeded_error"]),
        "verification_behavior": verification,
        "factual_reliability": {
            "label_status": (f"{len(reviewed)} of {n} cases human-reviewed; all other labels are PROVISIONAL "
                             "(assistant-authored from the policy text, not V3 output)"),
            "human_reviewed_cases": reviewed,
            "provisional": {"A_verification_only": factual("A", a), "B_with_correction": factual("B", b),
                            "correction_effects": {e: effects.get(e, 0) for e in EFFECTS},
                            "unsupported_claims_introduced": effects.get("introduced_incorrect_claim", 0),
                            "supported_claims_removed": effects.get("removed_useful_correct_information", 0)},
            "reviewed": "none" if not reviewed else "compute from human_review fields"},
        "correction_effectiveness": {
            "failed_initial_answers": len(failed_initial),
            "failed_initial_answers_revised_and_accepted": sum(b[i]["outcome"] == "accepted_revision"
                                                               for i in failed_initial),
            "revisions_accepted_by_v3": sum(b[i]["outcome"] == "accepted_revision" for i in ids),
            "revisions_factual_quality_provisional": dict(quality),
            "note": "V3 acceptance is not factual correctness; quality is judged from the provisional labels."},
        "performance": {"initial_generation": "not applicable: original answers were supplied by the evaluation",
                        "initial_extraction_freeze": {"latency": freeze_latency, "tokens": freeze_usage},
                        "A": performance(a, "A"), "B": perf_b,
                        "gemini_model": model, "tokens_total_freeze_plus_B": tokens_all,
                        "estimated_cost_usd": cost,
                        "cost_note": (f"pricing source {PRICING_SOURCE}" if cost is not None else
                                      f"no trusted price for model {model!r} in src/llm.py PRICING; cost not "
                                      "estimated")},
        "error_analysis_provisional": {c: causes.get(c, 0) for c in FAILURE_CAUSES},
    }


def manual_review_rows(frozen_cases, rows_a, rows_b, annotations):
    a = {r["case_id"]: r for r in rows_a}
    b = {r["case_id"]: r for r in rows_b}
    rows = []
    for case in frozen_cases:
        i = case["case_id"]
        if i not in a or i not in b:
            continue
        revision = (b[i]["verification"] or {}).get("revision") or {}

        def verdicts(row, stage):
            for record in (row["verification"] or {}).get("passes", []):
                if record["stage"] == stage:
                    return [{"claim": c["claim"], "v3_verdict": c["verdict"], "reason": t["reason"]}
                            for c, t in zip(record["verification"]["results"],
                                            (record["verification"].get("trace") or {}).get("claims", []))]
            return None

        ann = annotations.get(i, {})
        rows.append({
            "case_id": i, "category": case["category"], "answer_source": case["answer_source"],
            "seeded_error": case["seeded_error"], "question": case["question"],
            "evidence_section_ids": case["evidence_section_ids"],
            "expected_statements": case["expected"]["statements"],
            "original_answer": case["original_answer"], "initial_claims": case["initial_extraction"]["claims"],
            "v3_initial (not ground truth)": verdicts(a[i], "original"),
            "revised_answer": revision.get("answer"), "revised_citations": revision.get("citations"),
            "v3_revised (not ground truth)": verdicts(b[i], "revision"),
            "outcome_A": a[i]["outcome"], "outcome_B": b[i]["outcome"],
            "provisional_assessment": {k: ann.get(k) for k in ("original", "revised", "effects", "failure_causes",
                                                                "notes", "annotator")},
            "human_review": ann.get("human_review") or {"reviewer": None, "original_label": None,
                                                         "revised_label": None, "notes": ""},
        })
    return rows


def _pct(rate):
    return "n/a" if rate["rate"] is None else f"{rate['count']}/{rate['of']} ({rate['rate']:.0%})"


def render_report(manifest, frozen_cases, rows_a, rows_b, metrics, annotations):
    v, f = metrics["verification_behavior"], metrics["factual_reliability"]["provisional"]
    a = {r["case_id"]: r for r in rows_a}
    b = {r["case_id"]: r for r in rows_b}
    lines = [f"# M6.4C controlled self-correction evaluation — {manifest['run_id']}", "",
             f"{'**DRY RUN** (offline fake Gemini; not a result). ' if manifest['dry_run'] else ''}"
             f"{metrics['cases_evaluated']} cases; model {metrics['performance']['gemini_model']}; "
             f"V3 `{manifest['verifier'].get('identifier')}` ({manifest['verifier'].get('device')}).", "",
             "**Labels are provisional** (assistant-authored from policy text; "
             f"{metrics['factual_reliability']['label_status']}). V3 verdicts are not used as ground truth.", "",
             "## Verification behaviour", "", "| | Verification only (A) | With correction (B) |", "|---|---|---|",
             f"| Returned (V3 accepted) | {_pct(v['initial_v3_acceptance'])} | {_pct(v['post_correction_v3_acceptance'])} |",
             f"| Withheld | {_pct(v['initial_withheld'])} | {_pct(v['post_correction_withheld'])} |", "",
             f"Correction attempts: {_pct(v['correction_attempts'])}; revisions accepted by V3: "
             f"{_pct(v['revisions_accepted_by_v3'])}; correction API failures: {_pct(v['correction_api_failures'])}. "
             f"Initial verdicts identical in A and B: {v['initial_verdicts_identical_A_vs_B']}.", "",
             "## Factual reliability (provisional labels)", "", "| | A | B |", "|---|---|---|"]
    for key in ("correct_returned", "incorrect_or_unsupported_returned", "correct_withheld",
                "incorrect_or_unsupported_withheld"):
        lines.append(f"| {key.replace('_', ' ')} | {f['A_verification_only'][key]} | {f['B_with_correction'][key]} |")
    lines += ["", "Correction effects (provisional): " + ", ".join(f"{k.replace('_', ' ')} {n}"
                                                                    for k, n in f["correction_effects"].items()), "",
              "## Cases", "", "| Case | Category | Seeded | Original (prov.) | A | B | Revised (prov.) |",
              "|---|---|---|---|---|---|---|"]
    for case in frozen_cases:
        i = case["case_id"]
        if i in a and i in b:
            ann = annotations.get(i, {})
            lines.append(f"| {i} | {case['category']} | {'yes' if case['seeded_error'] else 'no'} | "
                         f"{(ann.get('original') or {}).get('label') or case['expected']['original_answer_label']} | "
                         f"{a[i]['outcome']} | {b[i]['outcome']} | {(ann.get('revised') or {}).get('label') or '-'} |")
    lines += ["", "## Per-case detail", ""]
    for case in frozen_cases:
        i = case["case_id"]
        if i not in a or i not in b:
            continue
        revision = (b[i]["verification"] or {}).get("revision") or {}
        lines += [f"### {i} ({case['category']}; {case['answer_source']})", "",
                  f"- Question: {case['question']}", f"- Original answer: {case['original_answer']}",
                  f"- Expected: {case['expected']['interpretation']}",
                  f"- A: {a[i]['outcome']}; B: {b[i]['outcome']}"]
        for record in (b[i]["verification"] or {}).get("passes", []):
            for c, t in zip(record["verification"]["results"],
                            (record["verification"].get("trace") or {}).get("claims", [])):
                labels = ", ".join(f"{p['section_id']}={p['nli_label']}" for p in t["premises"])
                lines.append(f"  - {record['stage']} {c['claim_id']} {c['verdict']} ({t['reason']}): "
                             f"{c['claim']} [{labels}]")
        if revision.get("answer") is not None:
            lines.append(f"- Revised answer: {revision['answer']} (citations {revision.get('citations')})")
        notes = (annotations.get(i) or {}).get("notes")
        if notes:
            lines.append(f"- Provisional assessment: {notes}")
        lines.append("")
    perf = metrics["performance"]
    lines += ["## Performance", "", f"- Freeze extraction (one per case): {perf['initial_extraction_freeze']}",
              f"- A: {json.dumps(perf['A'])}", f"- B: {json.dumps(perf['B'])}",
              f"- Cost: {perf['estimated_cost_usd'] if perf['estimated_cost_usd'] is not None else perf['cost_note']}",
              "", "## Error analysis (provisional)", "", json.dumps(metrics["error_analysis_provisional"]), ""]
    return "\n".join(lines)


def report(run_dir, annotations_path=None, cases_path=CASES_PATH):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    frozen = read_jsonl(run_dir / FILES["cases"])
    rows_a, rows_b = read_jsonl(run_dir / FILES["A"]), read_jsonl(run_dir / FILES["B"])
    annotations_path = Path(annotations_path) if annotations_path else run_dir / "provisional_annotations.json"
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))["cases"] if annotations_path.is_file() else {}
    metrics = compute_metrics(frozen, rows_a, rows_b, annotations)
    _, defined = load_cases(cases_path)
    completed = {r["case_id"] for r in rows_a} & {r["case_id"] for r in rows_b}
    metrics["completion"] = {"defined": len(defined), "completed": len(completed),
                             "pending": [c["case_id"] for c in defined if c["case_id"] not in completed],
                             "pending_categories": dict(Counter(c["category"] for c in defined
                                                                if c["case_id"] not in completed))}
    api_log = read_jsonl(run_dir / "api_failures.jsonl")
    superseded = read_jsonl(run_dir / "superseded_api_failures.jsonl")
    metrics["api_failure_log"] = {
        "stops": len(api_log), "kinds": dict(Counter((r["error"] or {}).get("kind") for r in api_log)),
        "superseded_rows": len(superseded),
        "note": "API/quota failures are infrastructure failures, kept apart from correction outcomes; the "
                "affected cases were left pending or re-queued, never re-asked after a successful response."}
    write_json(run_dir / "metrics.json", metrics)
    with open(run_dir / "manual_review.jsonl", "w", encoding="utf-8") as handle:
        for row in manual_review_rows(frozen, rows_a, rows_b, annotations):
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    text = render_report(manifest, frozen, rows_a, rows_b, metrics, annotations)
    done = metrics["completion"]
    if done["pending"]:
        text = text.replace("\n\n", f"\n\n**PARTIAL RUN: {done['completed']} of {done['defined']} cases completed; "
                            f"pending {', '.join(done['pending'])} (stopped by the Gemini daily request quota). "
                            "All rates below use the completed cases only.**\n\n", 1)
    (run_dir / "report.md").write_text(text,
                                       encoding="utf-8")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def default_index():
    from src.retrieve import build_search_index

    return build_search_index(methods=(RAG_METHOD,))


def main(argv=None, llm=None, verifier=None, index_factory=None, out_root=OUT_ROOT, cases_path=CASES_PATH,
         progress=print):
    parser = argparse.ArgumentParser(description="M6.4C controlled self-correction evaluation (evaluation only).")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--report", metavar="RUN_ID")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--confirm-live", action="store_true")
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--min-interval-s", type=float, default=0.0,
                        help="minimum seconds between real Gemini requests (per-minute quotas)")
    parser.add_argument("--retry-api-failures", action="store_true",
                        help="re-queue cases recorded as API/infrastructure failures (content failures are kept)")
    parser.add_argument("--record-api-failures", action="store_true",
                        help="record API failures as final outcomes instead of stopping")
    args = parser.parse_args(argv)

    data, cases = load_cases(cases_path)
    check_category_counts(cases)
    if args.report:
        metrics = report(Path(out_root) / args.report, args.annotations, cases_path)
        progress(json.dumps(metrics["verification_behavior"], indent=1))
        return 0

    run_id = args.run_id or (f"dryrun-{datetime.now():%Y%m%d-%H%M%S}" if args.dry_run else None)
    run_dir = Path(out_root) / run_id if run_id else None
    budget = plan(cases, run_dir if run_dir and run_dir.is_dir() else None)
    progress(json.dumps(budget, indent=1))
    if args.plan:
        return 0
    if args.live and not run_id:
        parser.error("--live needs --run-id (a new name, or an existing run to resume).")
    if args.dry_run and run_dir.exists():
        parser.error(f"{run_dir} exists; dry runs never reuse a directory.")
    if args.live:
        manifest_path = run_dir / "manifest.json"
        if manifest_path.is_file() and json.loads(manifest_path.read_text())["dry_run"]:
            parser.error(f"{run_dir} is a dry run; use a new --run-id for the live run.")
        if not args.confirm_live:
            progress("\nNo Gemini request was made. Re-run with --confirm-live to execute this plan.")
            return 0
        if llm is None:
            from src.llm import GeminiClient, load_env_file

            load_env_file(PROJECT_ROOT / ".env")
            if not os.environ.get(API_KEY_ENV):
                progress(f"{API_KEY_ENV} is not set; stopping before any live call. Everything else is ready.")
                return 1
            llm = GeminiClient()
    else:
        llm = llm or OfflineGemini()

    from src import local_verifier as lv

    verifier = verifier or lv.LocalNLIVerifier(lv.DEFAULT_V3_RUN, args.device)
    loaded = verifier.load() if hasattr(verifier, "load") else None
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    cases_sha = sha256_text(Path(cases_path).read_text(encoding="utf-8"))
    if previous and previous["cases_sha256"] != cases_sha:
        raise EvaluationError("Case definitions changed since this run started; use a new --run-id.")
    manifest = {
        "run_id": run_id, "purpose": "M6.4C controlled self-correction evaluation (exploratory; not a held-out test)",
        "dry_run": bool(args.dry_run), "created_at": previous.get("created_at", now()), "updated_at": now(),
        "git": git_state(), "cases_path": str(Path(cases_path).resolve().relative_to(PROJECT_ROOT))
        if Path(cases_path).resolve().is_relative_to(PROJECT_ROOT) else str(cases_path),
        "cases_sha256": cases_sha, "label_status": data["label_status"],
        "harness_sha256": sha256_text(Path(__file__).read_text(encoding="utf-8")),
        "harness_versions": sorted(set(previous.get("harness_versions", [previous["harness_sha256"]]
                                                    if previous.get("harness_sha256") else []))
                                   | {sha256_text(Path(__file__).read_text(encoding="utf-8"))}),
        "gemini_model": getattr(llm, "model", None), "verifier": loaded.info if loaded else {"name": verifier.name},
        "retrieval": {"method": RAG_METHOD, "evidence_k": DEFAULT_EVIDENCE_K, **retrieval_options(RAG_METHOD)},
        "conditions": {"A": "verify_and_correct(revise=False) on frozen inputs",
                       "B": "verify_and_correct(revise=True) on the same frozen inputs (one targeted correction)"},
        "plan_at_start": previous.get("plan_at_start", budget), "test_split": "not accessed",
    }
    write_json(manifest_path, manifest)
    if args.retry_api_failures:
        retry_api_failures(run_dir, progress)
    summary = run(cases, run_dir, llm, verifier, index_factory or default_index, bool(args.dry_run), progress,
                  args.min_interval_s, args.record_api_failures)
    manifest.update(last_execution=summary, updated_at=now())
    write_json(manifest_path, manifest)
    metrics = report(run_dir, args.annotations, cases_path)
    progress(f"\nRun directory: {run_dir}\n{json.dumps(summary)}\n"
             f"V3 returned: A {_pct(metrics['verification_behavior']['initial_v3_acceptance'])}, "
             f"B {_pct(metrics['verification_behavior']['post_correction_v3_acceptance'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
