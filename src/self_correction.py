"""Claim-level verification and one bounded, evidence-guided revision (M5).

    M4 answer (citations already validated)
       -> extract atomic claims -> verify each claim against the same evidence
            all SUPPORTED                -> accept the original answer
            otherwise                    -> ONE revision request (no new retrieval)
                -> validate structure and citation IDs
                -> extract claims again -> verify again (fresh verdicts)
                     all SUPPORTED       -> accept the revision
                     otherwise           -> abstain (the rejected text is not returned)

There is no loop: the revision step is written out once, so at most one
revision is ever requested. The GeminiClient's own bounded retries (transient
errors, at most 2) are the only retries.

Every way this can end is a distinct outcome (`verification.outcome`):

    accepted_original         every claim of the M4 answer SUPPORTED
    accepted_revision         the revision's claims all SUPPORTED on a fresh check
    abstained                 the M4 answer or the revision abstained (safe abstention)
    revision_unsupported      the revision still had unsupported claims -> abstention
    extraction_failure        claims could not be extracted (API error, invalid or empty output)
    verification_api_failure  the verifier could not run (API error, or VerifierUnavailable)
    invalid_verifier_output   the verifier's output failed validation (e.g. invented evidence IDs)
    citation_failure          citation-ID validation failed (M4 answer, or the revision)
    revision_api_failure      the revision request failed
    revision_invalid_output   the revision was not a valid {answer, citations, abstained} object
    generation_failure        M4 did not produce an answer (retrieval, LLM or output error)

`error.stage` says whether a failure was in the "original" or "revision"
pass. Failures are fail-closed: the answer is withheld (status
"verification_failed"), never returned as verified. "SUPPORTED" is only ever
recorded from a completed, validated verification.

The evidence is the `sources` list of the M4 result, the exact chunks the
answer was generated from, frozen into read-only mappings. Its fingerprint is
recorded and re-checked at the end, so no stage can retrieve or alter
evidence unnoticed.
"""

import hashlib
import json
import time
from types import MappingProxyType

from src.citations import FAIL, GroundedAnswer, InvalidStructuredOutput, parse_answer, validate_citations
from src.claims import GeminiClaimExtractor, InvalidClaimsOutput, call_llm_stage, parse_claims
from src.llm import LLMError, estimate_cost_usd
from src.prompts import build_revision_prompt
from src.rag import DEFAULT_EVIDENCE_K, RAG_METHOD, answer_question
from src.verification import (
    SUPPORTED,
    GeminiVerifier,
    InvalidVerifierOutput,
    VerifierUnavailable,
    parse_verdicts,
    verdict_counts,
)

ACCEPTED_ORIGINAL = "accepted_original"
ACCEPTED_REVISION = "accepted_revision"
ABSTAINED = "abstained"
REVISION_UNSUPPORTED = "revision_unsupported"
EXTRACTION_FAILURE = "extraction_failure"
VERIFICATION_API_FAILURE = "verification_api_failure"
INVALID_VERIFIER_OUTPUT = "invalid_verifier_output"
CITATION_FAILURE = "citation_failure"
REVISION_API_FAILURE = "revision_api_failure"
REVISION_INVALID_OUTPUT = "revision_invalid_output"
GENERATION_FAILURE = "generation_failure"
OUTCOMES = (ACCEPTED_ORIGINAL, ACCEPTED_REVISION, ABSTAINED, REVISION_UNSUPPORTED, EXTRACTION_FAILURE,
            VERIFICATION_API_FAILURE, INVALID_VERIFIER_OUTPUT, CITATION_FAILURE, REVISION_API_FAILURE,
            REVISION_INVALID_OUTPUT, GENERATION_FAILURE)

VERIFIED_STATUSES = ("verified_answer", "abstained", "verification_failed", "citation_failure", "invalid_output",
                     "llm_error", "retrieval_error")

ABSTENTION_MESSAGE = ("The answer could not be verified against the retrieved policy evidence, so no answer is "
                      "given. Please consult the policy documents directly or rephrase the question.")

MAX_REVISIONS = 1
_RAW_LIMIT = 2000


def evidence_snapshot(sources):
    """Read-only copies of the evidence chunks; any attempt to modify them raises TypeError."""
    keys = ("chunk_id", "section_id", "title", "section_title", "text")
    return tuple(MappingProxyType({key: source[key] for key in keys if key in source}) for source in sources)


def evidence_fingerprint(evidence):
    payload = json.dumps([[c.get("chunk_id"), c["section_id"], c["text"]] for c in evidence], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _error(kind, message, step, stage, api_error=False):
    return {"kind": kind, "message": message, "step": step, "stage": stage, "api_error": api_error}


def _raw(payload):
    if isinstance(payload, str):
        return payload[:_RAW_LIMIT]
    try:
        return json.dumps(payload)[:_RAW_LIMIT]
    except (TypeError, ValueError):
        return repr(payload)[:_RAW_LIMIT]


class _Run:
    """Accumulates per-step records, latency, token usage and LLM call counts for one verification."""

    def __init__(self):
        self.steps = []          # (model, usage, llm_call)
        self.llm_calls = 0
        self.latency_s = 0.0

    def record(self, record, output, elapsed):
        record["latency_s"] = elapsed
        self.latency_s += elapsed
        if output is not None:
            record.update(model=output.model, attempts=output.attempts, usage=dict(output.usage))
            self.steps.append((output.model, dict(output.usage), output.llm_call))
            if output.llm_call:
                self.llm_calls += 1

    def failed_call(self, record, elapsed, llm_call):
        record["latency_s"] = elapsed
        self.latency_s += elapsed
        if llm_call:
            self.llm_calls += 1
            self.steps.append((None, {}, True))   # usage unknown for a failed request

    def usage(self):
        totals = {}
        for _, usage, _ in self.steps:
            for key in ("input_tokens", "output_tokens", "thinking_tokens", "total_tokens"):
                if usage.get(key) is not None:
                    totals[key] = totals.get(key, 0) + usage[key]
        return totals

    def cost(self):
        """Sum of per-request estimates; None if any LLM request has no trusted price or unknown usage."""
        total = 0.0
        for model, usage, llm_call in self.steps:
            if not llm_call:
                continue
            cost = estimate_cost_usd(model, usage.get("input_tokens"), usage.get("output_tokens")) if model else None
            if cost is None:
                return None
            total += cost
        return total


def _verify_pass(stage, question, text, citations, evidence, supplied, extractor, verifier, run):
    """Extract and verify one answer. Returns (pass_record, failure_outcome or None, error or None)."""
    record = {"stage": stage, "answer": text, "citations": list(citations), "all_supported": None,
              "extraction": {"component": extractor.name, "status": None, "claims": [], "error": None},
              "verification": {"component": verifier.name, "status": "not_run", "results": [],
                               "verdict_counts": None, "error": None}}
    extraction, verification = record["extraction"], record["verification"]

    start = time.perf_counter()
    try:
        output = extractor.extract(question, text)
    except LLMError as error:
        run.failed_call(extraction, time.perf_counter() - start, llm_call=True)
        extraction.update(status="failed", error=_error(error.kind, str(error), "extraction", stage, api_error=True))
        return record, EXTRACTION_FAILURE, extraction["error"]
    run.record(extraction, output, time.perf_counter() - start)
    try:
        claims = parse_claims(output.payload, text, question)
    except InvalidClaimsOutput as error:
        extraction.update(status="failed", raw_output=_raw(output.payload),
                          error=_error(error.kind, str(error), "extraction", stage))
        return record, EXTRACTION_FAILURE, extraction["error"]
    extraction.update(status="ok", claims=[claim.to_dict() for claim in claims])

    start = time.perf_counter()
    try:
        output = verifier.verify(claims, evidence)
    except (LLMError, VerifierUnavailable) as error:
        run.failed_call(verification, time.perf_counter() - start, llm_call=isinstance(error, LLMError))
        verification.update(status="failed", error=_error(error.kind, str(error), "verification", stage,
                                                          api_error=isinstance(error, LLMError)))
        return record, VERIFICATION_API_FAILURE, verification["error"]
    run.record(verification, output, time.perf_counter() - start)
    try:
        results = parse_verdicts(output.payload, claims, supplied, citations)
    except InvalidVerifierOutput as error:
        verification.update(status="failed", raw_output=_raw(output.payload),
                            error=_error(error.kind, str(error), "verification", stage))
        return record, INVALID_VERIFIER_OUTPUT, verification["error"]
    verification.update(status="complete", results=[r.to_dict() for r in results],
                        verdict_counts=verdict_counts(results))
    record["all_supported"] = all(r.verdict == SUPPORTED for r in results)
    record["_results"] = results   # kept for the revision prompt; removed before returning
    return record, None, None


def verify_and_correct(question, sources, answer, extractor, verifier, reviser):
    """Verify a citation-valid, non-abstained GroundedAnswer and revise it at most once.

    `sources` is the M4 result's evidence list; `reviser` has the src/llm.py
    interface. Returns the `verification` dict described in the module docstring.
    """
    evidence = evidence_snapshot(sources)
    fingerprint = evidence_fingerprint(evidence)
    supplied = list(dict.fromkeys(chunk["section_id"] for chunk in evidence))
    run = _Run()
    verification = {
        "outcome": None, "final_answer": None, "final_citations": [], "final_abstained": None,
        "extractor": extractor.name, "verifier": verifier.name,
        "evidence_section_ids": supplied, "evidence_fingerprint": fingerprint,
        "passes": [], "revision_attempted": False, "revision": None, "error": None,
    }

    def finish(outcome, error=None, answer_text=None, citations=(), abstained=None):
        if evidence_fingerprint(evidence) != fingerprint:   # cannot happen with read-only mappings; checked anyway
            raise RuntimeError("The evidence snapshot changed during verification.")
        for record in verification["passes"]:
            record.pop("_results", None)
        verification.update(outcome=outcome, error=error, final_answer=answer_text,
                            final_citations=list(citations), final_abstained=abstained,
                            llm_calls=run.llm_calls, latency_s=run.latency_s, usage=run.usage(),
                            estimated_cost_usd=run.cost())
        return verification

    # Pass 1: the original answer.
    record, failure, error = _verify_pass("original", question, answer.answer, answer.citations, evidence, supplied,
                                          extractor, verifier, run)
    verification["passes"].append(record)
    if failure:
        return finish(failure, error)
    if record["all_supported"]:
        return finish(ACCEPTED_ORIGINAL, answer_text=answer.answer, citations=answer.citations, abstained=False)

    # The single revision. Same evidence snapshot; no retrieval.
    verification["revision_attempted"] = True
    revision = {"component": f"llm:{getattr(reviser, 'model', None)}", "status": None, "answer": None,
                "citations": [], "abstained": None, "citation_validation": None, "error": None}
    verification["revision"] = revision
    start = time.perf_counter()
    try:
        output = call_llm_stage(reviser, *build_revision_prompt(question, evidence, answer.answer, record["_results"]))
    except LLMError as error:
        run.failed_call(revision, time.perf_counter() - start, llm_call=True)
        revision.update(status="api_failure",
                        error=_error(error.kind, str(error), "revision", "revision", api_error=True))
        return finish(REVISION_API_FAILURE, revision["error"])
    run.record(revision, output, time.perf_counter() - start)
    try:
        revised = parse_answer(output.payload)
    except InvalidStructuredOutput as error:
        revision.update(status="invalid_output", raw_response=_raw(output.payload),
                        error=_error("invalid_output", str(error), "revision", "revision"))
        return finish(REVISION_INVALID_OUTPUT, revision["error"])
    report = validate_citations(revised, evidence)
    revision.update(answer=revised.answer, citations=list(revised.citations), abstained=revised.abstained,
                    citation_validation=report.to_dict())
    if report.status == FAIL:
        revision.update(status="citation_failure",
                        error=_error("citation", " ".join(report.issues), "revision", "revision"))
        return finish(CITATION_FAILURE, revision["error"])
    if revised.abstained:
        revision["status"] = "abstained"
        return finish(ABSTAINED, answer_text=revised.answer if not revised.citations else ABSTENTION_MESSAGE,
                      abstained=True)
    revision["status"] = "ok"

    # Pass 2: the revision, with fresh claims and fresh verdicts.
    record, failure, error = _verify_pass("revision", question, revised.answer, revised.citations, evidence,
                                          supplied, extractor, verifier, run)
    verification["passes"].append(record)
    if failure:
        return finish(failure, error)
    if record["all_supported"]:
        return finish(ACCEPTED_REVISION, answer_text=revised.answer, citations=revised.citations, abstained=False)
    return finish(REVISION_UNSUPPORTED, answer_text=ABSTENTION_MESSAGE, abstained=True)


def final_status(outcome):
    if outcome in (ACCEPTED_ORIGINAL, ACCEPTED_REVISION):
        return "verified_answer"
    if outcome in (ABSTAINED, REVISION_UNSUPPORTED):
        return "abstained"
    return "verification_failed"


def answer_question_verified(question, index, llm, extractor=None, verifier=None, reviser=None, method=RAG_METHOD,
                             evidence_k=DEFAULT_EVIDENCE_K, dense_weight=None, bm25_weight=None):
    """M4's answer_question() followed by claim verification and at most one revision.

    By default extraction, verification and revision use the same `llm` as
    generation. Top-level answer/citations/abstained are the user-facing
    result after verification; the untouched M4 result is under "baseline".
    """
    baseline = answer_question(question, index, llm, method, evidence_k, dense_weight, bm25_weight)
    extractor = extractor or GeminiClaimExtractor(llm)
    verifier = verifier or GeminiVerifier(llm)
    reviser = reviser or llm
    start = time.perf_counter()

    status, error = None, None
    if baseline["status"] in ("retrieval_error", "llm_error", "invalid_output", "citation_failure"):
        outcome = CITATION_FAILURE if baseline["status"] == "citation_failure" else GENERATION_FAILURE
        status = baseline["status"]
        error = {**baseline["error"], "step": "generation", "stage": "original",
                 "api_error": baseline["status"] == "llm_error"}
        verification = {"outcome": outcome, "final_answer": None, "final_citations": [], "final_abstained": None,
                        "error": error, "passes": [], "revision_attempted": False, "revision": None,
                        "llm_calls": 0, "latency_s": 0.0, "usage": {}, "estimated_cost_usd": 0.0}
    elif baseline["status"] == "abstained":
        # An abstention makes no policy claims to verify. If it still cites sections it may hide a
        # partial answer, so its text is replaced rather than passed through unverified.
        text = baseline["answer"] if not baseline["citations"] else ABSTENTION_MESSAGE
        verification = {"outcome": ABSTAINED, "final_answer": text, "final_citations": [], "final_abstained": True,
                        "error": None, "passes": [], "revision_attempted": False, "revision": None,
                        "llm_calls": 0, "latency_s": 0.0, "usage": {}, "estimated_cost_usd": 0.0}
    else:
        answer = GroundedAnswer(answer=baseline["answer"], citations=baseline["citations"], abstained=False)
        verification = verify_and_correct(question, baseline["sources"], answer, extractor, verifier, reviser)
    verification.setdefault("extractor", extractor.name)
    verification.setdefault("verifier", verifier.name)
    status = status or final_status(verification["outcome"])

    baseline_cost, verification_cost = baseline["estimated_cost_usd"], verification["estimated_cost_usd"]
    return {
        "question": question,
        "mode": "verified",
        "status": status,
        "answer": verification["final_answer"],
        "citations": verification["final_citations"],
        "abstained": verification["final_abstained"],
        "error": verification["error"],
        "model": baseline["model"],
        "verification": verification,
        "latency_s": {"baseline": baseline["latency_s"]["total"], "verification": verification["latency_s"],
                      "total": baseline["latency_s"]["total"] + (time.perf_counter() - start)},
        "usage": {"baseline": baseline["usage"], "verification": verification["usage"]},
        "estimated_cost_usd": {
            "baseline": baseline_cost, "verification": verification_cost,
            "total": None if baseline_cost is None or verification_cost is None else baseline_cost + verification_cost},
        "baseline": baseline,
    }
