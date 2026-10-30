"""Grounded question answering: retrieve policy chunks, ask Gemini, validate the structured answer.

Usage:
    python -m src.rag --question "What does FIN-205.3 say?"
    python -m src.rag --question "..." --method dense --evidence-k 3
    python -m src.rag --question "..." --model gemini-3.6-flash --timeout 30
    python -m src.rag --question "..." --json          # full structured result as JSON
    python -m src.rag --question "..." --verify        # claim verification with the local V3 NLI model (M6.4A)
    python -m src.rag --question "..." --verify --revise          # M6.4B: one Gemini correction guided by V3 feedback
    python -m src.rag --question "..." --verify --verifier gemini            # M5 Gemini verifier, no revision
    python -m src.rag --question "..." --verify --verifier gemini --revise   # M5 path with one revision

Pipeline: question -> retrieval (default hybrid 0.6 / 0.4, top 5, no reranking)
-> grounded prompt -> Gemini structured output -> schema validation -> citation
check -> result with sources, latency and token usage.

Each failure has its own status so they are never confused:
    retrieval_error   retrieval raised or returned no evidence
    llm_error         the Gemini call failed (configuration, auth, rate limit, timeout, API)
    invalid_output    the response was not a valid {answer, citations, abstained} object
    citation_failure  valid output, but it cites unsupplied sections or answers without citations
    abstained         valid output in which the model said the evidence is insufficient
    answered          valid output whose citations all refer to supplied sections
"answered" is not "correct": citations are checked for existence, not support.

--verify (off by default) adds Gemini claim extraction and claim-evidence
verification. The default verifier is the local fine-tuned V3 NLI model
(src/local_verifier.py), loaded before any Gemini request; if it cannot load,
the command stops with a fail-closed result and makes no Gemini call. An
answer with any claim that is not SUPPORTED is withheld; nothing is revised
unless --revise is given. With --revise, a failed answer gets exactly one Gemini
correction (guided by the local verifier's per-claim feedback, or by the M5
verdicts with --verifier gemini|mock), whose claims are extracted and verified
afresh against the same evidence; it is returned only if every claim is
SUPPORTED. See src/self_correction.py for the statuses. Without --verify the M4
path is unchanged.
"""

import argparse
import json
import sys
import time

from src.citations import FAIL, InvalidStructuredOutput, parse_answer, supplied_section_ids, validate_citations
from src.fusion import validate_weights
from src.llm import (
    DEFAULT_TIMEOUT_S,
    PRICING_SOURCE,
    GeminiClient,
    LLMError,
    estimate_cost_usd,
    load_env_file,
    sampling_warnings,
)
from src.prompts import build_prompt
from src.retrieve import ALL_METHODS, search, validate_top_k

RAG_METHOD = "hybrid"
RAG_WEIGHTS = (0.6, 0.4)       # the M2.5 / M3 comparison configuration
DEFAULT_EVIDENCE_K = 5
WEIGHTED_METHODS = ("hybrid", "hybrid-rerank")

VERIFIERS = ("nli", "gemini", "mock")
STATUSES = ("answered", "abstained", "citation_failure", "invalid_output", "llm_error", "retrieval_error")


def retrieval_options(method, dense_weight=None, bm25_weight=None):
    """search() keyword arguments for a method; weights apply only to hybrid methods."""
    if method not in WEIGHTED_METHODS:
        if dense_weight is not None or bm25_weight is not None:
            raise ValueError(f"Fusion weights apply only to {WEIGHTED_METHODS}, not {method!r}.")
        return {}
    dense_weight = RAG_WEIGHTS[0] if dense_weight is None else dense_weight
    bm25_weight = RAG_WEIGHTS[1] if bm25_weight is None else bm25_weight
    validate_weights({"dense": dense_weight, "bm25": bm25_weight})
    return {"dense_weight": dense_weight, "bm25_weight": bm25_weight}


def source_metadata(chunks):
    """The evidence supplied to the model, plus its retrieval rank and score (which the model never sees)."""
    return [{
        "ref": f"E{number}",
        "chunk_id": chunk["chunk_id"],
        "document_id": chunk["document_id"],
        "title": chunk["title"],
        "section_id": chunk["section_id"],
        "section_title": chunk["section_title"],
        "retrieval_rank": chunk["rank"],
        "retrieval_score": chunk.get("score", chunk.get("similarity")),
        "retrieval_score_type": chunk.get("score_type", "cosine"),
        "text": chunk["text"],
    } for number, chunk in enumerate(chunks, start=1)]


def answer_question(question, index, llm, method=RAG_METHOD, evidence_k=DEFAULT_EVIDENCE_K,
                    dense_weight=None, bm25_weight=None):
    """Run the full pipeline for one question and return a JSON-serializable result dict.

    `llm` is anything with `.model` and `.generate(system_instruction, user_prompt, schema)`
    (see src/llm.py). Failures are reported in the result's status and error
    fields rather than raised, except for invalid arguments.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")
    evidence_k = validate_top_k(evidence_k)
    options = retrieval_options(method, dense_weight, bm25_weight)

    result = {
        "question": question,
        "status": None,
        "answer": None,
        "citations": [],
        "abstained": None,
        "citation_validation": None,
        "error": None,
        "model": getattr(llm, "model", None),
        "retrieval": {"method": method, "evidence_k": evidence_k, **options},
        "sources": [],
        "latency_s": {"retrieval": None, "generation": None, "total": None},
        "usage": {},
        "estimated_cost_usd": None,
        "attempts": None,
        "semantic_support": "not evaluated (M5)",
    }
    start = time.perf_counter()

    def finish(status, error=None):
        result["status"] = status
        if error is not None:
            result["error"] = error
        result["latency_s"]["total"] = time.perf_counter() - start
        return result

    # 1-3. Retrieve and select evidence.
    try:
        chunks = search(question, index, method, top_k=evidence_k, **options)
    except Exception as error:  # noqa: BLE001 - reported as a retrieval failure
        result["latency_s"]["retrieval"] = time.perf_counter() - start
        return finish("retrieval_error", {"kind": "retrieval", "message": f"{type(error).__name__}: {error}"})
    result["latency_s"]["retrieval"] = time.perf_counter() - start
    if not chunks:
        return finish("retrieval_error", {"kind": "retrieval", "message": "Retrieval returned no evidence chunks."})
    result["sources"] = source_metadata(chunks)

    # 4-5. Build the grounded prompt and call the model.
    system_instruction, user_prompt, schema = build_prompt(question, chunks)
    generation_start = time.perf_counter()
    try:
        response = llm.generate(system_instruction, user_prompt, schema)
    except LLMError as error:
        result["latency_s"]["generation"] = time.perf_counter() - generation_start
        return finish("llm_error", {"kind": error.kind, "message": str(error)})
    result["latency_s"]["generation"] = time.perf_counter() - generation_start
    result["usage"] = dict(response.usage)
    result["attempts"] = response.attempts
    result["model"] = response.model
    result["estimated_cost_usd"] = estimate_cost_usd(response.model, response.usage.get("input_tokens"),
                                                     response.usage.get("output_tokens"))

    # 6. Parse and validate the structured output.
    try:
        parsed = parse_answer(response.text)
    except InvalidStructuredOutput as error:
        result["raw_response"] = response.text[:2000]
        return finish("invalid_output", {"kind": "invalid_output", "message": str(error)})
    result["answer"], result["citations"], result["abstained"] = parsed.answer, list(parsed.citations), parsed.abstained

    # 7. Check citations against the supplied evidence.
    report = validate_citations(parsed, chunks)
    result["citation_validation"] = report.to_dict()
    if report.status == FAIL:
        return finish("citation_failure", {"kind": "citation", "message": " ".join(report.issues)})
    return finish("abstained" if parsed.abstained else "answered")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def format_result(result):
    lines = ["", f"Question: {result['question']}"]
    retrieval = result["retrieval"]
    how = retrieval["method"]
    if "dense_weight" in retrieval:
        how += f" {retrieval['dense_weight']:.1f}/{retrieval['bm25_weight']:.1f}"
    lines.append(f"Model: {result['model']}   Retrieval: {how}, top {retrieval['evidence_k']} chunks")
    lines.append(f"Status: {result['status']}")
    if result["error"]:
        lines.append(f"Error ({result['error']['kind']}): {result['error']['message']}")
    if result["answer"] is not None:
        lines += ["", "Answer:", f"  {result['answer']}", ""]
        lines.append(f"Citations: {', '.join(result['citations']) or '(none)'}")
        lines.append(f"Abstained: {'yes' if result['abstained'] else 'no'}")
    validation = result["citation_validation"]
    if validation:
        lines.append(f"Citation check: {validation['status']}"
                     + (f" - {' '.join(validation['issues'])}" if validation["issues"] else ""))
    if result["sources"]:
        cited = set(result["citations"])
        lines += ["", "Sources supplied to the model:"]
        for source in result["sources"]:
            mark = "  [cited]" if source["section_id"] in cited else ""
            lines.append(f"  {source['ref']}  {source['section_id']:<10} {source['title']} - "
                         f"{source['section_title']}{mark}")
    latency = result["latency_s"]
    parts = [f"{name} {latency[name]:.3f} s" for name in ("retrieval", "generation", "total") if latency[name] is not None]
    lines += ["", "Latency: " + ", ".join(parts) + " (model loading excluded)"]
    usage = result["usage"]
    if usage:
        token_text = (f"Tokens: input {usage.get('input_tokens')}, output {usage.get('output_tokens')} "
                      f"(of which thinking {usage.get('thinking_tokens')}), total {usage.get('total_tokens')}")
        if result["estimated_cost_usd"] is not None:
            token_text += f"; est. paid-tier cost ${result['estimated_cost_usd']:.6f}"
        lines.append(token_text)
    lines.append("Note: the citation check confirms cited sections were supplied, not that they support the answer.")
    return "\n".join(lines)


def format_verified_result(result):
    """Text output for --verify: the user-facing answer first, then the verification trace."""
    verification = result["verification"]
    baseline = result["baseline"]
    lines = ["", f"Question: {result['question']}"]
    lines.append(f"Model: {result['model']}   Extractor: {verification.get('extractor')}   "
                 f"Verifier: {verification.get('verifier')}")
    lines.append(f"Status: {result['status']}   Outcome: {verification['outcome']}")
    if result["error"]:
        error = result["error"]
        lines.append(f"Error ({error.get('step')}/{error.get('stage')}, {error['kind']}): {error['message']}")
    lines.append("")
    if result["status"] == "verified_answer":
        lines += ["Verified answer:", f"  {result['answer']}", "",
                  f"Citations: {', '.join(result['citations']) or '(none)'}"]
    elif verification["outcome"] == "unverified":
        lines += ["Answer withheld: not every claim could be verified (verification only; no revision attempted).",
                  f"  {result['answer']}"]
    elif result["status"] == "abstained":
        lines += ["Abstained:", f"  {result['answer']}"]
    else:
        lines.append("Answer withheld: it could not be verified.")

    for record in verification["passes"]:
        title = "original answer" if record["stage"] == "original" else "revised answer"
        accepted = verification["outcome"] in ("accepted_original", "accepted_revision") and record["all_supported"]
        lines += ["", f"Verification trace - {title}" + ("" if accepted else " (diagnostic, not the answer)") + ":"]
        extraction, check = record["extraction"], record["verification"]
        if extraction["status"] != "ok":
            lines.append(f"  extraction failed: {extraction['error']['message']}")
            continue
        verdicts = {r["claim_id"]: r for r in check["results"]}
        trace = {t["claim_id"]: t for t in (check.get("trace") or {}).get("claims", [])}
        if check.get("trace"):
            model = check["trace"]["model"]
            integrity = model["integrity"]
            lines.append(f"  model {model['identifier']} on {model['device']}; checkpoint files {integrity['files']}, "
                         f"parameters {integrity['parameters']}")
        for claim in extraction["claims"]:
            item = verdicts.get(claim["claim_id"])
            verdict = item["verdict"] if item else "NOT VERIFIED"
            evidence = f" [{', '.join(item['evidence_section_ids'])}]" if item and item["evidence_section_ids"] else ""
            lines.append(f"  {claim['claim_id']} {verdict:<22}{evidence} {claim['text']}")
            if item and item["verdict"] != "SUPPORTED":
                lines.append(f"      {item['explanation']}")
            detail = trace.get(claim["claim_id"])
            if detail:
                lines.append(f"      reason: {detail['reason']}" + ("; possible compound claim ("
                                                                     + ", ".join(detail["compound_signals"]) + ")"
                                                                     if detail["possible_compound"] else ""))
                for premise in detail["premises"]:
                    label = premise["nli_label"] or f"unresolved ({premise['unresolved_reason']})"
                    lines.append(f"        {premise['section_id']} [{', '.join(premise['chunk_ids'])}]: {label}, "
                                 f"{premise['input_tokens']} tokens" + ("; incomplete section"
                                                                        if premise["known_incomplete"] else ""))
        if trace:
            lines.append("  (model labels per section; softmax scores in --json output are uncalibrated)")
        if check["status"] == "failed":
            lines.append(f"  verification failed: {check['error']['message']}")
    if verification["revision_attempted"]:
        revision = verification["revision"]
        lines.append(f"Revision: {revision['status']} (prompt: {revision.get('prompt', 'm5_verdicts')}; "
                     f"{len(revision.get('feedback') or [])} failed claim(s) sent; at most one revision is attempted)")

    latency = result["latency_s"]
    lines += ["", f"Latency: baseline {latency['baseline']:.3f} s, verification {latency['verification']:.3f} s, "
                  f"total {latency['total']:.3f} s (model loading excluded)"]
    usage = result["usage"]["verification"]
    lines.append(f"LLM calls for verification: {verification.get('llm_calls', 0)}"
                 + (f"; tokens: input {usage.get('input_tokens')}, output {usage.get('output_tokens')}" if usage else ""))
    costs = result["estimated_cost_usd"]
    if costs["total"] is not None:
        lines.append(f"Est. paid-tier cost: baseline ${costs['baseline']:.6f} + verification "
                     f"${costs['verification']:.6f} = ${costs['total']:.6f}")
    lines.append(f"Baseline (M4) status: {baseline['status']}")
    if verification.get("verifier", "").startswith("mock"):
        lines.append("Warning: mock verifier - a lexical heuristic for offline testing, not real verification.")
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(description="RAGForge grounded answering with Gemini (M4), optional "
                                                 "claim verification (M5).")
    parser.add_argument("--question", required=True, help="the question to answer")
    parser.add_argument("--method", choices=ALL_METHODS, default=RAG_METHOD,
                        help=f"retrieval method (default: {RAG_METHOD})")
    parser.add_argument("--evidence-k", type=int, default=DEFAULT_EVIDENCE_K,
                        help=f"number of retrieved chunks given to the model (default {DEFAULT_EVIDENCE_K})")
    parser.add_argument("--dense-weight", type=float, default=None,
                        help=f"hybrid methods only (default {RAG_WEIGHTS[0]})")
    parser.add_argument("--bm25-weight", type=float, default=None,
                        help=f"hybrid methods only (default {RAG_WEIGHTS[1]})")
    parser.add_argument("--model", default=None, help="Gemini model ID (default: $GEMINI_MODEL or the built-in default)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="sampling temperature; unset = model default (see README for Gemini 3 caveats)")
    parser.add_argument("--top-p", type=float, default=None, help="nucleus sampling; unset = model default")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="request timeout in seconds")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    parser.add_argument("--verify", action="store_true",
                        help="extract claims (Gemini) and verify them against the evidence; unverified answers are "
                             "withheld (off by default)")
    parser.add_argument("--verifier", choices=VERIFIERS, default=None,
                        help="with --verify: claim verifier (default nli = local fine-tuned V3 NLI model; gemini = M5 "
                             "LLM verifier; mock = offline lexical heuristic)")
    parser.add_argument("--nli-run", metavar="RUN_DIR", default=None,
                        help="with --verify and the nli verifier: M6.3 run directory holding best/ "
                             "(default: the selected V3 run)")
    parser.add_argument("--nli-device", choices=("auto", "mps", "cuda", "cpu"), default=None,
                        help="with --verify and the nli verifier: inference device (default auto: MPS, CUDA, CPU)")
    parser.add_argument("--revise", action="store_true",
                        help="with --verify: one Gemini correction of an answer that fails verification, guided by "
                             "the verifier's feedback, then a fresh check; returned only if fully verified "
                             "(off by default)")
    return parser


def main(argv=None, llm=None, index=None, verifier=None):
    """CLI entry point. `llm`, `index` and `verifier` can be injected (tests); otherwise they are built here."""
    from src.ingest import PROJECT_ROOT
    from src.retrieve import build_search_index

    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.question.strip():
        parser.error("--question must not be empty.")
    try:
        validate_top_k(args.evidence_k)
        retrieval_options(args.method, args.dense_weight, args.bm25_weight)
    except ValueError as error:
        parser.error(str(error))
    if not args.verify:
        for flag, value in (("--verifier", args.verifier), ("--nli-run", args.nli_run),
                            ("--nli-device", args.nli_device), ("--revise", args.revise)):
            if value:
                parser.error(f"{flag} requires --verify.")
    verifier_name = args.verifier or "nli"
    if verifier_name != "nli" and (args.nli_run or args.nli_device):
        parser.error("--nli-run and --nli-device apply only to the nli verifier.")
    load_s = None
    if args.verify and verifier_name == "nli":
        # Load (and integrity-check) the local model before any Gemini request, so a missing or
        # corrupted checkpoint costs nothing and never falls back to another verifier.
        from src.local_verifier import DEFAULT_V3_RUN, LocalNLIVerifier
        from src.verification import VerifierUnavailable

        verifier = verifier or LocalNLIVerifier(args.nli_run or DEFAULT_V3_RUN, args.nli_device or "auto")
        try:
            load_s = verifier.load().info["load_s"]
        except VerifierUnavailable as error:
            return _verifier_unavailable(args, verifier, error)

    load_env_file(PROJECT_ROOT / ".env")
    if llm is None:
        try:
            llm = GeminiClient(model=args.model, temperature=args.temperature, top_p=args.top_p,
                               timeout_s=args.timeout)
        except LLMError as error:
            parser.error(str(error))
    for warning in sampling_warnings(llm.model, args.temperature, args.top_p):
        print(f"Warning: {warning}", file=sys.stderr)

    if index is None:
        index = build_search_index(methods=(args.method,))
    if args.verify:
        return _main_verified(args, index, llm, verifier, load_s)
    result = answer_question(args.question, index, llm, args.method, args.evidence_k,
                             args.dense_weight, args.bm25_weight)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("RAGForge — Grounded Answer (Gemini)")
        print(format_result(result))
        if result["estimated_cost_usd"] is not None:
            print(f"Pricing source: {PRICING_SOURCE}")
    return 0 if result["status"] in ("answered", "abstained") else 1


def _verifier_unavailable(args, verifier, error):
    """Fail-closed result when the local verifier cannot load: no generation, no Gemini call, no answer."""
    result = {"question": args.question, "mode": "verified", "status": "verification_failed", "answer": None,
              "citations": [], "abstained": None, "verifier": verifier.name,
              "error": {"kind": error.kind, "message": str(error), "step": "verification", "stage": "setup",
                        "api_error": False}}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("RAGForge — Verified Answer (local NLI verification)")
        print(f"\nQuestion: {args.question}\nVerifier: {verifier.name}\nStatus: verification_failed\n"
              f"Error (verification/setup, {error.kind}): {error}\n\n"
              "No answer was generated: answers are only shown after verification, and the verifier is not "
              "available. No Gemini request was made. --verifier gemini selects the M5 LLM verifier explicitly.")
    return 1


def _main_verified(args, index, llm, verifier=None, load_s=None):
    from src.self_correction import answer_question_verified
    from src.verification import GeminiVerifier, MockVerifier

    if args.verifier == "mock":
        verifier = MockVerifier()
    elif args.verifier == "gemini":
        verifier = GeminiVerifier(llm)
    result = answer_question_verified(args.question, index, llm, verifier=verifier, method=args.method,
                                      evidence_k=args.evidence_k, dense_weight=args.dense_weight,
                                      bm25_weight=args.bm25_weight, revise=args.revise)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("RAGForge — Verified Answer (Gemini + claim verification)")
        print(format_verified_result(result))
        if load_s is not None:
            print(f"Local NLI model load (excluded from latency): {load_s:.2f} s")
        if result["estimated_cost_usd"]["total"] is not None:
            print(f"Pricing source: {PRICING_SOURCE}")
    return 0 if result["status"] in ("verified_answer", "abstained") else 1


if __name__ == "__main__":
    sys.exit(main())
