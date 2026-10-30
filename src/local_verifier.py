"""Local claim verification with the fine-tuned M6.3 V3 NLI model (M6.4A).

LocalNLIVerifier implements the M5 verifier contract,

    verifier.verify(claims, evidence) -> StageOutput(payload={"results": [...]}, trace={...})

so src/self_correction.py validates its output with parse_verdicts() exactly
like the Gemini and mock verifiers. It is the default verifier of
answer_question_verified() and `python -m src.rag --verify`.

Checkpoint (selected on validation in M6.3, evaluated once on the held-out test):

    artifacts/nli/m6.3/v3-full-seed42-20261028-055715/best/

Loading (load_v3), once per (run directory, device), never downloading:
  1. the checkpoint directory must exist (artifacts/ is not in Git, so a fresh
     clone has no checkpoint and verification fails closed);
  2. integrity: every file in best/ must match the SHA-256 recorded in the run's
     test_metrics.json, and the loaded parameters must match its parameter hash
     (skipped, and recorded as "not_available", only if that metadata is absent);
  3. the label mapping comes from the checkpoint's config.id2label (validated by
     src.nli.label_mapping) and must equal the label IDs recorded for the run;
  4. the model runs in eval() mode under torch.inference_mode(), with every
     requires_grad off. No optimizer is ever created.
Any failure raises VerifierUnavailable, which the pipeline reports as
verification_api_failure and withholds the answer. There is no fallback to Gemini.

Input format (identical to M6.3 training): premise = raw evidence text, no
document or section titles; hypothesis = the claim; max length 512 with
truncation="only_first". Evidence is grouped by section_id: chunks of the same
section are joined in chunk order, and different sections are never joined.
A pair longer than the limit is never truncated and classified: the premise is
marked unresolved ("token_limit"); sentence-window predictions are kept as
diagnostics only and never decide a verdict.

Decision rules (decide(); deterministic, argmax labels only, no thresholds):

    no evidence premises                              INSUFFICIENT_EVIDENCE  missing_evidence
    some premise contradicts and some entails         CONTRADICTED           conflicting_evidence
    some premise contradicts                          CONTRADICTED           contradiction
    some premise could not be evaluated (too long)    INSUFFICIENT_EVIDENCE  token_limit
    a complete premise entails, claim may be compound INSUFFICIENT_EVIDENCE  possible_compound
    a complete premise entails                        SUPPORTED              entailment
    only a known-incomplete section entails           INSUFFICIENT_EVIDENCE  incomplete_section
    otherwise                                         INSUFFICIENT_EVIDENCE  no_entailment

The model's own label for every premise is kept in the trace, separate from
the final verdict. Softmax scores are recorded as "uncalibrated_softmax": they
are not probabilities of correctness, and no decision uses them.
"""

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from src.claims import StageOutput
from src.ingest import PROJECT_ROOT
from src.nli import DEFAULT_NLI_MAX_LENGTH, NLI_TO_VERDICT, LocalNLI, label_mapping, select_device
from src.verification import CONTRADICTED, INSUFFICIENT_EVIDENCE, SUPPORTED, VerifierUnavailable

DEFAULT_V3_RUN = PROJECT_ROOT / "artifacts" / "nli" / "m6.3" / "v3-full-seed42-20261028-055715"
CHECKPOINT_DIR = "best"
TEST_METADATA = "test_metrics.json"
TRAINING_METADATA = "metrics.json"
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
SCORE_NOTE = "uncalibrated softmax of the model's logits; not a probability that the verdict is correct"

MISSING_EVIDENCE = "missing_evidence"
CONTRADICTION = "contradiction"
CONFLICTING_EVIDENCE = "conflicting_evidence"
ENTAILMENT = "entailment"
NO_ENTAILMENT = "no_entailment"
POSSIBLE_COMPOUND = "possible_compound"
TOKEN_LIMIT = "token_limit"
INCOMPLETE_SECTION = "incomplete_section"
REASONS = (MISSING_EVIDENCE, CONTRADICTION, CONFLICTING_EVIDENCE, ENTAILMENT, NO_ENTAILMENT, POSSIBLE_COMPOUND,
           TOKEN_LIMIT, INCOMPLETE_SECTION)


# ---------------------------------------------------------------------------
# Evidence grouping
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Premise:
    """One section's supplied text, built only from chunks already in the evidence set."""

    section_id: str
    document_ids: tuple
    chunk_ids: tuple
    text: str
    known_incomplete: bool     # True only when the chunk IDs prove that part of the section is missing
    coverage_note: str


def _chunk_index(chunk_id, section_id):
    """n from "<section_id>#<n>" (src/chunking.py), or None if the ID does not have that form."""
    if isinstance(chunk_id, str) and chunk_id.startswith(f"{section_id}#"):
        suffix = chunk_id[len(section_id) + 1:]
        if suffix.isdigit():
            return int(suffix)
    return None


def group_evidence(evidence):
    """Group evidence chunks by section_id (first-appearance order) and join each section's chunks in chunk order.

    Uses only the supplied chunks: nothing is retrieved, and different sections are
    never combined. Completeness is recorded only when it can be determined: chunk
    IDs that skip an index or start after #0 prove part of the section is missing.
    Chunks #0..#n with no gap may still be followed by unsupplied chunks, which the
    metadata cannot rule out, so that case is recorded as "not proven complete".
    """
    groups = {}
    for chunk in evidence:
        chunks = groups.setdefault(chunk["section_id"], [])
        if chunk.get("chunk_id") is None or all(c.get("chunk_id") != chunk.get("chunk_id") for c in chunks):
            chunks.append(chunk)
    premises = []
    for section_id, chunks in groups.items():
        indices = [_chunk_index(c.get("chunk_id"), section_id) for c in chunks]
        if all(index is not None for index in indices):
            ordered = [c for _, c in sorted(zip(indices, chunks), key=lambda pair: pair[0])]
            present = sorted(indices)
            missing = sorted(set(range(present[-1] + 1)) - set(present))
            known_incomplete = bool(missing)
            note = (f"chunks {', '.join(f'#{i}' for i in missing)} of this section were not supplied"
                    if missing else f"chunks #0-#{present[-1]} supplied; later chunks, if any, cannot be ruled out")
        else:
            ordered, known_incomplete = chunks, False
            note = "chunk order unknown from chunk IDs; supplied order used; completeness unknown"
        premises.append(Premise(
            section_id=section_id,
            document_ids=tuple(dict.fromkeys(c.get("document_id") for c in ordered if c.get("document_id"))),
            chunk_ids=tuple(c.get("chunk_id") for c in ordered),
            text=" ".join(c["text"].strip() for c in ordered),
            known_incomplete=known_incomplete,
            coverage_note=note))
    return premises


# ---------------------------------------------------------------------------
# Compound-claim signals
# ---------------------------------------------------------------------------

_VERB = (r"(?:must|may|can|cannot|can't|will|shall|should|is|are|was|were|has|have|had|requires?|required|"
         r"takes?|needs?|receives?|applies|apply|gets?|includes?|covers?|allows?|allowed)")
_JOINED_CLAUSES = re.compile(rf"\b{_VERB}\b.+?\b(?:and|but|while|whereas|or)\b\s+(?:\S+\s+){{0,6}}?{_VERB}\b",
                             re.IGNORECASE)
_SENTENCE_BREAK = re.compile(r"[.!?]\s+[A-Z]")


def compound_signals(text):
    """Deterministic signs that a claim joins more than one independently checkable statement.

    Conditions and exceptions ("unless", "if", "except") are deliberately not signals,
    because the extraction prompt keeps them with the rule they modify.
    """
    signals = []
    if ";" in text:
        signals.append("semicolon")
    if _SENTENCE_BREAK.search(text.strip()):
        signals.append("multiple_sentences")
    if _JOINED_CLAUSES.search(text):
        signals.append("coordinated_clauses")
    return signals


# ---------------------------------------------------------------------------
# Decision engine
# ---------------------------------------------------------------------------

def decide(claim_text, premise_records):
    """Final verdict for one claim from its per-premise records (see the module docstring).

    Each record has section_id, status ("evaluated" or "unresolved"), nli_label
    (None when unresolved) and known_incomplete.
    """
    signals = compound_signals(claim_text)
    evaluated = [r for r in premise_records if r["status"] == "evaluated"]
    unresolved = [r["section_id"] for r in premise_records if r["status"] != "evaluated"]
    contradicting = [r["section_id"] for r in evaluated if r["nli_label"] == "contradiction"]
    supporting = [r["section_id"] for r in evaluated if r["nli_label"] == "entailment" and not r["known_incomplete"]]
    partial = [r["section_id"] for r in evaluated if r["nli_label"] == "entailment" and r["known_incomplete"]]

    def result(verdict, reason, evidence_ids, explanation):
        return {"verdict": verdict, "reason": reason, "evidence_section_ids": list(dict.fromkeys(evidence_ids)),
                "supporting_section_ids": supporting, "partially_supporting_section_ids": partial,
                "contradicting_section_ids": contradicting, "unresolved_section_ids": unresolved,
                "possible_compound": bool(signals), "compound_signals": signals, "explanation": explanation}

    if not premise_records:
        return result(INSUFFICIENT_EVIDENCE, MISSING_EVIDENCE, [], "No evidence was supplied for this claim.")
    if contradicting and (supporting or partial):
        return result(CONTRADICTED, CONFLICTING_EVIDENCE, contradicting + supporting + partial,
                      f"Conflicting evidence: the model predicts contradiction on {', '.join(contradicting)} and "
                      f"entailment on {', '.join(supporting + partial)}; the claim is not accepted.")
    if contradicting:
        return result(CONTRADICTED, CONTRADICTION, contradicting,
                      f"The model predicts contradiction on {', '.join(contradicting)}.")
    if unresolved:
        return result(INSUFFICIENT_EVIDENCE, TOKEN_LIMIT, unresolved,
                      f"Evidence {', '.join(unresolved)} is too long to evaluate with this claim, so the claim "
                      f"is not accepted.")
    if supporting and signals:
        return result(INSUFFICIENT_EVIDENCE, POSSIBLE_COMPOUND, supporting,
                      f"The model predicts entailment on {', '.join(supporting)}, but the claim may combine several "
                      f"statements ({', '.join(signals)}), so one entailment is not accepted as complete support.")
    if supporting:
        return result(SUPPORTED, ENTAILMENT, supporting, f"The model predicts entailment on {', '.join(supporting)}.")
    if partial:
        return result(INSUFFICIENT_EVIDENCE, INCOMPLETE_SECTION, partial,
                      f"Entailment only from incomplete section text ({', '.join(partial)}), so the claim is "
                      f"not accepted.")
    return result(INSUFFICIENT_EVIDENCE, NO_ENTAILMENT, [], "The model predicts no entailment from any supplied section.")


# ---------------------------------------------------------------------------
# Checkpoint loading (once per run directory and device)
# ---------------------------------------------------------------------------

@dataclass
class LoadedModel:
    nli: LocalNLI
    info: dict


_CACHE = {}


def clear_cache():
    _CACHE.clear()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else None


def check_checkpoint(run_dir):
    """Existence and file-integrity checks before loading. Returns (integrity, training_record, test_record)."""
    run_dir = Path(run_dir)
    checkpoint = run_dir / CHECKPOINT_DIR
    if not (checkpoint / "config.json").is_file() or not any((checkpoint / n).is_file() for n in WEIGHT_FILES):
        raise VerifierUnavailable(
            f"The V3 NLI checkpoint was not found at {checkpoint}. Checkpoints live under artifacts/, which is not "
            "in Git; restore the M6.3 run directory there or pass another run with --nli-run. The answer is "
            "withheld rather than verified another way.")
    training, test = _read_json(run_dir / TRAINING_METADATA), _read_json(run_dir / TEST_METADATA)
    recorded = (test or {}).get("checkpoint_sha256") or {}
    if not recorded:
        return {"files": "not_available", "parameters": "not_available",
                "note": f"no checkpoint hashes in {TEST_METADATA}; integrity not verified"}, training, test
    present = {p.name for p in checkpoint.iterdir() if p.is_file() and not p.name.startswith(".")}
    if present != set(recorded):
        raise VerifierUnavailable(f"Checkpoint files {sorted(present)} differ from those recorded in {TEST_METADATA} "
                                  f"({sorted(recorded)}).")
    for name, digest in sorted(recorded.items()):
        if _sha256_file(checkpoint / name) != digest:
            raise VerifierUnavailable(f"Checkpoint file {name} does not match the SHA-256 recorded in {TEST_METADATA}.")
    return {"files": "verified", "parameters": "not_available", "source": TEST_METADATA}, training, test


def _load_model_and_tokenizer(checkpoint):
    """Local files only; fails if any weight would be missing, unexpected or re-initialized."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model, info = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True,
                                                                     output_loading_info=True)
    problems = {k: v for k, v in info.items() if k in ("missing_keys", "unexpected_keys", "mismatched_keys") and v}
    if problems:
        raise VerifierUnavailable(f"The checkpoint does not load cleanly: {problems}.")
    return model, tokenizer


def checkpoint_label_ids(id2label):
    """Verdict -> class index from the checkpoint's own id2label (never an assumed order)."""
    return {NLI_TO_VERDICT[name]: index for index, name in label_mapping(id2label).items()}


def load_v3(run_dir=DEFAULT_V3_RUN, device="auto", loader=None):
    """The V3 model as a LoadedModel, cached per (run directory, resolved device). Raises VerifierUnavailable."""
    run_dir = Path(run_dir).expanduser().resolve()
    resolved, device_note = select_device(device)
    key = (str(run_dir), resolved)
    if key in _CACHE:
        return _CACHE[key]
    start = time.perf_counter()
    try:
        integrity, training, test = check_checkpoint(run_dir)
        model, tokenizer = (loader or _load_model_and_tokenizer)(run_dir / CHECKPOINT_DIR)
        labels = checkpoint_label_ids(model.config.id2label)
        expected = (test or {}).get("label_ids") or (training or {}).get("label_ids")
        if expected and labels != expected:
            raise VerifierUnavailable(f"Checkpoint label IDs {labels} differ from the run's recorded {expected}.")
        if (test or {}).get("parameter_sha256") and hasattr(model, "state_dict"):
            from src.nli_train import parameter_fingerprint

            if parameter_fingerprint(model) != test["parameter_sha256"]:
                raise VerifierUnavailable(f"Loaded parameters do not match the hash recorded in {TEST_METADATA}.")
            integrity["parameters"] = "verified"
        if hasattr(model, "requires_grad_"):
            model.requires_grad_(False)
        max_length = ((test or {}).get("preprocessing", {}).get("max_length")
                      or (training or {}).get("config", {}).get("max_length") or DEFAULT_NLI_MAX_LENGTH)
        identifier = f"nli-v3:{run_dir.name}"
        nli = LocalNLI(model_name=identifier, revision=None, device=resolved, max_length=max_length,
                       tokenizer=tokenizer, model=model)
    except VerifierUnavailable:
        raise
    except Exception as error:   # noqa: BLE001 - every loading failure fails closed
        raise VerifierUnavailable(f"Could not load the V3 NLI checkpoint from {run_dir}: "
                                  f"{type(error).__name__}: {error}") from error
    info = {"identifier": identifier, "run_id": run_dir.name, "checkpoint": str(run_dir / CHECKPOINT_DIR),
            "variant": (training or {}).get("variant"), "best_epoch": (training or {}).get("best_epoch"),
            "device": resolved, "device_note": device_note, "label_ids": labels, "max_length": nli.max_length,
            "premise_format": "raw evidence text (no titles), grouped by section; claim as hypothesis",
            "truncation": "never: over-limit pairs are unresolved", "integrity": integrity,
            "score_note": SCORE_NOTE, "load_s": time.perf_counter() - start}
    _CACHE[key] = LoadedModel(nli, info)
    return _CACHE[key]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _prediction_fields(prediction):
    return {"nli_label": prediction.nli_label, "model_verdict": prediction.verdict, "labels": prediction.labels,
            "logits": prediction.logits, "uncalibrated_softmax": prediction.probabilities,
            "truncated": prediction.truncated}


def diagnostic_windows(nli, text, claim):
    """Sentence windows of an over-limit premise that fit with the claim, with predictions. Diagnostics only."""
    windows, current = [], []
    for sentence in (s for s in _SENTENCE_END.split(text.strip()) if s):
        candidate = " ".join(current + [sentence])
        if nli.pair_token_counts([(candidate, claim)])[0] <= nli.max_length:
            current.append(sentence)
            continue
        if current:
            windows.append(" ".join(current))
        current = [sentence] if nli.pair_token_counts([(sentence, claim)])[0] <= nli.max_length else []
    if current:
        windows.append(" ".join(current))
    predictions = nli.predict_pairs([(w, claim) for w in windows])
    return [{"window": i, "words": len(w.split()), "input_tokens": p.input_tokens, **_prediction_fields(p),
             "diagnostic_only": True} for i, (w, p) in enumerate(zip(windows, predictions))]


class LocalNLIVerifier:
    """The M5 verifier interface backed by the fine-tuned V3 checkpoint. The model loads on first use (or load())."""

    def __init__(self, run_dir=DEFAULT_V3_RUN, device="auto", loader=None, loaded=None):
        self.run_dir, self.device, self._loader, self._loaded = Path(run_dir), device, loader, loaded

    @property
    def name(self):
        return f"nli:v3:{self.run_dir.name}"

    def load(self):
        if self._loaded is None:
            self._loaded = load_v3(self.run_dir, self.device, self._loader)
        return self._loaded

    def verify(self, claims, evidence):
        start = time.perf_counter()
        loaded = self.load()
        try:
            results, records = self._verify(loaded.nli, claims, evidence)
        except VerifierUnavailable:
            raise
        except Exception as error:   # noqa: BLE001 - inference failures fail closed, never fall back
            raise VerifierUnavailable(f"Local NLI inference failed: {type(error).__name__}: {error}") from error
        return StageOutput(payload={"results": results}, model=loaded.info["identifier"],
                           latency_s=time.perf_counter() - start,
                           trace={"model": dict(loaded.info), "claims": records})

    def _verify(self, nli, claims, evidence):
        premises = group_evidence(evidence)
        pairs = [(claim, premise) for claim in claims for premise in premises]
        counts = nli.pair_token_counts([(p.text, c.text) for c, p in pairs])
        fits = [n <= nli.max_length for n in counts]
        predictions = iter(nli.predict_pairs([(p.text, c.text) for (c, p), ok in zip(pairs, fits) if ok],
                                             [p.section_id for (_, p), ok in zip(pairs, fits) if ok]))
        by_claim = {}
        for (claim, premise), tokens, ok in zip(pairs, counts, fits):
            record = {"section_id": premise.section_id, "document_ids": list(premise.document_ids),
                      "chunk_ids": list(premise.chunk_ids), "known_incomplete": premise.known_incomplete,
                      "coverage_note": premise.coverage_note, "input_tokens": tokens, "max_length": nli.max_length}
            if ok:
                prediction = next(predictions)
                if prediction.truncated:          # cannot happen after the length check; never accept truncated input
                    raise VerifierUnavailable(f"Input for {premise.section_id} was truncated.")
                record.update(status="evaluated", unresolved_reason=None, **_prediction_fields(prediction))
            else:
                record.update(status="unresolved", unresolved_reason=TOKEN_LIMIT, nli_label=None, model_verdict=None,
                              labels=None, logits=None, uncalibrated_softmax=None, truncated=False,
                              diagnostic_windows=diagnostic_windows(nli, premise.text, claim.text))
            by_claim.setdefault(claim.claim_id, []).append(record)
        results, records = [], []
        for claim in claims:
            premise_records = by_claim.get(claim.claim_id, [])
            decision = decide(claim.text, premise_records)
            results.append({"claim_id": claim.claim_id, "verdict": decision["verdict"],
                            "evidence_section_ids": decision["evidence_section_ids"],
                            "explanation": decision["explanation"]})
            records.append({"claim_id": claim.claim_id, "claim": claim.text, **decision,
                            "evidence_chunk_ids": [c for r in premise_records for c in r["chunk_ids"]],
                            "document_ids": list(dict.fromkeys(d for r in premise_records for d in r["document_ids"])),
                            "premises": premise_records, "model": nli.model_name})
        return results, records


def default_verifier():
    """The M6.4A default: V3 from DEFAULT_V3_RUN on the best available device (model cached across calls)."""
    return LocalNLIVerifier(DEFAULT_V3_RUN, "auto")
