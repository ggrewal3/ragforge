"""Claim-evidence verification (M5): one verdict per atomic claim, using only the supplied evidence.

Verdicts (src.prompts.VERDICTS):
    SUPPORTED              the supplied evidence entails the claim
    CONTRADICTED           the supplied evidence directly conflicts with the claim
    INSUFFICIENT_EVIDENCE  the evidence does not establish whether the claim is true or false
Missing evidence is INSUFFICIENT_EVIDENCE, not CONTRADICTED.

A verifier is anything with

    verifier.name                              short label recorded in results
    verifier.verify(claims, evidence) -> StageOutput

`claims` is a list of src.claims.Claim; `evidence` is the read-only snapshot of
the chunks the answer was generated from (each has section_id, title,
section_title, text). StageOutput.payload must decode to
{"results": [{"claim_id", "verdict", "evidence_section_ids", "explanation"}]}.
A verifier signals that it could not run by raising LLMError (API verifiers)
or VerifierUnavailable (local verifiers, for example a model that failed to
load). Implementations:

    GeminiVerifier   one batched LLM request per answer (M5)
    MockVerifier     deterministic lexical heuristic, offline; NOT a real verifier
    (M6)             a fine-tuned local NLI model can implement the same interface

parse_verdicts() validates every verifier's output the same way, so an
implementation cannot skip the checks:
  * exactly one result per claim, no unknown or duplicate claim IDs;
  * verdicts from the fixed set only;
  * every evidence section ID must be one that was supplied (exact match,
    never repaired) - an invented ID makes the whole output invalid;
  * SUPPORTED and CONTRADICTED must name at least one evidence section;
  * a non-empty explanation.
The claim text in each result is taken from the extraction, never from the
verifier, so a verifier cannot rewrite the claim it judges.

Valid output is not correct output. An LLM verifier can be wrong, which is
why it is measured against hand-labeled examples (src/evaluate_verification.py).
"""

import re
import time
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictStr, ValidationError

from src.claims import InvalidStageOutput, StageOutput, call_llm_stage, decode_json_object, numbers_in, schema_problems
from src.prompts import VERDICTS, build_verification_prompt

SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE = VERDICTS


class VerifierUnavailable(Exception):
    """A non-API verifier could not run (for example, a local model failed to load)."""

    kind = "verifier_unavailable"


class InvalidVerifierOutput(InvalidStageOutput):
    """The verifier's output is malformed or refers to unknown claims or evidence."""


@dataclass
class ClaimVerification:
    claim_id: str
    claim: str
    verdict: str
    evidence_section_ids: list = field(default_factory=list)
    explanation: str = ""
    status: str = "complete"
    support_cited: bool | None = None   # SUPPORTED only: is a supporting section among the answer's citations?

    def to_dict(self):
        return dict(self.__dict__)


class _VerdictItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: StrictStr
    verdict: Literal["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
    evidence_section_ids: list[StrictStr]
    explanation: StrictStr


class _VerdictList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    results: list[_VerdictItem]


def parse_verdicts(payload, claims, supplied_ids, cited_ids=()):
    """Validate verifier output against the claims and the supplied section IDs.

    Returns ClaimVerification objects in claim order; raises InvalidVerifierOutput.
    `cited_ids` are the answer's citations, used only to fill `support_cited`.
    """
    data = decode_json_object(payload, InvalidVerifierOutput)
    try:
        parsed = _VerdictList.model_validate(data)
    except ValidationError as error:
        raise InvalidVerifierOutput(f"Output does not match the verdict schema ({schema_problems(error)}).") from None

    expected = {claim.claim_id: claim for claim in claims}
    by_id = {}
    for item in parsed.results:
        if item.claim_id not in expected:
            raise InvalidVerifierOutput(f"Verdict for unknown claim ID {item.claim_id!r}.", kind="unknown_claim_id")
        if item.claim_id in by_id:
            raise InvalidVerifierOutput(f"More than one verdict for claim {item.claim_id}.", kind="duplicate_claim_id")
        by_id[item.claim_id] = item
    missing = [claim_id for claim_id in expected if claim_id not in by_id]
    if missing:
        raise InvalidVerifierOutput(f"No verdict for claims {missing}.", kind="missing_verdict")

    supplied, cited = set(supplied_ids), set(cited_ids)
    results = []
    for claim in claims:
        item = by_id[claim.claim_id]
        evidence = list(dict.fromkeys(item.evidence_section_ids))
        unknown = [section_id for section_id in evidence if section_id not in supplied]
        if unknown:
            raise InvalidVerifierOutput(f"Claim {claim.claim_id} refers to evidence that was not supplied: {unknown}.",
                                        kind="unknown_evidence_id")
        if item.verdict in (SUPPORTED, CONTRADICTED) and not evidence:
            raise InvalidVerifierOutput(f"Claim {claim.claim_id} is {item.verdict} but names no evidence section.",
                                        kind="missing_evidence_id")
        if not item.explanation.strip():
            raise InvalidVerifierOutput(f"Claim {claim.claim_id} has an empty explanation.")
        results.append(ClaimVerification(
            claim_id=claim.claim_id, claim=claim.text, verdict=item.verdict, evidence_section_ids=evidence,
            explanation=item.explanation.strip(),
            support_cited=bool(cited & set(evidence)) if item.verdict == SUPPORTED else None))
    return results


def verdict_counts(results):
    return {verdict: sum(r.verdict == verdict for r in results) for verdict in VERDICTS}


# ---------------------------------------------------------------------------
# Verifiers
# ---------------------------------------------------------------------------

class GeminiVerifier:
    """All claims of one answer in one structured LLM request. `llm` has the src/llm.py interface."""

    def __init__(self, llm):
        self.llm = llm
        self.name = f"llm:{llm.model}"

    def verify(self, claims, evidence):
        return call_llm_stage(self.llm, *build_verification_prompt(claims, evidence))


_WORD = re.compile(r"[a-z]+")
_STOPWORDS = frozenset(
    "a an the and or of to in on for by with at from as is are be been must may can will not no any all "
    "their they them it its this that these those which who whom than then there per each every after before "
    "within up only also have has had do does".split())


def _content_words(text):
    return {word for word in _WORD.findall(text.lower()) if word not in _STOPWORDS and len(word) > 2}


class MockVerifier:
    """Deterministic lexical heuristic for offline runs. It is NOT a verifier and its accuracy is not model accuracy.

    For each claim it picks the evidence chunk with the highest share of the
    claim's content words, then:
      * SUPPORTED if at least 75% of the content words and every number appear there;
      * CONTRADICTED if at least 50% appear, the chunk has numbers, and a claim number does not;
      * INSUFFICIENT_EVIDENCE otherwise.
    It ignores negation, conditions and which group a rule applies to, so it
    fails exactly the cases the benchmark is designed to test. It is useful
    as a floor that a real verifier must beat, and to exercise every pipeline path.
    """

    name = "mock-lexical (offline heuristic)"

    def verify(self, claims, evidence):
        start = time.perf_counter()
        prepared = [(chunk["section_id"], _content_words(chunk["text"]), numbers_in(chunk["text"])) for chunk in evidence]
        results = []
        for claim in claims:
            words, numbers = _content_words(claim.text), numbers_in(claim.text)
            best_id, best_cover, best_numbers = None, -1.0, set()
            for section_id, chunk_words, chunk_numbers in prepared:
                cover = len(words & chunk_words) / len(words) if words else 0.0
                if cover > best_cover:
                    best_id, best_cover, best_numbers = section_id, cover, chunk_numbers
            if best_cover >= 0.75 and numbers <= best_numbers:
                verdict, ids = SUPPORTED, [best_id]
            elif best_cover >= 0.5 and best_numbers and numbers and not numbers <= best_numbers:
                verdict, ids = CONTRADICTED, [best_id]
            else:
                verdict, ids = INSUFFICIENT_EVIDENCE, []
            results.append({"claim_id": claim.claim_id, "verdict": verdict, "evidence_section_ids": ids,
                            "explanation": f"Lexical heuristic: {best_cover:.0%} word overlap with {best_id}."})
        return StageOutput(payload={"results": results}, model=None, latency_s=time.perf_counter() - start)
