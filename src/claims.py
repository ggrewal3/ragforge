"""Atomic claim extraction (M5): split a generated answer into independently verifiable claims.

An extractor is anything with

    extractor.name                              short label recorded in results
    extractor.extract(question, answer) -> StageOutput

where StageOutput.payload is the raw output: JSON text from an LLM, or an
already-decoded object from a local component. Extractors never validate their
own output. The orchestrator (src/self_correction.py) always runs
parse_claims() on it, so a new extractor cannot skip validation.

parse_claims() is strict and fails closed:
  * the output must be {"claims": [{"claim_id": "C1", "text": "..."}, ...]}
    with no extra fields, IDs C1..Cn in order, non-empty unique texts;
  * an empty claim list is a failure, because parse_claims() is only called
    for substantive (non-abstained) answers;
  * a cheap faithfulness check compares the numbers in the claims with the
    numbers in the answer. A claim with a number the answer (and question)
    never mentions means the extractor changed or invented content; a number
    in the answer that no claim carries means a quantity would go unverified.
    Both are failures. This catches numeric drift only, not changed wording
    or dropped negations.

Nothing is repaired. A failed extraction means the answer is not verified.
"""

import json
import re
import time
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, StrictStr, ValidationError

from src.prompts import build_extraction_prompt


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str

    def to_dict(self):
        return {"claim_id": self.claim_id, "text": self.text}


@dataclass
class StageOutput:
    """Raw output of one extraction, verification or revision step, before validation."""

    payload: object                   # JSON text (LLM) or a decoded dict (local component)
    model: str | None = None
    latency_s: float = 0.0
    attempts: int | None = None
    usage: dict = field(default_factory=dict)
    llm_call: bool = False            # True when a (possibly billed) LLM request was made


def call_llm_stage(llm, system_instruction, user_prompt, schema):
    """One structured LLM request as a StageOutput. LLMError propagates to the caller."""
    start = time.perf_counter()
    response = llm.generate(system_instruction, user_prompt, schema)
    return StageOutput(payload=response.text, model=response.model, latency_s=time.perf_counter() - start,
                       attempts=response.attempts, usage=dict(response.usage), llm_call=True)


class InvalidStageOutput(ValueError):
    """Base for validation failures of M5 stage outputs. `kind` is a short machine-readable label."""

    def __init__(self, message, kind="invalid_output"):
        super().__init__(message)
        self.kind = kind


class InvalidClaimsOutput(InvalidStageOutput):
    """The extractor's output is not a valid, faithful claim list."""


def decode_json_object(payload, error_cls):
    """JSON text or an already-decoded dict -> dict. Raises error_cls with a readable reason."""
    if isinstance(payload, dict):
        return payload
    if not isinstance(payload, str) or not payload.strip():
        raise error_cls("Output is empty.")
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as error:
        raise error_cls(f"Output is not valid JSON ({error.msg} at position {error.pos}).") from None
    if not isinstance(data, dict):
        raise error_cls(f"Output JSON must be an object, got {type(data).__name__}.")
    return data


def schema_problems(error):
    return "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'output'}: {e['msg']}" for e in error.errors())


class _ExtractedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: StrictStr
    text: StrictStr


class _ClaimList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[_ExtractedClaim]


_SECTION_ID = re.compile(r"\b[A-Z]{2,4}-\d{3}(?:\.\d+)?\b")
_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def numbers_in(text):
    """Numbers mentioned in text, normalized ("$10,000" -> "10000"); section IDs such as FIN-205.3 are ignored."""
    text = _SECTION_ID.sub(" ", text)
    return {match.replace(",", "") for match in _NUMBER.findall(text)}


def parse_claims(payload, answer, question=""):
    """Validate extractor output for a substantive answer. Returns a list of Claim; raises InvalidClaimsOutput."""
    data = decode_json_object(payload, InvalidClaimsOutput)
    try:
        parsed = _ClaimList.model_validate(data)
    except ValidationError as error:
        raise InvalidClaimsOutput(f"Output does not match the claims schema ({schema_problems(error)}).") from None

    if not parsed.claims:
        raise InvalidClaimsOutput("No claims were extracted from a substantive answer.", kind="empty_claims")
    claims, seen_texts = [], set()
    for number, item in enumerate(parsed.claims, start=1):
        if item.claim_id != f"C{number}":
            raise InvalidClaimsOutput(f"Claim IDs must be C1..C{len(parsed.claims)} in order; "
                                      f"position {number} has {item.claim_id!r}.")
        text = item.text.strip()
        if not text:
            raise InvalidClaimsOutput(f"Claim {item.claim_id} is empty.")
        if text.casefold() in seen_texts:
            raise InvalidClaimsOutput(f"Claim {item.claim_id} duplicates an earlier claim.")
        seen_texts.add(text.casefold())
        claims.append(Claim(item.claim_id, text))

    source_numbers = numbers_in(answer) | numbers_in(question or "")
    for claim in claims:
        invented = numbers_in(claim.text) - source_numbers
        if invented:
            raise InvalidClaimsOutput(f"Claim {claim.claim_id} contains numbers not in the answer: {sorted(invented)}.",
                                      kind="extraction_drift")
    dropped = numbers_in(answer) - set().union(*(numbers_in(c.text) for c in claims))
    if dropped:
        raise InvalidClaimsOutput(f"Numbers in the answer are missing from every claim: {sorted(dropped)}.",
                                  kind="extraction_incomplete")
    return claims


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

class GeminiClaimExtractor:
    """LLM-based extraction with the CLAIM_EXTRACTION prompt. `llm` has the src/llm.py interface."""

    def __init__(self, llm):
        self.llm = llm
        self.name = f"llm:{llm.model}"

    def extract(self, question, answer):
        return call_llm_stage(self.llm, *build_extraction_prompt(question, answer))


_CITATION_GROUP = re.compile(r"\s*[\(\[]\s*(?:[A-Z]{2,4}-\d{3}(?:\.\d+)?[\s,;]*(?:and\s+)?)+[\)\]]")
_LEAD_IN = re.compile(r"^(?:according to|per|under|based on)\b[^:,]{0,60}[:,]\s*", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


class SentenceClaimExtractor:
    """Deterministic, offline extractor for dry runs and tests: one claim per sentence.

    It removes parenthesized citations and lead-ins such as "According to the
    policy:", and drops fragments of fewer than three words. It does not split
    compound sentences or resolve pronouns, so its claims are not atomic in
    general. It exists to exercise the pipeline without API calls, not to
    measure extraction quality.
    """

    name = "sentence-split (offline)"

    def extract(self, question, answer):
        start = time.perf_counter()
        claims = []
        for sentence in _SENTENCE_END.split(answer.strip()):
            sentence = _LEAD_IN.sub("", _CITATION_GROUP.sub("", sentence)).strip()
            if len(sentence.split()) < 3:
                continue
            sentence = sentence[0].upper() + sentence[1:]
            claims.append({"claim_id": f"C{len(claims) + 1}", "text": sentence})
        return StageOutput(payload={"claims": claims}, model=None, latency_s=time.perf_counter() - start)
