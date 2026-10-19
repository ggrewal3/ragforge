"""Parse Gemini's structured answer and check its citations against the evidence it was given.

Two separate steps:

1. parse_answer(): the response must be a JSON object with exactly the fields
   answer (non-empty string), citations (list of strings) and abstained
   (boolean). Types are strict: "false" is not a boolean. Nothing is repaired.

2. validate_citations(): every cited section ID must be one of the section IDs
   supplied in the evidence blocks. Matching is exact (no case folding, no
   stripping of "#0" chunk suffixes, no mapping of E1-style references), so a
   malformed citation is reported, never silently fixed. When several supplied
   chunks share a section ID (a long section split in two), the section counts
   once and citing it is valid.

Citation existence is not citation support. A citation that passes here only
means the model pointed at a section it was actually shown. Whether that
section's text supports the claim is M5's evidence verification.
"""

import json
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr, ValidationError


class GroundedAnswer(BaseModel):
    """The structured response contract. Validation checks shape, not truth."""

    model_config = ConfigDict(extra="forbid")

    answer: StrictStr
    citations: list[StrictStr]
    abstained: StrictBool


class InvalidStructuredOutput(ValueError):
    """The model's output is not a valid GroundedAnswer."""


def parse_answer(text):
    """Parse and validate the model's JSON text. Raises InvalidStructuredOutput with a readable reason."""
    if not isinstance(text, str) or not text.strip():
        raise InvalidStructuredOutput("Response text is empty.")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise InvalidStructuredOutput(f"Response is not valid JSON ({error.msg} at position {error.pos}).") from None
    if not isinstance(data, dict):
        raise InvalidStructuredOutput(f"Response JSON must be an object, got {type(data).__name__}.")
    try:
        parsed = GroundedAnswer.model_validate(data)
    except ValidationError as error:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'response'}: {e['msg']}" for e in error.errors())
        raise InvalidStructuredOutput(f"Response does not match the schema ({problems}).") from None
    if not parsed.answer.strip():
        raise InvalidStructuredOutput("The 'answer' field is empty.")
    return parsed


# Overall citation statuses
PASS = "pass"                        # answered, >= 1 citation, all supplied, no duplicates
PASS_ABSTENTION = "pass_abstention"  # abstained with no citations
WARN = "warn"                        # usable but flagged: duplicates, or an abstention that cites sections
FAIL = "fail"                        # unknown section IDs, or an answer with no citations


@dataclass
class CitationReport:
    status: str
    valid: list = field(default_factory=list)       # cited IDs that were supplied (unique, in order)
    unknown: list = field(default_factory=list)     # cited IDs that were not supplied
    duplicates: list = field(default_factory=list)  # IDs cited more than once
    missing_citation: bool = False                  # answered (not abstained) with no citations
    citations_with_abstention: bool = False         # abstained but still cited sections
    issues: list = field(default_factory=list)      # human-readable descriptions

    def to_dict(self):
        return dict(self.__dict__)


def supplied_section_ids(chunks):
    """Unique section IDs of the evidence chunks, in order of first appearance."""
    return list(dict.fromkeys(chunk["section_id"] for chunk in chunks))


def validate_citations(answer, chunks):
    """Check a parsed GroundedAnswer's citations against the evidence chunks actually supplied."""
    supplied = set(supplied_section_ids(chunks))
    seen, valid, unknown, duplicates = set(), [], [], []
    for citation in answer.citations:
        if citation in seen:
            if citation not in duplicates:
                duplicates.append(citation)
            continue
        seen.add(citation)
        (valid if citation in supplied else unknown).append(citation)

    report = CitationReport(status=PASS, valid=valid, unknown=unknown, duplicates=duplicates)
    if unknown:
        report.issues.append(f"Cited section IDs that were not in the supplied evidence: {unknown}.")
    if duplicates:
        report.issues.append(f"Duplicate citations: {duplicates}.")
    if answer.abstained and answer.citations:
        report.citations_with_abstention = True
        report.issues.append("The response abstained but still cited sections; review whether it is a partial answer.")
    if not answer.abstained and not answer.citations:
        report.missing_citation = True
        report.issues.append("The response answered without citing any section.")

    if unknown or report.missing_citation:
        report.status = FAIL
    elif duplicates or report.citations_with_abstention:
        report.status = WARN
    elif answer.abstained:
        report.status = PASS_ABSTENTION
    return report
