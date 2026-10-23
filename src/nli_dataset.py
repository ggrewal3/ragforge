"""Policy NLI dataset (M6.2): deterministic generation, validation, leakage audit, review queue and statistics.

Usage:
    python -m src.nli_dataset --generate --out data/nli/policy_nli_v1.jsonl
    python -m src.nli_dataset --validate --dataset data/nli/policy_nli_v1.jsonl
    python -m src.nli_dataset --audit --dataset data/nli/policy_nli_v1.jsonl --out data/nli/policy_nli_v1.audit.json
    python -m src.nli_dataset --review-queue --dataset data/nli/policy_nli_v1.jsonl --out data/nli/review_queue_v1.jsonl
    python -m src.nli_dataset --stats --dataset data/nli/policy_nli_v1.jsonl

No model is trained, no API is called and nothing is downloaded. Token
statistics use the M6.1 tokenizer only if it is already in the local Hugging
Face cache; otherwise they are labelled approximate.

Where the labels come from
--------------------------
Every claim and label is hand-authored in data/nli/policy_facts.json. Each
fact names its decisive evidence with exact quotes from one section. This
module only derives evidence formats and extra examples from those facts,
following fixed rules; it never invents a claim. All labels are therefore
"rule-generated from a hand-authored specification", with review status
"draft". The generated dataset file is never changed by reviews: human
decisions live in an append-only log and are applied only when a reviewed
version is frozen (src/nli_review.py). Nothing is marked reviewed automatically.

Derivation rules (the label is re-derived for each evidence, never copied blindly)
    authored, full_section     the section text; the authored label (authors make
                               labels hold against the whole section)
    authored, sentence         the single decisive sentence; the authored label.
                               Claims that need two or more sentences get no
                               sentence example with their label, and neither do
                               facts marked sentence_context, whose sentence leaves
                               the claim's subject to the surrounding section
                               ("Approval is reviewed every 12 months").
    partial_evidence           a SUPPORTED claim that needs several sentences, paired
                               with one of them -> INSUFFICIENT_EVIDENCE
    counterpart                a claim that names its group (scoped), paired with the
                               sibling policy's evidence (domestic vs international,
                               purchases vs reallocations, annual vs parental leave)
                               -> INSUFFICIENT_EVIDENCE. Directional: only when the
                               sibling sentence names its own group explicitly.
    distractor                 one claim per section paired with another sentence of
                               the same section that shares no number and few content
                               words with it -> INSUFFICIENT_EVIDENCE

Leakage control
---------------
Policy families (near-duplicate or cross-referencing documents grouped together,
defined in the facts file) are assigned to train / validation / test BEFORE any
example is generated, using only each family's authored-claim count and a fixed seed.
Every example derived from a fact, including counterpart pairs, stays inside its
family, so it stays in one split. Claims that nearly duplicate one of the 40 M5
benchmark claims are excluded at generation. The audit reports what grouping
cannot remove: cross-family references between documents and lexically similar
claims in different splits.
"""

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.claims import numbers_in
from src.ingest import PROJECT_ROOT

FACTS_PATH = PROJECT_ROOT / "data" / "nli" / "policy_facts.json"
DATASET_PATH = PROJECT_ROOT / "data" / "nli" / "policy_nli_v1.jsonl"
REVIEWS_PATH = PROJECT_ROOT / "data" / "nli" / "reviews.jsonl"
BENCHMARK_PATH = PROJECT_ROOT / "data" / "evaluations" / "verification_claims.json"
PROTECTED_DIR = PROJECT_ROOT / "data" / "evaluations"

GENERATOR_VERSION = "m6.2.0"
SEED = 6202
SPLITS = ("train", "validation", "test")
SPLIT_TARGETS = {"train": 0.70, "validation": 0.15, "test": 0.15}
LABELS = {"S": "SUPPORTED", "C": "CONTRADICTED", "I": "INSUFFICIENT_EVIDENCE"}
LABEL_VALUES = tuple(LABELS.values())
FORMATS = ("sentence", "full_section")
AUTHORED_PHENOMENA = {"direct": "direct", "paraphrase": "paraphrase", "numerical": "numerical",
                      "negation": "negation", "condition": "condition", "exception": "exception",
                      "missing": "missing_information", "similar_policy": "similar_policy",
                      "multi_condition": "multi_condition", "multi_sentence": "multi_sentence"}
DERIVED_PHENOMENA = ("multi_sentence_partial", "distractor_sentence")
PHENOMENA = tuple(AUTHORED_PHENOMENA.values()) + DERIVED_PHENOMENA
DERIVATIONS = ("authored", "partial_evidence", "counterpart", "distractor")
REVIEW_STATUSES = ("draft", "reviewed", "corrected", "rejected")
LABEL_SOURCE = "rule-generated from hand-authored fact specification (not human-reviewed)"
BENCHMARK_EXCLUSION_JACCARD = 0.8
NEAR_DUPLICATE_JACCARD = 0.7
MODEL_TOKEN_LIMIT = 512
REQUIRED_FIELDS = ("example_id", "evidence", "evidence_format", "claim", "label", "document_id", "document_title",
                   "section_id", "section_title", "fact_id", "claim_id", "family_id", "phenomenon", "split",
                   "rationale", "provenance", "review")

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_STOPWORDS = frozenset(
    "a an the and or of to in on for by with at from as is are be been must may can will their they them it its "
    "this that these those which who than then there per each every after before within up only also have has "
    "had do does any all".split())
_NEGATION = re.compile(r"\b(not|no|never|cannot|nor|without|prohibited|non-[a-z]+)\b|n't\b", re.IGNORECASE)
_DOCUMENT_REF = re.compile(r"\b([A-Z]{2,3}-\d{3})\b")
SCOPE_TERMS = ("domestic", "international", "full-time", "part-time", "birth parent", "non-birth", "adoptive",
               "surrogacy", "enterprise", "monthly", "annual plan", "public", "internal", "confidential",
               "restricted", "sev-1", "sev-2", "sev-3", "sev-4", "p1", "p2", "p3", "p4", "tier 1", "tier 2",
               "tier 3", "service account", "administrator", "standard user", "hybrid", "fully remote",
               "personal", "finance department", "customer support", "contractor", "vendor", "small supplier",
               "reallocat", "purchase", "parental leave", "annual leave", "travel", "warranty")


class DatasetError(ValueError):
    """The facts file, documents or dataset violate a hard constraint."""


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def split_sentences(text):
    return [s for s in _SENTENCE_END.split(text.strip()) if s]


def content_words(text):
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS}


def jaccard(a, b):
    a, b = content_words(a), content_words(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def coverage(claim, evidence):
    """Share of the claim's content words that appear in the evidence."""
    words = content_words(claim)
    return len(words & content_words(evidence)) / len(words) if words else 0.0


def has_negation(text):
    return bool(_NEGATION.search(text))


def scope_terms(text):
    lowered = text.lower()
    return {term for term in SCOPE_TERMS if re.search(rf"\b{re.escape(term)}", lowered)}


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash(*parts, length=12):
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:length]


def section_of(fact_id):
    if not re.fullmatch(r"[A-Z]{2,3}-\d{3}\.\d+-[a-z]", fact_id or ""):
        raise DatasetError(f"Invalid fact_id {fact_id!r}; expected e.g. 'REF-729.1-a'.")
    return fact_id.rsplit("-", 1)[0]


# ---------------------------------------------------------------------------
# Loading and validating the fact specification
# ---------------------------------------------------------------------------

def section_index(documents):
    sections = {}
    for document in documents:
        for section in document["sections"]:
            sections[section["section_id"]] = {**section, "document_id": document["document_id"],
                                               "document_title": document["title"]}
    return sections


def _resolve_quotes(quotes, section, where):
    """Exact quotes -> the distinct sentences of `section` that contain them, in section order."""
    if not isinstance(quotes, list) or not quotes:
        raise DatasetError(f"{where}: quotes must be a non-empty list.")
    sentences = split_sentences(section["text"])
    found = []
    for quote in quotes:
        if not isinstance(quote, str) or not quote.strip():
            raise DatasetError(f"{where}: empty quote.")
        if section["text"].count(quote) != 1:
            raise DatasetError(f"{where}: quote {quote!r} must occur exactly once in {section['section_id']}.")
        matches = [i for i, sentence in enumerate(sentences) if quote in sentence]
        if len(matches) != 1:
            raise DatasetError(f"{where}: quote {quote!r} must lie inside one sentence of {section['section_id']}.")
        found.append(matches[0])
    return [sentences[i] for i in sorted(set(found))]


def load_facts(path=FACTS_PATH, documents=None):
    """Load and validate the fact specification. Returns (spec, resolved facts by ID, family by document)."""
    from src.ingest import load_documents

    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    documents = documents if documents is not None else load_documents()
    sections = section_index(documents)
    family_of_document = {}
    for family_id, family in spec.get("families", {}).items():
        for document_id in family["documents"]:
            if document_id in family_of_document:
                raise DatasetError(f"Document {document_id} is in two families.")
            family_of_document[document_id] = family_id
    missing_documents = {d["document_id"] for d in documents} - set(family_of_document)
    if missing_documents:
        raise DatasetError(f"Documents without a family: {sorted(missing_documents)}.")

    facts = {}
    for raw in spec.get("facts", []):
        fact_id = raw.get("fact_id")
        section_id = section_of(fact_id)
        if fact_id in facts:
            raise DatasetError(f"Duplicate fact_id {fact_id}.")
        if section_id not in sections:
            raise DatasetError(f"{fact_id}: unknown section {section_id}.")
        section = sections[section_id]
        fact = {"fact_id": fact_id, "section": section, "family_id": family_of_document[section["document_id"]],
                "sentences": _resolve_quotes(raw.get("q"), section, fact_id), "quotes": raw["q"],
                "counterpart": raw.get("counterpart"), "sentence_context": raw.get("sentence_context"),
                "claims": []}
        if set(raw) - {"fact_id", "q", "counterpart", "sentence_context", "claims"}:
            raise DatasetError(f"{fact_id}: unknown fact fields {sorted(set(raw) - set(fact) - {'q'})}.")
        seen = set()
        for index, claim in enumerate(raw.get("claims", [])):
            where = f"{fact_id} claim {index}"
            if not isinstance(claim, list) or len(claim) not in (4, 5):
                raise DatasetError(f"{where}: expected [label, phenomenon, text, why, options?].")
            label, phenomenon, text, why = claim[:4]
            options = claim[4] if len(claim) == 5 else {}
            if label not in LABELS:
                raise DatasetError(f"{where}: invalid label {label!r}.")
            if phenomenon not in AUTHORED_PHENOMENA:
                raise DatasetError(f"{where}: invalid phenomenon {phenomenon!r}.")
            if not isinstance(text, str) or not text.strip() or not isinstance(why, str) or not why.strip():
                raise DatasetError(f"{where}: claim text and rationale must be non-empty.")
            if not isinstance(options, dict) or set(options) - {"q", "scoped"}:
                raise DatasetError(f"{where}: unknown options {options!r}.")
            if text.casefold() in seen:
                raise DatasetError(f"{where}: duplicate claim text in fact.")
            seen.add(text.casefold())
            quotes = options.get("q", raw["q"])
            fact["claims"].append({
                "index": index, "label": LABELS[label], "phenomenon": AUTHORED_PHENOMENA[phenomenon],
                "text": text.strip(), "why": why.strip(), "scoped": bool(options.get("scoped")), "quotes": quotes,
                "sentences": _resolve_quotes(quotes, section, where),
                "claim_id": f"{fact_id}/{_hash(text.strip(), length=6)}"})
        facts[fact_id] = fact

    for fact in facts.values():
        counterpart = fact["counterpart"]
        if counterpart is None:
            continue
        if counterpart not in facts or counterpart == fact["fact_id"]:
            raise DatasetError(f"{fact['fact_id']}: unknown counterpart {counterpart!r}.")
        other = facts[counterpart]
        if other["family_id"] != fact["family_id"]:
            raise DatasetError(f"{fact['fact_id']}: counterpart {counterpart} is in another family.")
        if other["section"]["section_id"] == fact["section"]["section_id"] or len(other["sentences"]) != 1:
            raise DatasetError(f"{fact['fact_id']}: counterpart must be one sentence from another section.")
    return spec, facts, family_of_document


def assign_splits(family_sizes, seed=SEED):
    """Families -> splits, before any example exists. Greedy on authored-claim counts, seeded order, deterministic.

    Families are visited largest first (ties in a seeded shuffle) and each goes
    to the split furthest below its target share of facts.
    """
    if not family_sizes:
        raise DatasetError("No policy families to split.")
    order = sorted(family_sizes)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda family: -family_sizes[family])      # stable: seeded order breaks ties
    total = sum(family_sizes.values())
    assigned = {split: 0 for split in SPLITS}
    result = {}
    for family in order:
        split = max(SPLITS, key=lambda s: (SPLIT_TARGETS[s] * total - assigned[s], -SPLITS.index(s)))
        result[family] = split
        assigned[split] += family_sizes[family]
    return dict(sorted(result.items()))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

LABEL_MEANING = {"SUPPORTED": "the evidence establishes the claim",
                 "CONTRADICTED": "the evidence conflicts with the claim",
                 "INSUFFICIENT_EVIDENCE": "the evidence does not establish whether the claim is true or false"}


def load_benchmark_claims(path=BENCHMARK_PATH):
    if not Path(path).is_file():
        return []
    return [(e["example_id"], e["claim"]) for e in json.loads(Path(path).read_text(encoding="utf-8"))["examples"]]


def _example(fact, claim, evidence_section, evidence, evidence_format, label, phenomenon, derivation, rationale,
             decisive_sentences, split):
    evidence_id = evidence_section["section_id"]
    example_id = "nli-" + _hash(fact["fact_id"], claim["text"], evidence_format, evidence_id, evidence)
    return {
        "example_id": example_id,
        "evidence": evidence,
        "evidence_format": evidence_format,
        "claim": claim["text"],
        "label": label,
        "document_id": evidence_section["document_id"],
        "document_title": evidence_section["document_title"],
        "section_id": evidence_id,
        "section_title": evidence_section["section_title"],
        "fact_id": fact["fact_id"],
        "claim_id": claim["claim_id"],
        "family_id": fact["family_id"],
        "phenomenon": phenomenon,
        "split": split,
        "rationale": rationale,
        "provenance": {
            "derivation": derivation,
            "authored_label": claim["label"],
            "claim_section_id": fact["section"]["section_id"],
            "decisive_sentences": decisive_sentences,
            "section_text_sha256": sha256(evidence_section["text"]),
            "fact_source": "data/nli/policy_facts.json",
            "generator": "src/nli_dataset.py",
            "generator_version": GENERATOR_VERSION,
            "seed": SEED,
        },
        "review": {"status": "draft", "label_source": LABEL_SOURCE, "ai_label": label, "human_label": None,
                   "reviewer": None, "note": None, "reviewed_at": None,
                   "fingerprint": fingerprint(evidence, claim["text"], label)},
    }


def fingerprint(evidence, claim, label):
    """Identifies the exact generated content a review was made against."""
    return _hash(evidence, claim, label, length=16)


def _quote(sentences):
    return " / ".join(f'"{s}"' for s in sentences)


def generate(facts, family_splits, benchmark_claims=(), reviews=None):
    """Build all examples deterministically. Returns (examples, generation report)."""
    examples, excluded, partial_skipped, context_skipped = [], [], [], []
    by_section = defaultdict(list)
    for fact in facts.values():
        by_section[fact["section"]["section_id"]].append(fact)

    for fact_id in sorted(facts):
        fact = facts[fact_id]
        section, split = fact["section"], family_splits[fact["family_id"]]
        for claim in fact["claims"]:
            near = [(bid, round(jaccard(claim["text"], text), 3)) for bid, text in benchmark_claims
                    if claim["text"].casefold() == text.casefold()
                    or jaccard(claim["text"], text) >= BENCHMARK_EXCLUSION_JACCARD]
            if near:
                excluded.append({"claim_id": claim["claim_id"], "claim": claim["text"], "benchmark_matches": near})
                continue
            label, sentences = claim["label"], claim["sentences"]
            base = f"{label}: {LABEL_MEANING[label]}. {claim['why']} Decisive evidence: {_quote(sentences)}."
            examples.append(_example(fact, claim, section, section["text"], "full_section", label, claim["phenomenon"],
                                     "authored", base, sentences, split))
            if len(sentences) == 1 and fact["sentence_context"]:
                context_skipped.append(claim["claim_id"])
            elif len(sentences) == 1:
                examples.append(_example(fact, claim, section, sentences[0], "sentence", label, claim["phenomenon"],
                                         "authored", base, sentences, split))
            elif label == "SUPPORTED":
                for sentence in sentences:
                    rationale = (f"INSUFFICIENT_EVIDENCE: the claim needs {len(sentences)} sentences "
                                 f"({_quote(sentences)}); with only one supplied, the rest of the claim is not "
                                 f"established. Authored label with full evidence: SUPPORTED.")
                    examples.append(_example(fact, claim, section, sentence, "sentence", "INSUFFICIENT_EVIDENCE",
                                             "multi_sentence_partial", "partial_evidence", rationale, sentences, split))
            else:
                partial_skipped.append(claim["claim_id"])   # a lone sentence might still contradict: not derived

            if claim["scoped"] and fact["counterpart"]:
                other = facts[fact["counterpart"]]
                other_section = other["section"]
                rationale = (f"INSUFFICIENT_EVIDENCE: the claim is about the policy in {section['section_id']} "
                             f"({section['document_title']}), but the evidence is from {other_section['section_id']} "
                             f"({other_section['document_title']}), a similar policy for a different group or "
                             "purpose. A rule for one group neither supports nor contradicts a claim about another.")
                for evidence, evidence_format in ((other["sentences"][0], "sentence"),
                                                  (other_section["text"], "full_section")):
                    examples.append(_example(fact, claim, other_section, evidence, evidence_format,
                                             "INSUFFICIENT_EVIDENCE", "similar_policy", "counterpart", rationale,
                                             [], split))

    included = {e["claim_id"] for e in examples}
    for section_id in sorted(by_section):
        candidates = [(fact, claim) for fact in sorted(by_section[section_id], key=lambda f: f["fact_id"])
                      for claim in fact["claims"] if claim["claim_id"] in included and claim["label"] != "INSUFFICIENT_EVIDENCE"]
        if not candidates:
            continue
        rng = random.Random(f"{SEED}:{section_id}")
        fact, claim = candidates[0]
        decisive = {s for c in fact["claims"] for s in c["sentences"]}
        options = [s for s in split_sentences(fact["section"]["text"])
                   if s not in decisive and not (numbers_in(s) & numbers_in(claim["text"]))
                   and coverage(claim["text"], s) <= 0.34]
        if not options:
            continue
        sentence = rng.choice(options)
        rationale = (f"INSUFFICIENT_EVIDENCE: this sentence comes from the same section but does not address the "
                     f"claim; the decisive sentence ({_quote(claim['sentences'])}) is withheld.")
        examples.append(_example(fact, claim, fact["section"], sentence, "sentence", "INSUFFICIENT_EVIDENCE",
                                 "distractor_sentence", "distractor", rationale, [], family_splits[fact["family_id"]]))

    examples.sort(key=lambda e: (e["fact_id"], e["claim_id"], DERIVATIONS.index(e["provenance"]["derivation"]),
                                 e["evidence_format"], e["section_id"], e["example_id"]))
    applied = apply_reviews(examples, reviews or [])
    return examples, {"excluded_benchmark_near_duplicates": excluded,
                      "multi_sentence_claims_without_sentence_example": partial_skipped,
                      "claims_without_sentence_example_needing_context": context_skipped, "reviews": applied}


def apply_reviews(examples, reviews):
    """Apply the latest review per example, in place. Stale reviews (fingerprint mismatch) are reported, not applied.

    ai_label always keeps the generated label; human_label is the reviewer's
    label (None for a rejection). Only an explicit review changes status.
    """
    latest = {}
    for review in reviews:
        latest[review["example_id"]] = review
    by_id = {e["example_id"]: e for e in examples}
    applied, stale, unknown = 0, [], []
    for example_id, review in sorted(latest.items()):
        example = by_id.get(example_id)
        if example is None:
            unknown.append(example_id)
            continue
        if review["fingerprint"] != example["review"]["fingerprint"]:
            stale.append(example_id)
            continue
        status = review["status"]
        human_label = {"reviewed": example["label"], "corrected": review["corrected_label"], "rejected": None}[status]
        example["review"].update(status=status, reviewer=review["reviewer"], note=review.get("note"),
                                 reviewed_at=review["reviewed_at"], human_label=human_label,
                                 decision_id=review.get("decision_id"),
                                 label_source="human-reviewed" if status in ("reviewed", "corrected") else LABEL_SOURCE)
        if status == "corrected":
            example["label"] = human_label
        applied += 1
    return {"applied": applied, "stale": stale, "unknown_example_ids": unknown}


def load_reviews(path=REVIEWS_PATH):
    path = Path(path)
    if not path.is_file():
        return []
    reviews = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        review = json.loads(line)
        if review.get("status") not in REVIEW_STATUSES[1:]:
            raise DatasetError(f"{path}:{number}: invalid review status {review.get('status')!r}.")
        if review["status"] == "corrected" and review.get("corrected_label") not in LABEL_VALUES:
            raise DatasetError(f"{path}:{number}: a correction needs a valid corrected_label.")
        if not review.get("reviewer") or not review.get("fingerprint") or not review.get("example_id"):
            raise DatasetError(f"{path}:{number}: example_id, fingerprint and reviewer are required.")
        reviews.append(review)
    return reviews


def build(facts_path=FACTS_PATH, reviews_path=None, documents=None, benchmark_path=BENCHMARK_PATH):
    """Load everything and generate the AI-authored drafts. Returns (examples, manifest).

    reviews_path is applied only when given explicitly; the dataset file is always generated without it.
    """
    spec, facts, _ = load_facts(facts_path, documents)
    sizes = Counter(fact["family_id"] for fact in facts.values() for _ in fact["claims"])
    family_splits = assign_splits(dict(sizes))
    reviews = load_reviews(reviews_path) if reviews_path else []
    examples, report = generate(facts, family_splits, load_benchmark_claims(benchmark_path), reviews)
    manifest = {
        "dataset": "RAGForge policy NLI",
        "generator_version": GENERATOR_VERSION,
        "seed": SEED,
        "facts_version": spec.get("version"),
        "facts_sha256": sha256(Path(facts_path).read_text(encoding="utf-8")),
        "split_unit": "policy family (assigned before generation, by authored-claim counts)",
        "family_splits": family_splits,
        "family_documents": {f: spec["families"][f]["documents"] for f in sorted(spec["families"])},
        "examples": len(examples),
        "label_source": LABEL_SOURCE,
        **report,
    }
    return examples, manifest


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(examples, documents=None, family_splits=None):
    """Hard checks. Returns a list of error strings (empty means valid)."""
    from src.ingest import load_documents

    sections = section_index(documents if documents is not None else load_documents())
    errors = []
    if not examples:
        return ["Dataset is empty."]
    seen_ids, pairs = set(), {}
    fact_split, family_split, fact_family = {}, {}, {}
    evidence_split = defaultdict(set)
    for example in examples:
        where = example.get("example_id", "<no id>")
        missing = [f for f in REQUIRED_FIELDS if f not in example]
        if missing:
            errors.append(f"{where}: missing fields {missing}.")
            continue
        if where in seen_ids:
            errors.append(f"{where}: duplicate example_id.")
        seen_ids.add(where)
        if example["label"] not in LABEL_VALUES:
            errors.append(f"{where}: invalid label {example['label']!r}.")
        if example["evidence_format"] not in FORMATS:
            errors.append(f"{where}: invalid evidence_format {example['evidence_format']!r}.")
        if example["split"] not in SPLITS:
            errors.append(f"{where}: invalid split {example['split']!r}.")
        if example["phenomenon"] not in PHENOMENA:
            errors.append(f"{where}: invalid phenomenon {example['phenomenon']!r}.")
        for field in ("evidence", "claim", "rationale"):
            if not isinstance(example[field], str) or not example[field].strip():
                errors.append(f"{where}: empty {field}.")
        review = example["review"]
        if review.get("status") not in REVIEW_STATUSES:
            errors.append(f"{where}: invalid review status {review.get('status')!r}.")
        if review.get("status") == "draft" and review.get("label_source") != LABEL_SOURCE:
            errors.append(f"{where}: a draft example must not claim a human label source.")
        if review.get("status") in ("reviewed", "corrected") and not review.get("reviewer"):
            errors.append(f"{where}: a reviewed example needs a reviewer.")
        if review.get("ai_label") not in LABEL_VALUES:
            errors.append(f"{where}: the AI-authored label (review.ai_label) is missing.")
        if review.get("status") == "draft" and review.get("human_label") is not None:
            errors.append(f"{where}: a draft example must not have a human label.")
        if review.get("status") in ("reviewed", "corrected") and review.get("human_label") != example["label"]:
            errors.append(f"{where}: a reviewed example's label must equal its human label.")
        if review.get("status") == "reviewed" and review.get("human_label") != review.get("ai_label"):
            errors.append(f"{where}: an accepted example must keep the AI label; use 'corrected' to change it.")
        if review.get("status") == "corrected" and review.get("human_label") == review.get("ai_label"):
            errors.append(f"{where}: a correction must change the label.")
        provenance = example["provenance"]
        if provenance.get("derivation") not in DERIVATIONS or provenance.get("generator_version") != GENERATOR_VERSION:
            errors.append(f"{where}: invalid provenance.")

        section = sections.get(example["section_id"])
        if section is None:
            errors.append(f"{where}: unknown section {example['section_id']!r}.")
        else:
            if example["document_id"] != section["document_id"]:
                errors.append(f"{where}: document_id does not match section.")
            if provenance.get("section_text_sha256") != sha256(section["text"]):
                errors.append(f"{where}: section text changed since generation.")
            if example["evidence_format"] == "full_section" and example["evidence"] != section["text"]:
                errors.append(f"{where}: full_section evidence is not the exact section text.")
            if example["evidence_format"] == "sentence" and example["evidence"] not in split_sentences(section["text"]):
                errors.append(f"{where}: sentence evidence is not one exact sentence of {example['section_id']}.")
        try:
            claim_section = section_of(example["fact_id"])
        except DatasetError as error:
            errors.append(f"{where}: {error}")
            claim_section = None
        if claim_section and provenance.get("claim_section_id") != claim_section:
            errors.append(f"{where}: fact_id does not match the claim's section.")
        if not example["claim_id"].startswith(example["fact_id"] + "/"):
            errors.append(f"{where}: claim_id is not derived from fact_id.")

        key = (example["evidence"], example["claim"])
        if key in pairs:
            errors.append(f"{where}: duplicate evidence-claim pair (also {pairs[key]}).")
        pairs[key] = where
        for mapping, key_, value, name in ((fact_split, example["fact_id"], example["split"], "fact"),
                                           (family_split, example["family_id"], example["split"], "family"),
                                           (fact_family, example["fact_id"], example["family_id"], "fact family")):
            if mapping.setdefault(key_, value) != value:
                errors.append(f"{where}: {name} {key_} appears in more than one split/family.")
        evidence_split[example["evidence"]].add(example["split"])
        if family_splits is not None and family_splits.get(example["family_id"]) != example["split"]:
            errors.append(f"{where}: split differs from the manifest's family assignment.")
    for evidence, splits in evidence_split.items():
        if len(splits) > 1:
            errors.append(f"Identical evidence text appears in splits {sorted(splits)}: {evidence[:80]!r}.")
    return errors


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def label_flags(example):
    """Heuristic warnings about a label. They prompt review; they prove nothing either way."""
    label, claim = example["label"], example["claim"]
    derivation = example["provenance"]["derivation"]
    evidence = " ".join(example["provenance"]["decisive_sentences"]) or example["evidence"]
    flags = []
    if label == "SUPPORTED":
        if numbers_in(claim) - numbers_in(evidence):
            flags.append("supported_with_numbers_not_in_evidence")
        if has_negation(claim) != has_negation(evidence):
            flags.append("negation_mismatch")
        if scope_terms(claim) - scope_terms(evidence):
            flags.append("scope_terms_not_in_evidence")
    elif label == "CONTRADICTED":
        if coverage(claim, evidence) < 0.3:
            flags.append("contradiction_with_low_overlap")
    elif derivation != "counterpart":
        if coverage(claim, evidence) >= 0.8 and numbers_in(claim) <= numbers_in(evidence):
            flags.append("insufficient_but_looks_supported")
        if derivation == "distractor" and coverage(claim, example["evidence"]) >= 0.2:
            flags.append("distractor_overlaps_claim")
    return flags


def audit(examples, documents=None, benchmark_claims=None, manifest=None):
    """Leakage and label-quality audit report (a dict)."""
    from src.ingest import load_documents

    documents = documents if documents is not None else load_documents()
    benchmark_claims = load_benchmark_claims() if benchmark_claims is None else benchmark_claims
    split_of_family = {e["family_id"]: e["split"] for e in examples}
    family_of_document = {}
    for example in examples:
        family_of_document[example["document_id"]] = example["family_id"]

    # Exact duplicates and conflicting labels
    pairs = defaultdict(list)
    for e in examples:
        pairs[(e["evidence"], e["claim"])].append(e)
    duplicates = [[x["example_id"] for x in group] for group in pairs.values() if len(group) > 1]
    conflicting = [[x["example_id"] for x in group] for group in pairs.values() if len({x["label"] for x in group}) > 1]

    def crossing(key):
        splits = defaultdict(set)
        for e in examples:
            splits[key(e)].add(e["split"])
        return sorted(k for k, v in splits.items() if len(v) > 1)

    # Sentence-level evidence overlap across splits (catches shared sentences between sibling documents)
    sentence_splits = defaultdict(set)
    for e in examples:
        for sentence in split_sentences(e["evidence"]):
            sentence_splits[sentence].add(e["split"])
    shared_sentences = sorted(s for s, v in sentence_splits.items() if len(v) > 1)

    # Near-duplicate claims across splits
    claims_by_split = defaultdict(set)
    for e in examples:
        claims_by_split[e["split"]].add(e["claim"])
    near = []
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            for claim_a in sorted(claims_by_split[a]):
                for claim_b in sorted(claims_by_split[b]):
                    score = jaccard(claim_a, claim_b)
                    if score >= NEAR_DUPLICATE_JACCARD:
                        near.append({"splits": [a, b], "jaccard": round(score, 3), "claims": [claim_a, claim_b]})
    near.sort(key=lambda x: -x["jaccard"])

    # Cross-family document references in evidence (content that grouping cannot separate)
    references = []
    seen_refs = set()
    for e in examples:
        for document_id in sorted(set(_DOCUMENT_REF.findall(e["evidence"]))):
            target = family_of_document.get(document_id)
            if target and target != e["family_id"]:
                key = (e["section_id"], document_id)
                if key in seen_refs:
                    continue
                seen_refs.add(key)
                references.append({"section_id": e["section_id"], "family": e["family_id"], "split": e["split"],
                                   "references": document_id, "referenced_family": target,
                                   "referenced_split": split_of_family[target],
                                   "crosses_splits": split_of_family[target] != e["split"]})

    # Relationship to the M5 benchmark (kept separate and unchanged)
    benchmark_rows = []
    claims_split = [(e["claim"], e["split"]) for e in examples]
    for benchmark_id, text in benchmark_claims:
        best = {split: 0.0 for split in SPLITS}
        for claim, split in claims_split:
            best[split] = max(best[split], jaccard(text, claim))
        benchmark_rows.append({"benchmark_id": benchmark_id, "max_jaccard_by_split": {k: round(v, 3) for k, v in best.items()}})

    flags = {e["example_id"]: label_flags(e) for e in examples}
    flag_counts = Counter(flag for values in flags.values() for flag in values)
    families_per_split = {split: sorted({e["family_id"] for e in examples if e["split"] == split}) for split in SPLITS}
    limitations = []
    for split in ("validation", "test"):
        if len(families_per_split[split]) < 4:
            limitations.append(
                f"The {split} split contains only {len(families_per_split[split])} policy families "
                f"({', '.join(families_per_split[split])}). Its scores will reflect few policy types and have high "
                "variance; it is not a broad held-out benchmark.")
    if any(r["crosses_splits"] for r in references):
        limitations.append("Some sections reference documents in another split's families. Grouping cannot remove "
                           "this; the referencing sentences mostly name the other policy rather than restate it, but "
                           "a few restate its rule (see cross_family_references).")
    if any(max(r["max_jaccard_by_split"].values()) >= 0.5 for r in benchmark_rows):
        limitations.append("The M5 benchmark is built from the same 24 documents. Near-duplicate claims were "
                           "excluded, but many benchmark facts are also dataset facts, so the M5 benchmark is not an "
                           "independent test for a model trained on this dataset.")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "examples": len(examples),
        "independent_groups": {"families": len(split_of_family),
                               "facts": len({e["fact_id"] for e in examples}),
                               "families_per_split": families_per_split},
        "exact_duplicate_pairs": duplicates,
        "conflicting_label_pairs": conflicting,
        "facts_crossing_splits": crossing(lambda e: e["fact_id"]),
        "claims_crossing_splits": crossing(lambda e: e["claim_id"]),
        "families_crossing_splits": crossing(lambda e: e["family_id"]),
        "sections_crossing_splits": crossing(lambda e: e["section_id"]),
        "evidence_texts_crossing_splits": crossing(lambda e: e["evidence"]),
        "evidence_sentences_crossing_splits": shared_sentences,
        "near_duplicate_claims_across_splits": near,
        "cross_family_references": references,
        "benchmark": {
            "excluded_near_duplicate_claims": (manifest or {}).get("excluded_benchmark_near_duplicates"),
            "similarity_to_dataset_claims": benchmark_rows,
        },
        "label_flags": {"counts": dict(sorted(flag_counts.items())),
                        "flagged_examples": sum(bool(v) for v in flags.values()),
                        "by_example": {k: v for k, v in flags.items() if v},
                        "note": "Heuristic warnings for review prioritisation, not proof that a label is wrong or right."},
        "limitations": limitations,
    }


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------

PRIORITY = {"negation": 3, "numerical": 3, "exception": 3, "multi_sentence": 3, "multi_sentence_partial": 3,
            "similar_policy": 3, "condition": 2, "multi_condition": 2, "missing_information": 2,
            "distractor_sentence": 1, "paraphrase": 1, "direct": 0}


def review_queue(examples, documents=None):
    """Draft examples ordered by review priority: risky phenomena and automated flags first."""
    from src.ingest import load_documents

    sections = section_index(documents if documents is not None else load_documents())
    queue = []
    for example in examples:
        if example["review"]["status"] != "draft":
            continue
        flags = label_flags(example)
        reasons = [f"phenomenon:{example['phenomenon']}"] + [f"flag:{f}" for f in flags]
        queue.append({
            "priority": PRIORITY[example["phenomenon"]] + 4 * len(flags),
            "reasons": reasons,
            "example_id": example["example_id"],
            "split": example["split"],
            "evidence_format": example["evidence_format"],
            "evidence": example["evidence"],
            "claim": example["claim"],
            "proposed_label": example["label"],
            "rationale": example["rationale"],
            "section_id": example["section_id"],
            "original_section_text": sections[example["section_id"]]["text"],
            "review_status": example["review"]["status"],
            "fingerprint": example["review"]["fingerprint"],
        })
    queue.sort(key=lambda item: (-item["priority"], item["example_id"]))
    return queue


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def token_counter(model_name=None):
    """(count(evidence, claim) -> int, description). Uses a cached tokenizer only; never downloads."""
    if model_name:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            return (lambda evidence, claim: len(tokenizer(evidence, claim)["input_ids"]),
                    f"{model_name} tokenizer (local cache)")
        except (OSError, ValueError, ImportError):
            pass
    return (lambda evidence, claim: round(1.3 * (len(evidence.split()) + len(claim.split()))) + 4,
            "approximate (1.3 tokens per word + 4 specials); tokenizer not available locally")


def compute_stats(examples, count_tokens=None, token_source="approximate"):
    count_tokens = count_tokens or token_counter(None)[0]
    by = lambda key, rows=examples: dict(sorted(Counter(key(e) for e in rows).items()))   # noqa: E731
    per_split = {}
    for split in SPLITS:
        rows = [e for e in examples if e["split"] == split]
        per_split[split] = {
            "examples": len(rows),
            "share": round(len(rows) / len(examples), 3) if examples else None,
            "labels": by(lambda e: e["label"], rows),
            "label_shares": {k: round(v / len(rows), 3) for k, v in by(lambda e: e["label"], rows).items()} if rows else {},
            "formats": by(lambda e: e["evidence_format"], rows),
            "phenomena": by(lambda e: e["phenomenon"], rows),
            "families": len({e["family_id"] for e in rows}),
            "facts": len({e["fact_id"] for e in rows}),
        }
    lengths = defaultdict(list)
    for e in examples:
        lengths[e["evidence_format"]].append(count_tokens(e["evidence"], e["claim"]))
    token_stats = {fmt: {"mean": round(statistics.fmean(v), 1), "max": max(v),
                         "p95": sorted(v)[int(0.95 * (len(v) - 1))],
                         "over_512": sum(x > MODEL_TOKEN_LIMIT for x in v), "over_384": sum(x > 384 for x in v)}
                   for fmt, v in sorted(lengths.items())}
    return {
        "examples": len(examples),
        "splits": per_split,
        "labels": by(lambda e: e["label"]),
        "formats": by(lambda e: e["evidence_format"]),
        "phenomena": by(lambda e: e["phenomenon"]),
        "derivations": by(lambda e: e["provenance"]["derivation"]),
        "label_by_phenomenon": {p: by(lambda e: e["label"], [e for e in examples if e["phenomenon"] == p])
                                for p in sorted({e["phenomenon"] for e in examples})},
        "independent_groups": {"families": len({e["family_id"] for e in examples}),
                               "facts": len({e["fact_id"] for e in examples}),
                               "claims": len({e["claim_id"] for e in examples}),
                               "documents": len({e["document_id"] for e in examples}),
                               "sections": len({e["section_id"] for e in examples})},
        "tokens": {"source": token_source, "limit": MODEL_TOKEN_LIMIT, "by_format": token_stats},
        "review_status": by(lambda e: e["review"]["status"]),
        "label_sources": by(lambda e: e["review"]["label_source"]),
    }


def print_stats(stats, audit_report=None):
    print("\nRAGForge — Policy NLI Dataset (M6.2)")
    print(f"Examples: {stats['examples']}   Labels: " + ", ".join(f"{k} {v}" for k, v in stats["labels"].items()))
    print("Formats: " + ", ".join(f"{k} {v}" for k, v in stats["formats"].items())
          + "   Derivations: " + ", ".join(f"{k} {v}" for k, v in stats["derivations"].items()))
    groups = stats["independent_groups"]
    print(f"Independent groups: {groups['families']} families, {groups['facts']} facts, {groups['claims']} claims, "
          f"{groups['documents']} documents, {groups['sections']} sections")
    print(f"\n  {'split':<11}{'examples':>9}{'share':>7}{'families':>9}{'facts':>7}   "
          f"{'SUPPORTED':>10}{'CONTRADICTED':>13}{'INSUFFICIENT':>13}   sentence/full")
    for split, s in stats["splits"].items():
        shares = s["label_shares"]
        print(f"  {split:<11}{s['examples']:>9}{s['share']:>7.2f}{s['families']:>9}{s['facts']:>7}   "
              f"{shares.get('SUPPORTED', 0):>10.2f}{shares.get('CONTRADICTED', 0):>13.2f}"
              f"{shares.get('INSUFFICIENT_EVIDENCE', 0):>13.2f}   "
              f"{s['formats'].get('sentence', 0)}/{s['formats'].get('full_section', 0)}")
    print("\n  Phenomenon (labels S/C/I)")
    for phenomenon, labels in stats["label_by_phenomenon"].items():
        print(f"    {phenomenon:<24}{sum(labels.values()):>5}   "
              f"{labels.get('SUPPORTED', 0)}/{labels.get('CONTRADICTED', 0)}/{labels.get('INSUFFICIENT_EVIDENCE', 0)}")
    print(f"\n  Tokens per evidence+claim pair ({stats['tokens']['source']}):")
    for fmt, t in stats["tokens"]["by_format"].items():
        print(f"    {fmt:<13} mean {t['mean']}, p95 {t['p95']}, max {t['max']}, "
              f"over 512: {t['over_512']}, over 384: {t['over_384']}")
    print("\n  Review status: " + ", ".join(f"{k} {v}" for k, v in stats["review_status"].items()))
    print("  Label source: " + "; ".join(f"{k} ({v})" for k, v in stats["label_sources"].items()))
    if audit_report:
        print("\n  Leakage: " + ", ".join(f"{k} {len(audit_report[k])}" for k in (
            "exact_duplicate_pairs", "conflicting_label_pairs", "facts_crossing_splits", "families_crossing_splits",
            "evidence_texts_crossing_splits", "evidence_sentences_crossing_splits",
            "near_duplicate_claims_across_splits")))
        crossing = [r for r in audit_report["cross_family_references"] if r["crosses_splits"]]
        print(f"  Cross-family references crossing splits: {len(crossing)} "
              f"(of {len(audit_report['cross_family_references'])})")
        print(f"  Heuristic label flags: {audit_report['label_flags']['flagged_examples']} examples "
              f"{audit_report['label_flags']['counts']}")
        for limitation in audit_report["limitations"]:
            print(f"  LIMITATION: {limitation}")


# ---------------------------------------------------------------------------
# I/O and CLI
# ---------------------------------------------------------------------------

def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def manifest_path(dataset_path):
    return Path(dataset_path).with_suffix(".manifest.json")


def _check_output(parser, path, force):
    path = Path(path).resolve()
    if PROTECTED_DIR.resolve() in path.parents:
        parser.error(f"Refusing to write into {PROTECTED_DIR} (benchmark files are protected).")
    if path.exists() and not force:
        parser.error(f"{path} exists; pass --force to overwrite it.")


def build_parser():
    parser = argparse.ArgumentParser(description="RAGForge M6.2 policy NLI dataset tools (no training, no API calls). "
                                                 "Human review: python -m src.nli_review.", allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate", action="store_true", help="generate the dataset and its manifest")
    mode.add_argument("--validate", action="store_true", help="hard checks plus deterministic regeneration")
    mode.add_argument("--audit", action="store_true", help="leakage and label-quality audit report")
    mode.add_argument("--review-queue", action="store_true", help="write the prioritised review queue")
    mode.add_argument("--stats", action="store_true", help="print dataset statistics")
    parser.add_argument("--dataset", default=str(DATASET_PATH), help="dataset JSONL (default %(default)s)")
    parser.add_argument("--out", default=None, help="output path for --generate, --audit, --review-queue, --stats")
    parser.add_argument("--facts", default=str(FACTS_PATH), help="fact specification (default %(default)s)")
    parser.add_argument("--tokenizer", default="cross-encoder/nli-MiniLM2-L6-H768",
                        help="tokenizer for --stats, used only if cached locally")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output file")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.generate:
        out = args.out or args.dataset
        _check_output(parser, out, args.force)
        examples, manifest = build(args.facts)
        errors = validate(examples, family_splits=manifest["family_splits"])
        if errors:
            print("\n".join(errors[:20]), file=sys.stderr)
            parser.error(f"Generated dataset failed validation ({len(errors)} errors); nothing written.")
        write_jsonl(out, examples)
        manifest_path(out).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {len(examples)} examples to {out} and {manifest_path(out)}.")
        print(f"Excluded {len(manifest['excluded_benchmark_near_duplicates'])} claims that nearly duplicate the M5 "
              "benchmark. Human review decisions are kept separately (python -m src.nli_review).")
        return 0

    dataset = Path(args.dataset)
    if not dataset.is_file():
        parser.error(f"{dataset} not found; run --generate first.")
    examples = read_jsonl(dataset)

    if args.validate:
        manifest = json.loads(manifest_path(dataset).read_text(encoding="utf-8")) if manifest_path(dataset).is_file() else None
        errors = validate(examples, family_splits=manifest["family_splits"] if manifest else None)
        regenerated, _ = build(args.facts)
        if json.dumps(regenerated, ensure_ascii=False) != json.dumps(examples, ensure_ascii=False):
            errors.append("Regenerating from the facts file does not reproduce the dataset file exactly.")
        if manifest is None:
            errors.append(f"Manifest {manifest_path(dataset)} is missing.")
        for error in errors:
            print(f"ERROR {error}")
        print(f"{len(examples)} examples: " + ("valid; regeneration is deterministic." if not errors
                                              else f"{len(errors)} errors."))
        return 0 if not errors else 1

    if args.audit:
        manifest = json.loads(manifest_path(dataset).read_text(encoding="utf-8")) if manifest_path(dataset).is_file() else None
        report = audit(examples, manifest=manifest)
        out = args.out or str(dataset.with_suffix(".audit.json"))
        _check_output(parser, out, True)
        Path(out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print_stats(compute_stats(examples), report)
        print(f"\nAudit report: {out}")
        return 0

    if args.review_queue:
        queue = review_queue(examples)
        out = args.out or str(PROJECT_ROOT / "data" / "nli" / "review_queue_v1.jsonl")
        _check_output(parser, out, True)
        write_jsonl(out, queue)
        top = Counter(r for item in queue[:100] for r in item["reasons"])
        print(f"Wrote {len(queue)} draft examples to {out}, highest priority first.")
        print("Top-100 reasons: " + ", ".join(f"{k} {v}" for k, v in top.most_common()))
        return 0

    count, source = token_counter(args.tokenizer)
    stats = compute_stats(examples, count, source)
    print_stats(stats)
    if args.out:
        _check_output(parser, args.out, True)
        Path(args.out).write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
