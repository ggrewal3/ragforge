"""Local Natural Language Inference (M6.1): a pretrained NLI cross-encoder as a zero-shot claim verifier.

An NLI model reads a premise and a hypothesis together,

    [CLS] premise [SEP] hypothesis [SEP] -> transformer -> 3 logits

and classifies the pair as entailment, contradiction or neutral. Here the
premise is one evidence section and the hypothesis is one atomic claim:

    entailment    -> SUPPORTED
    contradiction -> CONTRADICTED
    neutral       -> INSUFFICIENT_EVIDENCE

The class order is read from the checkpoint's config.id2label, never assumed.
For cross-encoder/nli-MiniLM2-L6-H768 (revision b95119c) it is
{0: contradiction, 1: entailment, 2: neutral}. Label metadata that does not
name exactly these three classes is rejected (for example "LABEL_0").

Premise text (fixed before any benchmark run, not tuned on it):
    "{document title}, {section title}: {section text}"
The titles carry context the section text leaves implicit (a sentence in the
International Refund Policy may not repeat "international"). Section IDs are
left out because they mean nothing to an NLI model.

Length: inputs are truncated to max_length tokens (512 by default, the model's
limit), cutting only the premise (truncation="only_first") so the claim is
always seen whole. Every prediction records its token count and whether it was
truncated. Policy sections here are at most about 200 tokens.

Multiple evidence sections are NOT concatenated. Each section is scored
against the claim separately and the per-section verdicts are aggregated
with a fixed, conservative precedence (aggregate_verdict):

    any section CONTRADICTED -> CONTRADICTED
    else any section SUPPORTED -> SUPPORTED
    else INSUFFICIENT_EVIDENCE

Contradiction comes first because a false SUPPORTED is the most harmful error:
when a similar policy (say the domestic 30-day rule) seems to entail a wrong
claim about another group while the right section contradicts it, the claim
must not be accepted. The cost is that a distractor section misread as
contradicting can override real support.

The model is zero-shot: trained on general NLI data (SNLI/MultiNLI), never on
RAGForge policies. Its softmax probabilities are not calibrated confidences.
"""

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from src.claims import StageOutput
from src.verification import CONTRADICTED, INSUFFICIENT_EVIDENCE, SUPPORTED

DEFAULT_NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
DEFAULT_NLI_REVISION = "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"   # pinned for reproducibility
DEFAULT_NLI_BATCH_SIZE = 16
DEFAULT_NLI_MAX_LENGTH = 512

NLI_LABELS = ("entailment", "contradiction", "neutral")
NLI_TO_VERDICT = {"entailment": SUPPORTED, "contradiction": CONTRADICTED, "neutral": INSUFFICIENT_EVIDENCE}


class NLIModelError(ValueError):
    """The checkpoint's label metadata or output cannot be used safely."""


# ---------------------------------------------------------------------------
# Pure helpers (no torch needed)
# ---------------------------------------------------------------------------

def label_mapping(id2label):
    """config.id2label -> {class index: NLI label}, validated. Raises NLIModelError.

    Requires exactly three classes with indices 0, 1, 2 whose names are, after
    lower-casing, exactly entailment, contradiction and neutral (each once).
    """
    if not isinstance(id2label, dict) or len(id2label) != 3:
        raise NLIModelError(f"Expected 3 NLI labels, got {id2label!r}.")
    mapping = {}
    for index, name in id2label.items():
        try:
            index = int(index)
        except (TypeError, ValueError):
            raise NLIModelError(f"Label index {index!r} is not an integer.") from None
        label = str(name).strip().lower()
        if label not in NLI_TO_VERDICT:
            raise NLIModelError(f"Unsupported NLI label {name!r}; expected {NLI_LABELS}.")
        mapping[index] = label
    if sorted(mapping) != [0, 1, 2] or sorted(mapping.values()) != sorted(NLI_LABELS):
        raise NLIModelError(f"Labels must cover indices 0-2 and each NLI class once, got {id2label!r}.")
    return dict(sorted(mapping.items()))


def softmax(logits):
    """Numerically stable row-wise softmax in float64."""
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def check_logits(logits, expected_rows):
    """Validate raw model output: shape (n, 3) and finite. Returns a float64 array."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (expected_rows, 3):
        raise NLIModelError(f"Expected logits of shape ({expected_rows}, 3), got {logits.shape}.")
    if not np.all(np.isfinite(logits)):
        raise NLIModelError("Model returned NaN or infinite logits.")
    return logits


def select_device(preference="auto", mps_available=None, cuda_available=None):
    """(device, note). 'auto' prefers Apple MPS, then CUDA, then CPU; an unavailable request falls back to CPU."""
    if preference not in ("auto", "mps", "cuda", "cpu"):
        raise ValueError(f"device must be auto, mps, cuda or cpu, got {preference!r}.")
    if mps_available is None or cuda_available is None:
        import torch

        mps_available = torch.backends.mps.is_available() if mps_available is None else mps_available
        cuda_available = torch.cuda.is_available() if cuda_available is None else cuda_available
    available = {"mps": mps_available, "cuda": cuda_available, "cpu": True}
    if preference == "auto":
        for device in ("mps", "cuda", "cpu"):
            if available[device]:
                return device, None
    if available[preference]:
        return preference, None
    return "cpu", f"{preference} requested but not available; using cpu."


def premise_text(chunk):
    """The premise for one evidence chunk: document title, section title and text (no section ID)."""
    return f"{chunk['title']}, {chunk['section_title']}: {chunk['text']}"


def _check_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


@dataclass
class NLIPrediction:
    premise_id: str | None
    logits: list                 # raw logits in the checkpoint's class order (see `labels`)
    labels: list                 # NLI label for each logit index
    probabilities: dict          # NLI label -> softmax probability
    nli_label: str
    verdict: str
    model: str
    device: str
    input_tokens: int
    truncated: bool
    latency_s: float             # this pair's share of its batch's wall time

    def to_dict(self):
        return asdict(self)


def aggregate_verdict(predictions):
    """(verdict, deciding section IDs, explanation) from per-section predictions. See module docstring."""
    if not predictions:
        raise ValueError("Cannot aggregate zero predictions.")
    for verdict, nli_label in ((CONTRADICTED, "contradiction"), (SUPPORTED, "entailment")):
        deciding = [p for p in predictions if p.verdict == verdict]
        if deciding:
            ids = list(dict.fromkeys(p.premise_id for p in deciding if p.premise_id is not None))
            best = max(deciding, key=lambda p: p.probabilities[nli_label])
            return verdict, ids, f"NLI {nli_label} p={best.probabilities[nli_label]:.2f} on {best.premise_id}."
    best = max(predictions, key=lambda p: p.probabilities["neutral"])
    return INSUFFICIENT_EVIDENCE, [], f"NLI neutral on every section (max p={best.probabilities['neutral']:.2f})."


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LocalNLI:
    """A pretrained NLI checkpoint run locally with PyTorch (eval mode, inference_mode, MPS when available).

    `tokenizer` and `model` can be injected (tests); otherwise they are loaded
    with transformers' Auto classes. The first load downloads the checkpoint to
    the Hugging Face cache; pass local_files_only=True to forbid downloads.
    """

    def __init__(self, model_name=DEFAULT_NLI_MODEL, revision=DEFAULT_NLI_REVISION, device="auto",
                 batch_size=DEFAULT_NLI_BATCH_SIZE, max_length=DEFAULT_NLI_MAX_LENGTH, tokenizer=None, model=None,
                 local_files_only=False):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}.")
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 16:
            raise ValueError(f"max_length must be an integer >= 16, got {max_length!r}.")
        self.model_name, self.revision = model_name, revision
        self.batch_size, self.max_length = batch_size, max_length
        self.device, self.device_note = select_device(device)
        start = time.perf_counter()
        if tokenizer is None or model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name, revision=revision,
                                                                   local_files_only=local_files_only)
            model = model or AutoModelForSequenceClassification.from_pretrained(
                model_name, revision=revision, local_files_only=local_files_only)
        self.tokenizer = tokenizer
        self.labels = label_mapping(model.config.id2label)   # validate before moving weights
        self.model = model.to(self.device)
        self.model.eval()
        self.load_s = time.perf_counter() - start
        model_limit = getattr(tokenizer, "model_max_length", None)
        if isinstance(model_limit, int) and model_limit < self.max_length:
            self.max_length = model_limit

    @property
    def name(self):
        return f"nli:{self.model_name}"

    def predict_pairs(self, pairs, premise_ids=None):
        """[(premise, hypothesis), ...] -> [NLIPrediction], in order, batched by batch_size."""
        import torch

        if not pairs:
            return []
        pairs = [(_check_text(p, "Evidence (premise)"), _check_text(h, "Claim (hypothesis)")) for p, h in pairs]
        premise_ids = list(premise_ids) if premise_ids is not None else [None] * len(pairs)
        if len(premise_ids) != len(pairs):
            raise ValueError("premise_ids must have one entry per pair.")
        order = [self.labels[i] for i in range(3)]
        predictions = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start:start + self.batch_size]
            premises, hypotheses = [p for p, _ in batch], [h for _, h in batch]
            lengths = [len(ids) for ids in self.tokenizer(premises, hypotheses, truncation=False)["input_ids"]]
            encoded = self.tokenizer(premises, hypotheses, truncation="only_first", max_length=self.max_length,
                                     padding=True, return_tensors="pt")
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            began = time.perf_counter()
            with torch.inference_mode():
                output = self.model(**encoded)
                logits = output.logits.float().cpu().numpy()   # .cpu() waits for the device to finish
            share = (time.perf_counter() - began) / len(batch)
            logits = check_logits(logits, len(batch))
            for offset, (row, probs) in enumerate(zip(logits, softmax(logits))):
                index = int(np.argmax(row))
                nli_label = self.labels[index]
                predictions.append(NLIPrediction(
                    premise_id=premise_ids[start + offset], logits=[float(x) for x in row], labels=order,
                    probabilities={order[i]: float(probs[i]) for i in range(3)}, nli_label=nli_label,
                    verdict=NLI_TO_VERDICT[nli_label], model=self.model_name, device=self.device,
                    input_tokens=min(lengths[offset], self.max_length), truncated=lengths[offset] > self.max_length,
                    latency_s=share))
        return predictions

    def predict(self, premise, hypothesis, premise_id=None):
        return self.predict_pairs([(premise, hypothesis)], [premise_id])[0]

    def classify_claim(self, claim, evidence):
        """One claim against evidence chunks: (verdict, deciding section IDs, explanation, per-section predictions)."""
        _check_text(claim, "Claim (hypothesis)")
        if not evidence:
            raise ValueError("At least one evidence chunk is required.")
        predictions = self.predict_pairs([(premise_text(chunk), claim) for chunk in evidence],
                                         [chunk["section_id"] for chunk in evidence])
        return (*aggregate_verdict(predictions), predictions)


@dataclass
class NLIVerifier:
    """The M5 verifier interface (verify(claims, evidence) -> StageOutput) backed by LocalNLI.

    Provided so M6.4 can plug it into src/self_correction.py; M5 does not use it yet.
    Its payload passes the same parse_verdicts() validation as every other verifier.
    """

    nli: LocalNLI
    details: list = field(default_factory=list)   # per-claim section predictions from the last call

    @property
    def name(self):
        return self.nli.name

    def verify(self, claims, evidence):
        start = time.perf_counter()
        results, self.details = [], []
        for claim in claims:
            verdict, ids, explanation, predictions = self.nli.classify_claim(claim.text, evidence)
            results.append({"claim_id": claim.claim_id, "verdict": verdict, "evidence_section_ids": ids,
                            "explanation": explanation})
            self.details.append([p.to_dict() for p in predictions])
        return StageOutput(payload={"results": results}, model=self.nli.model_name,
                           latency_s=time.perf_counter() - start)
