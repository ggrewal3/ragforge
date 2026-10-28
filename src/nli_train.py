"""NLI fine-tuning experiments (M6.3): pretrained baseline vs frozen encoder vs full fine-tuning.

Usage:
    python -m src.nli_train --baseline                      # V1: pretrained model, validation only, no training
    python -m src.nli_train --train frozen                  # V2: frozen encoder, trainable pretrained head
    python -m src.nli_train --train full                    # V3: full fine-tuning
    python -m src.nli_train --evaluate artifacts/nli/m6.3/<run>   # re-evaluate a saved checkpoint on validation
    python -m src.nli_train --compare                       # table of completed runs + cue-only baseline
    python -m src.nli_train --test artifacts/nli/m6.3/<run>       # FINAL held-out test of the selected checkpoint (once)

Data: the experimental frozen dataset (default v0.1-exp), loaded through
src.nli_review.load_frozen, which verifies every file hash first. Only the
train split drives gradient updates and only the validation split is used for
checkpoint selection and early stopping. load_split refuses the test split.
The only path to it is the explicit --test RUN_DIR command (load_test_split),
which evaluates an already-selected checkpoint once, with no training, no
optimizer and no checkpoint selection, and refuses to overwrite earlier results.
No other command reads or triggers it.

Model: cross-encoder/nli-MiniLM2-L6-H768 at the pinned M6.1 revision,
RobertaForSequenceClassification with its pretrained 3-way classification head.
V2 and V3 both start from freshly loaded pretrained weights (never from each
other), and loading fails if any head weight would be randomly initialized.
Class indices come from the checkpoint's config.id2label
({0: contradiction, 1: entailment, 2: neutral}) through src.nli's validated
mapping: SUPPORTED -> entailment, CONTRADICTED -> contradiction,
INSUFFICIENT_EVIDENCE -> neutral.

Input: premise = the example's evidence text (the exact sentence or section the
label was written against), hypothesis = the claim; truncation="only_first" at
512 tokens so the claim is never cut. This format was fixed before any run.

Variants (identical seed, data order and evaluation):
    V1 pretrained   no parameter updates
    V2 frozen       encoder (model.roberta) frozen and kept in eval mode; the pretrained
                    head (classifier.dense + classifier.out_proj, 592,899 parameters) trains.
                    AdamW lr 1e-3, batch 16, up to 10 epochs
    V3 full         every parameter trains (82,120,707). AdamW lr 2e-5, batch 8, up to 5 epochs
Both: cross-entropy loss, gradient clipping at norm 1.0, weight decay 0.01,
validation after every epoch, early stopping on validation macro-F1 with
patience 2, seed 42. No hyperparameter search.

The dataset is experimentally frozen and mostly AI-labeled (94 of 1,270 labels
human-reviewed). Scores measure agreement with those labels, not verified accuracy,
and no model from this module is production-ready.
"""

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src import nli_dataset as nd
from src import nli_review as nr
from src.ingest import PROJECT_ROOT
from src.nli import (
    DEFAULT_NLI_MAX_LENGTH,
    DEFAULT_NLI_MODEL,
    DEFAULT_NLI_REVISION,
    NLI_TO_VERDICT,
    label_mapping,
    select_device,
    softmax,
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "nli" / "m6.3"
DATASET_VERSION = "v0.1-exp"
TRAINING_SPLITS = ("train", "validation")
TEST_SPLIT = "test"
TEST_ARTIFACTS = ("test_metrics.json", "test_predictions.jsonl")
LABELS = nd.LABEL_VALUES                     # SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE
EXPECTED_ID2LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}
PREMISE_FORMAT = "evidence text only (the exact sentence or section the label was written against)"
REPORTED_CUE_BASELINE = {"validation_accuracy": 0.758, "test_accuracy": 0.730,
                         "source": "M6.2 shortcut audit, draft labels before the v0.1-exp freeze"}


@dataclass(frozen=True)
class TrainConfig:
    variant: str                 # "frozen" (V2) or "full" (V3)
    learning_rate: float
    batch_size: int
    max_epochs: int
    patience: int = 2
    seed: int = 42
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    eval_batch_size: int = 32
    max_length: int = DEFAULT_NLI_MAX_LENGTH


VARIANTS = {
    "frozen": TrainConfig("frozen", learning_rate=1e-3, batch_size=16, max_epochs=10),
    "full": TrainConfig("full", learning_rate=2e-5, batch_size=8, max_epochs=5),
}
VARIANT_NAMES = {"pretrained": "V1 pretrained", "frozen": "V2 frozen encoder", "full": "V3 full fine-tuning"}


class ExperimentError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Data and labels
# ---------------------------------------------------------------------------

def load_split(split, version=DATASET_VERSION, frozen_dir=nr.FROZEN_DIR):
    """Rows of the train or validation split of a frozen version. The test split is never loaded here."""
    if split not in TRAINING_SPLITS:
        raise ExperimentError(f"M6.3 uses only {TRAINING_SPLITS}; the {split!r} split is reserved and not loaded.")
    rows, manifest = nr.load_frozen(version, split, frozen_dir)
    return _checked(rows, manifest, version)


def load_test_split(version=DATASET_VERSION, frozen_dir=nr.FROZEN_DIR):
    """Rows of the held-out test split. Only for the final --test evaluation of an already-selected checkpoint."""
    rows, manifest = nr.load_frozen(version, TEST_SPLIT, frozen_dir, allow_test=True)
    return _checked(rows, manifest, version)


def _checked(rows, manifest, version):
    if manifest.get("dataset_type") != nr.EXPERIMENTAL_TYPE:
        raise ExperimentError(f"{version} is not an experimental sample-audited freeze.")
    for row in rows:
        if row["label"] not in LABELS:
            raise ExperimentError(f"{row['example_id']}: invalid label {row['label']!r}.")
    return rows, manifest


def dataset_info(version, frozen_dir, manifest, rows_by_split):
    manifest_path = Path(frozen_dir) / version / "manifest.json"
    return {
        "version": version,
        "dataset_type": manifest["dataset_type"],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "file_sha256": {f"{s}.jsonl": manifest["file_sha256"][f"{s}.jsonl"] for s in rows_by_split},
        "examples": {s: len(rows) for s, rows in rows_by_split.items()},
        "label_sources": {s: dict(Counter(r["review"]["label_source"] for r in rows)) for s, rows in rows_by_split.items()},
    }


def label_ids(id2label):
    """Verdict -> class index, from the checkpoint's own label metadata (validated by src.nli.label_mapping)."""
    mapping = label_mapping(id2label)
    return {NLI_TO_VERDICT[name]: index for index, name in mapping.items()}


def make_collate(tokenizer, label_to_id, max_length=DEFAULT_NLI_MAX_LENGTH):
    """Batch of dataset rows -> tokenized tensors (premise = evidence, hypothesis = claim) plus label IDs."""
    import torch

    def collate(rows):
        encoded = tokenizer([r["evidence"] for r in rows], [r["claim"] for r in rows], truncation="only_first",
                            max_length=max_length, padding=True, return_tensors="pt")
        batch = dict(encoded)
        batch["labels"] = torch.tensor([label_to_id[r["label"]] for r in rows], dtype=torch.long)
        return batch

    return collate


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def model_inputs(batch):
    return {key: value for key, value in batch.items() if key != "labels"}


# ---------------------------------------------------------------------------
# Reproducibility, model preparation, checkpoints
# ---------------------------------------------------------------------------

def seed_everything(seed):
    """Seed Python, NumPy and PyTorch (CPU and MPS). MPS kernels are not guaranteed bit-for-bit deterministic."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_pretrained(model_name=DEFAULT_NLI_MODEL, revision=DEFAULT_NLI_REVISION, local_files_only=True):
    """Fresh pretrained model and tokenizer. Fails rather than silently re-initializing any weight."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model, info = AutoModelForSequenceClassification.from_pretrained(
        model_name, revision=revision, local_files_only=local_files_only, output_loading_info=True)
    if info["missing_keys"] or info["mismatched_keys"]:
        raise ExperimentError(f"Checkpoint weights would be re-initialized: missing {sorted(info['missing_keys'])}, "
                              f"mismatched {info['mismatched_keys']}.")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, local_files_only=local_files_only)
    check_compatibility(model, tokenizer)
    return model, tokenizer


def check_compatibility(model, tokenizer):
    mapping = label_mapping(model.config.id2label)
    if mapping != EXPECTED_ID2LABEL:
        raise ExperimentError(f"Unexpected label mapping {mapping}; expected {EXPECTED_ID2LABEL} for the pinned checkpoint.")
    if tokenizer.vocab_size > model.config.vocab_size:
        raise ExperimentError("Tokenizer vocabulary is larger than the model's embedding table.")
    if not hasattr(model, model.base_model_prefix) or not hasattr(model, "classifier"):
        raise ExperimentError("Expected an encoder plus a 'classifier' head.")
    return mapping


def configure_trainable(model, variant):
    """Set requires_grad for a variant and return the names of trainable parameters.

    frozen: only the classification head (everything outside the encoder) trains.
    full:   every parameter trains.
    """
    if variant not in VARIANTS:
        raise ExperimentError(f"Unknown variant {variant!r}; expected one of {sorted(VARIANTS)}.")
    encoder = getattr(model, model.base_model_prefix)
    for parameter in model.parameters():
        parameter.requires_grad = True
    if variant == "frozen":
        for parameter in encoder.parameters():
            parameter.requires_grad = False
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise ExperimentError("No trainable parameters: the whole model was frozen.")
    if variant == "frozen" and any(name.startswith(model.base_model_prefix + ".") for name in trainable):
        raise ExperimentError("Encoder parameters are still trainable in the frozen variant.")
    return trainable


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def set_train_mode(model, variant):
    """Dropout on where weights train. A frozen encoder stays in eval mode so it is a fixed feature extractor."""
    model.train()
    if variant == "frozen":
        getattr(model, model.base_model_prefix).eval()


def save_checkpoint(model, tokenizer, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(directory)


def load_checkpoint(directory, tokenizer=None):
    """A saved checkpoint (model + tokenizer) from a local directory; never touches the network."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(directory, local_files_only=True)
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(directory, local_files_only=True)
    label_mapping(model.config.id2label)
    return model, tokenizer


class EarlyStopper:
    """Track the best validation macro-F1; stop after `patience` checks without strict improvement."""

    def __init__(self, patience):
        if patience < 1:
            raise ValueError("patience must be at least 1.")
        self.patience, self.best, self.best_epoch, self.bad_checks = patience, None, None, 0

    def update(self, score, epoch):
        """Record one validation check. Returns True when it is a new best (save a checkpoint)."""
        if self.best is None or score > self.best:
            self.best, self.best_epoch, self.bad_checks = score, epoch, 0
            return True
        self.bad_checks += 1
        return False

    @property
    def should_stop(self):
        return self.bad_checks >= self.patience


# ---------------------------------------------------------------------------
# Metrics and evaluation
# ---------------------------------------------------------------------------

def _rate(numerator, denominator):
    return None if denominator == 0 else numerator / denominator


def classification_metrics(gold, predicted):
    """Accuracy, macro-F1, per-class precision/recall/F1, confusion matrix and false-SUPPORTED rate.

    false_supported_rate = non-SUPPORTED examples predicted SUPPORTED / all non-SUPPORTED examples.
    A class never predicted has precision 0 (counted as 0 in macro-F1); a class absent from `gold`
    is left out of macro-F1.
    """
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have the same length.")
    confusion = {g: {p: 0 for p in LABELS} for g in LABELS}
    for g, p in zip(gold, predicted):
        confusion[g][p] += 1
    per_class, f1s = {}, []
    for label in LABELS:
        tp = confusion[label][label]
        predicted_n = sum(confusion[g][label] for g in LABELS)
        support = sum(confusion[label].values())
        precision = tp / predicted_n if predicted_n else 0.0
        recall = _rate(tp, support)
        f1 = (2 * precision * recall / (precision + recall) if precision + recall else 0.0) if support else None
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support,
                            "predicted": predicted_n}
        if f1 is not None:
            f1s.append(f1)
    not_supported = [p for g, p in zip(gold, predicted) if g != "SUPPORTED"]
    false_supported = sum(p == "SUPPORTED" for p in not_supported)
    return {
        "n": len(gold),
        "accuracy": _rate(sum(g == p for g, p in zip(gold, predicted)), len(gold)),
        "macro_f1": statistics.fmean(f1s) if f1s else None,
        "per_class": per_class,
        "confusion": confusion,
        "false_supported": false_supported,
        "false_supported_rate": _rate(false_supported, len(not_supported)),
    }


def sliced_metrics(rows, predicted):
    """Overall metrics plus slices by evidence format, policy family, gold label and label source."""
    gold = [r["label"] for r in rows]

    def by(key):
        groups = {}
        for value in sorted({key(r) for r in rows}):
            idx = [i for i, r in enumerate(rows) if key(r) == value]
            m = classification_metrics([gold[i] for i in idx], [predicted[i] for i in idx])
            groups[value] = {"n": m["n"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                             "false_supported_rate": m["false_supported_rate"]}
        return groups

    return {
        **classification_metrics(gold, predicted),
        "by_format": by(lambda r: r["evidence_format"]),
        "by_family": by(lambda r: r["family_id"]),
        "by_label": {label: {"n": sum(g == label for g in gold),
                             "recall": classification_metrics(gold, predicted)["per_class"][label]["recall"]}
                     for label in LABELS},
        "by_label_source": by(lambda r: "human-reviewed" if r["review"]["status"] in ("reviewed", "corrected")
                              else "ai-draft"),
        "by_phenomenon": by(lambda r: r["phenomenon"]),
    }


def evaluate(model, rows, collate, device, id_to_label, batch_size=32, warmup=True):
    """Validation pass without gradients: (metrics, per-example predictions).

    Loss is mean cross-entropy over examples; latency is wall time per example
    after one untimed warm-up batch.
    """
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    loader = DataLoader(rows, batch_size=batch_size, shuffle=False, collate_fn=collate)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    predictions, logits_all, total_loss, timed, elapsed = [], [], 0.0, 0, 0.0
    with torch.inference_mode():
        if warmup and rows:
            model(**model_inputs(to_device(collate(rows[:min(len(rows), batch_size)]), device)))
        for batch in loader:
            batch = to_device(batch, device)
            start = time.perf_counter()
            logits = model(**model_inputs(batch)).logits
            logits_cpu = logits.float().cpu()          # .cpu() waits for the device, so the timing is complete
            elapsed += time.perf_counter() - start
            timed += len(batch["labels"])
            total_loss += loss_fn(logits_cpu, batch["labels"].cpu()).item()
            logits_all.append(logits_cpu.numpy())
    logits = np.concatenate(logits_all) if logits_all else np.zeros((0, 3))
    probabilities = softmax(logits) if len(logits) else logits
    predicted = [id_to_label[int(i)] for i in np.argmax(logits, axis=1)] if len(logits) else []
    metrics = sliced_metrics(rows, predicted)
    metrics["loss"] = _rate(total_loss, len(rows))
    metrics["latency_ms_per_example"] = 1000 * elapsed / timed if timed else None
    for row, label, probs, raw in zip(rows, predicted, probabilities, logits):
        predictions.append({"example_id": row["example_id"], "gold": row["label"], "predicted": label,
                            "probabilities": {id_to_label[i]: float(probs[i]) for i in range(3)},
                            "logits": [float(x) for x in raw], "evidence_format": row["evidence_format"],
                            "family_id": row["family_id"]})
    return metrics, predictions


def cue_baseline(train_rows, validation_rows):
    """The M6.2 cue-only shortcut rule, refit on the frozen train split and scored on frozen validation."""
    predict, majority = nr.fit_cue_rule(train_rows)
    metrics = sliced_metrics(validation_rows, [predict(r) for r in validation_rows])
    metrics["majority_class"] = majority
    metrics["majority_class_accuracy"] = _rate(sum(r["label"] == majority for r in validation_rows),
                                               len(validation_rows))
    return metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config, model, tokenizer, train_rows, validation_rows, device, run_dir, progress=print):
    """Train one variant with an explicit PyTorch loop. Returns the experiment record (also written to run_dir)."""
    import torch
    from torch.utils.data import DataLoader

    seed_everything(config.seed)
    label_to_id = label_ids(model.config.id2label)
    id_to_label = {i: label for label, i in label_to_id.items()}
    trainable_names = configure_trainable(model, config.variant)
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if optimizer_ids != {id(p) for p in trainable}:
        raise ExperimentError("The optimizer does not hold exactly the trainable parameters.")
    collate = make_collate(tokenizer, label_to_id, config.max_length)
    generator = torch.Generator().manual_seed(config.seed)     # same shuffling order for every variant
    loader = DataLoader(train_rows, batch_size=config.batch_size, shuffle=True, generator=generator,
                        collate_fn=collate)
    loss_fn = torch.nn.CrossEntropyLoss()
    stopper = EarlyStopper(config.patience)
    history, best_metrics, started = [], None, time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        set_train_mode(model, config.variant)
        epoch_loss, seen, epoch_start = 0.0, 0, time.perf_counter()
        for batch in loader:                                            # 1. load a batch
            batch = to_device(batch, device)                            # 2. move tensors to the device
            logits = model(**model_inputs(batch)).logits                # 3. forward pass
            loss = loss_fn(logits, batch["labels"])                     # 4. cross-entropy loss
            optimizer.zero_grad(set_to_none=True)                       # 5. clear old gradients
            loss.backward()                                             # 6. backpropagation
            torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()                                            # 7. parameter update
            epoch_loss += loss.item() * len(batch["labels"])            # 8. track training loss
            seen += len(batch["labels"])
        metrics, predictions = evaluate(model, validation_rows, collate, device, id_to_label,  # 9. no-grad validation
                                        config.eval_batch_size, warmup=False)
        improved = stopper.update(metrics["macro_f1"], epoch)
        if improved:                                                    # 10. keep the best checkpoint
            save_checkpoint(model, tokenizer, Path(run_dir) / "best")
            best_metrics = metrics
            _write_json(Path(run_dir) / "validation_predictions.json", predictions)
        history.append({"epoch": epoch, "train_loss": epoch_loss / seen, "validation_loss": metrics["loss"],
                        "validation_accuracy": metrics["accuracy"], "validation_macro_f1": metrics["macro_f1"],
                        "validation_false_supported_rate": metrics["false_supported_rate"],
                        "epoch_seconds": time.perf_counter() - epoch_start, "new_best": improved})
        progress(f"  epoch {epoch}: train loss {epoch_loss / seen:.4f}, val loss {metrics['loss']:.4f}, "
                 f"val acc {metrics['accuracy']:.3f}, val macro-F1 {metrics['macro_f1']:.3f}, "
                 f"false-SUPPORTED {metrics['false_supported_rate']:.3f}" + ("  (best)" if improved else ""))
        if stopper.should_stop:
            progress(f"  early stopping: no macro-F1 improvement for {config.patience} validation checks")
            break

    # Re-time the best checkpoint's inference with a warm-up, so latency is comparable with V1.
    best_model, _ = load_checkpoint(Path(run_dir) / "best", tokenizer=tokenizer)
    best_model.to(device)
    timed_metrics, _ = evaluate(best_model, validation_rows, collate, device, id_to_label, config.eval_batch_size)
    best_metrics["latency_ms_per_example"] = timed_metrics["latency_ms_per_example"]
    return {
        "history": history,
        "best_epoch": stopper.best_epoch,
        "stopped_early": stopper.should_stop,
        "epochs_run": len(history),
        "training_seconds": time.perf_counter() - started,
        "trainable_parameter_names": trainable_names,
        "validation": best_metrics,
    }


# ---------------------------------------------------------------------------
# Experiment records
# ---------------------------------------------------------------------------

def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def new_run_dir(root, name):
    """Create a fresh run directory; never reuse or overwrite an existing one."""
    path = Path(root) / name
    if path.exists():
        raise ExperimentError(f"{path} already exists; experiment results are never overwritten. Choose another --run-name.")
    path.mkdir(parents=True)
    return path


def environment(device, device_note=None):
    import torch
    import transformers

    return {"device": device, "device_note": device_note, "torch": torch.__version__,
            "transformers": transformers.__version__, "python": sys.version.split()[0], "git_commit": _git_commit()}


def run_baseline(model, tokenizer, validation_rows, device, run_dir, info, env, progress=print):
    """V1: evaluate the pretrained model on validation without updating anything."""
    label_to_id = label_ids(model.config.id2label)
    id_to_label = {i: label for label, i in label_to_id.items()}
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.to(device)
    collate = make_collate(tokenizer, label_to_id)
    metrics, predictions = evaluate(model, validation_rows, collate, device, id_to_label)
    record = {"variant": "pretrained", "name": VARIANT_NAMES["pretrained"], "model": DEFAULT_NLI_MODEL,
              "revision": DEFAULT_NLI_REVISION, "label_ids": label_to_id, "premise_format": PREMISE_FORMAT,
              "parameters": parameter_counts(model), "dataset": info, "environment": env,
              "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "validation": metrics}
    _write_json(Path(run_dir) / "metrics.json", record)
    _write_json(Path(run_dir) / "validation_predictions.json", predictions)
    progress(format_metrics(record))
    return record


def run_training(config, model, tokenizer, train_rows, validation_rows, device, run_dir, info, env, progress=print):
    label_to_id = label_ids(model.config.id2label)
    configure_trainable(model, config.variant)
    record = {"variant": config.variant, "name": VARIANT_NAMES[config.variant], "model": DEFAULT_NLI_MODEL,
              "revision": DEFAULT_NLI_REVISION, "initialized_from": "pretrained checkpoint (fresh load)",
              "label_ids": label_to_id, "premise_format": PREMISE_FORMAT, "config": asdict(config),
              "parameters": parameter_counts(model), "dataset": info, "environment": env,
              "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "status": "running"}
    _write_json(Path(run_dir) / "config.json", record)
    progress(f"{record['name']}: {record['parameters']['trainable']:,} of {record['parameters']['total']:,} "
             f"parameters trainable; device {env['device']}; up to {config.max_epochs} epochs")
    result = train(config, model, tokenizer, train_rows, validation_rows, device, run_dir, progress)
    record.update(status="complete", **result)
    _write_json(Path(run_dir) / "metrics.json", record)
    progress(format_metrics(record))
    return record


def _f(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def format_metrics(record, split="validation"):
    m = record[split]
    lines = [f"\n{record['name']} - {split} ({m['n']} examples)"]
    if "best_epoch" in record:
        lines.append(f"  best epoch {record['best_epoch']} of {record['epochs_run']}, training "
                     f"{record['training_seconds']:.1f} s, {record['parameters']['trainable']:,} trainable parameters")
    lines.append(f"  accuracy {_f(m['accuracy'])}  macro-F1 {_f(m['macro_f1'])}  false-SUPPORTED rate "
                 f"{_f(m['false_supported_rate'])} ({m['false_supported']})  loss {_f(m.get('loss'))}  "
                 f"latency {_f(m.get('latency_ms_per_example'), 2)} ms/example")
    for label, s in m["per_class"].items():
        lines.append(f"    {label:<22} precision {_f(s['precision'])} recall {_f(s['recall'])} f1 {_f(s['f1'])} "
                     f"(support {s['support']})")
    lines.append("  confusion (rows gold, columns predicted): " + "; ".join(
        f"{g[:5]} " + "/".join(str(m["confusion"][g][p]) for p in LABELS) for g in LABELS))
    for name in ("by_format", "by_label_source"):
        lines.append(f"  {name.replace('by_', 'by ')}: " + ", ".join(
            f"{k} acc {_f(v['accuracy'])} F1 {_f(v['macro_f1'])} (n={v['n']})" for k, v in m[name].items()))
    lines.append("  by family: " + ", ".join(f"{k} {_f(v['accuracy'])} (n={v['n']})" for k, v in m["by_family"].items()))
    return "\n".join(lines)


def load_runs(root=ARTIFACTS_DIR):
    runs = []
    for path in sorted(Path(root).glob("*/metrics.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("validation"):
            runs.append({**record, "run_dir": str(path.parent)})
    return runs


def compare(runs, cue):
    order = {"pretrained": 0, "frozen": 1, "full": 2}
    runs = sorted(runs, key=lambda r: (order.get(r["variant"], 9), r["created_at"]))
    lines = ["\nM6.3 comparison - frozen v0.1-exp validation split (mostly AI-authored labels)",
             f"  {'model':<34}{'acc':>7}{'macro-F1':>10}{'false-SUP':>11}{'val loss':>10}{'ms/ex':>8}"
             f"{'trainable':>13}  sentence/full acc"]
    rows = [(f"cue-only rule (recomputed)", cue, None)] + [
        (f"{r['name']} [{Path(r['run_dir']).name}]", r["validation"], r["parameters"]["trainable"]) for r in runs]
    for name, m, trainable in rows:
        formats = m["by_format"]
        lines.append(f"  {name[:34]:<34}{_f(m['accuracy']):>7}{_f(m['macro_f1']):>10}{_f(m['false_supported_rate']):>11}"
                     f"{_f(m.get('loss')):>10}{_f(m.get('latency_ms_per_example'), 2):>8}"
                     f"{'-' if trainable is None else f'{trainable:,}':>13}  "
                     f"{_f(formats.get('sentence', {}).get('accuracy'))}/{_f(formats.get('full_section', {}).get('accuracy'))}")
    lines.append(f"\n  Recall by gold label (SUPPORTED / CONTRADICTED / INSUFFICIENT):")
    for name, m, _ in rows:
        lines.append(f"    {name[:40]:<40}" + " / ".join(_f(m["per_class"][label]["recall"]) for label in LABELS))
    lines.append(f"\n  Previously reported cue-only accuracy (draft labels, M6.2): validation "
                 f"{REPORTED_CUE_BASELINE['validation_accuracy']}, test {REPORTED_CUE_BASELINE['test_accuracy']}. "
                 f"The 'recomputed' row is refit on frozen v0.1-exp train and scored on its validation split.")
    lines.append(f"  Majority-class validation accuracy: {_f(cue['majority_class_accuracy'])}.")
    lines.append("  Do not choose a model on accuracy alone: weigh macro-F1, the false-SUPPORTED rate, per-class recall, "
                 "the format and family slices, and the margin over the cue-only rule. Validation has only 3 policy "
                 "families and mostly AI-authored labels.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Final held-out test (explicit --test only)
# ---------------------------------------------------------------------------

def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_fingerprint(model):
    """SHA-256 over every parameter and buffer, used to prove the test pass changed nothing."""
    import torch

    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def check_test_run(run_dir, version, frozen_dir):
    """Validate the selected run before any test data is read. Returns its training record.

    Requires a completed V2/V3 training run whose best/ checkpoint was chosen on
    validation, trained on the same frozen version (same manifest hash), and with
    no earlier test results (test results are never overwritten).
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise ExperimentError(f"{run_dir} is not a run directory.")
    existing = [name for name in TEST_ARTIFACTS if (run_dir / name).exists()]
    if existing:
        raise ExperimentError(f"{run_dir} already has {', '.join(existing)}. The held-out test is run once per selected "
                              "checkpoint and its results are never overwritten; read the existing files instead.")
    best = run_dir / "best"
    if not (best / "config.json").is_file() or not any((best / n).is_file()
                                                        for n in ("model.safetensors", "pytorch_model.bin")):
        raise ExperimentError(f"{best} does not contain a saved checkpoint (config.json and model weights).")
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        raise ExperimentError(f"{metrics_path} is missing; only a completed training run can be tested.")
    record = json.loads(metrics_path.read_text(encoding="utf-8"))
    if record.get("variant") not in VARIANTS or record.get("status") != "complete" or not record.get("validation"):
        raise ExperimentError(f"{run_dir} is not a completed V2/V3 training run with a validation-selected checkpoint.")
    trained_on = record.get("dataset", {})
    if trained_on.get("version") != version:
        raise ExperimentError(f"{run_dir} was trained on {trained_on.get('version')!r}, not {version!r}.")
    manifest_path = Path(frozen_dir) / version / "manifest.json"
    if not manifest_path.is_file():
        raise ExperimentError(f"No frozen version at {manifest_path.parent}.")
    if trained_on.get("manifest_sha256") != _sha256_file(manifest_path):
        raise ExperimentError(f"The {version} manifest no longer matches the one {run_dir.name} was trained against.")
    return record


def run_test(run_dir, record, test_rows, info, device, env, progress=print, checkpoint_loader=load_checkpoint):
    """Evaluate the selected checkpoint once on the test split: no optimizer, no gradients, no selection.

    Reuses the training run's collate (evidence + claim, same max_length and label IDs) and the
    shared evaluate() metrics. Writes RUN_DIR/test_predictions.jsonl and RUN_DIR/test_metrics.json
    with exclusive-create, so existing test results can never be replaced.
    """
    run_dir = Path(run_dir)
    best = run_dir / "best"
    checkpoint_sha256 = {p.name: _sha256_file(p) for p in sorted(best.iterdir()) if p.is_file()}
    model, tokenizer = checkpoint_loader(best)
    label_to_id = label_ids(model.config.id2label)
    if label_to_id != record["label_ids"]:
        raise ExperimentError(f"Checkpoint label IDs {label_to_id} differ from the training run's {record['label_ids']}.")
    id_to_label = {i: label for label, i in label_to_id.items()}
    config = record["config"]
    model.requires_grad_(False)
    model.to(device)
    model.eval()
    before = parameter_fingerprint(model)
    collate = make_collate(tokenizer, label_to_id, config["max_length"])
    metrics, predictions = evaluate(model, test_rows, collate, device, id_to_label, config["eval_batch_size"])
    if parameter_fingerprint(model) != before:
        raise ExperimentError("Model parameters changed during the test pass; no results were written.")
    validation = record["validation"]
    result = {
        "split": TEST_SPLIT,
        "evaluation": "final held-out test of a checkpoint selected on validation; no training, no optimizer, "
                      "no checkpoint selection",
        "run_id": run_dir.name, "run_dir": str(run_dir), "variant": record["variant"],
        "name": record["name"], "model": record["model"], "revision": record["revision"],
        "checkpoint": str(best), "checkpoint_sha256": checkpoint_sha256, "parameter_sha256": before,
        "selected_best_epoch": record["best_epoch"], "selection_metric": "validation macro-F1",
        "validation_at_selection": {k: validation.get(k) for k in ("n", "accuracy", "macro_f1",
                                                                    "false_supported_rate", "loss")},
        "label_ids": label_to_id, "premise_format": record["premise_format"],
        "preprocessing": {"truncation": "only_first", "max_length": config["max_length"], "padding": True,
                          "eval_batch_size": config["eval_batch_size"]},
        "training_config": config, "training_dataset": record["dataset"], "dataset": info,
        "test_examples": len(test_rows), "environment": env,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test": metrics,
    }
    with open(run_dir / "test_predictions.jsonl", "x", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    with open(run_dir / "test_metrics.json", "x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n")
    progress(format_metrics({"name": f"{record['name']} [{run_dir.name}]", "test": metrics}, split="test"))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description="RAGForge M6.3 NLI experiments. --baseline, --train, --evaluate and "
                                                 "--compare use train/validation only; --test is the separate, "
                                                 "one-time final held-out evaluation.", allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--baseline", action="store_true", help="V1: evaluate the pretrained model on validation")
    mode.add_argument("--train", choices=sorted(VARIANTS), help="V2 'frozen' or V3 'full'")
    mode.add_argument("--evaluate", metavar="RUN_DIR", help="re-evaluate a saved run's best checkpoint on validation")
    mode.add_argument("--compare", action="store_true", help="compare completed runs and the cue-only baseline")
    mode.add_argument("--test", metavar="RUN_DIR",
                      help="FINAL held-out evaluation of RUN_DIR/best on the test split, without training or "
                           "checkpoint selection. Run once, only after choosing the model on validation; writes "
                           "RUN_DIR/test_metrics.json and RUN_DIR/test_predictions.jsonl and refuses to overwrite them")
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    parser.add_argument("--frozen-dir", default=str(nr.FROZEN_DIR))
    parser.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR))
    parser.add_argument("--run-name", default=None, help="run directory name (default: variant + seed + timestamp)")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--allow-download", action="store_true",
                        help="allow downloading the checkpoint if it is not in the local Hugging Face cache")
    return parser


def main(argv=None, model_loader=None, progress=print, checkpoint_loader=load_checkpoint):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.test:
        return test_command(args, progress, checkpoint_loader)
    train_rows, manifest = load_split("train", args.dataset_version, args.frozen_dir)
    validation_rows, _ = load_split("validation", args.dataset_version, args.frozen_dir)
    info = dataset_info(args.dataset_version, args.frozen_dir, manifest,
                        {"train": train_rows, "validation": validation_rows})

    if args.compare:
        runs = load_runs(args.artifacts_dir)
        progress(compare(runs, cue_baseline(train_rows, validation_rows)))
        if not runs:
            progress("\nNo completed runs yet. Start with: python -m src.nli_train --baseline")
        return 0

    device, note = select_device(args.device)
    env = environment(device, note)
    loader = model_loader or (lambda: load_pretrained(local_files_only=not args.allow_download))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.evaluate:
        run_dir = Path(args.evaluate)
        model, tokenizer = load_checkpoint(run_dir / "best")
        label_to_id = label_ids(model.config.id2label)
        model.to(device)
        metrics, _ = evaluate(model, validation_rows, make_collate(tokenizer, label_to_id), device,
                              {i: label for label, i in label_to_id.items()})
        record = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")) if (run_dir / "metrics.json").is_file() \
            else {"name": run_dir.name}
        progress(format_metrics({"name": f"{record.get('name', run_dir.name)} (re-evaluated)", "validation": metrics}))
        return 0

    model, tokenizer = loader()
    if args.baseline:
        run_dir = new_run_dir(args.artifacts_dir, args.run_name or f"v1-pretrained-{stamp}")
        run_baseline(model, tokenizer, validation_rows, device, run_dir, info, env, progress)
    else:
        config = VARIANTS[args.train]
        prefix = {"frozen": "v2-frozen", "full": "v3-full"}[args.train]
        run_dir = new_run_dir(args.artifacts_dir, args.run_name or f"{prefix}-seed{config.seed}-{stamp}")
        run_training(config, model, tokenizer, train_rows, validation_rows, device, run_dir, info, env, progress)
    progress(f"\nResults: {run_dir}")
    return 0


def test_command(args, progress=print, checkpoint_loader=load_checkpoint):
    """--test: validate the run, then load only the test split and evaluate once. Never trains."""
    run_dir = Path(args.test)
    record = check_test_run(run_dir, args.dataset_version, args.frozen_dir)    # before any test data is read
    test_rows, manifest = load_test_split(args.dataset_version, args.frozen_dir)
    info = dataset_info(args.dataset_version, args.frozen_dir, manifest, {TEST_SPLIT: test_rows})
    device, note = select_device(args.device)
    progress(f"Final held-out test: {record['name']} [{run_dir.name}], best epoch {record['best_epoch']} "
             f"(selected on validation), {len(test_rows)} test examples, device {device}")
    run_test(run_dir, record, test_rows, info, device, environment(device, note), progress, checkpoint_loader)
    progress(f"\nTest results: {run_dir / TEST_ARTIFACTS[0]} and {run_dir / TEST_ARTIFACTS[1]}\n"
             "These are final numbers. Do not retune, retrain or re-select the model using them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
