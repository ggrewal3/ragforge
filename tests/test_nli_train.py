"""Tests for M6.3 NLI training: data loading, label mapping, batching, trainable parameters, gradient updates,
checkpoints, metrics, early stopping, reproducibility and test-split protection.

A tiny randomly initialized RoBERTa classifier (built from a config, nothing downloaded) and a
hash-based fake tokenizer stand in for the pretrained checkpoint. Frozen data is a synthetic
experimental freeze in a temporary directory, apart from read-only checks of the real v0.1-exp.
"""

import json
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

import torch
from transformers import RobertaConfig, RobertaForSequenceClassification
from transformers.utils import logging as hf_logging

from src import nli_dataset as nd
from src import nli_review as nr
from src import nli_train as nt
from test_nli_dataset import DOCUMENTS
from test_nli_review import NOW, Workspace

ID2LABEL = {0: "contradiction", 1: "entailment", 2: "neutral"}
hf_logging.disable_progress_bar()


class FakeTokenizer:
    """Word-hash tokenizer with the call signature the training code uses. Records its kwargs."""

    vocab_size = 60
    model_max_length = 512

    def __init__(self):
        self.calls = []

    def _ids(self, text):
        return [zlib.crc32(w.encode()) % 55 + 3 for w in text.lower().split()][:15]

    def __call__(self, premises, hypotheses, truncation=None, max_length=None, padding=None, return_tensors=None):
        self.calls.append({"truncation": truncation, "max_length": max_length, "padding": padding,
                           "return_tensors": return_tensors, "size": len(premises)})
        rows = [[0] + self._ids(p) + [2, 2] + self._ids(h) + [2] for p, h in zip(premises, hypotheses)]
        width = max(len(r) for r in rows)
        ids = torch.tensor([r + [1] * (width - len(r)) for r in rows])
        mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows])
        return {"input_ids": ids, "attention_mask": mask}

    def save_pretrained(self, directory):
        Path(directory, "fake_tokenizer.json").write_text("{}")


def tiny_model(seed=0, id2label=None):
    torch.manual_seed(seed)
    id2label = id2label or ID2LABEL
    config = RobertaConfig(vocab_size=60, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                           intermediate_size=32, max_position_embeddings=48, type_vocab_size=1, pad_token_id=1,
                           num_labels=3, id2label=id2label, label2id={v: k for k, v in id2label.items()})
    return RobertaForSequenceClassification(config)


def row(i, label, evidence_format="sentence", family="F1-returns", status="draft", split="train"):
    words = {"SUPPORTED": "refunds take thirty days", "CONTRADICTED": "refunds take ninety days",
             "INSUFFICIENT_EVIDENCE": "badges are blue today"}[label]
    return {"example_id": f"x{i}", "evidence": "refunds take thirty days for domestic customers",
            "claim": f"{words} item {i}", "label": label, "evidence_format": evidence_format, "family_id": family,
            "phenomenon": "direct", "split": split, "review": {"status": status, "label_source": "x"}}


ROWS = [row(i, label, fmt) for i, (label, fmt) in enumerate(
    [(l, f) for l in nd.LABEL_VALUES for f in nd.FORMATS] * 3)]
VAL = [row(100 + i, label, fmt, family="F2-passwords", split="validation") for i, (label, fmt) in enumerate(
    [(l, f) for l in nd.LABEL_VALUES for f in nd.FORMATS])]


class LabelAndBatchTests(unittest.TestCase):
    def test_label_mapping_from_metadata(self):
        self.assertEqual(nt.label_ids(ID2LABEL), {"CONTRADICTED": 0, "SUPPORTED": 1, "INSUFFICIENT_EVIDENCE": 2})
        self.assertEqual(nt.label_ids({0: "entailment", 1: "neutral", 2: "contradiction"}),
                         {"SUPPORTED": 0, "INSUFFICIENT_EVIDENCE": 1, "CONTRADICTED": 2})
        with self.assertRaises(ValueError):
            nt.label_ids({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
        tokenizer = FakeTokenizer()
        nt.check_compatibility(tiny_model(), tokenizer)
        with self.assertRaises(nt.ExperimentError):
            nt.check_compatibility(tiny_model(id2label={0: "entailment", 1: "contradiction", 2: "neutral"}), tokenizer)

    def test_tokenization_and_batching(self):
        tokenizer = FakeTokenizer()
        collate = nt.make_collate(tokenizer, nt.label_ids(ID2LABEL), max_length=512)
        batch = collate(ROWS[:4])
        self.assertEqual(batch["input_ids"].shape[0], 4)
        self.assertEqual(batch["attention_mask"].shape, batch["input_ids"].shape)
        self.assertEqual(batch["labels"].tolist(), [1, 1, 0, 0])            # SUPPORTED=1, CONTRADICTED=0
        self.assertEqual(tokenizer.calls[-1], {"truncation": "only_first", "max_length": 512, "padding": True,
                                               "return_tensors": "pt", "size": 4})
        self.assertEqual(set(nt.model_inputs(batch)), {"input_ids", "attention_mask"})


class TrainableParameterTests(unittest.TestCase):
    def test_frozen_variant(self):
        model = tiny_model()
        names = nt.configure_trainable(model, "frozen")
        self.assertTrue(names and all(n.startswith("classifier.") for n in names))
        self.assertTrue(all(not p.requires_grad for p in model.roberta.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.classifier.parameters()))
        counts = nt.parameter_counts(model)
        self.assertEqual(counts["trainable"], sum(p.numel() for p in model.classifier.parameters()))
        self.assertGreater(counts["frozen"], 0)

    def test_full_variant(self):
        model = tiny_model()
        nt.configure_trainable(model, "frozen")
        names = nt.configure_trainable(model, "full")                     # re-enables the encoder
        self.assertEqual(len(names), len(list(model.parameters())))
        self.assertEqual(nt.parameter_counts(model)["frozen"], 0)
        with self.assertRaises(nt.ExperimentError):
            nt.configure_trainable(model, "everything")

    def test_frozen_encoder_stays_in_eval_mode(self):
        model = tiny_model()
        nt.set_train_mode(model, "frozen")
        self.assertFalse(model.roberta.training)
        self.assertTrue(model.classifier.training)
        nt.set_train_mode(model, "full")
        self.assertTrue(model.roberta.training)

    def test_gradient_updates(self):
        for variant in ("frozen", "full"):
            with self.subTest(variant):
                model = tiny_model()
                nt.configure_trainable(model, variant)
                encoder_before = {n: p.detach().clone() for n, p in model.roberta.named_parameters()}
                head_before = {n: p.detach().clone() for n, p in model.classifier.named_parameters()}
                trainable = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(trainable, lr=1e-2)
                batch = nt.make_collate(FakeTokenizer(), nt.label_ids(ID2LABEL))(ROWS[:6])
                nt.set_train_mode(model, variant)
                loss = torch.nn.CrossEntropyLoss()(model(**nt.model_inputs(batch)).logits, batch["labels"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                head_changed = any(not torch.equal(head_before[n], p) for n, p in model.classifier.named_parameters())
                encoder_changed = any(not torch.equal(encoder_before[n], p) for n, p in model.roberta.named_parameters())
                self.assertTrue(head_changed)
                self.assertEqual(encoder_changed, variant == "full")


class MetricTests(unittest.TestCase):
    def test_classification_metrics(self):
        gold = ["SUPPORTED", "SUPPORTED", "CONTRADICTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
        pred = ["SUPPORTED", "CONTRADICTED", "SUPPORTED", "CONTRADICTED", "SUPPORTED"]
        m = nt.classification_metrics(gold, pred)
        self.assertAlmostEqual(m["accuracy"], 2 / 5)
        self.assertEqual(m["false_supported"], 2)                          # 1 contradicted + 1 insufficient
        self.assertAlmostEqual(m["false_supported_rate"], 2 / 3)
        s, c, i = (m["per_class"][label] for label in nd.LABEL_VALUES)
        self.assertAlmostEqual(s["precision"], 1 / 3)
        self.assertAlmostEqual(s["recall"], 1 / 2)
        self.assertAlmostEqual(c["precision"], 1 / 2)
        self.assertEqual((i["precision"], i["recall"], i["f1"]), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(m["macro_f1"], (0.4 + 0.5 + 0.0) / 3)
        self.assertEqual(m["confusion"]["INSUFFICIENT_EVIDENCE"]["SUPPORTED"], 1)
        perfect = nt.classification_metrics(gold, gold)
        self.assertEqual((perfect["accuracy"], perfect["macro_f1"], perfect["false_supported_rate"]), (1.0, 1.0, 0.0))
        with self.assertRaises(ValueError):
            nt.classification_metrics(gold, pred[:2])

    def test_slices(self):
        rows = ROWS[:6]
        m = nt.sliced_metrics(rows, [r["label"] for r in rows])
        self.assertEqual(set(m["by_format"]), set(nd.FORMATS))
        self.assertEqual(m["by_label"]["SUPPORTED"]["recall"], 1.0)
        self.assertEqual(m["by_label_source"]["ai-draft"]["n"], 6)
        self.assertEqual(m["by_family"]["F1-returns"]["accuracy"], 1.0)


class EarlyStoppingTests(unittest.TestCase):
    def test_patience(self):
        stopper = nt.EarlyStopper(patience=2)
        self.assertTrue(stopper.update(0.50, 1))
        self.assertTrue(stopper.update(0.60, 2))
        self.assertFalse(stopper.update(0.60, 3))                         # a tie is not an improvement
        self.assertFalse(stopper.should_stop)
        self.assertFalse(stopper.update(0.55, 4))
        self.assertTrue(stopper.should_stop)
        self.assertEqual((stopper.best, stopper.best_epoch), (0.60, 2))
        stopper = nt.EarlyStopper(patience=2)
        for epoch, score in enumerate([0.5, 0.4, 0.6, 0.5], start=1):
            stopper.update(score, epoch)
        self.assertFalse(stopper.should_stop)                             # the counter resets on improvement
        with self.assertRaises(ValueError):
            nt.EarlyStopper(0)


class TrainingLoopTests(unittest.TestCase):
    def run_train(self, variant, root, seed=42, epochs=3):
        config = nt.TrainConfig(variant, learning_rate=5e-3, batch_size=4, max_epochs=epochs, seed=seed)
        run_dir = nt.new_run_dir(root, f"{variant}-{seed}")
        result = nt.train(config, tiny_model(), FakeTokenizer(), ROWS, VAL, "cpu", run_dir, progress=lambda *a: None)
        return result, run_dir

    def test_train_checkpoint_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            result, run_dir = self.run_train("full", directory)
            self.assertTrue((run_dir / "best" / "model.safetensors").is_file())
            self.assertTrue((run_dir / "best" / "config.json").is_file())
            self.assertTrue((run_dir / "validation_predictions.json").is_file())
            self.assertEqual(len(result["history"]), result["epochs_run"])
            self.assertIn(result["best_epoch"], [h["epoch"] for h in result["history"]])
            best = max(result["history"], key=lambda h: h["validation_macro_f1"])
            self.assertEqual(result["validation"]["macro_f1"], best["validation_macro_f1"])
            for key in ("train_loss", "validation_loss", "validation_accuracy", "validation_false_supported_rate"):
                self.assertIn(key, result["history"][0])
            self.assertIsNotNone(result["validation"]["latency_ms_per_example"])
            reloaded, _ = nt.load_checkpoint(run_dir / "best", tokenizer=FakeTokenizer())
            self.assertEqual(reloaded.config.id2label, ID2LABEL)
            self.assertTrue(all(isinstance(v, torch.Tensor) for v in reloaded.state_dict().values()))

    def test_saved_checkpoint_matches_trained_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            model = tiny_model()
            nt.save_checkpoint(model, FakeTokenizer(), Path(directory) / "best")
            reloaded, _ = nt.load_checkpoint(Path(directory) / "best", tokenizer=FakeTokenizer())
            for (name, a), (_, b) in zip(model.state_dict().items(), reloaded.state_dict().items()):
                self.assertTrue(torch.equal(a, b), name)

    def test_reproducible_with_same_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            first, _ = self.run_train("full", directory, seed=7, epochs=2)
            second, _ = self.run_train("full", Path(directory) / "again", seed=7, epochs=2)
            self.assertEqual([h["train_loss"] for h in first["history"]], [h["train_loss"] for h in second["history"]])
            self.assertEqual(first["validation"]["accuracy"], second["validation"]["accuracy"])

    def test_frozen_training_leaves_encoder_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            model = tiny_model()
            encoder = {n: p.detach().clone() for n, p in model.roberta.named_parameters()}
            config = nt.TrainConfig("frozen", learning_rate=1e-2, batch_size=4, max_epochs=2)
            nt.train(config, model, FakeTokenizer(), ROWS, VAL, "cpu", Path(directory) / "r", progress=lambda *a: None)
            self.assertTrue(all(torch.equal(encoder[n], p) for n, p in model.roberta.named_parameters()))

    def test_run_directories_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            nt.new_run_dir(directory, "run")
            with self.assertRaises(nt.ExperimentError):
                nt.new_run_dir(directory, "run")

    def test_variant_configs(self):
        frozen, full = nt.VARIANTS["frozen"], nt.VARIANTS["full"]
        self.assertEqual((frozen.learning_rate, frozen.batch_size, frozen.max_epochs, frozen.patience, frozen.seed),
                         (1e-3, 16, 10, 2, 42))
        self.assertEqual((full.learning_rate, full.batch_size, full.max_epochs, full.patience, full.seed),
                         (2e-5, 8, 5, 2, 42))


class SyntheticFreeze(Workspace):
    """A synthetic experimental freeze in a temporary directory (never the real test split)."""

    def setUp(self):
        super().setUp()
        sample = nr.stratified_sample(self.examples, 10)
        by_id = {e["example_id"]: e for e in self.examples}
        for example_id in sample["example_ids"]:
            self.decide(by_id[example_id])
        nr.freeze_experimental("v0.1-exp", self.dataset, self.reviews, self.frozen, self.facts, documents=DOCUMENTS,
                               now=NOW, sample_size=10)

    def cli(self, *argv):
        output = []
        loader = lambda: (tiny_model(), FakeTokenizer())     # noqa: E731
        with mock.patch("src.nli_review.load_frozen", wraps=nr.load_frozen) as spy:
            code = nt.main(["--frozen-dir", str(self.frozen), "--artifacts-dir", str(Path(self.tmp.name) / "art"),
                            "--device", "cpu", *argv], model_loader=loader, progress=output.append)
        self.assertNotIn("test", [call.args[1] for call in spy.call_args_list])
        return code, "\n".join(output)


class FrozenDataTests(SyntheticFreeze):
    """The M6.3 loaders and CLI on top of the synthetic freeze."""

    def test_loading_and_test_split_protection(self):
        train, manifest = nt.load_split("train", "v0.1-exp", self.frozen)
        validation, _ = nt.load_split("validation", "v0.1-exp", self.frozen)
        self.assertEqual(len(train), manifest["counts"]["by_split"]["train"])
        self.assertEqual(len(validation), manifest["counts"]["by_split"]["validation"])
        with self.assertRaises(nt.ExperimentError):
            nt.load_split("test", "v0.1-exp", self.frozen)
        info = nt.dataset_info("v0.1-exp", self.frozen, manifest, {"train": train, "validation": validation})
        self.assertEqual(set(info["file_sha256"]), {"train.jsonl", "validation.jsonl"})
        self.assertEqual(len(info["manifest_sha256"]), 64)

    def test_cli_end_to_end_without_test_split(self):
        code, out = self.cli("--baseline", "--run-name", "v1")
        self.assertEqual(code, 0)
        self.assertIn("V1 pretrained - validation", out)
        record = json.loads((Path(self.tmp.name) / "art" / "v1" / "metrics.json").read_text())
        self.assertEqual(record["parameters"]["trainable"], 0)
        self.assertEqual(record["dataset"]["version"], "v0.1-exp")
        with mock.patch.dict(nt.VARIANTS, {"frozen": nt.TrainConfig("frozen", 5e-3, 4, 2)}):
            code, out = self.cli("--train", "frozen", "--run-name", "v2")
        self.assertEqual(code, 0)
        run = Path(self.tmp.name) / "art" / "v2"
        record = json.loads((run / "metrics.json").read_text())
        for key in ("config", "history", "best_epoch", "training_seconds", "parameters", "environment", "dataset",
                    "revision", "trainable_parameter_names"):
            self.assertIn(key, record)
        self.assertEqual(record["config"]["seed"], 42)
        self.assertEqual(record["environment"]["device"], "cpu")
        self.assertTrue(all(n.startswith("classifier.") for n in record["trainable_parameter_names"]))
        code, out = self.cli("--evaluate", str(run))
        self.assertIn("(re-evaluated)", out)
        code, out = self.cli("--compare")
        self.assertIn("cue-only rule (recomputed)", out)
        self.assertIn("V2 frozen encoder", out)
        self.assertIn("0.758", out)
        with self.assertRaises(nt.ExperimentError):
            self.cli("--baseline", "--run-name", "v1")                     # never overwrite results

    def test_no_test_option(self):
        parser = nt.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--evaluate", "x", "--split", "test"])
        self.assertNotIn("--split", parser.format_help())



def _digests(directory):
    return {str(f.relative_to(directory)): f.read_bytes() for f in sorted(Path(directory).rglob("*")) if f.is_file()}


class TestCommandTests(SyntheticFreeze):
    """--test RUN_DIR on a synthetic freeze: validation, isolation, artifacts, overwrite protection, no training."""

    def setUp(self):
        super().setUp()
        self.art = Path(self.tmp.name) / "art"
        with mock.patch.dict(nt.VARIANTS, {"frozen": nt.TrainConfig("frozen", 5e-3, 4, 2)}):
            self.cli("--train", "frozen", "--run-name", "v2")
        self.run = self.art / "v2"

    def run_test_cli(self, *argv):
        output = []
        loader = lambda d: nt.load_checkpoint(d, tokenizer=FakeTokenizer())     # noqa: E731
        forbidden = lambda *a, **k: self.fail("training code was called by --test")   # noqa: E731
        with mock.patch("src.nli_review.load_frozen", wraps=nr.load_frozen) as spy, \
                mock.patch.object(nt, "evaluate", wraps=nt.evaluate) as evaluate_spy, \
                mock.patch.object(nt, "train", side_effect=forbidden), \
                mock.patch.object(nt, "run_training", side_effect=forbidden), \
                mock.patch.object(nt, "run_baseline", side_effect=forbidden), \
                mock.patch.object(nt, "configure_trainable", side_effect=forbidden), \
                mock.patch.object(nt, "save_checkpoint", side_effect=forbidden), \
                mock.patch("torch.optim.AdamW", side_effect=forbidden), \
                mock.patch("torch.Tensor.backward", side_effect=forbidden):
            try:
                code = nt.main(["--frozen-dir", str(self.frozen), "--device", "cpu", *argv],
                               progress=output.append, checkpoint_loader=loader)
            finally:
                self.load_calls = [(c.args[1], c.kwargs) for c in spy.call_args_list]
                self.evaluate_calls = evaluate_spy.call_args_list
        return code, "\n".join(output)

    def test_parser(self):
        parser = nt.build_parser()
        self.assertEqual(parser.parse_args(["--test", "runs/x"]).test, "runs/x")
        for other in (["--baseline"], ["--train", "full"], ["--evaluate", "y"], ["--compare"]):
            with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
                parser.parse_args(["--test", "runs/x", *other])
        self.assertIn("held-out", parser.format_help())
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            parser.parse_args(["--tes", "runs/x"])                          # no prefix abbreviation

    def test_test_split_protection(self):
        with self.assertRaises(nt.ExperimentError):
            nt.load_split("test", "v0.1-exp", self.frozen)                  # training loader still refuses it
        with self.assertRaises(ValueError):
            nr.load_frozen("v0.1-exp", "test", self.frozen)                 # so does the default frozen loader
        with mock.patch("src.nli_review.load_frozen", wraps=nr.load_frozen) as spy:
            rows, manifest = nt.load_test_split("v0.1-exp", self.frozen)
        self.assertEqual(spy.call_args.args[1], "test")
        self.assertTrue(spy.call_args.kwargs["allow_test"])
        self.assertEqual(len(rows), manifest["counts"]["by_split"]["test"])
        self.assertTrue(rows and all(r["split"] == "test" for r in rows))

    def test_end_to_end(self):
        before = _digests(self.run)
        code, out = self.run_test_cli("--test", str(self.run))
        self.assertEqual(code, 0)
        self.assertEqual(self.load_calls, [("test", {"allow_test": True})])    # train/validation not even loaded
        self.assertEqual(len(self.evaluate_calls), 1)                         # one pass, no selection
        test_rows, _ = nt.load_test_split("v0.1-exp", self.frozen)
        self.assertEqual([r["example_id"] for r in self.evaluate_calls[0].args[1]],
                         [r["example_id"] for r in test_rows])
        self.assertIn("- test (", out)
        self.assertIn("Do not retune", out)
        after = _digests(self.run)
        for name, data in before.items():                                   # validation artifacts untouched
            self.assertEqual(after[name], data, name)
        self.assertEqual(set(after) - set(before), set(nt.TEST_ARTIFACTS))
        result = json.loads((self.run / "test_metrics.json").read_text())
        predictions = [json.loads(line) for line in (self.run / "test_predictions.jsonl").read_text().splitlines()]
        training = json.loads((self.run / "metrics.json").read_text())
        self.assertEqual(result["split"], "test")
        self.assertEqual(result["run_id"], "v2")
        self.assertEqual(result["checkpoint"], str(self.run / "best"))
        self.assertEqual(result["test_examples"], len(test_rows))
        self.assertEqual(result["dataset"]["version"], "v0.1-exp")
        self.assertEqual(set(result["dataset"]["file_sha256"]), {"test.jsonl"})
        self.assertEqual(result["training_dataset"], training["dataset"])
        self.assertEqual(result["selected_best_epoch"], training["best_epoch"])
        self.assertEqual(result["preprocessing"]["max_length"], training["config"]["max_length"])
        self.assertEqual(result["label_ids"], training["label_ids"])
        self.assertIn("evaluated_at", result)
        self.assertIn("model.safetensors", result["checkpoint_sha256"])
        m = result["test"]
        for key in ("accuracy", "macro_f1", "per_class", "confusion", "false_supported", "false_supported_rate",
                    "loss", "latency_ms_per_example", "by_format", "by_family", "by_label_source"):
            self.assertIn(key, m)
        self.assertEqual(len(predictions), len(test_rows))
        gold = [p["gold"] for p in predictions]
        predicted = [p["predicted"] for p in predictions]
        self.assertEqual(gold, [r["label"] for r in test_rows])
        not_supported = [p for g, p in zip(gold, predicted) if g != "SUPPORTED"]
        self.assertEqual(m["false_supported"], sum(p == "SUPPORTED" for p in not_supported))
        self.assertAlmostEqual(m["false_supported_rate"], m["false_supported"] / len(not_supported))
        self.assertEqual(m["accuracy"], sum(g == p for g, p in zip(gold, predicted)) / len(gold))

        # A second run is refused before any test data is read, and nothing changes.
        snapshot = _digests(self.run)
        with self.assertRaisesRegex(nt.ExperimentError, "never overwritten"):
            self.run_test_cli("--test", str(self.run))
        self.assertEqual(self.load_calls, [])
        self.assertEqual(_digests(self.run), snapshot)
        (self.run / "test_metrics.json").unlink()                       # either file alone also blocks a rerun
        with self.assertRaisesRegex(nt.ExperimentError, "test_predictions.jsonl"):
            self.run_test_cli("--test", str(self.run))

    def test_run_directory_validation(self):
        def refused(run_dir, pattern, version="v0.1-exp"):
            with self.assertRaisesRegex(nt.ExperimentError, pattern):
                nt.check_test_run(run_dir, version, self.frozen)

        refused(self.art / "missing", "not a run directory")
        nt.check_test_run(self.run, "v0.1-exp", self.frozen)                   # the real run passes
        refused(self.run, "trained on", version="v0.2-exp")
        metrics = json.loads((self.run / "metrics.json").read_text())
        for change, pattern in ((dict(variant="pretrained"), "completed V2/V3"), (dict(status="running"), "completed"),
                                (dict(dataset={**metrics["dataset"], "manifest_sha256": "0" * 64}), "no longer matches")):
            (self.run / "metrics.json").write_text(json.dumps({**metrics, **change}))
            refused(self.run, pattern)
        (self.run / "metrics.json").unlink()
        refused(self.run, "metrics.json is missing")
        self.cli("--baseline", "--run-name", "v1")                         # V1 has no best/ checkpoint
        refused(self.art / "v1", "does not contain a saved checkpoint")
        (self.run / "best" / "model.safetensors").unlink()
        refused(self.run, "does not contain a saved checkpoint")

    def test_invalid_run_never_loads_test_data(self):
        (self.run / "best" / "model.safetensors").unlink()
        with self.assertRaises(nt.ExperimentError):
            self.run_test_cli("--test", str(self.run))
        self.assertEqual(self.load_calls, [])

    def test_no_writes_if_parameters_change(self):
        record = nt.check_test_run(self.run, "v0.1-exp", self.frozen)
        rows, manifest = nt.load_test_split("v0.1-exp", self.frozen)
        info = nt.dataset_info("v0.1-exp", self.frozen, manifest, {"test": rows})
        real_evaluate = nt.evaluate

        def tampering_evaluate(model, *args, **kwargs):
            result = real_evaluate(model, *args, **kwargs)
            with torch.no_grad():
                model.classifier.out_proj.bias.add_(1.0)
            return result

        loader = lambda d: nt.load_checkpoint(d, tokenizer=FakeTokenizer())     # noqa: E731
        with mock.patch.object(nt, "evaluate", tampering_evaluate), \
                self.assertRaisesRegex(nt.ExperimentError, "parameters changed"):
            nt.run_test(self.run, record, rows, info, "cpu", {}, lambda *a: None, loader)
        for name in nt.TEST_ARTIFACTS:
            self.assertFalse((self.run / name).exists())

    def test_model_is_frozen_and_in_eval_mode(self):
        record = nt.check_test_run(self.run, "v0.1-exp", self.frozen)
        rows, manifest = nt.load_test_split("v0.1-exp", self.frozen)
        info = nt.dataset_info("v0.1-exp", self.frozen, manifest, {"test": rows})
        seen = {}

        def checking_evaluate(model, *args, **kwargs):
            seen["requires_grad"] = any(p.requires_grad for p in model.parameters())
            seen["training"] = model.training
            seen["grad_enabled"] = torch.is_grad_enabled()
            return nt.evaluate.__wrapped__(model, *args, **kwargs)

        checking_evaluate.__wrapped__ = nt.evaluate
        loader = lambda d: nt.load_checkpoint(d, tokenizer=FakeTokenizer())     # noqa: E731
        with mock.patch.object(nt, "evaluate", checking_evaluate):
            nt.run_test(self.run, record, rows, info, "cpu", {}, lambda *a: None, loader)
        self.assertEqual(seen, {"requires_grad": False, "training": False, "grad_enabled": True})

    def test_false_supported_definition(self):
        gold = ["CONTRADICTED", "INSUFFICIENT_EVIDENCE", "SUPPORTED", "CONTRADICTED", "SUPPORTED"]
        pred = ["SUPPORTED", "SUPPORTED", "CONTRADICTED", "CONTRADICTED", "SUPPORTED"]
        m = nt.classification_metrics(gold, pred)
        self.assertEqual(m["false_supported"], 2)                          # gold non-SUPPORTED predicted SUPPORTED
        self.assertAlmostEqual(m["false_supported_rate"], 2 / 3)           # ... over all gold non-SUPPORTED


class RealFrozenDatasetTests(unittest.TestCase):
    """Read-only checks against the real frozen v0.1-exp (train and validation only)."""

    @classmethod
    def setUpClass(cls):
        if not (nr.FROZEN_DIR / "v0.1-exp" / "manifest.json").is_file():
            raise unittest.SkipTest("frozen v0.1-exp not present")
        cls.train, cls.manifest = nt.load_split("train")
        cls.validation, _ = nt.load_split("validation")

    def test_counts_labels_and_sources(self):
        self.assertEqual((len(self.train), len(self.validation)), (884, 190))
        self.assertEqual(self.manifest["dataset_type"], nr.EXPERIMENTAL_TYPE)
        self.assertEqual({r["label"] for r in self.train}, set(nd.LABEL_VALUES))
        self.assertEqual({r["split"] for r in self.validation}, {"validation"})
        self.assertFalse({r["family_id"] for r in self.train} & {r["family_id"] for r in self.validation})
        reviewed = [r for r in self.validation if r["review"]["status"] == "reviewed"]
        self.assertTrue(all(r["review"]["human_label"] == r["label"] for r in reviewed))

    def test_cue_baseline_recomputed(self):
        cue = nt.cue_baseline(self.train, self.validation)
        self.assertEqual(cue["n"], 190)
        self.assertGreater(cue["accuracy"], cue["majority_class_accuracy"])


if __name__ == "__main__":
    unittest.main()
