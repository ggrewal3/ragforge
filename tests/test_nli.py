"""Tests for M6.1: local NLI inference, label mapping, aggregation, metrics and the evaluation CLI.

The fast tests use a fake tokenizer and a fake model built on real torch
tensors, so no weights are downloaded and no network is used. One test loads
the real checkpoint from the local Hugging Face cache, only with
RAGFORGE_MODEL_TESTS=1 (it never downloads: local_files_only=True).
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from src import evaluate_local_nli as evalnli
from src import evaluate_verification as evalver
from src.claims import SentenceClaimExtractor
from src.citations import GroundedAnswer
from src.nli import (
    DEFAULT_NLI_MODEL,
    LocalNLI,
    NLIModelError,
    NLIPrediction,
    NLIVerifier,
    aggregate_verdict,
    check_logits,
    label_mapping,
    premise_text,
    select_device,
    softmax,
)
from src.self_correction import verify_and_correct
from src.verification import parse_verdicts
from test_generation import chunk, run_cli
from test_verification import RoutedLLM

RUN_MODEL_TESTS = os.environ.get("RAGFORGE_MODEL_TESTS") == "1"
CHECKPOINT_LABELS = {0: "contradiction", 1: "entailment", 2: "neutral"}   # cross-encoder/nli-MiniLM2-L6-H768


class FakeTokenizer:
    """Mimics a Hugging Face tokenizer called on (premises, hypotheses). One token per word plus 3 specials."""

    model_max_length = 512

    def __init__(self):
        self.calls = []
        self.last_batch = None

    def __call__(self, premises, hypotheses, truncation=False, max_length=None, padding=False, return_tensors=None):
        self.calls.append({"truncation": truncation, "max_length": max_length, "padding": padding,
                           "return_tensors": return_tensors, "size": len(premises)})
        lengths = [len(p.split()) + len(h.split()) + 3 for p, h in zip(premises, hypotheses)]
        if return_tensors != "pt":
            return {"input_ids": [[0] * n for n in lengths]}
        self.last_batch = list(zip(premises, hypotheses))
        width = min(max(lengths), max_length or max(lengths))
        return {"input_ids": torch.zeros((len(premises), width), dtype=torch.long),
                "attention_mask": torch.ones((len(premises), width), dtype=torch.long)}


def keyword_rule(premise, hypothesis):
    """Semantic logits for the fake model: 'not' in the claim -> contradiction, shared words -> entailment."""
    if " not " in f" {hypothesis.lower()} ":
        return {"contradiction": 3.0, "entailment": -1.0, "neutral": 0.0}
    if set(hypothesis.lower().rstrip(".").split()) <= set(premise.lower().rstrip(".").split()):
        return {"contradiction": -2.0, "entailment": 2.5, "neutral": 0.5}
    return {"contradiction": -1.0, "entailment": -0.5, "neutral": 2.0}


class FakeNLIModel:
    """Returns logits in the order its id2label declares, computed by `rule` from the tokenizer's last batch."""

    def __init__(self, tokenizer, id2label=None, rule=keyword_rule, output=None):
        self.config = SimpleNamespace(id2label=id2label or dict(CHECKPOINT_LABELS))
        self.tokenizer, self.rule, self.output = tokenizer, rule, output
        self.calls, self.device, self.eval_called, self.inference_mode = 0, None, False, []

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, **encoded):
        self.calls += 1
        self.inference_mode.append(torch.is_inference_mode_enabled())
        if self.output is not None:
            return SimpleNamespace(logits=torch.tensor(self.output, dtype=torch.float32))
        order = [str(self.config.id2label[i]).lower() for i in range(3)]
        rows = [[self.rule(p, h)[label] for label in order] for p, h in self.tokenizer.last_batch]
        return SimpleNamespace(logits=torch.tensor(rows, dtype=torch.float32))


def fake_nli(id2label=None, output=None, batch_size=16, max_length=512, device="cpu", rule=keyword_rule):
    tokenizer = FakeTokenizer()
    model = FakeNLIModel(tokenizer, id2label=id2label, output=output, rule=rule)
    return LocalNLI("fake/nli", revision="test", device=device, batch_size=batch_size, max_length=max_length,
                    tokenizer=tokenizer, model=model), tokenizer, model


class LabelMappingTests(unittest.TestCase):
    def test_checkpoint_mapping_and_string_keys(self):
        self.assertEqual(label_mapping({"0": "contradiction", "1": "entailment", "2": "neutral"}), CHECKPOINT_LABELS)
        self.assertEqual(label_mapping({0: "ENTAILMENT", 1: "Neutral", 2: "contradiction"}),
                         {0: "entailment", 1: "neutral", 2: "contradiction"})

    def test_unsupported_mappings_rejected(self):
        cases = {
            "generic labels": {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},
            "two classes": {0: "entailment", 1: "not_entailment"},
            "four classes": {0: "entailment", 1: "neutral", 2: "contradiction", 3: "other"},
            "duplicate class": {0: "entailment", 1: "entailment", 2: "neutral"},
            "index gap": {0: "entailment", 1: "neutral", 3: "contradiction"},
            "non-integer index": {"a": "entailment", 1: "neutral", 2: "contradiction"},
            "not a dict": ["entailment", "neutral", "contradiction"],
        }
        for name, id2label in cases.items():
            with self.subTest(name), self.assertRaises(NLIModelError):
                label_mapping(id2label)
        with self.assertRaises(NLIModelError):
            fake_nli(id2label={0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})

    def test_verdict_follows_metadata_not_index_order(self):
        """The same semantic output gives the same verdict whatever class order the checkpoint uses."""
        for id2label in (CHECKPOINT_LABELS, {0: "entailment", 1: "neutral", 2: "contradiction"},
                         {0: "neutral", 1: "contradiction", 2: "entailment"}):
            with self.subTest(id2label):
                nli, _, _ = fake_nli(id2label=id2label)
                self.assertEqual(nli.predict("Refunds take 45 days.", "Refunds take 45 days.").verdict, "SUPPORTED")
                self.assertEqual(nli.predict("Refunds take 45 days.", "Refunds are not allowed.").verdict,
                                 "CONTRADICTED")
                self.assertEqual(nli.predict("Refunds take 45 days.", "Shipping is free.").verdict,
                                 "INSUFFICIENT_EVIDENCE")


class InferenceTests(unittest.TestCase):
    def test_softmax(self):
        probs = softmax([[2.0, 1.0, 0.0], [1000.0, 0.0, -1000.0]])
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0])
        expected = np.exp([2.0, 1.0, 0.0]) / np.exp([2.0, 1.0, 0.0]).sum()
        np.testing.assert_allclose(probs[0], expected)
        np.testing.assert_allclose(probs[1], [1.0, 0.0, 0.0], atol=1e-12)   # no overflow

    def test_prediction_fields(self):
        nli, _, model = fake_nli()
        prediction = nli.predict("Refunds take 45 days.", "Refunds take 45 days.", premise_id="REF-729.1")
        self.assertEqual(prediction.labels, ["contradiction", "entailment", "neutral"])
        self.assertEqual(prediction.logits, [-2.0, 2.5, 0.5])
        self.assertEqual(prediction.nli_label, "entailment")
        self.assertEqual(prediction.verdict, "SUPPORTED")
        expected = softmax([[-2.0, 2.5, 0.5]])[0]
        self.assertAlmostEqual(prediction.probabilities["entailment"], expected[1], places=6)
        self.assertAlmostEqual(sum(prediction.probabilities.values()), 1.0, places=6)
        self.assertEqual((prediction.model, prediction.device, prediction.premise_id),
                         ("fake/nli", "cpu", "REF-729.1"))
        self.assertGreaterEqual(prediction.latency_s, 0.0)
        self.assertTrue(model.eval_called)
        self.assertEqual(model.device, "cpu")
        self.assertEqual(model.inference_mode, [True])

    def test_invalid_model_output(self):
        for output in ([[1.0, 2.0]], [[float("nan"), 0.0, 1.0]], [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]):
            with self.subTest(output):
                nli, _, _ = fake_nli(output=output)
                with self.assertRaises(NLIModelError):
                    nli.predict("Premise.", "Hypothesis.")
        with self.assertRaises(NLIModelError):
            check_logits(np.zeros((2, 3)), expected_rows=3)

    def test_empty_evidence_and_claim(self):
        nli, _, model = fake_nli()
        for premise, hypothesis in (("", "Claim."), ("   ", "Claim."), ("Evidence.", ""), ("Evidence.", None)):
            with self.subTest((premise, hypothesis)), self.assertRaises(ValueError):
                nli.predict(premise, hypothesis)
        with self.assertRaises(ValueError):
            nli.classify_claim("A claim.", [])
        with self.assertRaises(ValueError):
            nli.classify_claim(" ", [chunk("REF-729.1")])
        self.assertEqual(model.calls, 0)
        self.assertEqual(nli.predict_pairs([]), [])

    def test_batch_inference(self):
        nli, tokenizer, model = fake_nli(batch_size=2)
        pairs = [(f"Rule {n} applies.", f"Rule {n} applies.") for n in range(5)]
        predictions = nli.predict_pairs(pairs, premise_ids=[f"S{n}" for n in range(5)])
        self.assertEqual(model.calls, 3)                                     # 2 + 2 + 1
        self.assertEqual([p.premise_id for p in predictions], ["S0", "S1", "S2", "S3", "S4"])
        encode_calls = [c for c in tokenizer.calls if c["return_tensors"] == "pt"]
        self.assertEqual([c["size"] for c in encode_calls], [2, 2, 1])
        self.assertTrue(all(c["truncation"] == "only_first" and c["max_length"] == 512 and c["padding"]
                            for c in encode_calls))
        with self.assertRaises(ValueError):
            nli.predict_pairs(pairs, premise_ids=["S0"])

    def test_truncation_is_recorded_and_length_capped_by_tokenizer(self):
        nli, _, _ = fake_nli(max_length=20)
        long_premise = " ".join(["word"] * 40) + "."
        prediction = nli.predict(long_premise, "A short claim.")
        self.assertTrue(prediction.truncated)
        self.assertEqual(prediction.input_tokens, 20)
        self.assertFalse(nli.predict("Short.", "Short claim.").truncated)
        nli, _, _ = fake_nli(max_length=4096)
        self.assertEqual(nli.max_length, 512)

    def test_configuration_validation(self):
        for kwargs in ({"batch_size": 0}, {"batch_size": True}, {"max_length": 8}):
            with self.subTest(kwargs), self.assertRaises(ValueError):
                fake_nli(**kwargs)

    def test_serialization(self):
        nli, _, _ = fake_nli()
        prediction = nli.predict("Refunds take 45 days.", "Refunds take 45 days.")
        restored = json.loads(json.dumps(prediction.to_dict()))
        self.assertEqual(NLIPrediction(**restored), prediction)


class DeviceTests(unittest.TestCase):
    def test_selection_and_fallback(self):
        self.assertEqual(select_device("auto", mps_available=True, cuda_available=False), ("mps", None))
        self.assertEqual(select_device("auto", mps_available=False, cuda_available=False), ("cpu", None))
        device, note = select_device("mps", mps_available=False, cuda_available=False)
        self.assertEqual(device, "cpu")
        self.assertIn("not available", note)
        self.assertEqual(select_device("cpu", mps_available=True, cuda_available=True), ("cpu", None))
        with self.assertRaises(ValueError):
            select_device("tpu", mps_available=False, cuda_available=False)

    def test_model_moved_to_fallback_device(self):
        with mock.patch("torch.backends.mps.is_available", return_value=False), \
                mock.patch("torch.cuda.is_available", return_value=False):
            nli, _, model = fake_nli(device="mps")
        self.assertEqual((nli.device, model.device), ("cpu", "cpu"))
        self.assertIn("mps requested", nli.device_note)


def prediction(verdict, section_id, probability=0.9):
    label = {"SUPPORTED": "entailment", "CONTRADICTED": "contradiction", "INSUFFICIENT_EVIDENCE": "neutral"}[verdict]
    probs = {"entailment": 0.05, "contradiction": 0.05, "neutral": 0.05, label: probability}
    return NLIPrediction(section_id, [0.0, 0.0, 0.0], ["contradiction", "entailment", "neutral"], probs, label,
                         verdict, "fake/nli", "cpu", 10, False, 0.0)


class AggregationTests(unittest.TestCase):
    def test_contradiction_first(self):
        verdict, ids, explanation = aggregate_verdict([prediction("SUPPORTED", "REF-728.1"),
                                                       prediction("CONTRADICTED", "REF-729.1", 0.7)])
        self.assertEqual((verdict, ids), ("CONTRADICTED", ["REF-729.1"]))
        self.assertIn("REF-729.1", explanation)

    def test_entailment_then_neutral(self):
        verdict, ids, _ = aggregate_verdict([prediction("INSUFFICIENT_EVIDENCE", "A-1.1"),
                                             prediction("SUPPORTED", "A-1.2"), prediction("SUPPORTED", "A-1.2")])
        self.assertEqual((verdict, ids), ("SUPPORTED", ["A-1.2"]))
        verdict, ids, _ = aggregate_verdict([prediction("INSUFFICIENT_EVIDENCE", "A-1.1")])
        self.assertEqual((verdict, ids), ("INSUFFICIENT_EVIDENCE", []))
        with self.assertRaises(ValueError):
            aggregate_verdict([])

    def test_each_section_scored_separately_with_exact_ids(self):
        nli, tokenizer, _ = fake_nli()
        evidence = [chunk("REF-728.1", "Domestic refunds take 30 days.", "Domestic Refund Policy", "Eligibility"),
                    chunk("REF-729.1", "International refunds take 45 days.", "International Refund Policy",
                          "Eligibility")]
        verdict, ids, _, predictions = nli.classify_claim("International refunds take 45 days.", evidence)
        self.assertEqual([p.premise_id for p in predictions], ["REF-728.1", "REF-729.1"])
        self.assertEqual((verdict, ids), ("SUPPORTED", ["REF-729.1"]))
        self.assertEqual(tokenizer.last_batch[1][0],
                         "International Refund Policy, Eligibility: International refunds take 45 days.")
        self.assertEqual(premise_text(evidence[0]), "Domestic Refund Policy, Eligibility: Domestic refunds take 30 days.")

    def test_m5_verifier_interface(self):
        nli, _, _ = fake_nli()
        verifier = NLIVerifier(nli)
        sources = [chunk("REF-729.1", "International refunds take 45 days.", "Refunds", "Rules")]
        answer = GroundedAnswer(answer="International refunds take 45 days.", citations=["REF-729.1"],
                                abstained=False)
        result = verify_and_correct("q?", sources, answer, SentenceClaimExtractor(), verifier, RoutedLLM())
        self.assertEqual(result["outcome"], "accepted_original")
        self.assertEqual(result["verifier"], "nli:fake/nli")
        self.assertEqual(result["llm_calls"], 0)
        from src.claims import Claim

        output = verifier.verify([Claim("C1", "Refunds are not allowed.")], sources)
        self.assertEqual(parse_verdicts(output.payload, [Claim("C1", "x")], ["REF-729.1"])[0].verdict,
                         "CONTRADICTED")
        self.assertEqual(len(verifier.details[0]), 1)


class MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples, cls.sections = evalver.load_benchmark()

    def rows(self, predict):
        return [{"example_id": e["example_id"], "verifier": "x", "model": "fake/nli", "revision": "r",
                 "device": "cpu", "status": "complete", "verdict": predict(e), "evidence_section_ids": [],
                 "latency_s": 0.01, "usage": {}, "estimated_cost_usd": None, "sections": []}
                for e in self.examples]

    def test_false_supported_count_and_rate(self):
        summary = evalnli.score(self.rows(lambda e: "SUPPORTED"), self.examples)
        not_supported = sum(e["label"] != "SUPPORTED" for e in self.examples)
        self.assertEqual(summary["not_supported_examples"], not_supported)
        self.assertEqual(summary["false_supported_count"], not_supported)
        self.assertEqual(summary["false_supported_rate"], 1.0)
        # Never predicting SUPPORTED: no false SUPPORTED, and SUPPORTED precision/F1 count as 0.
        summary = evalnli.score(self.rows(lambda e: "CONTRADICTED"), self.examples)
        self.assertEqual((summary["false_supported_count"], summary["false_supported_rate"]), (0, 0.0))
        self.assertEqual(summary["per_verdict"]["SUPPORTED"]["f1"], 0.0)

    def test_perfect_predictions(self):
        summary = evalnli.score(self.rows(lambda e: e["label"]), self.examples)
        self.assertEqual((summary["accuracy_completed"], summary["macro_f1"]), (1.0, 1.0))
        self.assertEqual(summary["false_supported_count"], 0)

    def test_macro_f1_by_hand(self):
        flip = {"V001": "CONTRADICTED", "V002": "SUPPORTED"}   # one SUPPORTED and one CONTRADICTED label swapped
        summary = evalnli.score(self.rows(lambda e: flip.get(e["example_id"], e["label"])), self.examples)
        # With one swap each way, precision = recall = (n - 1) / n for both classes, so F1 = (n - 1) / n.
        n_s, n_c = summary["label_counts"]["SUPPORTED"], summary["label_counts"]["CONTRADICTED"]
        self.assertAlmostEqual(summary["macro_f1"], ((n_s - 1) / n_s + (n_c - 1) / n_c + 1.0) / 3)
        self.assertEqual(summary["false_supported"], ["V002"])
        self.assertAlmostEqual(summary["false_supported_rate"], 1 / summary["not_supported_examples"])


class CommandLineTests(unittest.TestCase):
    def test_single_pair(self):
        nli, _, _ = fake_nli()
        code, out, _ = run_cli(evalnli.main, ["--premise", "Refunds take 45 days.", "--hypothesis",
                                              "Refunds take 45 days."], nli=nli)
        self.assertEqual(code, 0)
        for text in ("entailment -> SUPPORTED", "logits: contradiction -2.000, entailment +2.500",
                     "probabilities:", "on cpu"):
            self.assertIn(text, out)
        code, _, err = run_cli(evalnli.main, ["--premise", "x"], nli=nli)
        self.assertEqual(code, 2)
        self.assertIn("--hypothesis", err)

    def test_example_and_interactive(self):
        nli, _, _ = fake_nli()
        code, out, _ = run_cli(evalnli.main, ["--example", "V004"], nli=nli)
        self.assertEqual(code, 0)
        self.assertIn("[REF-728.2]", out)
        self.assertIn("[REF-729.2]", out)
        self.assertIn("Draft label: CONTRADICTED", out)
        code, _, err = run_cli(evalnli.main, ["--example", "V999"], nli=nli)
        self.assertEqual(code, 2)
        answers = iter(["Refunds take 45 days.", "Refunds take 45 days.", " ", "x", "exit"])
        code, out, _ = run_cli(evalnli.main, ["--interactive"], nli=nli, input_fn=lambda prompt: next(answers))
        self.assertEqual(code, 0)
        self.assertIn("entailment -> SUPPORTED", out)
        self.assertIn("Error:", out)

    def test_benchmark_run_writes_label_free_predictions(self):
        nli, _, _ = fake_nli()
        with tempfile.TemporaryDirectory() as directory:
            out_path = Path(directory, "nli.jsonl")
            code, out, _ = run_cli(evalnli.main, ["--out", str(out_path)], nli=nli)
            self.assertEqual(code, 0)
            rows = [json.loads(line) for line in out_path.read_text().splitlines()]
            code, summary_out, _ = run_cli(evalnli.main, ["--summarize", str(out_path)])
        self.assertEqual(len(rows), 40)
        self.assertTrue(all("label" not in row and "rationale" not in row for row in rows))
        self.assertTrue(all(len(row["sections"]) == len(row_example["evidence_section_ids"])
                            for row, row_example in zip(rows, evalver.load_benchmark()[0])))
        for text in ("zero-shot, not fine-tuned", "Macro-F1", "False SUPPORTED", "Labels: draft", "Device: cpu"):
            self.assertIn(text, out)
        self.assertIn("Macro-F1", summary_out)


@unittest.skipUnless(RUN_MODEL_TESTS, "set RAGFORGE_MODEL_TESTS=1 to load the cached NLI checkpoint")
class RealCheckpointTests(unittest.TestCase):
    def test_label_metadata_and_brief_examples(self):
        try:
            nli = LocalNLI(DEFAULT_NLI_MODEL, local_files_only=True)
        except OSError:
            self.skipTest("NLI checkpoint not in the local cache; run python -m src.evaluate_local_nli once.")
        self.assertEqual(nli.labels, CHECKPOINT_LABELS)
        self.assertEqual(nli.predict("Employees become eligible after six months.",
                                     "Employees become eligible after three months.").verdict, "CONTRADICTED")
        self.assertEqual(nli.predict("International customers may request refunds within 45 days.",
                                     "International customers have 45 days to request refunds.").verdict, "SUPPORTED")


if __name__ == "__main__":
    unittest.main()
