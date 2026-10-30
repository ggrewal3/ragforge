"""Tests for M6.4A: the local V3 NLI verifier, its decision rules and traces, verification-only mode and the CLI.

Everything here is offline. Inference uses the fake tokenizer and fake NLI model
from test_nli (one token per word; logits chosen by a rule over the actual
premise and claim). Checkpoint loading uses synthetic run directories in a
temporary folder with injected loaders, so no weights are read or downloaded.
Gemini is replaced by RoutedLLM. One smoke test loads the real V3 checkpoint,
only with RAGFORGE_MODEL_TESTS=1 and only if it exists locally.
"""

import hashlib
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from src import local_verifier as lv
from src.citations import GroundedAnswer
from src.claims import Claim, GeminiClaimExtractor
from src.nli import LocalNLI, premise_text, raw_premise_text
from src.rag import main as rag_main
from src.retrieve import build_search_index, search
from src.self_correction import (
    ABSTENTION_MESSAGE,
    answer_question_verified,
    evidence_fingerprint,
    evidence_snapshot,
    verify_and_correct,
)
from src.verification import GeminiVerifier, MockVerifier, VerifierUnavailable, parse_verdicts
from test_generation import chunk, run_cli
from test_hybrid import FakeModel
from test_nli import CHECKPOINT_LABELS, FakeNLIModel, FakeTokenizer, fake_nli
from test_nli_train import tiny_model
from test_verification import RoutedLLM, claims, grounded

S, C, I = "SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"
LABEL_IDS = {"CONTRADICTED": 0, "SUPPORTED": 1, "INSUFFICIENT_EVIDENCE": 2}
LOGITS = {"entailment": {"contradiction": -2.0, "entailment": 3.0, "neutral": 0.0},
          "contradiction": {"contradiction": 3.0, "entailment": -2.0, "neutral": 0.0},
          "neutral": {"contradiction": -1.0, "entailment": -1.0, "neutral": 2.0}}


def marker_rule(premise, hypothesis):
    """Fake semantics: a premise containing 'contradicts' contradicts, one containing 'supports' entails."""
    text = premise.lower()
    if "contradicts" in text:
        return LOGITS["contradiction"]
    if "supports" in text:
        return LOGITS["entailment"]
    return LOGITS["neutral"]


def entail_if(fragment):
    return lambda premise, hypothesis: LOGITS["entailment" if fragment in premise else "neutral"]


def verifier_from(nli):
    info = {"identifier": nli.model_name, "run_id": "fake-run", "device": nli.device,
            "integrity": {"files": "verified", "parameters": "not_available"}, "label_ids": LABEL_IDS,
            "max_length": nli.max_length, "score_note": lv.SCORE_NOTE, "load_s": 0.0}
    return lv.LocalNLIVerifier("fake-run", "cpu", loaded=lv.LoadedModel(nli, info))


def fake_verifier(rule=marker_rule, max_length=512, output=None, id2label=None):
    nli, tokenizer, model = fake_nli(rule=rule, max_length=max_length, output=output, id2label=id2label)
    return verifier_from(nli), tokenizer, model


def verify(verifier, claim_texts, evidence):
    claim_list = [Claim(f"C{n}", text) for n, text in enumerate(claim_texts, start=1)]
    output = verifier.verify(claim_list, evidence)
    supplied = list(dict.fromkeys(c["section_id"] for c in evidence))
    return parse_verdicts(output.payload, claim_list, supplied), output


class FakeCheckpoint:
    """A synthetic M6.3 run directory: best/ files plus metrics.json and test_metrics.json with real hashes."""

    def __init__(self, root, name="v3-fake", files=None, label_ids=None, test_metadata=True, parameter_sha256=None):
        self.run = Path(root) / name
        best = self.run / "best"
        best.mkdir(parents=True)
        for file_name, data in (files or {"config.json": b"{}", "model.safetensors": b"weights",
                                          "tokenizer.json": b"tok"}).items():
            (best / file_name).write_bytes(data)
        label_ids = label_ids or LABEL_IDS
        (self.run / "metrics.json").write_text(json.dumps({"variant": "full", "best_epoch": 2, "label_ids": label_ids,
                                                           "config": {"max_length": 512}}))
        if test_metadata:
            hashes = {p.name: lv._sha256_file(p) for p in best.iterdir()}
            record = {"checkpoint_sha256": hashes, "label_ids": label_ids, "preprocessing": {"max_length": 512}}
            if parameter_sha256:
                record["parameter_sha256"] = parameter_sha256
            (self.run / "test_metrics.json").write_text(json.dumps(record))
        self.calls = 0

    def loader(self, model=None, id2label=None):
        def load(checkpoint):
            self.calls += 1
            tokenizer = FakeTokenizer()
            return model or FakeNLIModel(tokenizer, id2label=id2label, rule=marker_rule), tokenizer
        return load


class LoadingTests(unittest.TestCase):
    def setUp(self):
        lv.clear_cache()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(lv.clear_cache)

    def test_loads_once_per_checkpoint_and_device(self):
        checkpoint = FakeCheckpoint(self.tmp.name)
        with mock.patch("torch.optim.AdamW", side_effect=AssertionError("optimizer created")):
            first = lv.load_v3(checkpoint.run, "cpu", checkpoint.loader())
            second = lv.load_v3(checkpoint.run, "cpu", checkpoint.loader())
            verifier = lv.LocalNLIVerifier(checkpoint.run, "cpu", checkpoint.loader())
            verify(verifier, ["Refunds take 30 days."], [chunk("REF-1.1", "This supports it.")])
        self.assertIs(first, second)
        self.assertIs(verifier.load(), first)
        self.assertEqual(checkpoint.calls, 1)
        self.assertTrue(first.nli.model.eval_called)
        self.assertEqual(first.nli.device, "cpu")
        self.assertEqual(first.info["integrity"]["files"], "verified")
        self.assertEqual(first.info["label_ids"], LABEL_IDS)
        self.assertEqual((first.info["variant"], first.info["best_epoch"], first.info["max_length"]), ("full", 2, 512))
        lv.clear_cache()
        lv.load_v3(checkpoint.run, "cpu", checkpoint.loader())
        self.assertEqual(checkpoint.calls, 2)

    def test_real_model_is_frozen_and_parameters_checked(self):
        from src.nli_train import parameter_fingerprint

        model = tiny_model()
        checkpoint = FakeCheckpoint(self.tmp.name, parameter_sha256=parameter_fingerprint(model))
        loaded = lv.load_v3(checkpoint.run, "cpu", checkpoint.loader(model=model))
        self.assertEqual(loaded.info["integrity"]["parameters"], "verified")
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        self.assertFalse(model.training)
        bad = FakeCheckpoint(self.tmp.name, name="bad", parameter_sha256="0" * 64)
        with self.assertRaisesRegex(VerifierUnavailable, "parameters do not match"):
            lv.load_v3(bad.run, "cpu", bad.loader(model=tiny_model()))

    def test_missing_checkpoint(self):
        checkpoint = FakeCheckpoint(self.tmp.name)
        for path in (Path(self.tmp.name) / "nowhere", checkpoint.run):
            if path == checkpoint.run:
                (checkpoint.run / "best" / "model.safetensors").unlink()
            with self.subTest(path=path.name), self.assertRaisesRegex(VerifierUnavailable, "not found"):
                lv.load_v3(path, "cpu", checkpoint.loader())
        self.assertEqual(checkpoint.calls, 0)

    def test_integrity_mismatch(self):
        cases = {"changed": lambda best: (best / "model.safetensors").write_bytes(b"tampered"),
                 "extra": lambda best: (best / "extra.bin").write_bytes(b"x"),
                 "removed": lambda best: (best / "tokenizer.json").unlink()}
        for name, damage in cases.items():
            checkpoint = FakeCheckpoint(self.tmp.name, name=name)
            damage(checkpoint.run / "best")
            with self.subTest(name), self.assertRaises(VerifierUnavailable):
                lv.load_v3(checkpoint.run, "cpu", checkpoint.loader())
            self.assertEqual(checkpoint.calls, 0)                      # refused before the weights are loaded
        finder = FakeCheckpoint(self.tmp.name, name="dsstore")
        (finder.run / "best" / ".DS_Store").write_bytes(b"finder metadata")     # hidden files are ignored
        lv.load_v3(finder.run, "cpu", finder.loader())

    def test_without_test_metadata_integrity_is_reported_unverified(self):
        checkpoint = FakeCheckpoint(self.tmp.name, test_metadata=False)
        loaded = lv.load_v3(checkpoint.run, "cpu", checkpoint.loader())
        self.assertEqual(loaded.info["integrity"]["files"], "not_available")

    def test_label_mapping_from_checkpoint(self):
        self.assertEqual(lv.checkpoint_label_ids(CHECKPOINT_LABELS), LABEL_IDS)
        permuted = {0: "neutral", 1: "contradiction", 2: "entailment"}
        checkpoint = FakeCheckpoint(self.tmp.name, label_ids=lv.checkpoint_label_ids(permuted))
        loaded = lv.load_v3(checkpoint.run, "cpu", checkpoint.loader(id2label=permuted))
        verifier = lv.LocalNLIVerifier(checkpoint.run, "cpu", loaded=loaded)
        results, _ = verify(verifier, ["x y z."], [chunk("A-1.1", "This contradicts it."),
                                                   chunk("B-1.1", "Unrelated text.")])
        self.assertEqual(results[0].verdict, C)                       # follows id2label, not index order
        mismatch = FakeCheckpoint(self.tmp.name, name="mismatch")     # recorded IDs are the original order
        with self.assertRaisesRegex(VerifierUnavailable, "label IDs"):
            lv.load_v3(mismatch.run, "cpu", mismatch.loader(id2label=permuted))
        bad = FakeCheckpoint(self.tmp.name, name="bad")
        with self.assertRaisesRegex(VerifierUnavailable, "Unsupported NLI label"):
            lv.load_v3(bad.run, "cpu", bad.loader(id2label={0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}))

    def test_loader_errors_become_verifier_unavailable(self):
        checkpoint = FakeCheckpoint(self.tmp.name)

        def broken(_):
            raise OSError("disk read failed")

        with self.assertRaisesRegex(VerifierUnavailable, "OSError: disk read failed"):
            lv.load_v3(checkpoint.run, "cpu", broken)


class InputFormatTests(unittest.TestCase):
    def test_raw_evidence_premise_and_order(self):
        verifier, tokenizer, _ = fake_verifier()
        evidence = [chunk("REF-729.1", "International refunds take 45 days.", "International Refund Policy",
                          "Refund Eligibility")]
        verify(verifier, ["International refunds take 45 days."], evidence)
        self.assertEqual(tokenizer.last_batch, [("International refunds take 45 days.",
                                                 "International refunds take 45 days.")])   # (evidence, claim)
        call = tokenizer.calls[-1]
        self.assertEqual((call["truncation"], call["max_length"], call["padding"]), ("only_first", 512, True))
        self.assertNotIn("International Refund Policy", tokenizer.last_batch[0][0])

    def test_m61_premise_unchanged_and_optional(self):
        item = chunk("REF-729.1", "Refunds take 45 days.", "Refund Policy", "Eligibility")
        self.assertEqual(premise_text(item), "Refund Policy, Eligibility: Refunds take 45 days.")
        self.assertEqual(raw_premise_text(item), "Refunds take 45 days.")
        nli, tokenizer, _ = fake_nli()
        nli.classify_claim("Refunds take 45 days.", [item])
        self.assertEqual(tokenizer.last_batch[0][0], premise_text(item))      # M6.1 default
        nli.classify_claim("Refunds take 45 days.", [item], premise_fn=raw_premise_text)
        self.assertEqual(tokenizer.last_batch[0][0], "Refunds take 45 days.")


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.verifier, _, self.model = fake_verifier()

    def judge(self, claim_text, evidence):
        results, output = verify(self.verifier, [claim_text], evidence)
        return results[0], output.trace["claims"][0]

    def test_supported(self):
        result, trace = self.judge("Refunds take 30 days.", [chunk("REF-1.1", "This supports it."),
                                                             chunk("REF-1.2", "Other rule.")])
        self.assertEqual((result.verdict, trace["reason"]), (S, "entailment"))
        self.assertEqual(result.evidence_section_ids, ["REF-1.1"])

    def test_contradicted(self):
        result, trace = self.judge("Refunds take 30 days.", [chunk("REF-1.1", "This contradicts it."),
                                                             chunk("REF-1.2", "Other rule.")])
        self.assertEqual((result.verdict, trace["reason"]), (C, "contradiction"))
        self.assertEqual(result.evidence_section_ids, ["REF-1.1"])

    def test_insufficient(self):
        result, trace = self.judge("Refunds take 30 days.", [chunk("REF-1.1", "Other rule."),
                                                             chunk("REF-1.2", "Another rule.")])
        self.assertEqual((result.verdict, trace["reason"]), (I, "no_entailment"))
        self.assertEqual(result.evidence_section_ids, [])

    def test_conflicting_evidence(self):
        result, trace = self.judge("Refunds take 30 days.", [chunk("REF-1.1", "This supports it."),
                                                             chunk("REF-2.1", "This contradicts it.")])
        self.assertEqual((result.verdict, trace["reason"]), (C, "conflicting_evidence"))
        self.assertEqual(trace["supporting_section_ids"], ["REF-1.1"])
        self.assertEqual(trace["contradicting_section_ids"], ["REF-2.1"])
        self.assertEqual(result.evidence_section_ids, ["REF-2.1", "REF-1.1"])
        self.assertEqual([p["nli_label"] for p in trace["premises"]], ["entailment", "contradiction"])

    def test_possible_compound(self):
        self.assertEqual(lv.compound_signals("Employees must submit receipts and managers must approve them."),
                         ["coordinated_clauses"])
        self.assertIn("semicolon", lv.compound_signals("Refunds take 30 days; exchanges take 60 days."))
        self.assertIn("multiple_sentences", lv.compound_signals("Refunds take 30 days. Exchanges are free."))
        for atomic in ("Opened electronics carry a 15 percent restocking fee unless the item is defective.",
                       "Laptops and phones are covered for two years.",
                       "Refunds may be issued to the original card or as store credit."):
            self.assertEqual(lv.compound_signals(atomic), [], atomic)
        result, trace = self.judge("Refunds are issued within 30 days and are sent to the original card.",
                                   [chunk("REF-1.1", "This supports it.")])
        self.assertEqual((result.verdict, trace["reason"]), (I, "possible_compound"))
        self.assertTrue(trace["possible_compound"])
        self.assertEqual(trace["premises"][0]["nli_label"], "entailment")     # the model label is kept separately
        result, trace = self.judge("Refunds are issued within 30 days and are sent to the original card.",
                                   [chunk("REF-1.1", "This contradicts it.")])
        self.assertEqual(result.verdict, C)                                   # compound flag never helps a claim

    def test_split_section_grouping(self):
        evidence = [chunk("REF-1.2", "second half supports it.", n=1), chunk("REF-3.1", "Unrelated."),
                    chunk("REF-1.2", "First half.", n=0)]
        premises = lv.group_evidence(evidence)
        self.assertEqual([p.section_id for p in premises], ["REF-1.2", "REF-3.1"])
        self.assertEqual(premises[0].chunk_ids, ("REF-1.2#0", "REF-1.2#1"))
        self.assertEqual(premises[0].text, "First half. second half supports it.")
        self.assertEqual(premises[0].document_ids, ("REF-1",))
        self.assertFalse(premises[0].known_incomplete)
        self.assertIn("cannot be ruled out", premises[0].coverage_note)
        result, trace = self.judge("Refunds take 30 days.", evidence)
        self.assertEqual(result.verdict, S)
        self.assertEqual(trace["premises"][0]["chunk_ids"], ["REF-1.2#0", "REF-1.2#1"])
        self.assertEqual(self.model.calls, 1)
        self.assertEqual(len(self.model.tokenizer.last_batch), 2)             # 2 sections -> 2 premises, not 3

    def test_known_incomplete_section_is_not_accepted(self):
        partial = [chunk("REF-1.2", "Second half supports it.", n=1)]
        self.assertTrue(lv.group_evidence(partial)[0].known_incomplete)
        result, trace = self.judge("Refunds take 30 days.", partial)
        self.assertEqual((result.verdict, trace["reason"]), (I, "incomplete_section"))
        self.assertIn("#0", trace["premises"][0]["coverage_note"])
        result, _ = self.judge("Refunds take 30 days.", [chunk("REF-1.2", "Second half contradicts it.", n=1)])
        self.assertEqual(result.verdict, C)
        result, trace = self.judge("Refunds take 30 days.", partial + [chunk("REF-2.1", "This contradicts it.")])
        self.assertEqual((result.verdict, trace["reason"]), (C, "conflicting_evidence"))

    def test_missing_evidence(self):
        result, trace = self.judge("Refunds take 30 days.", [])
        self.assertEqual((result.verdict, trace["reason"]), (I, "missing_evidence"))
        self.assertEqual(self.model.calls, 0)
        self.assertEqual(lv.decide("x", [])["reason"], "missing_evidence")

    def test_overlength_evidence_is_unresolved(self):
        verifier, tokenizer, model = fake_verifier(max_length=16)
        long_text = "This supports it fully here. Second sentence five words. Third sentence has five words."
        results, output = verify(verifier, ["Refunds take thirty days."], [chunk("REF-1.1", long_text)])
        trace = output.trace["claims"][0]
        premise = trace["premises"][0]
        self.assertEqual((results[0].verdict, trace["reason"]), (I, "token_limit"))
        self.assertEqual((premise["status"], premise["unresolved_reason"]), ("unresolved", "token_limit"))
        self.assertGreater(premise["input_tokens"], 16)
        self.assertIsNone(premise["nli_label"])
        self.assertFalse(premise["truncated"])
        self.assertEqual(premise["chunk_ids"], ["REF-1.1#0"])
        windows = premise["diagnostic_windows"]
        self.assertTrue(windows and all(w["diagnostic_only"] and w["input_tokens"] <= 16 for w in windows))
        self.assertEqual(windows[0]["nli_label"], "entailment")              # a window entails, but does not decide
        self.assertTrue(all(c["truncation"] == "only_first" for c in tokenizer.calls if c["return_tensors"] == "pt"))
        mixed = [chunk("REF-1.1", long_text), chunk("REF-2.1", "Short supports it.")]
        results, output = verify(verifier, ["Refunds take thirty days."], mixed)
        self.assertEqual((results[0].verdict, output.trace["claims"][0]["reason"]), (I, "token_limit"))
        results, _ = verify(verifier, ["Refunds take thirty days."],
                            [chunk("REF-1.1", long_text), chunk("REF-2.1", "It contradicts.")])
        self.assertEqual(results[0].verdict, C)

    def test_inference_failures(self):
        for name, verifier in (("nan", fake_verifier(output=[[math.nan, 0.0, 0.0]])[0]),
                               ("shape", fake_verifier(output=[[1.0, 2.0]])[0])):
            with self.subTest(name), self.assertRaisesRegex(VerifierUnavailable, "inference failed"):
                verify(verifier, ["Refunds take 30 days."], [chunk("REF-1.1", "Text.")])
        verifier, _, model = fake_verifier()
        with mock.patch.object(type(model), "__call__", side_effect=RuntimeError("MPS out of memory")):
            with self.assertRaisesRegex(VerifierUnavailable, "RuntimeError: MPS out of memory"):
                verify(verifier, ["Refunds take 30 days."], [chunk("REF-1.1", "Text.")])

    def test_provenance_and_trace(self):
        evidence = [chunk("REF-729.1", "This supports it.", n=0), chunk("REF-728.1", "Other rule.")]
        claim_list = [Claim("C1", "International refunds take 45 days."), Claim("C2", "Refunds take 30 days.")]
        output = self.verifier.verify(claim_list, evidence)
        results = parse_verdicts(output.payload, claim_list, ["REF-729.1", "REF-728.1"], cited_ids=["REF-729.1"])
        self.assertTrue(results[0].support_cited)
        self.assertEqual(output.model, "fake/nli")
        self.assertEqual(output.trace["model"]["score_note"], lv.SCORE_NOTE)
        record = output.trace["claims"][0]
        for key in ("claim_id", "claim", "verdict", "reason", "possible_compound", "evidence_section_ids",
                    "evidence_chunk_ids", "document_ids", "premises", "model"):
            self.assertIn(key, record)
        self.assertEqual(record["evidence_chunk_ids"], ["REF-729.1#0", "REF-728.1#0"])
        self.assertEqual(record["document_ids"], ["REF-729", "REF-728"])
        premise = record["premises"][0]
        for key in ("section_id", "chunk_ids", "document_ids", "nli_label", "model_verdict", "logits", "labels",
                    "uncalibrated_softmax", "input_tokens", "truncated", "status", "known_incomplete"):
            self.assertIn(key, premise)
        self.assertEqual(premise["labels"], ["contradiction", "entailment", "neutral"])   # checkpoint order
        self.assertEqual(len(premise["logits"]), 3)
        self.assertNotIn('"probabilities"', json.dumps(output.trace))         # scores are labelled uncalibrated
        json.dumps(output.trace)


QUESTION = "How long do international customers have to request refunds?"
ANSWER = "International customers have 45 days to request refunds."


class PipelineTests(unittest.TestCase):
    """verify_and_correct / answer_question_verified with the local verifier; Gemini is RoutedLLM."""

    @classmethod
    def setUpClass(cls):
        cls.index = build_search_index(("dense", "bm25", "hybrid"), model=FakeModel())
        cls.question = "What does FIN-205.3 say?"
        cls.top = search(cls.question, cls.index, "hybrid", top_k=5, dense_weight=0.6, bm25_weight=0.4)
        cls.first = cls.top[0]["section_id"]

    def llm(self, answer="Reallocations up to $10,000 need no approval."):
        return RoutedLLM(generate=[grounded(answer, [self.first])], extract=[claims(answer)],
                         revise=[AssertionError("revision must not be requested")],
                         verify=[AssertionError("the Gemini verifier must not be called")])

    def run_default(self, rule):
        verifier, _, _ = fake_verifier(rule=rule)
        llm = self.llm()
        with mock.patch("src.local_verifier.default_verifier", return_value=verifier) as default:
            result = answer_question_verified(self.question, self.index, llm)
        default.assert_called_once()
        return result, llm

    def test_default_is_local_verifier_and_verification_only(self):
        result, llm = self.run_default(entail_if(self.top[0]["text"][:40]))
        self.assertEqual(result["status"], "verified_answer")
        self.assertEqual(result["verification"]["outcome"], "accepted_original")
        self.assertEqual(result["verification"]["verifier"], "nli:v3:fake-run")
        self.assertFalse(result["verification"]["revise"])
        self.assertEqual(llm.routes(), ["generate", "extract"])
        trace = result["verification"]["passes"][0]["verification"]["trace"]
        self.assertEqual(trace["claims"][0]["reason"], "entailment")
        self.assertEqual(result["verification"]["llm_calls"], 1)                # extraction only
        json.dumps(result)

    def test_unverified_answer_is_withheld_without_revision(self):
        result, llm = self.run_default(lambda p, h: LOGITS["neutral"])
        verification = result["verification"]
        self.assertEqual(verification["outcome"], "unverified")
        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["answer"], ABSTENTION_MESSAGE)
        self.assertTrue(result["abstained"])
        self.assertEqual(result["citations"], [])
        self.assertFalse(verification["revision_attempted"])
        self.assertIsNone(verification["revision"])
        self.assertEqual(llm.routes(), ["generate", "extract"])                # no Gemini verify, no revision
        self.assertEqual(verification["passes"][0]["verification"]["results"][0]["verdict"], I)
        self.assertIn("trace", verification["passes"][0]["verification"])
        self.assertEqual(result["baseline"]["answer"], "Reallocations up to $10,000 need no approval.")

    def test_verifier_failure_fails_closed_without_fallback(self):
        llm = self.llm()
        missing = lv.LocalNLIVerifier(Path(tempfile.gettempdir()) / "ragforge-no-such-run", "cpu")
        result = answer_question_verified(self.question, self.index, llm, verifier=missing)
        self.assertEqual(result["status"], "verification_failed")
        self.assertEqual(result["verification"]["outcome"], "verification_api_failure")
        self.assertEqual(result["error"]["kind"], "verifier_unavailable")
        self.assertIn("not found", result["error"]["message"])
        self.assertIsNone(result["answer"])
        self.assertEqual(llm.routes(), ["generate", "extract"])
        broken, _, _ = fake_verifier(output=[[math.nan, 0.0, 0.0]])
        llm = self.llm()
        result = answer_question_verified(self.question, self.index, llm, verifier=broken)
        self.assertEqual(result["verification"]["outcome"], "verification_api_failure")
        self.assertEqual(llm.routes(), ["generate", "extract"])

    def test_revise_flag_preserves_m5_path(self):
        answer = GroundedAnswer(answer=ANSWER, citations=["REF-729.1"], abstained=False)
        sources = [chunk("REF-729.1", "Something else.")]
        llm = RoutedLLM(extract=[claims(ANSWER), claims("Not covered by policy details.")],
                        verify=[{"results": [{"claim_id": "C1", "verdict": I, "evidence_section_ids": [],
                                              "explanation": "Missing."}]}],
                        revise=[grounded("Not covered.", [], abstained=True)])
        result = verify_and_correct(QUESTION, sources, answer, GeminiClaimExtractor(llm), GeminiVerifier(llm), llm)
        self.assertTrue(result["revision_attempted"])                          # revise=True is still the default here
        self.assertEqual(llm.routes(), ["extract", "verify", "revise"])
        with self.assertRaises(ValueError):
            verify_and_correct(QUESTION, sources, answer, GeminiClaimExtractor(llm), GeminiVerifier(llm), None)
        llm = RoutedLLM(extract=[claims(ANSWER)], verify=[{"results": [{"claim_id": "C1", "verdict": I,
                                                                        "evidence_section_ids": [],
                                                                        "explanation": "Missing."}]}])
        result = verify_and_correct(QUESTION, sources, answer, GeminiClaimExtractor(llm), GeminiVerifier(llm),
                                    revise=False)
        self.assertEqual(result["outcome"], "unverified")
        self.assertEqual(llm.routes(), ["extract", "verify"])

    def test_explicit_gemini_and_mock_verifiers(self):
        answer = "Reallocations up to $10,000 need no approval."
        llm = RoutedLLM(generate=[grounded(answer, [self.first])], extract=[claims(answer)],
                        verify=[{"results": [{"claim_id": "C1", "verdict": S, "evidence_section_ids": [self.first],
                                              "explanation": "Stated."}]}])
        result = answer_question_verified(self.question, self.index, llm, verifier=GeminiVerifier(llm))
        self.assertEqual(result["status"], "verified_answer")
        self.assertEqual(llm.routes(), ["generate", "extract", "verify"])
        llm = RoutedLLM(generate=[grounded(answer, [self.first])], extract=[claims(answer)])
        result = answer_question_verified(self.question, self.index, llm, verifier=MockVerifier())
        self.assertTrue(result["verification"]["verifier"].startswith("mock"))
        self.assertNotIn("trace", result["verification"]["passes"][0]["verification"])

    def test_evidence_snapshot_adds_document_id_and_keeps_fingerprint(self):
        sources = [dict(chunk("REF-729.1", "Rule one.", n=0), ref="E1"), chunk("REF-729.1", "Rule two.", n=1)]
        snapshot = evidence_snapshot(sources)
        self.assertEqual(snapshot[0]["document_id"], "REF-729")
        self.assertNotIn("ref", snapshot[0])
        m5 = hashlib.sha256(json.dumps(
            [[c["chunk_id"], c["section_id"], c["text"]] for c in sources], ensure_ascii=False).encode()).hexdigest()
        self.assertEqual(evidence_fingerprint(snapshot), m5)                    # the M5 definition, unchanged
        without = evidence_snapshot([{k: v for k, v in s.items() if k != "document_id"} for s in sources])
        self.assertEqual(evidence_fingerprint(without), evidence_fingerprint(snapshot))
        with self.assertRaises(TypeError):
            snapshot[0]["document_id"] = "X"


class CommandLineTests(unittest.TestCase):
    QUESTION = "What does FIN-205.3 say?"

    @classmethod
    def setUpClass(cls):
        cls.index = build_search_index(("dense", "bm25", "hybrid"), model=FakeModel())
        cls.top = search(cls.QUESTION, cls.index, "hybrid", top_k=5, dense_weight=0.6, bm25_weight=0.4)
        cls.first = cls.top[0]["section_id"]

    def llm(self):
        answer = "Reallocations up to $10,000 need no approval."
        return RoutedLLM(generate=[grounded(answer, [self.first])], extract=[claims(answer)],
                         verify=[{"results": [{"claim_id": "C1", "verdict": S, "evidence_section_ids": [self.first],
                                               "explanation": "Stated."}]}],
                         revise=[grounded("Not covered.", [], abstained=True)])

    def cli(self, *argv, verifier=None, llm=None):
        llm = llm or self.llm()
        code, out, err = run_cli(rag_main, ["--question", self.QUESTION, *argv], llm=llm, index=self.index,
                                 verifier=verifier)
        return code, out, err, llm

    def test_default_verify_uses_local_model(self):
        verifier, _, _ = fake_verifier(rule=entail_if(self.top[0]["text"][:40]))
        code, out, _, llm = self.cli("--verify", verifier=verifier)
        self.assertEqual(code, 0)
        for text in ("Verifier: nli:v3:fake-run", "Status: verified_answer", "C1 SUPPORTED", "reason: entailment",
                     "model fake/nli on cpu", "uncalibrated", "Local NLI model load"):
            self.assertIn(text, out)
        self.assertEqual(llm.routes(), ["generate", "extract"])
        code, out, _, llm = self.cli("--verify", "--json", verifier=fake_verifier(rule=lambda p, h: LOGITS["neutral"])[0])
        result = json.loads(out)
        self.assertEqual((result["status"], result["verification"]["outcome"]), ("abstained", "unverified"))
        self.assertIn("uncalibrated_softmax", json.dumps(result))
        self.assertEqual(llm.routes(), ["generate", "extract"])

    def test_unverified_text_output(self):
        code, out, _, _ = self.cli("--verify", verifier=fake_verifier(rule=lambda p, h: LOGITS["neutral"])[0])
        self.assertEqual(code, 0)
        self.assertIn("Answer withheld: not every claim could be verified", out)
        self.assertIn("reason: no_entailment", out)
        self.assertNotIn("Verified answer:", out)

    def test_missing_checkpoint_fails_closed_before_gemini(self):
        missing = str(Path(tempfile.gettempdir()) / "ragforge-no-such-run")
        for extra in ((), ("--json",)):
            code, out, _, llm = self.cli("--verify", "--nli-run", missing, "--nli-device", "cpu", *extra)
            self.assertEqual(code, 1)
            self.assertEqual(llm.calls, [])                                    # no Gemini request at all
            self.assertIn("not found", out)
            self.assertIn("verification_failed", out)
        self.assertIsNone(json.loads(out)["answer"])

    def test_gemini_and_mock_verifiers(self):
        code, out, _, llm = self.cli("--verify", "--verifier", "gemini")
        self.assertEqual(code, 0)
        self.assertIn("Verifier: llm:fake-llm", out)
        self.assertEqual(llm.routes(), ["generate", "extract", "verify"])
        code, out, _, _ = self.cli("--verify", "--verifier", "mock")
        self.assertIn("Verifier: mock-lexical", out)
        self.assertIn("Warning: mock verifier", out)
        llm = self.llm()
        llm.queues["verify"] = [{"results": [{"claim_id": "C1", "verdict": I, "evidence_section_ids": [],
                                              "explanation": "Missing."}]}]
        code, out, _, llm = self.cli("--verify", "--verifier", "gemini", "--revise", llm=llm)
        self.assertIn("revise", llm.routes())                                  # explicit M5 revision path

    def test_argument_validation(self):
        for argv in (("--verifier", "nli"), ("--nli-run", "x"), ("--nli-device", "cpu"), ("--revise",),
                     ("--verify", "--verifier", "gemini", "--nli-run", "x"),
                     ("--verify", "--verifier", "mock", "--nli-device", "cpu")):
                # (--verify --revise with the nli verifier is valid since M6.4B; see test_self_correction.py)
            with self.subTest(argv=argv):
                code, _, err, llm = self.cli(*argv)
                self.assertEqual(code, 2)
                self.assertEqual(llm.calls, [])

    def test_normal_generation_unchanged(self):
        with mock.patch("src.local_verifier.load_v3", side_effect=AssertionError("model loaded")):
            code, out, _, llm = self.cli()
        self.assertEqual(code, 0)
        self.assertIn("Status: answered", out)
        self.assertNotIn("Verification trace", out)
        self.assertEqual(llm.routes(), ["generate"])


@unittest.skipUnless(os.environ.get("RAGFORGE_MODEL_TESTS") == "1" and (lv.DEFAULT_V3_RUN / "best").is_dir(),
                     "set RAGFORGE_MODEL_TESTS=1 with the local V3 checkpoint to run")
class RealCheckpointSmokeTest(unittest.TestCase):
    """Synthetic evidence-claim pairs through the real V3 checkpoint (local files only, CPU)."""

    def test_synthetic_pairs(self):
        verifier = lv.LocalNLIVerifier(lv.DEFAULT_V3_RUN, "cpu")
        evidence = [chunk("TST-100.1", "Contractors must complete security training within 14 days of starting.",
                          "Synthetic Policy", "Training")]
        results, output = verify(verifier, ["Contractors must complete security training within 14 days of starting.",
                                            "Contractors must complete security training within 60 days of starting.",
                                            "Contractors receive a laptop on their first day."], evidence)
        self.assertEqual(output.trace["model"]["integrity"], {"files": "verified", "parameters": "verified",
                                                              "source": "test_metrics.json"})
        self.assertEqual([r.verdict for r in results], [S, C, I])
        self.assertFalse(any(p.requires_grad for p in verifier.load().nli.model.parameters()))


if __name__ == "__main__":
    unittest.main()
