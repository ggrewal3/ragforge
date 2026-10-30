"""Offline tests for the M6.4C evaluation harness (scripts/m64c_eval.py).

Gemini is RoutedLLM, V3 is the scripted fake local verifier from test_self_correction,
and retrieval is patched to return a fixed snapshot. Nothing touches the network.
"""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.llm import LLMRequestError, LLMResponse
from test_generation import chunk
from test_self_correction import Table, local
from test_verification import RoutedLLM, claims, grounded

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("m64c_eval", ROOT / "scripts" / "m64c_eval.py")
m64c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m64c)

ELIGIBILITY = "Domestic customers may request a refund within 30 days of delivery."
PROCESSING = "Domestic refunds are issued within 5 business days after the returned item is received."
WRONG = "Domestic customers may request a refund within 14 days of delivery."
RIGHT = "Domestic customers may request a refund within 30 days of delivery."
CHUNKS = [dict(chunk("REF-728.1", ELIGIBILITY, "Domestic Refund Policy", "Refund Eligibility"), rank=1),
          dict(chunk("REF-728.3", PROCESSING, "Domestic Refund Policy", "Refund Processing"), rank=2)]


def case(case_id="GC1", answer=WRONG, label="factually_incorrect", excerpt="within 30 days of delivery",
         citations=("REF-728.1",), category="genuine_contradiction"):
    return {"case_id": case_id, "category": category, "question": "How long do domestic customers have?",
            "original_answer": answer, "citations": list(citations), "answer_source": "synthetic",
            "seeded_error": label != "factually_supported", "provenance_note": "test",
            "expected": {"original_answer_label": label, "interpretation": "30 days.", "ideal_correction": "30 days.",
                         "statements": [{"statement": answer, "label": label, "section_id": "REF-728.1",
                                         "excerpt": excerpt}]}}


def verifier():
    return local({WRONG: {ELIGIBILITY: "contradiction"}, RIGHT: {ELIGIBILITY: "entailment"}})


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run_dir = Path(self.tmp.name) / "run"
        patcher = mock.patch("src.retrieve.search", return_value=[dict(c) for c in CHUNKS])
        self.search = patcher.start()
        self.addCleanup(patcher.stop)

    def execute(self, cases, llm, v=None):
        return m64c.run(cases, self.run_dir, llm, v or verifier(), lambda: "index", dry_run=False,
                        progress=lambda *a: None)

    def rows(self, name):
        return m64c.read_jsonl(self.run_dir / m64c.FILES[name])

    def test_conditions_share_frozen_inputs(self):
        llm = RoutedLLM(extract=[claims(WRONG), claims(RIGHT)], revise=[grounded(RIGHT, ["REF-728.1"])])
        summary = self.execute([case()], llm)
        self.assertEqual(llm.routes(), ["extract", "revise", "extract"])         # freeze, then B only
        self.assertEqual(summary["gemini_requests_made"], 3)
        frozen, = self.rows("cases")
        a, = self.rows("A")
        b, = self.rows("B")
        self.assertEqual(frozen["initial_extraction"]["claims"], [{"claim_id": "C1", "text": WRONG}])
        self.assertEqual(frozen["evidence_section_ids"], ["REF-728.1", "REF-728.3"])
        self.assertEqual((a["outcome"], a["returned"]), ("unverified", False))
        self.assertEqual((b["outcome"], b["returned"], b["final_answer"]), ("accepted_revision", True, RIGHT))
        for row in (a, b):
            self.assertEqual(row["verification"]["evidence_fingerprint"], frozen["evidence_fingerprint"])
            self.assertEqual(row["verification"]["passes"][0]["extraction"]["claims"],
                             frozen["initial_extraction"]["claims"])
        self.assertEqual(a["verification"]["passes"][0]["verification"]["results"],
                         b["verification"]["passes"][0]["verification"]["results"])
        self.assertEqual(a["verification"]["llm_calls"], 0)
        self.assertEqual(b["verification"]["llm_calls"], 2)                       # correction + re-extraction
        self.assertIn("trace", b["verification"]["passes"][1]["verification"])    # full traces kept

    def test_resume_does_not_repeat_calls(self):
        llm = RoutedLLM(extract=[claims(WRONG), claims(RIGHT)], revise=[grounded(RIGHT, ["REF-728.1"])])
        self.execute([case()], llm)
        again = RoutedLLM(extract=[AssertionError("no call")], revise=[AssertionError("no call")])
        summary = self.execute([case()], again)
        self.assertEqual(again.calls, [])
        self.assertEqual(summary["gemini_requests_made"], 0)
        self.assertEqual(len(self.rows("A")), 1)
        self.assertEqual(len(self.rows("B")), 1)

    def test_interrupted_case_replays_cached_responses(self):
        llm = RoutedLLM(extract=[claims(WRONG), claims(RIGHT)], revise=[grounded(RIGHT, ["REF-728.1"])])
        self.execute([case()], llm)
        (self.run_dir / m64c.FILES["B"]).unlink()                                 # B lost mid-run
        again = RoutedLLM(extract=[AssertionError("no call")], revise=[AssertionError("no call")])
        summary = self.execute([case()], again)
        self.assertEqual(again.calls, [])
        self.assertEqual(summary["cache_hits"], 2)
        self.assertEqual(self.rows("B")[0]["outcome"], "accepted_revision")

    def test_case_definition_change_is_refused(self):
        self.execute([case()], RoutedLLM(extract=[claims(WRONG), claims(RIGHT)],
                                         revise=[grounded(RIGHT, ["REF-728.1"])]))
        with self.assertRaises(m64c.EvaluationError):
            self.execute([case(answer=RIGHT)], RoutedLLM(extract=[claims(RIGHT)]))

    def test_api_failure_stops_the_run_and_leaves_cases_pending(self):
        second = case(case_id="GC2")
        llm = RoutedLLM(extract=[claims(WRONG)], revise=[LLMRequestError("429 RESOURCE_EXHAUSTED")])
        summary = self.execute([case(), second], llm)
        self.assertEqual(summary["stopped_on_api_failure"]["case_id"], "GC1")
        self.assertEqual(len(self.rows("A")), 1)
        self.assertEqual(self.rows("B"), [])                                     # GC1 B stays pending
        self.assertEqual([r["case_id"] for r in self.rows("cases")], ["GC1"])    # GC2 never started
        log, = m64c.read_jsonl(self.run_dir / "api_failures.jsonl")
        self.assertEqual((log["case_id"], log["stage"]), ("GC1", "B"))
        resumed = RoutedLLM(extract=[claims(RIGHT), claims(WRONG), claims(RIGHT)],
                            revise=[grounded(RIGHT, ["REF-728.1"])])
        self.execute([case(), second], resumed)
        self.assertEqual(resumed.routes(), ["revise", "extract", "extract", "revise", "extract"])   # no cross-case reuse
        self.assertEqual(sorted(r["case_id"] for r in self.rows("B")), ["GC1", "GC2"])

    def test_freeze_api_failure_is_not_frozen(self):
        llm = RoutedLLM(extract=[LLMRequestError("429 RESOURCE_EXHAUSTED")])
        summary = self.execute([case()], llm)
        self.assertEqual(summary["stopped_on_api_failure"]["stage"], "freeze")
        self.assertEqual(self.rows("cases"), [])

    def test_retry_api_failures_requeues_only_infrastructure_failures(self):
        llm = RoutedLLM(extract=[claims(WRONG), LLMRequestError("429 RESOURCE_EXHAUSTED")],
                        revise=[grounded(RIGHT, ["REF-728.1"])])
        m64c.run([case()], self.run_dir, llm, verifier(), lambda: "index", dry_run=False, progress=lambda *a: None,
                 record_api_failures=True)
        self.assertEqual(self.rows("B")[0]["outcome"], "extraction_failure")
        content = case(case_id="GC9")
        m64c.run([content], self.run_dir, RoutedLLM(extract=["garbage"]), verifier(), lambda: "index",
                 dry_run=False, progress=lambda *a: None)                         # a content failure
        dropped = m64c.retry_api_failures(self.run_dir, progress=lambda *a: None)
        self.assertEqual(dropped["B"], {"GC1"})
        self.assertEqual(dropped["cases"], set())
        self.assertEqual(sorted(r["case_id"] for r in self.rows("cases")), ["GC1", "GC9"])
        superseded, = m64c.read_jsonl(self.run_dir / "superseded_api_failures.jsonl")
        self.assertEqual(superseded["row"]["outcome"], "extraction_failure")
        again = RoutedLLM(extract=[claims(RIGHT)], revise=[AssertionError("cached correction is replayed")])
        self.execute([case(), content], again)
        self.assertEqual(again.routes(), ["extract"])                             # only the failed re-extraction
        self.assertEqual({r["case_id"]: r["outcome"] for r in self.rows("B")},
                         {"GC9": "extraction_failure", "GC1": "accepted_revision"})

    def test_pacing_between_real_requests(self):
        llm = RoutedLLM(extract=[claims(WRONG), claims(RIGHT)], revise=[grounded(RIGHT, ["REF-728.1"])])
        with mock.patch.object(m64c.time, "sleep") as sleep:
            m64c.run([case()], self.run_dir, llm, verifier(), lambda: "index", dry_run=False,
                     progress=lambda *a: None, min_interval_s=10.0)
        self.assertEqual(sleep.call_count, 2)                                    # before the 2nd and 3rd request
        self.assertTrue(all(0 < c.args[0] <= 10.0 for c in sleep.call_args_list))

    def test_api_failures_recorded_when_requested(self):
        llm = RoutedLLM(extract=[claims(WRONG)], revise=[LLMRequestError("Gemini server error (503).")])
        m64c.run([case()], self.run_dir, llm, verifier(), lambda: "index", dry_run=False, progress=lambda *a: None,
                 record_api_failures=True)
        b, = self.rows("B")
        self.assertEqual(b["outcome"], "revision_api_failure")
        self.assertTrue(b["error"]["api_error"])
        metrics = m64c.compute_metrics(self.rows("cases"), self.rows("A"), self.rows("B"), {})
        self.assertEqual(metrics["verification_behavior"]["correction_api_failures"]["count"], 1)
        self.assertEqual(metrics["verification_behavior"]["any_api_failure_cases"], ["GC1"])

    def test_freeze_extraction_failure(self):
        llm = RoutedLLM(extract=["garbage"], revise=[AssertionError("no correction")])
        self.execute([case()], llm)
        frozen, = self.rows("cases")
        self.assertEqual(frozen["initial_extraction"]["status"], "failed")
        self.assertEqual([r["outcome"] for r in self.rows("A") + self.rows("B")],
                         ["extraction_failure", "extraction_failure"])
        self.assertEqual(llm.routes(), ["extract"])

    def test_excerpts_and_citations_are_checked(self):
        with self.assertRaisesRegex(m64c.EvaluationError, "excerpt not found"):
            self.execute([case(excerpt="within 45 days of delivery")], RoutedLLM(extract=[claims(WRONG)]))
        with self.assertRaisesRegex(m64c.EvaluationError, "citations fail"):
            self.execute([case(case_id="X", citations=("REF-999.1",))], RoutedLLM(extract=[claims(WRONG)]))

    def test_replay_extractor_guards(self):
        replay = m64c.ReplayExtractor([{"claim_id": "C1", "text": WRONG}], WRONG)
        with self.assertRaises(m64c.EvaluationError):
            replay.extract("q", "a different answer")
        replay = m64c.ReplayExtractor([{"claim_id": "C1", "text": WRONG}], WRONG)
        replay.extract("q", WRONG)
        with self.assertRaises(m64c.EvaluationError):
            replay.extract("q", RIGHT)                                           # no live extractor configured

    def test_caching_llm_persists(self):
        class Counting:
            model = "fake"
            calls = 0

            def generate(self, system, user, schema):
                Counting.calls += 1
                return LLMResponse(text="{}", model="fake", latency_s=0.1, attempts=1, usage={"input_tokens": 3})

        path = Path(self.tmp.name) / "cache.jsonl"
        first = m64c.CachingLLM(Counting(), path)
        first.generate("s", "u", {})
        first.generate("s", "u", {})
        second = m64c.CachingLLM(Counting(), path)
        response = second.generate("s", "u", {})
        self.assertEqual(Counting.calls, 1)
        self.assertEqual(response.usage, {"input_tokens": 3})
        second.generate("s", "different", {})
        self.assertEqual(Counting.calls, 2)
        m64c.CachingLLM(Counting(), path, namespace="GC1:B").generate("s", "u", {})   # another stage or case: fresh
        self.assertEqual(Counting.calls, 3)

    def test_legacy_stage_only_cache_rows_are_replayed(self):
        class Counting:
            model = "fake"
            calls = 0

            def generate(self, system, user, schema):
                Counting.calls += 1
                return LLMResponse(text="{}", model="fake", latency_s=0.1, attempts=1)

        path = Path(self.tmp.name) / "cache.jsonl"
        m64c.CachingLLM(Counting(), path, namespace="B").generate("s", "u", {})       # old-style key
        m64c.CachingLLM(Counting(), path, namespace="GC1:B").generate("s", "u", {})   # replayed, not re-requested
        m64c.CachingLLM(Counting(), path, namespace="GC1:freeze").generate("s", "u", {})   # other stage: fresh
        self.assertEqual(Counting.calls, 2)

    def test_identical_revision_is_extracted_afresh(self):
        llm = RoutedLLM(extract=[claims(WRONG), claims(WRONG)], revise=[grounded(WRONG, ["REF-728.1"])])
        self.execute([case()], llm)
        self.assertEqual(llm.routes(), ["extract", "revise", "extract"])         # not served from the freeze cache
        self.assertEqual(self.rows("B")[0]["outcome"], "revision_unsupported")

    def test_metrics_use_provisional_labels_not_v3(self):
        supported = case(case_id="AS1", answer=RIGHT, label="factually_supported", category="already_supported")
        llm = RoutedLLM(extract=[claims(WRONG), claims(RIGHT), claims(RIGHT)],
                        revise=[grounded(RIGHT, ["REF-728.1"])])
        self.execute([case(), supported], llm)
        rows_a, rows_b, frozen = self.rows("A"), self.rows("B"), self.rows("cases")
        metrics = m64c.compute_metrics(frozen, rows_a, rows_b, {})
        provisional = metrics["factual_reliability"]["provisional"]
        self.assertEqual(provisional["B_with_correction"]["unlabeled"], ["GC1"])  # revised answer not yet annotated
        self.assertIn("0 of 2 cases human-reviewed", metrics["factual_reliability"]["label_status"])
        annotations = {"GC1": {"revised": {"label": "factually_supported"},
                               "effects": {"fixed_incorrect_claim": True}, "failure_causes": []}}
        metrics = m64c.compute_metrics(frozen, rows_a, rows_b, annotations)
        provisional = metrics["factual_reliability"]["provisional"]
        self.assertEqual(provisional["A_verification_only"]["correct_returned"], 1)
        self.assertEqual(provisional["A_verification_only"]["incorrect_or_unsupported_withheld"], 1)
        self.assertEqual(provisional["B_with_correction"]["correct_returned"], 2)
        self.assertEqual(provisional["correction_effects"]["fixed_incorrect_claim"], 1)
        self.assertEqual(metrics["correction_effectiveness"]["revisions_factual_quality_provisional"],
                         {"improved": 1})
        self.assertIsNone(metrics["performance"]["estimated_cost_usd"])          # no trusted price for fake model
        self.assertTrue(metrics["verification_behavior"]["initial_verdicts_identical_A_vs_B"])

    def test_report_files(self):
        llm = RoutedLLM(extract=[claims(WRONG), claims(RIGHT)], revise=[grounded(RIGHT, ["REF-728.1"])])
        self.execute([case()], llm)
        m64c.write_json(self.run_dir / "manifest.json", {"run_id": "run", "dry_run": False,
                                                         "verifier": {"identifier": "fake", "device": "cpu"}})
        m64c.report(self.run_dir)
        for name in ("metrics.json", "manual_review.jsonl", "report.md"):
            self.assertTrue((self.run_dir / name).is_file())
        review, = m64c.read_jsonl(self.run_dir / "manual_review.jsonl")
        self.assertEqual(review["human_review"]["reviewer"], None)
        self.assertEqual(review["revised_answer"], RIGHT)
        self.assertIn("not ground truth", " ".join(review))
        self.assertIn("provisional", (self.run_dir / "report.md").read_text().lower())


class CaseFileAndCommandLineTests(unittest.TestCase):
    def test_case_definitions(self):
        data, cases = m64c.load_cases()
        m64c.check_category_counts(cases)
        self.assertEqual(len(cases), 12)
        self.assertIn("PROVISIONAL", data["label_status"])
        for item in cases:
            self.assertIn(item["answer_source"].split(":")[0], ("synthetic", "adapted_benchmark", "adapted_live_run"))
            self.assertTrue(item["provenance_note"])
            self.assertTrue(item["expected"]["statements"])
            if item["seeded_error"]:
                self.assertNotEqual(item["expected"]["original_answer_label"], "factually_supported")

    def test_live_needs_confirmation_and_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            out = []
            code = m64c.main(["--live", "--run-id", "r1"], out_root=directory, progress=out.append)
            self.assertEqual(code, 0)
            self.assertIn("No Gemini request was made", "\n".join(out))
            self.assertFalse((Path(directory) / "r1").exists())
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch("src.llm.load_env_file"):
                out = []
                code = m64c.main(["--live", "--run-id", "r1", "--confirm-live"], out_root=directory,
                                 progress=out.append)
            self.assertEqual(code, 1)
            self.assertIn("not set", "\n".join(out))
            self.assertFalse((Path(directory) / "r1").exists())

    def test_plan_reports_budget(self):
        out = []
        self.assertEqual(m64c.main(["--plan"], progress=out.append), 0)
        budget = json.loads(out[0])
        self.assertEqual(budget["cases"], 12)
        self.assertEqual(budget["max_gemini_requests"]["total"], 36)
        self.assertEqual(budget["condition_A_gemini_requests"], 0)


if __name__ == "__main__":
    unittest.main()
