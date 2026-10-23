"""Tests for the M6.2 quality gate: review decisions, resumable sessions, audit history, shortcut audit and freeze.

All decisions here are written by the tests themselves into temporary logs, using a
synthetic corpus; the real data/nli/reviews.jsonl is never touched. No network, no models.
"""

import copy
import json
import os
import stat
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from src import nli_dataset as nd
from src import nli_review as nr
from test_generation import run_cli
from test_nli_dataset import DOCUMENTS, SPEC

NOW = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)


class Workspace(unittest.TestCase):
    """A temporary facts file, generated draft dataset and empty decision log."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.facts = root / "facts.json"
        self.facts.write_text(json.dumps(SPEC))
        self.dataset = root / "drafts.jsonl"
        self.reviews = root / "reviews.jsonl"
        self.frozen = root / "frozen"
        examples, _ = nd.build(self.facts, documents=DOCUMENTS, benchmark_path=root / "none.json")
        nd.write_jsonl(self.dataset, examples)
        self.examples = examples
        self.by_split = {split: [e for e in examples if e["split"] == split] for split in nd.SPLITS}
        self.patch = mock.patch("src.nli_dataset.load_benchmark_claims", return_value=[])
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        for path in Path(self.tmp.name).rglob("*"):
            if path.is_file():
                path.chmod(stat.S_IWUSR | stat.S_IRUSR)
        self.tmp.cleanup()

    def decide(self, example, status="reviewed", label=None, reason=None, **kwargs):
        return nr.record_decision(self.reviews, example, status, "test-reviewer", label, reason, now=NOW, **kwargs)

    def freeze(self, version="v1.0"):
        return nr.freeze(version, self.dataset, self.reviews, self.frozen, self.facts, documents=DOCUMENTS, now=NOW)

    def decide_all_evaluation(self):
        evaluation = self.by_split["validation"] + self.by_split["test"]
        for example in evaluation:
            self.decide(example)
        return evaluation


class DecisionTests(Workspace):
    def test_workspace_has_all_splits(self):
        self.assertTrue(all(self.by_split[split] for split in nd.SPLITS))

    def test_accept(self):
        example = self.examples[0]
        record = self.decide(example)
        self.assertEqual((record["status"], record["ai_label"], record["human_label"]),
                         ("reviewed", example["label"], example["label"]))
        self.assertEqual(record["reviewer"], "test-reviewer")
        self.assertEqual(record["reviewed_at"], NOW.isoformat(timespec="seconds"))
        applied = copy.deepcopy(self.examples)
        nd.apply_reviews(applied, list(nr.effective_decisions(nr.load_decisions(self.reviews)).values()))
        self.assertEqual(applied[0]["review"]["label_source"], "human-reviewed")
        self.assertEqual(nd.validate(applied, DOCUMENTS), [])

    def test_correct(self):
        example = self.examples[0]
        other = next(label for label in nd.LABEL_VALUES if label != example["label"])
        with self.assertRaises(nr.ReviewError):
            self.decide(example, "corrected", example["label"], "same label")
        with self.assertRaises(nr.ReviewError):
            self.decide(example, "corrected", other, None)                 # reason required
        record = self.decide(example, "corrected", other, "the evidence says otherwise")
        self.assertEqual((record["ai_label"], record["human_label"], record["reason"]),
                         (example["label"], other, "the evidence says otherwise"))
        applied = copy.deepcopy(self.examples)
        nd.apply_reviews(applied, [record])
        self.assertEqual(applied[0]["label"], other)
        self.assertEqual(applied[0]["review"]["ai_label"], example["label"])      # original AI label kept
        self.assertEqual(self.examples[0]["label"], example["label"])               # drafts untouched
        self.assertEqual(nd.validate(applied, DOCUMENTS), [])

    def test_reject(self):
        with self.assertRaises(nr.ReviewError):
            self.decide(self.examples[0], "rejected")
        record = self.decide(self.examples[0], "rejected", reason="ambiguous evidence")
        self.assertIsNone(record["human_label"])
        with self.assertRaises(nr.ReviewError):
            self.decide(self.examples[1], "approved")
        with self.assertRaises(nr.ReviewError):
            nr.record_decision(self.reviews, self.examples[1], "reviewed", "  ")         # reviewer required

    def test_duplicate_prevention_and_history(self):
        example = self.examples[0]
        first = self.decide(example)
        with self.assertRaises(nr.DuplicateReviewError):
            self.decide(example)
        with self.assertRaises(nr.ReviewError):
            self.decide(example, "rejected", reason=None, redo=True)
        with self.assertRaises(nr.ReviewError):
            self.decide(self.examples[1], redo=True, reason="nothing to redo")
        second = self.decide(example, "rejected", reason="second look: ambiguous", redo=True)
        self.assertEqual(second["supersedes"], first["decision_id"])
        records = nr.history(self.reviews, example["example_id"])
        self.assertEqual([r["status"] for r in records], ["reviewed", "rejected"])
        self.assertEqual(nr.effective_decisions(nr.load_decisions(self.reviews))[example["example_id"]]["status"],
                         "rejected")

    def test_tampering_is_detected(self):
        self.decide(self.examples[0])
        self.decide(self.examples[1])
        lines = self.reviews.read_text().splitlines()
        edited = json.loads(lines[0])
        edited["human_label"] = "CONTRADICTED" if edited["human_label"] != "CONTRADICTED" else "SUPPORTED"
        self.reviews.write_text(json.dumps(edited) + "\n" + lines[1] + "\n")
        with self.assertRaises(nr.ReviewIntegrityError):
            nr.load_decisions(self.reviews)
        self.reviews.write_text(lines[1] + "\n")                                   # first line deleted
        with self.assertRaises(nr.ReviewIntegrityError):
            nr.load_decisions(self.reviews)


class SessionTests(Workspace):
    def session(self, keys):
        keys = iter(keys)
        output = []
        plan, _ = nr.review_plan(self.examples, DOCUMENTS)
        summary = nr.run_session(self.examples, plan, self.reviews, "test-reviewer",
                                 input_fn=lambda prompt: next(keys), output_fn=output.append, now=NOW)
        return summary, "\n".join(output), plan

    def test_display_skip_and_resume(self):
        summary, output, plan = self.session(["s", "a", "", "q"])
        self.assertEqual((summary["recorded"], summary["skipped"]), (1, 1))
        first, second = plan[0], plan[1]
        for field in (first["example_id"], first["evidence"], first["claim"], first["proposed_label"],
                      "Rationale:", "Family:", "Format:", "Original section", "Warnings:", "decided 0/"):
            self.assertIn(field, output)
        decided = nr.effective_decisions(nr.load_decisions(self.reviews))
        self.assertNotIn(first["example_id"], decided)                       # skipped: not reviewed
        self.assertIn(second["example_id"], decided)
        # Resume: the skipped example comes back, the decided one does not.
        summary, output, _ = self.session(["q"])
        self.assertIn(f"Example: {first['example_id']}", output)
        self.assertNotIn(f"Example: {second['example_id']}", output)
        self.assertIn("decided 1/", output)

    def test_correct_and_reject_in_session(self):
        plan, _ = nr.review_plan(self.examples, DOCUMENTS)
        proposed = plan[0]["proposed_label"]
        key = next(k for k, v in nr.LABEL_KEYS.items() if v != proposed)
        summary, _, _ = self.session(["c", "9", key, "", "wrong value", "r", "", "unclear claim", "q"])
        records = nr.load_decisions(self.reviews)
        self.assertEqual([r["status"] for r in records], ["corrected", "rejected"])
        self.assertEqual(records[0]["human_label"], nr.LABEL_KEYS[key])
        self.assertEqual(records[1]["reason"], "unclear claim")
        self.assertEqual(summary["recorded"], 2)

    def test_interrupt_keeps_saved_decisions(self):
        keys = iter(["a", ""])

        def keyboard(prompt):
            try:
                return next(keys)
            except StopIteration:
                raise KeyboardInterrupt from None

        plan, _ = nr.review_plan(self.examples, DOCUMENTS)
        summary = nr.run_session(self.examples, plan, self.reviews, "test-reviewer", input_fn=keyboard,
                                 output_fn=lambda *a: None, now=NOW)
        self.assertTrue(summary["stopped"])
        self.assertEqual(len(nr.load_decisions(self.reviews)), 1)

    def test_plan_order_and_deduplication(self):
        with mock.patch.object(nr, "HIGH_RISK_COUNT", 3):      # the synthetic set is smaller than 70 examples
            plan, report = nr.review_plan(self.examples, DOCUMENTS)
        self.assertEqual(sum("high_risk" in item["groups"] for item in plan), 3)
        self.assertTrue(all("high_risk" in item["groups"] for item in plan[:3]))
        ids = [item["example_id"] for item in plan]
        self.assertEqual(len(ids), len(set(ids)))
        evaluation = {e["example_id"] for e in self.by_split["validation"] + self.by_split["test"]}
        self.assertTrue(evaluation <= set(ids))
        first_test = next(i for i, item in enumerate(plan) if item["groups"] == ["test"])
        self.assertTrue(all(item["groups"] != ["validation"] for item in plan[first_test:]))
        self.assertEqual(report["unique_examples"], len(ids))
        self.assertEqual(report["additional_flagged_training_examples"], len(report["additional_flagged_training_ids"]))
        self.assertFalse(set(report["additional_flagged_training_ids"]) & set(ids))


class FreezeTests(Workspace):
    def test_prerequisites(self):
        with self.assertRaises(nr.FreezeError) as context:
            self.freeze()
        self.assertIn("have no human decision", str(context.exception))
        self.assertFalse((self.frozen / "v1.0").exists())
        for example in self.by_split["validation"]:
            self.decide(example)
        with self.assertRaises(nr.FreezeError) as context:
            self.freeze()
        self.assertIn(f"test {len(self.by_split['test'])}", str(context.exception))
        with self.assertRaises(nr.FreezeError):
            self.freeze("version-one")

    def test_training_review_not_required_and_skips_do_not_count(self):
        self.decide_all_evaluation()
        manifest = self.freeze()
        self.assertEqual(manifest["review_status_by_split"]["train"], {"draft": len(self.by_split["train"])})

    def test_stale_and_modified_sources_refused(self):
        self.decide_all_evaluation()
        rows = nd.read_jsonl(self.dataset)
        rows[0]["claim"] += " Changed."
        nd.write_jsonl(self.dataset, rows)
        with self.assertRaises(nr.FreezeError) as context:
            self.freeze()
        self.assertIn("does not regenerate", str(context.exception))

    def test_freeze_outputs_hashes_and_immutability(self):
        evaluation = self.decide_all_evaluation()
        corrected, rejected = evaluation[0], evaluation[1]
        other = next(label for label in nd.LABEL_VALUES if label != corrected["label"])
        self.decide(corrected, "corrected", other, "wrong label", redo=True)
        self.decide(rejected, "rejected", reason="ambiguous", redo=True)
        manifest = self.freeze()
        target = self.frozen / "v1.0"
        self.assertEqual(sorted(p.name for p in target.iterdir()),
                         ["excluded.jsonl", "manifest.json", "reviews.snapshot.jsonl", "test.jsonl", "train.jsonl",
                          "validation.jsonl"])
        self.assertTrue(all(not os.access(p, os.W_OK) for p in target.iterdir()))
        self.assertEqual(nr.verify_frozen(target), [])
        rows = {e["example_id"]: e for split in nd.SPLITS for e in nd.read_jsonl(target / f"{split}.jsonl")}
        self.assertNotIn(rejected["example_id"], rows)
        self.assertEqual(rows[corrected["example_id"]]["label"], other)
        self.assertEqual(rows[corrected["example_id"]]["review"]["ai_label"], corrected["label"])
        self.assertEqual(manifest["counts"]["excluded_rejected"], 1)
        self.assertEqual(manifest["counts"]["corrected_labels"], 1)
        self.assertEqual(sum(manifest["counts"]["by_split"].values()), len(self.examples) - 1)
        self.assertIn("by_split_format_label", manifest["counts"])
        self.assertEqual(manifest["source_hashes"]["facts_sha256"], nr._sha256(self.facts.read_bytes()))
        self.assertEqual(manifest["source_hashes"]["draft_dataset_sha256"], nr._sha256(self.dataset.read_text()))
        effective = nr.effective_decisions(nr.load_decisions(self.reviews))
        self.assertEqual(manifest["review_hashes"]["effective_decisions_sha256"], nr.decisions_hash(effective))
        self.assertTrue(all(v == 0 for v in manifest["leakage_checks"].values()))
        # Tampering with a frozen file is detected.
        path = target / "test.jsonl"
        path.chmod(stat.S_IWUSR | stat.S_IRUSR)
        path.write_text(path.read_text().replace("SUPPORTED", "CONTRADICTED", 1))
        self.assertEqual(nr.verify_frozen(target), ["test.jsonl is missing or was modified."])

    def test_decision_hash_changes_with_decisions(self):
        evaluation = self.decide_all_evaluation()
        before = nr.decisions_hash(nr.effective_decisions(nr.load_decisions(self.reviews)))
        self.assertEqual(before, nr.decisions_hash(nr.effective_decisions(nr.load_decisions(self.reviews))))
        self.decide(evaluation[0], "rejected", reason="changed my mind", redo=True)
        self.assertNotEqual(before, nr.decisions_hash(nr.effective_decisions(nr.load_decisions(self.reviews))))

    def test_overwrite_protection(self):
        self.decide_all_evaluation()
        self.freeze()
        before = {p.name: p.read_bytes() for p in (self.frozen / "v1.0").iterdir()}
        with self.assertRaises(nr.FreezeError) as context:
            self.freeze()
        self.assertIn("never overwritten", str(context.exception))
        self.assertEqual(before, {p.name: p.read_bytes() for p in (self.frozen / "v1.0").iterdir()})
        self.assertEqual(self.freeze("v1.1")["version"], "v1.1")

    def test_cross_split_leakage_blocks_freeze(self):
        self.decide_all_evaluation()
        real_audit = nd.audit

        def leaky(*args, **kwargs):
            report = real_audit(*args, **kwargs)
            report["facts_crossing_splits"] = ["AAA-100.1-a"]
            return report

        with mock.patch("src.nli_dataset.audit", side_effect=leaky):
            with self.assertRaises(nr.FreezeError) as context:
                self.freeze()
        self.assertIn("leakage check failed: facts_crossing_splits", str(context.exception))
        self.assertFalse((self.frozen / "v1.0").exists())
        leaked = copy.deepcopy(self.examples)
        next(e for e in leaked if e["split"] == "train")["split"] = "test"
        self.assertTrue(nd.audit(leaked, DOCUMENTS, benchmark_claims=[])["facts_crossing_splits"]
                        or nd.audit(leaked, DOCUMENTS, benchmark_claims=[])["families_crossing_splits"])


class ShortcutAuditTests(Workspace):
    def test_tables_flags_and_no_rewrites(self):
        before = json.dumps(self.examples)
        report = nr.shortcut_audit(self.examples)
        self.assertEqual(json.dumps(self.examples), before)
        table = report["phenomenon_by_label"]
        self.assertTrue(table["direct"]["single_label"])
        self.assertEqual(sum(table[p]["n"] for p in table), len(self.examples))
        self.assertIn("direct", report["single_label_phenomena"])
        for cue in ("numbers_not_in_evidence", "direct_quotation", "claim_has_negation", "scope_terms_not_in_evidence",
                    "evidence_format", "claim_length"):
            self.assertIn(cue, report["cue_by_label"])
        self.assertIn("train", report["cue_only_baseline"])
        self.assertTrue(report["recommendations"])

    def test_cues(self):
        cues = nr.text_cues({"claim": "Refunds are not paid within 60 days.", "evidence": "Refunds are paid within 30 days.",
                             "evidence_format": "sentence"})
        self.assertTrue(cues["numbers_not_in_evidence"])
        self.assertTrue(cues["claim_has_negation"] and cues["negation_differs_from_evidence"])
        self.assertFalse(cues["direct_quotation"])
        self.assertEqual(cues["claim_length"], "<=8 words")


class CommandLineTests(Workspace):
    def setUp(self):
        super().setUp()
        loader = mock.patch("src.ingest.load_documents", return_value=DOCUMENTS)
        loader.start()
        self.addCleanup(loader.stop)

    def cli(self, *argv, **kwargs):
        return run_cli(nr.main, ["--dataset", str(self.dataset), "--reviews", str(self.reviews),
                                 "--frozen-dir", str(self.frozen), "--facts", str(self.facts), *argv], **kwargs)

    def test_commands(self):
        code, out, _ = self.cli("--plan")
        self.assertEqual(code, 0)
        self.assertIn("Unique examples to review", out)
        code, out, _ = self.cli("--status")
        self.assertIn("NOT ready", out)
        code, _, err = self.cli("--review", "--reviewer", "")
        self.assertEqual(code, 2)
        self.assertIn("--reviewer is required", err)
        with mock.patch.dict(os.environ, {nr.REVIEWER_ENV: "env-reviewer"}):
            code, out, _ = run_cli(nr.main, ["--dataset", str(self.dataset), "--reviews", str(self.reviews), "--review"],
                                   input_fn=lambda prompt, keys=iter(["a", "", "q"]): next(keys))
        self.assertEqual(nr.load_decisions(self.reviews)[0]["reviewer"], "env-reviewer")
        example_id = nr.load_decisions(self.reviews)[0]["example_id"]
        code, out, _ = self.cli("--history", example_id)
        self.assertIn("env-reviewer", out)
        code, _, err = self.cli("--freeze")
        self.assertEqual(code, 2)
        code, _, err = self.cli("--freeze", "--version", "v1.0")
        self.assertEqual(code, 1)
        self.assertIn("Freeze refused", err)


class RealDatasetPlanTests(unittest.TestCase):
    def test_real_plan_counts(self):
        examples = nd.read_jsonl(nd.DATASET_PATH)
        plan, report = nr.review_plan(examples)
        self.assertEqual((report["high_risk"], report["validation"], report["test"]), (70, 190, 196))
        self.assertEqual(report["unique_examples"], len({item["example_id"] for item in plan}))
        self.assertTrue(all(e["review"]["status"] == "draft" and e["review"]["human_label"] is None for e in examples))


if __name__ == "__main__":
    unittest.main()
