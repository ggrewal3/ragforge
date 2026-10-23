"""Tests for the M6.2 random quality audit and experimental freeze.

Decisions are written by the tests into temporary logs for a synthetic corpus;
the real data/nli/reviews.jsonl is never read for writing. No network, no models.
"""

import os
import unittest
from pathlib import Path
from unittest import mock

from src import nli_dataset as nd
from src import nli_review as nr
from test_generation import run_cli
from test_nli_dataset import DOCUMENTS
from test_nli_review import NOW, Workspace

SIZE = 10


class SamplingTests(Workspace):
    def test_stratified_unique_and_reproducible(self):
        sample = nr.stratified_sample(self.examples, SIZE, seed=7)
        ids = sample["example_ids"]
        self.assertEqual(len(ids), SIZE)
        self.assertEqual(len(set(ids)), SIZE)
        self.assertEqual(sample, nr.stratified_sample(self.examples, SIZE, seed=7))
        self.assertEqual(sum(sample["allocation"].values()), SIZE)
        by_id = {e["example_id"]: e for e in self.examples}
        chosen = [by_id[i] for i in ids]
        self.assertEqual({e["review"]["ai_label"] for e in chosen}, set(nd.LABEL_VALUES))
        self.assertEqual({e["evidence_format"] for e in chosen}, set(nd.FORMATS))
        self.assertEqual({e["split"] for e in chosen}, set(nd.SPLITS))
        self.assertNotEqual(ids, nr.stratified_sample(self.examples, SIZE, seed=8)["example_ids"])

    def test_allocation_is_proportional(self):
        sample = nr.stratified_sample(self.examples, SIZE, seed=7)
        for stratum, count in sample["allocation"].items():
            quota = SIZE * sample["population_by_stratum"][stratum] / len(self.examples)
            self.assertLessEqual(abs(count - quota), 1.0 + 1e-9, stratum)

    def test_sample_ignores_priority_and_decisions(self):
        before = nr.stratified_sample(self.examples, SIZE, seed=7)
        for example in self.examples[:5]:
            self.decide(example)
        self.assertEqual(before, nr.stratified_sample(self.examples, SIZE, seed=7))
        with self.assertRaises(ValueError):
            nr.stratified_sample(self.examples, 0)
        with self.assertRaises(ValueError):
            nr.stratified_sample(self.examples, len(self.examples) + 1)

    def test_real_dataset_sample(self):
        examples = nd.read_jsonl(nd.DATASET_PATH)
        sample = nr.stratified_sample(examples)
        dist = nr.sample_distribution(examples, sample)
        self.assertEqual(len(set(sample["example_ids"])), 50)
        self.assertEqual(set(dist["label"]), set(nd.LABEL_VALUES))
        self.assertEqual(set(dist["format"]), set(nd.FORMATS))
        self.assertEqual(set(dist["split"]), set(nd.SPLITS))
        queue_top = [item["example_id"] for item in nd.review_queue(examples)[:50]]
        self.assertLess(len(set(queue_top) & set(sample["example_ids"])), 25)   # not the risk-ordered queue


class SampleReviewTests(Workspace):
    def setUp(self):
        super().setUp()
        self.sample = nr.stratified_sample(self.examples, SIZE, seed=7)
        self.plan = nr.sample_plan(self.examples, self.sample, DOCUMENTS)
        self.by_id = {e["example_id"]: e for e in self.examples}

    def test_progress_resume_and_existing_decisions(self):
        self.decide(self.by_id[self.sample["example_ids"][0]])            # an earlier decision is reused
        keys = iter(["s", "a", "", "q"])
        output = []
        summary = nr.run_session(self.examples, self.plan, self.reviews, "test-reviewer",
                                 input_fn=lambda prompt: next(keys), output_fn=output.append, now=NOW)
        text = "\n".join(output)
        self.assertIn(f"decided 1/{SIZE}", text)
        self.assertIn("audit-sample", text)
        self.assertNotIn(f"Example: {self.sample['example_ids'][0]}", text)      # no duplicate work
        self.assertEqual((summary["recorded"], summary["skipped"]), (1, 1))
        decided = nr.effective_decisions(nr.load_decisions(self.reviews))
        self.assertNotIn(self.sample["example_ids"][1], decided)                  # skipped stays undecided
        self.assertIn(self.sample["example_ids"][2], decided)
        stats = nr.progress(self.plan, decided)
        self.assertEqual(stats["audit_sample"], {"decided": 2, "total": SIZE})

    def test_agreement_calculations(self):
        ids = self.sample["example_ids"]
        self.decide(self.by_id[ids[0]])
        self.decide(self.by_id[ids[1]])
        self.decide(self.by_id[ids[2]])
        wrong = self.by_id[ids[3]]
        other = next(label for label in nd.LABEL_VALUES if label != wrong["label"])
        self.decide(wrong, "corrected", other, "number differs")
        self.decide(self.by_id[ids[4]], "rejected", reason="claim is ambiguous")
        self.decide(next(e for e in self.examples if e["example_id"] not in ids))      # outside the sample
        report = nr.sample_report(self.examples, self.sample, nr.effective_decisions(nr.load_decisions(self.reviews)))
        self.assertEqual((report["reviewed"], report["accepted"], report["corrected"], report["rejected"]), (5, 3, 1, 1))
        self.assertEqual(report["label_agreement"], 0.75)                  # 3 of 4 labelled (rejections excluded)
        self.assertEqual(report["acceptance_rate"], 0.6)
        low, high = report["ci95"]
        self.assertTrue(0 < low < 0.75 < high <= 1)
        self.assertEqual(report["remaining_in_sample"], SIZE - 5)
        self.assertEqual(report["decisions_outside_sample"], 1)
        self.assertEqual(report["corrections"][0]["human_label"], other)
        self.assertEqual(report["rejections"][0]["reason"], "claim is ambiguous")
        cls = report["by_class"][wrong["label"]]
        self.assertGreaterEqual(cls["corrected"], 1)
        self.assertEqual(sum(v["reviewed"] for v in report["by_format"].values()), 5)
        self.assertEqual(sum(v["reviewed"] for v in report["by_split"].values()), 5)
        self.assertTrue(any("below 90%" in w for w in report["warnings"]))
        self.assertTrue(any("rejected" in w for w in report["warnings"]))
        self.assertTrue(any("incomplete" in w for w in report["warnings"]))

    def test_wilson_interval(self):
        self.assertEqual(nr.wilson_interval(0, 0), (None, None))
        low, high = nr.wilson_interval(48, 50)
        self.assertTrue(0.85 < low < 0.96 < high <= 1.0)


class ExperimentalFreezeTests(Workspace):
    def setUp(self):
        super().setUp()
        self.sample = nr.stratified_sample(self.examples, SIZE)
        self.by_id = {e["example_id"]: e for e in self.examples}

    def freeze_exp(self, version="v0.1-exp"):
        return nr.freeze_experimental(version, self.dataset, self.reviews, self.frozen, self.facts,
                                      documents=DOCUMENTS, now=NOW, sample_size=SIZE)

    def decide_sample(self, correct_first=False):
        for n, example_id in enumerate(self.sample["example_ids"]):
            example = self.by_id[example_id]
            if correct_first and n == 0:
                other = next(label for label in nd.LABEL_VALUES if label != example["label"])
                self.decide(example, "corrected", other, "wrong label")
            else:
                self.decide(example)

    def test_requirements(self):
        with self.assertRaises(nr.FreezeError) as context:
            self.freeze_exp()
        self.assertIn(f"{SIZE} of {SIZE} sampled examples have no human decision", str(context.exception))
        with self.assertRaises(nr.FreezeError):
            self.freeze_exp("v0.1")                                   # must be marked experimental
        self.decide_sample()
        self.decide(self.by_id[self.sample["example_ids"][0]], "rejected", reason="defective", redo=True)
        with self.assertRaises(nr.FreezeError) as context:
            self.freeze_exp()
        self.assertIn("sampled examples were rejected", str(context.exception))
        self.assertFalse((self.frozen / "v0.1-exp").exists())

    def test_tampered_log_stale_and_leakage_block_freeze(self):
        self.decide_sample()
        lines = self.reviews.read_text().splitlines()
        self.reviews.write_text("\n".join(lines[1:]) + "\n")
        with self.assertRaises(nr.FreezeError) as context:
            self.freeze_exp()
        self.assertIn("hash chain is broken", str(context.exception))
        self.reviews.write_text("\n".join(lines) + "\n")
        real_audit = nd.audit

        def leaky(*args, **kwargs):
            report = real_audit(*args, **kwargs)
            report["evidence_texts_crossing_splits"] = ["x"]
            return report

        with mock.patch("src.nli_dataset.audit", side_effect=leaky), self.assertRaises(nr.FreezeError) as context:
            self.freeze_exp()
        self.assertIn("leakage check failed", str(context.exception))
        rows = nd.read_jsonl(self.dataset)
        rows[0]["claim"] += " Edited."
        nd.write_jsonl(self.dataset, rows)
        with self.assertRaises(nr.FreezeError) as context:
            self.freeze_exp()
        self.assertIn("does not regenerate", str(context.exception))

    def test_freeze_contents_and_corrections(self):
        self.decide_sample(correct_first=True)
        outside = next(e for e in self.examples if e["example_id"] not in self.sample["example_ids"])
        self.decide(outside, "rejected", reason="defective outside the sample")
        manifest = self.freeze_exp()
        target = self.frozen / "v0.1-exp"
        self.assertEqual(manifest["dataset_type"], "experimental_ai_labeled_sample_audited")
        self.assertEqual(sorted(p.name for p in target.iterdir()),
                         ["audit_sample.json", "excluded.jsonl", "manifest.json", "reviews.snapshot.jsonl",
                          "test.jsonl", "train.jsonl", "validation.jsonl"])
        self.assertTrue(all(not os.access(p, os.W_OK) for p in target.iterdir()))
        self.assertEqual(nr.verify_frozen(target), [])
        sample = manifest["audit_sample"]
        self.assertEqual((sample["size"], sample["reviewed"], sample["corrected"], sample["rejected"]), (SIZE, SIZE, 1, 0))
        self.assertEqual(sample["seed"], nr.AUDIT_SAMPLE_SEED)
        labels = manifest["labels"]
        self.assertEqual(labels["human_reviewed"], SIZE)
        self.assertEqual(labels["rejected_excluded"], 1)
        self.assertEqual(labels["remaining_draft"], len(self.examples) - SIZE - 1)
        rows = {e["example_id"]: e for split in nd.SPLITS for e in nd.read_jsonl(target / f"{split}.jsonl")}
        self.assertNotIn(outside["example_id"], rows)
        corrected = rows[self.sample["example_ids"][0]]
        self.assertNotEqual(corrected["label"], corrected["review"]["ai_label"])
        self.assertEqual(corrected["label"], corrected["review"]["human_label"])
        drafts = [e for e in rows.values() if e["example_id"] not in self.sample["example_ids"]]
        self.assertTrue(all(e["review"]["status"] == "draft" and e["review"]["human_label"] is None for e in drafts))
        self.assertTrue(all(e["review"]["label_source"] == nd.LABEL_SOURCE for e in drafts))
        self.assertEqual(manifest["source_hashes"]["draft_dataset_sha256"], nr._sha256(self.dataset.read_text()))
        self.assertEqual(manifest["review_hashes"]["review_log_sha256"], nr._sha256(self.reviews.read_bytes()))
        self.assertEqual(sum(manifest["counts"]["by_split"].values()), len(self.examples) - 1)
        self.assertTrue(manifest["known_limitations"])
        self.assertTrue(all(v == 0 for v in manifest["leakage_checks"].values()))

    def test_overwrite_protection_and_strict_mode_unchanged(self):
        self.decide_sample()
        self.freeze_exp()
        with self.assertRaises(nr.FreezeError) as context:
            self.freeze_exp()
        self.assertIn("never overwritten", str(context.exception))
        # The strict freeze still requires every validation and test example and rejects -exp names.
        evaluation = [e for e in self.examples if e["split"] in nr.EVALUATION_SPLITS]
        if any(e["example_id"] not in self.sample["example_ids"] for e in evaluation):
            with self.assertRaises(nr.FreezeError) as context:
                nr.freeze("v1.0", self.dataset, self.reviews, self.frozen, self.facts, documents=DOCUMENTS)
            self.assertIn("have no human decision", str(context.exception))
        with self.assertRaises(nr.FreezeError):
            nr.freeze("v0.1-exp", self.dataset, self.reviews, self.frozen, self.facts, documents=DOCUMENTS)

    def test_load_frozen_for_training(self):
        self.decide_sample()
        self.freeze_exp()
        train, manifest = nr.load_frozen("v0.1-exp", "train", self.frozen)
        validation, _ = nr.load_frozen("v0.1-exp", "validation", self.frozen)
        self.assertEqual(len(train), manifest["counts"]["by_split"]["train"])
        self.assertEqual(len(validation), manifest["counts"]["by_split"]["validation"])
        self.assertTrue(all(e["label"] in nd.LABEL_VALUES for e in train))
        with self.assertRaises(ValueError):
            nr.load_frozen("v0.1-exp", "test", self.frozen)                       # reserved
        test_rows, _ = nr.load_frozen("v0.1-exp", "test", self.frozen, allow_test=True)
        self.assertEqual(len(test_rows), manifest["counts"]["by_split"]["test"])
        path = self.frozen / "v0.1-exp" / "train.jsonl"
        path.chmod(0o644)
        path.write_text(path.read_text() + "\n")
        with self.assertRaises(ValueError):
            nr.load_frozen("v0.1-exp", "train", self.frozen)                      # verification fails
        with self.assertRaises(FileNotFoundError):
            nr.load_frozen("v9.9-exp", "train", self.frozen)


class SampleCommandLineTests(Workspace):
    def setUp(self):
        super().setUp()
        loader = mock.patch("src.ingest.load_documents", return_value=DOCUMENTS)
        loader.start()
        self.addCleanup(loader.stop)
        size = mock.patch.object(nr, "AUDIT_SAMPLE_SIZE", SIZE)
        size.start()
        self.addCleanup(size.stop)
        path = mock.patch.object(nr, "AUDIT_SAMPLE_PATH", Path(self.tmp.name) / "sample.json")
        path.start()
        self.addCleanup(path.stop)

    def cli(self, *argv, **kwargs):
        return run_cli(nr.main, ["--dataset", str(self.dataset), "--reviews", str(self.reviews), "--frozen-dir",
                                 str(self.frozen), "--facts", str(self.facts), *argv], **kwargs)

    def test_sample_review_report_and_freeze(self):
        code, out, _ = self.cli("--sample")
        self.assertEqual(code, 0, out)
        self.assertIn(f"Audit sample: {SIZE} unique examples", out)
        code, _, err = self.cli("--review-sample")
        self.assertEqual(code, 2)
        self.assertIn("--reviewer", err)
        keys = iter(["a", ""] * SIZE)
        code, out, _ = self.cli("--review-sample", "--reviewer", "test-reviewer",
                                input_fn=lambda prompt: next(keys))
        self.assertEqual(code, 0)
        self.assertEqual(len(nr.load_decisions(self.reviews)), SIZE)
        code, out, _ = self.cli("--sample-report")
        self.assertIn(f"{SIZE} of {SIZE} sampled examples reviewed", out)
        self.assertIn("100.0%", out)
        code, out, _ = self.cli("--status")
        self.assertIn(f"Audit sample: {SIZE}/{SIZE} decided", out)
        code, out, err = self.cli("--freeze-experimental", "--version", "v0.1-exp")
        self.assertEqual(code, 0, err)
        self.assertIn("experimental_ai_labeled_sample_audited", out)
        code, out, _ = self.cli("--verify-frozen", "v0.1-exp")
        self.assertEqual(code, 0)
        code, _, err = self.cli("--freeze-experimental", "--version", "v0.1-exp")
        self.assertEqual(code, 1)
        self.assertIn("never overwritten", err)


if __name__ == "__main__":
    unittest.main()
