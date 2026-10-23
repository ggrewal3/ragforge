"""Tests for M6.2: the policy NLI dataset generator, validation, leakage audit, review workflow and statistics.

Synthetic documents and fact specifications exercise each rule in isolation;
a few tests check the real facts file and corpus. No network, no models.
"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from src import nli_dataset as nd
from test_generation import run_cli

DOCUMENTS = [
    {"document_id": "AAA-100", "title": "Domestic Returns Policy", "department": "X", "effective_date": "2026-01-01",
     "sections": [
         {"section_id": "AAA-100.1", "section_title": "Window",
          "text": "Domestic customers may return items within 30 days. Returns need a receipt. "
                  "Staff wear blue shirts on Fridays."},
         {"section_id": "AAA-100.2", "section_title": "Payment",
          "text": "Domestic refunds are paid within 5 business days. Gift cards are non-refundable."}]},
    {"document_id": "BBB-200", "title": "International Returns Policy", "department": "X", "effective_date": "2026-01-01",
     "sections": [
         {"section_id": "BBB-200.1", "section_title": "Window",
          "text": "International customers may return items within 45 days. Each return needs a customs form."}]},
    {"document_id": "CCC-300", "title": "Password Policy", "department": "Y", "effective_date": "2026-01-01",
     "sections": [
         {"section_id": "CCC-300.1", "section_title": "Length",
          "text": "Passwords must be at least 14 characters long. Lockout lasts 15 minutes after 10 failures."}]},
    {"document_id": "DDD-400", "title": "Badge Policy", "department": "Z", "effective_date": "2026-01-01",
     "sections": [
         {"section_id": "DDD-400.1", "section_title": "Badges",
          "text": "Visitors must wear a badge at all times. Lost badges are reported to reception within 1 day."}]},
]

SPEC = {
    "version": "test",
    "families": {"F1-returns": {"documents": ["AAA-100", "BBB-200"], "why": "siblings"},
                 "F2-passwords": {"documents": ["CCC-300"], "why": "x"},
                 "F3-badges": {"documents": ["DDD-400"], "why": "x"}},
    "facts": [
        {"fact_id": "AAA-100.1-a", "q": ["Domestic customers may return items within 30 days"],
         "counterpart": "BBB-200.1-a", "claims": [
            ["S", "direct", "Domestic customers may return items within 30 days.", "Restates.", {"scoped": True}],
            ["C", "numerical", "Domestic customers may return items within 60 days.", "30, not 60.", {"scoped": True}],
            ["I", "missing", "Domestic returns are inspected by a quality team.", "Not mentioned."]]},
        {"fact_id": "AAA-100.1-b", "q": ["Domestic customers may return items within 30 days", "Returns need a receipt"],
         "claims": [["S", "multi_sentence", "Domestic customers returning within 30 days need a receipt.", "Both."]]},
        {"fact_id": "AAA-100.2-a", "q": ["Domestic refunds are paid within 5 business days"], "claims": [
            ["S", "direct", "Domestic refunds are paid within 5 business days.", "Restates."]]},
        {"fact_id": "BBB-200.1-a", "q": ["International customers may return items within 45 days"], "claims": [
            ["S", "paraphrase", "International customers have 45 days to return items.", "Restates.", {"scoped": True}]]},
        {"fact_id": "BBB-200.1-b", "q": ["Each return needs a customs form"], "sentence_context": "'Each return' is unscoped.",
         "claims": [["S", "direct", "International returns need a customs form.", "Restates."]]},
        {"fact_id": "CCC-300.1-a", "q": ["Passwords must be at least 14 characters long"], "claims": [
            ["S", "direct", "Passwords must be at least 14 characters long.", "Restates."],
            ["C", "numerical", "Passwords must be at least 20 characters long.", "14, not 20."]]},
        {"fact_id": "DDD-400.1-a", "q": ["Visitors must wear a badge at all times"], "claims": [
            ["S", "direct", "Visitors must wear a badge at all times.", "Restates."],
            ["C", "negation", "Visitors may remove their badge in meeting rooms.", "At all times."]]},
    ],
}


class SyntheticCase(unittest.TestCase):
    def build(self, spec=None, benchmark=(), reviews=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "facts.json")
            path.write_text(json.dumps(spec or SPEC))
            _, facts, _ = nd.load_facts(path, DOCUMENTS)
        family_splits = {"F1-returns": "train", "F2-passwords": "validation", "F3-badges": "test"}
        examples, report = nd.generate(facts, family_splits, benchmark, reviews)
        return examples, report, family_splits

    def errors(self, examples):
        return nd.validate(examples, DOCUMENTS)


class SpecificationTests(SyntheticCase):
    def bad(self, mutate):
        spec = copy.deepcopy(SPEC)
        mutate(spec)
        with self.assertRaises(nd.DatasetError):
            self.build(spec)

    def test_invalid_source_identifiers(self):
        self.bad(lambda s: s["facts"][0].update(fact_id="ZZZ-999.1-a"))           # unknown section
        self.bad(lambda s: s["facts"][0].update(fact_id="AAA-100.1"))             # malformed fact ID
        self.bad(lambda s: s["facts"].append(copy.deepcopy(s["facts"][0])))       # duplicate fact ID
        self.bad(lambda s: s["facts"][0].update(counterpart="CCC-300.1-a"))       # counterpart in another family
        self.bad(lambda s: s["facts"][0].update(counterpart="NOPE-1.1-a"))
        self.bad(lambda s: s["families"]["F3-badges"]["documents"].remove("DDD-400"))   # document without family

    def test_quotes_must_be_exact_and_within_one_sentence(self):
        self.bad(lambda s: s["facts"][0].update(q=["Domestic customers may return items within 31 days"]))
        self.bad(lambda s: s["facts"][0].update(q=["within 30 days. Returns need"]))
        self.bad(lambda s: s["facts"][0].update(q=[]))

    def test_claim_shape_and_labels(self):
        self.bad(lambda s: s["facts"][0]["claims"].append(["X", "direct", "A claim.", "why"]))
        self.bad(lambda s: s["facts"][0]["claims"].append(["S", "made_up", "A claim.", "why"]))
        self.bad(lambda s: s["facts"][0]["claims"].append(["S", "direct", "  ", "why"]))
        self.bad(lambda s: s["facts"][0]["claims"].append(["S", "direct", "A claim.", ""]))
        self.bad(lambda s: s["facts"][0]["claims"].append(["S", "direct", "A claim.", "why", {"weird": 1}]))
        self.bad(lambda s: s["facts"][0]["claims"].append(list(s["facts"][0]["claims"][0][:4])))   # duplicate text


class GenerationTests(SyntheticCase):
    def test_schema_and_draft_status(self):
        examples, _, _ = self.build()
        self.assertEqual(self.errors(examples), [])
        for example in examples:
            self.assertEqual(set(nd.REQUIRED_FIELDS) - set(example), set())
            self.assertEqual(example["review"]["status"], "draft")
            self.assertEqual(example["review"]["label_source"], nd.LABEL_SOURCE)
            self.assertTrue(example["rationale"].startswith(example["label"]))

    def test_formats_and_rederived_labels(self):
        examples, report, _ = self.build()
        by = {(e["claim"], e["evidence_format"], e["provenance"]["derivation"], e["section_id"]): e for e in examples}
        # authored claims appear in both formats with the authored label
        direct = "Domestic customers may return items within 30 days."
        self.assertEqual(by[(direct, "sentence", "authored", "AAA-100.1")]["evidence"],
                         "Domestic customers may return items within 30 days.")
        self.assertEqual(by[(direct, "full_section", "authored", "AAA-100.1")]["evidence"], DOCUMENTS[0]["sections"][0]["text"])
        # a multi-sentence SUPPORTED claim: full section keeps the label, each lone sentence is INSUFFICIENT
        multi = "Domestic customers returning within 30 days need a receipt."
        self.assertEqual(by[(multi, "full_section", "authored", "AAA-100.1")]["label"], "SUPPORTED")
        partials = [e for e in examples if e["claim"] == multi and e["provenance"]["derivation"] == "partial_evidence"]
        self.assertEqual(len(partials), 2)
        self.assertTrue(all(e["label"] == "INSUFFICIENT_EVIDENCE" and e["evidence_format"] == "sentence"
                            for e in partials))
        # counterpart: the scoped domestic claim against international evidence is INSUFFICIENT, never CONTRADICTED
        counterpart = [e for e in examples if e["provenance"]["derivation"] == "counterpart"]
        self.assertEqual({e["section_id"] for e in counterpart}, {"BBB-200.1"})
        self.assertTrue(all(e["label"] == "INSUFFICIENT_EVIDENCE" for e in counterpart))
        self.assertIn("Domestic customers may return items within 60 days.", {e["claim"] for e in counterpart})
        self.assertTrue(all(e["family_id"] == "F1-returns" and e["split"] == "train" for e in counterpart))
        # sentence_context facts get no sentence example
        self.assertNotIn(("International returns need a customs form.", "sentence", "authored", "BBB-200.1"), by)
        self.assertEqual(len(report["claims_without_sentence_example_needing_context"]), 1)

    def test_distractor_rules(self):
        examples, _, _ = self.build()
        distractors = [e for e in examples if e["provenance"]["derivation"] == "distractor"]
        self.assertTrue(distractors)
        for e in distractors:
            self.assertEqual((e["label"], e["evidence_format"]), ("INSUFFICIENT_EVIDENCE", "sentence"))
            self.assertFalse(nd.numbers_in(e["evidence"]) & nd.numbers_in(e["claim"]))
        texts = {e["evidence"] for e in distractors}
        self.assertIn("Staff wear blue shirts on Fridays.", texts)

    def test_deterministic_and_stable_ids(self):
        first, _, _ = self.build()
        second, _, _ = self.build()
        self.assertEqual(json.dumps(first), json.dumps(second))
        self.assertEqual(len({e["example_id"] for e in first}), len(first))

    def test_benchmark_near_duplicates_excluded(self):
        examples, report, _ = self.build(benchmark=[("V900", "Passwords must be at least 14 characters long.")])
        self.assertNotIn("Passwords must be at least 14 characters long.", {e["claim"] for e in examples})
        self.assertEqual(report["excluded_benchmark_near_duplicates"][0]["benchmark_matches"][0][0], "V900")

    def test_group_based_splitting(self):
        sizes = {"F1": 50, "F2": 10, "F3": 12, "F4": 9, "F5": 8, "F6": 11}
        splits = nd.assign_splits(sizes)
        self.assertEqual(splits, nd.assign_splits(sizes))       # deterministic
        self.assertEqual(set(splits), set(sizes))
        self.assertEqual(set(splits.values()), set(nd.SPLITS))
        self.assertEqual(splits["F1"], "train")                 # the largest family goes to train first
        with self.assertRaises(nd.DatasetError):
            nd.assign_splits({})
        examples, _, family_splits = self.build()
        self.assertTrue(all(e["split"] == family_splits[e["family_id"]] for e in examples))


class ValidationAndAuditTests(SyntheticCase):
    def setUp(self):
        self.examples, _, _ = self.build()

    def test_empty_dataset(self):
        self.assertEqual(nd.validate([], DOCUMENTS), ["Dataset is empty."])

    def test_label_format_and_text_validation(self):
        cases = {"label": "MAYBE", "evidence_format": "paragraph", "split": "dev", "phenomenon": "vibes",
                 "claim": " ", "rationale": ""}
        for field, value in cases.items():
            with self.subTest(field):
                broken = copy.deepcopy(self.examples)
                broken[0][field] = value
                self.assertTrue(self.errors(broken))
        broken = copy.deepcopy(self.examples)
        del broken[0]["provenance"]
        self.assertIn("missing fields", self.errors(broken)[0])

    def test_evidence_format_consistency_and_provenance(self):
        sentence = next(i for i, e in enumerate(self.examples) if e["evidence_format"] == "sentence")
        full = next(i for i, e in enumerate(self.examples) if e["evidence_format"] == "full_section")
        mutations = [
            (sentence, lambda e: e.update(evidence=e["evidence"][:-1])),                 # not an exact sentence
            (full, lambda e: e.update(evidence=e["evidence"] + " Extra.")),              # not the exact section
            (full, lambda e: e.update(section_id="ZZZ-999.1")),                          # invalid source identifier
            (full, lambda e: e["provenance"].update(section_text_sha256="0" * 64)),      # changed source text
            (full, lambda e: e["provenance"].update(derivation="invented")),
            (full, lambda e: e.update(claim_id="OTHER-1.1-a/abc")),
            (full, lambda e: e.update(fact_id="CCC-300.1-a")),                           # fact not from this claim
        ]
        for index, mutate in mutations:
            broken = copy.deepcopy(self.examples)
            mutate(broken[index])
            self.assertTrue(self.errors(broken), mutate)

    def test_duplicate_detection(self):
        duplicate = copy.deepcopy(self.examples[0])
        duplicate["example_id"] = "nli-duplicate"
        self.assertTrue(any("duplicate evidence-claim pair" in e for e in self.errors(self.examples + [duplicate])))
        report = nd.audit(self.examples + [duplicate], DOCUMENTS, benchmark_claims=[])
        self.assertEqual(len(report["exact_duplicate_pairs"]), 1)
        duplicate["label"] = "CONTRADICTED" if duplicate["label"] != "CONTRADICTED" else "SUPPORTED"
        report = nd.audit(self.examples + [duplicate], DOCUMENTS, benchmark_claims=[])
        self.assertEqual(len(report["conflicting_label_pairs"]), 1)

    def test_cross_split_leakage_detection(self):
        leaked = copy.deepcopy(self.examples)
        target = next(e for e in leaked if e["fact_id"] == "AAA-100.1-a")
        target["split"] = "test"
        errors = self.errors(leaked)
        self.assertTrue(any("more than one split" in e for e in errors))
        report = nd.audit(leaked, DOCUMENTS, benchmark_claims=[])
        self.assertIn("AAA-100.1-a", report["facts_crossing_splits"])
        self.assertIn("F1-returns", report["families_crossing_splits"])
        self.assertTrue(report["evidence_sentences_crossing_splits"] or report["evidence_texts_crossing_splits"])
        family_splits = {"F1-returns": "train", "F2-passwords": "validation", "F3-badges": "test"}
        self.assertTrue(any("manifest" in e for e in nd.validate(leaked, DOCUMENTS, family_splits)))

    def test_clean_audit_and_small_split_limitation(self):
        report = nd.audit(self.examples, DOCUMENTS, benchmark_claims=[("V1", "Visitors must always wear badges.")])
        for key in ("exact_duplicate_pairs", "conflicting_label_pairs", "facts_crossing_splits",
                    "families_crossing_splits", "evidence_texts_crossing_splits"):
            self.assertEqual(report[key], [], key)
        self.assertEqual(report["independent_groups"]["families"], 3)
        self.assertTrue(any("only 1 policy families" in limitation for limitation in report["limitations"]))
        self.assertEqual(report["benchmark"]["similarity_to_dataset_claims"][0]["benchmark_id"], "V1")

    def test_near_duplicate_claims_across_splits(self):
        moved = copy.deepcopy(self.examples)
        clone = copy.deepcopy(next(e for e in moved if e["split"] == "validation"))
        clone.update(split="test", family_id="F3-badges", claim=clone["claim"].replace(".", " today."))
        report = nd.audit(moved + [clone], DOCUMENTS, benchmark_claims=[])
        self.assertTrue(report["near_duplicate_claims_across_splits"])

    def test_heuristic_label_flags(self):
        def example(label, claim, evidence, derivation="authored"):
            return {"label": label, "claim": claim, "evidence": evidence,
                    "provenance": {"derivation": derivation, "decisive_sentences": [evidence]}}

        self.assertIn("supported_with_numbers_not_in_evidence",
                      nd.label_flags(example("SUPPORTED", "Refunds take 60 days.", "Refunds take 30 days.")))
        self.assertIn("negation_mismatch",
                      nd.label_flags(example("SUPPORTED", "Gift cards are refundable.", "Gift cards are not refundable.")))
        self.assertIn("scope_terms_not_in_evidence",
                      nd.label_flags(example("SUPPORTED", "International refunds take 30 days.", "Refunds take 30 days.")))
        self.assertIn("contradiction_with_low_overlap",
                      nd.label_flags(example("CONTRADICTED", "Badges are blue.", "Passwords expire yearly.")))
        self.assertIn("insufficient_but_looks_supported",
                      nd.label_flags(example("INSUFFICIENT_EVIDENCE", "Refunds take 30 days.", "Refunds take 30 days.")))
        self.assertEqual(nd.label_flags(example("SUPPORTED", "Refunds take 30 days.", "Refunds take 30 days.")), [])


class ReviewTests(SyntheticCase):
    def review(self, example, status, corrected=None, fingerprint=None):
        return {"example_id": example["example_id"], "fingerprint": fingerprint or example["review"]["fingerprint"],
                "status": status, "corrected_label": corrected, "reviewer": "tester", "note": "n",
                "reviewed_at": "2026-10-22T00:00:00+00:00"}

    def test_corrections_are_traceable_and_stale_reviews_ignored(self):
        examples, _, _ = self.build()
        first, second, third = examples[0], examples[1], examples[2]
        reviews = [self.review(first, "reviewed"), self.review(second, "corrected", "INSUFFICIENT_EVIDENCE"),
                   self.review(third, "reviewed", fingerprint="stale"),
                   {**self.review(first, "rejected"), "reviewed_at": "2026-10-23T00:00:00+00:00"}]   # latest wins
        reviewed, report, _ = self.build(reviews=reviews)
        by_id = {e["example_id"]: e for e in reviewed}
        self.assertEqual(by_id[first["example_id"]]["review"]["status"], "rejected")
        corrected = by_id[second["example_id"]]
        self.assertEqual(corrected["label"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(corrected["review"]["ai_label"], second["label"])
        self.assertEqual(corrected["review"]["human_label"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(corrected["review"]["label_source"], "human-reviewed")
        self.assertEqual(by_id[third["example_id"]]["review"]["status"], "draft")
        self.assertEqual(report["reviews"]["stale"], [third["example_id"]])
        self.assertEqual(self.errors(reviewed), [])

    def test_review_status_validation(self):
        examples, _, _ = self.build()
        broken = copy.deepcopy(examples)
        broken[0]["review"]["status"] = "approved"
        self.assertTrue(self.errors(broken))
        broken = copy.deepcopy(examples)
        broken[0]["review"]["label_source"] = "human-reviewed"       # draft must not claim human review
        self.assertTrue(self.errors(broken))
        broken = copy.deepcopy(examples)
        broken[0]["review"].update(status="corrected", reviewer="x", human_label=broken[0]["label"])
        self.assertTrue(self.errors(broken))                          # a correction must change the label
        broken = copy.deepcopy(examples)
        broken[0]["review"]["human_label"] = broken[0]["label"]      # draft with a human label
        self.assertTrue(self.errors(broken))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "reviews.jsonl")
            path.write_text(json.dumps({"example_id": "x", "fingerprint": "f", "status": "approved",
                                        "reviewer": "r"}) + "\n")
            with self.assertRaises(nd.DatasetError):
                nd.load_reviews(path)
            path.write_text(json.dumps({"example_id": "x", "fingerprint": "f", "status": "corrected",
                                        "corrected_label": None, "reviewer": "r"}) + "\n")
            with self.assertRaises(nd.DatasetError):
                nd.load_reviews(path)
            self.assertEqual(nd.load_reviews(Path(directory, "missing.jsonl")), [])

    def test_queue_priorities(self):
        examples, _, _ = self.build()
        queue = nd.review_queue(examples, DOCUMENTS)
        self.assertEqual(len(queue), len(examples))
        self.assertEqual([q["priority"] for q in queue], sorted((q["priority"] for q in queue), reverse=True))
        self.assertTrue(all(q["original_section_text"] for q in queue))
        reviewed = nd.review_queue([dict(e, review={**e["review"], "status": "reviewed"}) for e in examples], DOCUMENTS)
        self.assertEqual(reviewed, [])


class StatisticsTests(SyntheticCase):
    def test_stats(self):
        examples, _, _ = self.build()
        stats = nd.compute_stats(examples)
        self.assertEqual(stats["examples"], len(examples))
        self.assertEqual(sum(s["examples"] for s in stats["splits"].values()), len(examples))
        self.assertEqual(sum(stats["labels"].values()), len(examples))
        for split in stats["splits"].values():
            if split["examples"]:
                self.assertAlmostEqual(sum(split["label_shares"].values()), 1.0, places=2)
        self.assertEqual(stats["independent_groups"]["families"], 3)
        self.assertEqual(stats["review_status"], {"draft": len(examples)})
        self.assertEqual(set(stats["tokens"]["by_format"]), {"sentence", "full_section"})
        self.assertEqual(stats["tokens"]["by_format"]["sentence"]["over_512"], 0)
        count, source = nd.token_counter(None)
        self.assertIn("approximate", source)
        self.assertGreater(count("a b c", "d"), 4)


class RealDatasetTests(unittest.TestCase):
    """The committed facts file must generate a valid, leak-free dataset."""

    @classmethod
    def setUpClass(cls):
        cls.examples, cls.manifest = nd.build()

    def test_valid_and_deterministic(self):
        self.assertEqual(nd.validate(self.examples, family_splits=self.manifest["family_splits"]), [])
        again, _ = nd.build()
        self.assertEqual(json.dumps(again), json.dumps(self.examples))
        self.assertTrue(900 <= len(self.examples) <= 1400)

    def test_coverage_and_leakage(self):
        phenomena = {e["phenomenon"] for e in self.examples}
        self.assertEqual(phenomena, set(nd.PHENOMENA))
        self.assertEqual({e["label"] for e in self.examples}, set(nd.LABEL_VALUES))
        self.assertEqual({e["evidence_format"] for e in self.examples}, set(nd.FORMATS))
        report = nd.audit(self.examples, manifest=self.manifest)
        for key in ("exact_duplicate_pairs", "conflicting_label_pairs", "facts_crossing_splits",
                    "families_crossing_splits", "evidence_texts_crossing_splits", "evidence_sentences_crossing_splits"):
            self.assertEqual(report[key], [], key)
        self.assertTrue(all(e["review"]["status"] == "draft" for e in self.examples))

    def test_benchmark_and_held_out_split_untouched(self):
        text = nd.BENCHMARK_PATH.read_text(encoding="utf-8")
        self.assertEqual(len(json.loads(text)["examples"]), 40)
        claims = {e["claim"].casefold() for e in self.examples}
        for _, benchmark_claim in nd.load_benchmark_claims():
            self.assertNotIn(benchmark_claim.casefold(), claims)


class CommandLineTests(unittest.TestCase):
    def test_generate_validate_and_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory, "ds.jsonl")
            code, stdout, _ = run_cli(nd.main, ["--generate", "--out", str(out)])
            self.assertEqual(code, 0, stdout)
            self.assertTrue(nd.manifest_path(out).is_file())
            code, _, err = run_cli(nd.main, ["--generate", "--out", str(out)])
            self.assertEqual(code, 2)
            self.assertIn("--force", err)
            code, stdout, _ = run_cli(nd.main, ["--validate", "--dataset", str(out)])
            self.assertEqual(code, 0, stdout)
            self.assertIn("regeneration is deterministic", stdout)
            rows = nd.read_jsonl(out)
            rows[0]["label"] = "SUPPORTED" if rows[0]["label"] != "SUPPORTED" else "CONTRADICTED"
            nd.write_jsonl(out, rows)
            code, stdout, _ = run_cli(nd.main, ["--validate", "--dataset", str(out)])
            self.assertEqual(code, 1)
            self.assertIn("does not reproduce", stdout)
            queue = Path(directory, "queue.jsonl")
            code, _, _ = run_cli(nd.main, ["--review-queue", "--dataset", str(out), "--out", str(queue)])
            self.assertEqual(code, 0)
            code, _, err = run_cli(nd.main, ["--review", "--dataset", str(out)])   # moved to src.nli_review
            self.assertEqual(code, 2)
        code, _, err = run_cli(nd.main, ["--generate", "--out", str(nd.PROTECTED_DIR / "x.jsonl"), "--force"])
        self.assertEqual(code, 2)
        self.assertIn("protected", err)


if __name__ == "__main__":
    unittest.main()
