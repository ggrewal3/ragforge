"""Tests for RAGForge M1.

Most tests use only Python and NumPy and run in well under a second.
Tests that need the real embedding model are in ModelIntegrationTests and run
only when RAGFORGE_MODEL_TESTS=1 is set:

    python -m unittest discover -s tests -v
    RAGFORGE_MODEL_TESTS=1 python -m unittest discover -s tests -v
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.chunking import DEFAULT_MAX_WORDS, chunk_documents, split_text
from src.embeddings import normalize_rows
from src.ingest import DocumentValidationError, REQUIRED_DOCUMENT_FIELDS, load_documents, validate_document
from src.evaluate import (
    CATEGORIES,
    SPLITS,
    assign_splits,
    failure_analysis,
    first_relevant_rank,
    hit_at_k,
    load_evaluation_questions,
    recall_at_k,
    reciprocal_rank_at_k,
    score_question,
    summarize,
    unique_document_ids,
    validate_questions,
)
from src.retrieve import cosine_similarities, cosine_similarity, rank_chunks, retrieve

RUN_MODEL_TESTS = os.environ.get("RAGFORGE_MODEL_TESTS") == "1"


def make_document(document_id="DOC-1", **overrides):
    document = {
        "document_id": document_id,
        "title": "Test Policy",
        "department": "Testing",
        "effective_date": "2026-01-01",
        "sections": [{"section_id": f"{document_id}.1", "section_title": "Rules", "text": "Some rule text."}],
    }
    document.update(overrides)
    return document


class IngestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents = load_documents()

    def test_exactly_24_documents_load(self):
        self.assertEqual(len(self.documents), 24)

    def test_document_ids_are_unique(self):
        ids = [document["document_id"] for document in self.documents]
        self.assertEqual(len(ids), len(set(ids)))

    def test_required_metadata_exists(self):
        for document in self.documents:
            for field in REQUIRED_DOCUMENT_FIELDS:
                self.assertTrue(document[field], f"{document.get('document_id')} missing {field}")
            self.assertGreaterEqual(len(document["sections"]), 3, document["document_id"])

    def test_four_departments_with_six_documents_each(self):
        counts = {}
        for document in self.documents:
            counts[document["department"]] = counts.get(document["department"], 0) + 1
        self.assertEqual(
            counts,
            {"Human Resources": 6, "Finance": 6, "Security": 6, "Customer Policies": 6},
        )

    def test_missing_field_is_rejected(self):
        raw = make_document()
        del raw["title"]
        with self.assertRaisesRegex(DocumentValidationError, "title"):
            validate_document(raw, "bad.json")

    def test_empty_section_text_is_rejected(self):
        raw = make_document(sections=[{"section_id": "X.1", "section_title": "Rules", "text": "   "}])
        with self.assertRaisesRegex(DocumentValidationError, "text"):
            validate_document(raw, "bad.json")

    def test_empty_sections_are_rejected(self):
        with self.assertRaisesRegex(DocumentValidationError, "sections"):
            validate_document(make_document(sections=[]), "bad.json")

    def test_invalid_date_is_rejected(self):
        with self.assertRaisesRegex(DocumentValidationError, "effective_date"):
            validate_document(make_document(effective_date="January 2026"), "bad.json")

    def test_duplicate_document_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("a.json", "b.json"):
                Path(directory, name).write_text(json.dumps(make_document("DUP-1")))
            with self.assertRaisesRegex(DocumentValidationError, "Duplicate document_id"):
                load_documents(directory)


class ChunkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents = load_documents()
        cls.chunks = chunk_documents(cls.documents)
        cls.documents_by_id = {document["document_id"]: document for document in cls.documents}

    def test_chunk_ids_are_unique(self):
        ids = [chunk["chunk_id"] for chunk in self.chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_chunk_preserves_its_source(self):
        for chunk in self.chunks:
            document = self.documents_by_id[chunk["document_id"]]
            self.assertEqual(chunk["title"], document["title"])
            self.assertEqual(chunk["department"], document["department"])
            section = next(s for s in document["sections"] if s["section_id"] == chunk["section_id"])
            self.assertEqual(chunk["section_title"], section["section_title"])
            # The chunk text must come from that section only (no mixing across sources).
            self.assertIn(chunk["text"], " ".join(section["text"].split()))

    def test_chunk_text_is_nonempty(self):
        for chunk in self.chunks:
            self.assertTrue(chunk["text"].strip(), chunk["chunk_id"])

    def test_chunk_sizes_respect_default_limit(self):
        for chunk in self.chunks:
            self.assertLessEqual(len(chunk["text"].split()), DEFAULT_MAX_WORDS, chunk["chunk_id"])

    def test_long_sections_are_split(self):
        self.assertGreater(len(self.chunks), sum(len(d["sections"]) for d in self.documents))

    def test_chunk_sizes_respect_custom_limit_without_losing_words(self):
        chunks = chunk_documents(self.documents, max_words=20)
        for chunk in chunks:
            self.assertLessEqual(len(chunk["text"].split()), 20)
        for document in self.documents:
            original = " ".join(" ".join(s["text"].split()) for s in document["sections"])
            rebuilt = " ".join(c["text"] for c in chunks if c["document_id"] == document["document_id"])
            self.assertEqual(rebuilt, original)

    def test_split_text_never_breaks_words(self):
        text = "Alpha beta gamma. Delta epsilon zeta eta theta. Iota."
        pieces = split_text(text, max_words=4)
        self.assertEqual(" ".join(pieces).split(), text.split())
        for piece in pieces:
            self.assertLessEqual(len(piece.split()), 4)

    def test_split_text_long_sentence_falls_back_to_words(self):
        pieces = split_text("one two three four five six seven", max_words=3)
        self.assertEqual(pieces, ["one two three", "four five six", "seven"])

    def test_invalid_max_words_is_rejected(self):
        for bad in (0, -1, 2.5, None):
            with self.assertRaises(ValueError):
                split_text("some text", max_words=bad)


class CosineSimilarityTests(unittest.TestCase):
    def test_identical_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0, places=6)

    def test_scaled_vectors_have_similarity_one(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 2.0], [10.0, 20.0]), 1.0, places=6)

    def test_orthogonal_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, places=6)

    def test_opposite_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 1.0], [-1.0, -1.0]), -1.0, places=6)

    def test_known_angle(self):
        # 45 degrees between (1, 0) and (1, 1): cos = 1 / sqrt(2).
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 1.0]), 1 / np.sqrt(2), places=6)

    def test_zero_norm_vector_is_rejected(self):
        with self.assertRaises(ValueError):
            cosine_similarity([0.0, 0.0], [1.0, 0.0])

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])

    def test_vectorized_matches_scalar(self):
        rng = np.random.default_rng(0)
        query = rng.normal(size=8)
        matrix = rng.normal(size=(5, 8))
        expected = [cosine_similarity(query, row) for row in matrix]
        np.testing.assert_allclose(cosine_similarities(query, matrix), expected, rtol=1e-5)

    def test_normalized_dot_product_equals_cosine(self):
        rng = np.random.default_rng(1)
        matrix = normalize_rows(rng.normal(size=(4, 6)))
        np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=1e-6)
        np.testing.assert_allclose(matrix @ matrix[0], cosine_similarities(matrix[0], matrix), rtol=1e-5)

    def test_vectorized_rejects_zero_rows_and_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            cosine_similarities([1.0, 0.0], np.array([[1.0, 0.0], [0.0, 0.0]]))
        with self.assertRaises(ValueError):
            cosine_similarities([1.0, 0.0, 0.0], np.eye(2))

    def test_normalize_rows_rejects_zero_vector(self):
        with self.assertRaises(ValueError):
            normalize_rows(np.array([[1.0, 0.0], [0.0, 0.0]]))


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {"chunk_id": f"C{i}", "document_id": f"D{i}", "title": "t", "department": "d",
             "section_id": f"D{i}.1", "section_title": "s", "text": f"chunk {i}"}
            for i in range(4)
        ]
        # Angles chosen so the similarity order to the query (1, 0) is C2, C0, C3, C1.
        self.embeddings = normalize_rows(np.array([[1.0, 0.5], [0.0, 1.0], [1.0, 0.0], [1.0, 2.0]]))
        self.query = np.array([1.0, 0.0], dtype=np.float32)

    def test_results_sorted_by_similarity(self):
        results = rank_chunks(self.query, self.embeddings, self.chunks, top_k=4)
        self.assertEqual([r["chunk_id"] for r in results], ["C2", "C0", "C3", "C1"])
        scores = [r["similarity"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual([r["rank"] for r in results], [1, 2, 3, 4])
        self.assertTrue(all(np.isfinite(scores)))

    def test_top_k_limits_results(self):
        self.assertEqual(len(rank_chunks(self.query, self.embeddings, self.chunks, top_k=2)), 2)
        self.assertEqual(len(rank_chunks(self.query, self.embeddings, self.chunks, top_k=1)), 1)

    def test_top_k_larger_than_collection_returns_all(self):
        self.assertEqual(len(rank_chunks(self.query, self.embeddings, self.chunks, top_k=50)), 4)

    def test_invalid_top_k_is_rejected(self):
        for bad in (0, -3, 2.0, "5", None, True):
            with self.assertRaises(ValueError):
                rank_chunks(self.query, self.embeddings, self.chunks, top_k=bad)

    def test_empty_collection_is_rejected(self):
        with self.assertRaises(ValueError):
            rank_chunks(self.query, np.zeros((0, 2)), [], top_k=5)

    def test_chunk_embedding_count_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            rank_chunks(self.query, self.embeddings[:3], self.chunks, top_k=2)

    def test_results_keep_chunk_metadata(self):
        result = rank_chunks(self.query, self.embeddings, self.chunks, top_k=1)[0]
        self.assertEqual(result["document_id"], "D2")
        self.assertEqual(result["text"], "chunk 2")

    def test_empty_query_is_rejected_before_model_use(self):
        # model=None proves the question is validated before any embedding happens.
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                retrieve(bad, None, self.chunks, self.embeddings)

    def test_invalid_top_k_is_rejected_before_model_use(self):
        with self.assertRaises(ValueError):
            retrieve("valid question", None, self.chunks, self.embeddings, top_k=0)


def toy_chunk(section_id, department="Dept", similarity=0.5):
    document_id = section_id.split(".")[0]
    return {"chunk_id": f"{section_id}#0", "document_id": document_id, "section_id": section_id,
            "title": f"{document_id} title", "section_title": "s", "department": department,
            "similarity": similarity}


def toy_question(question_id="T1", sections=("A.1",), documents=None, category="exact_keyword", split="development"):
    sections = list(sections)
    if documents is None:
        documents = sorted({s.split(".")[0] for s in sections})
    return {"question_id": question_id, "question": "toy?", "relevant_document_ids": documents,
            "relevant_section_ids": sections, "category": category, "split": split}


ORIGINAL_QUESTION_IDS = [f"Q{i:03d}" for i in range(1, 30)]
ORIGINAL_DOCUMENT_LABELS = {
    "Q001": ["FIN-203"], "Q002": ["HR-104"], "Q003": ["SEC-302"], "Q004": ["HR-102"], "Q005": ["FIN-201"],
    "Q006": ["ESC-733"], "Q007": ["SUB-730"], "Q008": ["REF-729"], "Q009": ["HR-102"],
    "Q010": ["SEC-305", "SEC-304"], "Q011": ["HR-105"], "Q012": ["SEC-304"], "Q013": ["SHP-732"],
    "Q014": ["HR-101"], "Q015": ["SEC-301"], "Q016": ["HR-103"], "Q017": ["REF-729"], "Q018": ["FIN-202"],
    "Q019": ["WAR-731"], "Q020": ["FIN-205"], "Q021": ["SEC-306"], "Q022": ["REF-728"], "Q023": ["REF-729"],
    "Q024": ["HR-104"], "Q025": ["FIN-204"], "Q026": ["FIN-203"], "Q027": ["FIN-206"], "Q028": ["HR-106"],
    "Q029": ["FIN-205"],
}


class BenchmarkDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents = load_documents()
        # Loading with documents runs the full reference validation.
        cls.questions = load_evaluation_questions(documents=cls.documents)
        cls.section_to_document = {
            s["section_id"]: d["document_id"] for d in cls.documents for s in d["sections"]
        }

    def test_question_ids_are_unique(self):
        ids = [q["question_id"] for q in self.questions]
        self.assertEqual(len(ids), len(set(ids)))

    def test_document_labels_exist(self):
        known = {d["document_id"] for d in self.documents}
        for q in self.questions:
            self.assertTrue(set(q["relevant_document_ids"]) <= known, q["question_id"])

    def test_section_labels_exist(self):
        for q in self.questions:
            self.assertTrue(set(q["relevant_section_ids"]) <= set(self.section_to_document), q["question_id"])

    def test_sections_belong_to_labeled_documents(self):
        for q in self.questions:
            owners = {self.section_to_document[s] for s in q["relevant_section_ids"]}
            self.assertEqual(owners, set(q["relevant_document_ids"]), q["question_id"])

    def test_both_splits_exist_with_every_category(self):
        for split in SPLITS:
            categories = {q["category"] for q in self.questions if q["split"] == split}
            self.assertEqual(categories, set(CATEGORIES), split)
        test_share = sum(q["split"] == "test" for q in self.questions) / len(self.questions)
        self.assertTrue(0.25 <= test_share <= 0.35, test_share)

    def test_categories_are_valid(self):
        for q in self.questions:
            self.assertIn(q["category"], CATEGORIES, q["question_id"])

    def test_original_questions_preserved(self):
        by_id = {q["question_id"]: q for q in self.questions}
        for question_id in ORIGINAL_QUESTION_IDS:
            self.assertIn(question_id, by_id)
            self.assertEqual(by_id[question_id]["relevant_document_ids"], ORIGINAL_DOCUMENT_LABELS[question_id])

    def test_stored_split_matches_deterministic_recipe(self):
        expected = assign_splits(self.questions)
        self.assertEqual({q["question_id"]: q["split"] for q in self.questions}, expected)
        self.assertEqual(assign_splits(list(reversed(self.questions))), expected)

    def test_benchmark_size(self):
        self.assertGreaterEqual(len(self.questions), 50)
        self.assertTrue(any(len(q["relevant_section_ids"]) > 1 for q in self.questions))


class LabelValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.documents = load_documents()

    def assertRejected(self, question, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            validate_questions([question], self.documents)

    def test_valid_question_passes(self):
        validate_questions([toy_question(sections=["REF-729.1"])], self.documents)

    def test_empty_section_labels_rejected(self):
        self.assertRejected(toy_question(sections=[], documents=["REF-729"]), "relevant_section_ids")

    def test_empty_document_labels_rejected(self):
        self.assertRejected(toy_question(sections=["REF-729.1"], documents=[]), "relevant_document_ids")

    def test_blank_label_rejected(self):
        self.assertRejected(toy_question(sections=["REF-729.1", " "], documents=["REF-729"]), "empty or non-string")

    def test_duplicate_label_rejected(self):
        self.assertRejected(toy_question(sections=["REF-729.1", "REF-729.1"], documents=["REF-729"]), "duplicate")

    def test_unknown_section_rejected(self):
        self.assertRejected(toy_question(sections=["REF-729.9"], documents=["REF-729"]), "unknown sections")

    def test_unknown_document_rejected(self):
        self.assertRejected(toy_question(sections=["REF-729.1"], documents=["REF-729", "NOPE-1"]), "unknown documents")

    def test_section_outside_labeled_documents_rejected(self):
        self.assertRejected(toy_question(sections=["REF-728.1"], documents=["REF-729"]), "not labeled relevant")

    def test_document_without_section_rejected(self):
        self.assertRejected(toy_question(sections=["REF-729.1"], documents=["REF-729", "REF-728"]), "no labeled section")

    def test_bad_category_and_split_rejected(self):
        self.assertRejected(toy_question(sections=["REF-729.1"], category="bm25_friendly"), "category")
        self.assertRejected(toy_question(sections=["REF-729.1"], split="train"), "split")

    def test_missing_field_and_duplicate_id_rejected(self):
        question = toy_question(sections=["REF-729.1"])
        del question["relevant_section_ids"]
        self.assertRejected(question, "missing")
        duplicate = toy_question(sections=["REF-729.1"])
        with self.assertRaisesRegex(ValueError, "Duplicate question_id"):
            validate_questions([duplicate, dict(duplicate)], self.documents)


class MetricTests(unittest.TestCase):
    def test_document_level_metrics_unchanged(self):
        self.assertEqual(recall_at_k(["A", "B", "C"], ["B", "Z"]), 0.5)
        self.assertEqual(recall_at_k(["A", "B", "C", "D", "E", "F"], ["F"], k=5), 0.0)
        self.assertEqual(reciprocal_rank_at_k(["A", "B", "C"], ["C", "B"]), 0.5)
        self.assertAlmostEqual(reciprocal_rank_at_k(["A", "B", "C", "D", "E"], ["E"], k=5), 0.2)
        self.assertEqual(unique_document_ids([{"document_id": d} for d in "ABACB"]), ["A", "B", "C"])

    def test_section_recall_at_1(self):
        ranking = [toy_chunk(s) for s in ["A.1", "A.2", "B.1"]]
        self.assertEqual(score_question(toy_question(sections=["A.1"]), ranking)["section_recall@1"], 1.0)
        self.assertEqual(score_question(toy_question(sections=["A.2"]), ranking)["section_recall@1"], 0.0)
        # Two relevant sections, only one can be at rank 1.
        self.assertEqual(score_question(toy_question(sections=["A.1", "B.1"]), ranking)["section_recall@1"], 0.5)

    def test_section_hit_at_1_differs_from_recall_at_1(self):
        ranking = [toy_chunk(s) for s in ["A.1", "B.1"]]
        scores = score_question(toy_question(sections=["A.1", "B.1"]), ranking)
        self.assertEqual(scores["section_hit@1"], 1.0)
        self.assertEqual(scores["section_recall@1"], 0.5)
        self.assertEqual(hit_at_k(["X", "A"], ["A"], k=1), 0.0)
        self.assertEqual(hit_at_k(["X", "A"], ["A"], k=2), 1.0)

    def test_section_recall_at_5(self):
        ranking = [toy_chunk(s) for s in ["A.1", "B.1", "C.1", "D.1", "E.1", "F.1"]]
        self.assertEqual(score_question(toy_question(sections=["E.1"]), ranking)["section_recall@5"], 1.0)
        # Rank 6 is outside the top 5.
        self.assertEqual(score_question(toy_question(sections=["F.1"]), ranking)["section_recall@5"], 0.0)

    def test_section_mrr_at_5_uses_chunk_ranks(self):
        # A.1 is split into two chunks; the second document appears at chunk rank 3, not "document rank 2".
        ranking = [toy_chunk(s) for s in ["A.1", "A.1", "B.2", "C.1"]]
        scores = score_question(toy_question(sections=["B.2"]), ranking)
        self.assertAlmostEqual(scores["section_mrr@5"], 1 / 3)
        self.assertAlmostEqual(scores["document_mrr@5"], 1 / 2)
        ranking = [toy_chunk(s) for s in ["A.1", "B.1", "C.1", "D.1", "E.1", "F.2"]]
        self.assertEqual(score_question(toy_question(sections=["F.2"]), ranking)["section_mrr@5"], 0.0)

    def test_duplicate_chunks_do_not_inflate_recall(self):
        ranking = [toy_chunk(s) for s in ["A.1", "A.1", "A.1", "C.1", "D.1"]]
        scores = score_question(toy_question(sections=["A.1", "B.1"]), ranking)
        self.assertEqual(scores["section_recall@5"], 0.5)
        self.assertEqual(recall_at_k(["A.1", "A.1"], ["A.1"]), 1.0)

    def test_multi_section_partial_credit(self):
        ranking = [toy_chunk(s) for s in ["X.1", "A.2", "Y.1", "A.3", "Z.1"]]
        scores = score_question(toy_question(sections=["A.1", "A.2", "A.3"]), ranking)
        self.assertAlmostEqual(scores["section_recall@5"], 2 / 3)
        self.assertEqual(scores["section_recall@1"], 0.0)
        self.assertEqual(scores["section_hit@1"], 0.0)
        self.assertAlmostEqual(scores["section_mrr@5"], 0.5)
        # Document level sees A at document rank 2 and counts it fully: evidence-level is stricter.
        self.assertEqual(scores["document_recall@5"], 1.0)

    def test_missing_relevant_sections_score_zero(self):
        ranking = [toy_chunk(s) for s in ["X.1", "Y.1", "Z.1"]]
        scores = score_question(toy_question(sections=["A.1", "B.1"]), ranking)
        for name in ("section_recall@1", "section_hit@1", "section_recall@5", "section_mrr@5",
                     "document_recall@5", "document_mrr@5"):
            self.assertEqual(scores[name], 0.0, name)

    def test_empty_relevant_labels_rejected_by_metrics(self):
        for metric in (recall_at_k, hit_at_k, reciprocal_rank_at_k):
            with self.assertRaises(ValueError):
                metric(["A"], [])

    def test_summary_averages_overall_and_by_category(self):
        questions = [
            toy_question("Q1", ["A.1"], category="exact_keyword"),
            toy_question("Q2", ["B.1", "C.1"], category="exact_keyword"),
            toy_question("Q3", ["Z.1"], category="multi_section"),
        ]
        ranking = [toy_chunk(s) for s in ["A.1", "B.1"]]
        summary = summarize(questions, {"Q1": ranking, "Q2": ranking, "Q3": ranking})
        # Section Recall@5: Q1=1, Q2=0.5, Q3=0.  Section MRR@5: Q1=1, Q2=0.5, Q3=0.
        self.assertAlmostEqual(summary["by_category"]["exact_keyword"]["section_recall@5"], 0.75)
        self.assertAlmostEqual(summary["by_category"]["exact_keyword"]["section_mrr@5"], 0.75)
        self.assertEqual(summary["by_category"]["multi_section"]["count"], 1)
        self.assertAlmostEqual(summary["overall"]["section_recall@5"], 0.5)
        self.assertAlmostEqual(summary["overall"]["section_hit@1"], 1 / 3)
        self.assertEqual(summary["overall"]["count"], 3)
        for name, value in summary["overall"].items():
            if name != "count":
                self.assertTrue(0.0 <= value <= 1.0, name)


class FailureAnalysisTests(unittest.TestCase):
    def test_first_relevant_rank(self):
        ranking = [toy_chunk(s) for s in ["X.1", "Y.1", "A.1"]]
        self.assertEqual(first_relevant_rank(ranking, ["A.1"]), 3)
        self.assertIsNone(first_relevant_rank(ranking, ["B.1"]))

    def test_reports_top1_misses_and_partial_coverage(self):
        departments = {"A": "Finance", "B": "Finance", "C": "Security"}
        questions = [
            toy_question("Q1", ["A.1"]),                                  # hit at rank 1
            toy_question("Q2", ["A.2"]),                                  # wrong section, same document
            toy_question("Q3", ["A.1"], category="exact_identifier"),    # top chunk from C
            toy_question("Q4", ["A.1", "C.9"], category="multi_section"),  # hit@1 but misses C.9
        ]
        ranking = [toy_chunk("A.1", "Finance"), toy_chunk("B.1", "Finance"), toy_chunk("A.2", "Finance")]
        other = [toy_chunk("C.1", "Security"), toy_chunk("A.1", "Finance")]
        summary = summarize(questions, {"Q1": ranking, "Q2": ranking, "Q3": other, "Q4": ranking})
        misses, partial = failure_analysis(summary, departments)

        by_id = {m["question"]["question_id"]: m for m in misses}
        self.assertEqual(sorted(by_id), ["Q2", "Q3"])
        self.assertEqual(by_id["Q2"]["diagnosis"], "wrong section in correct document")
        self.assertEqual(by_id["Q2"]["first_relevant_rank"], 3)
        self.assertEqual(by_id["Q3"]["diagnosis"], "exact identifier mismatch")
        self.assertEqual(by_id["Q3"]["relation"], "different department")
        self.assertEqual([p["question"]["question_id"] for p in partial], ["Q4"])
        self.assertEqual(partial[0]["missing_sections"], ["C.9"])

    def test_disambiguation_miss_inside_correct_document_is_not_policy_confusion(self):
        departments = {"A": "Finance", "B": "Finance"}
        same_document = summarize([toy_question("Q1", ["A.2"], category="similar_policy_disambiguation")],
                                  {"Q1": [toy_chunk("A.1", "Finance"), toy_chunk("A.2", "Finance")]})
        other_policy = summarize([toy_question("Q2", ["A.2"], category="similar_policy_disambiguation")],
                                 {"Q2": [toy_chunk("B.1", "Finance"), toy_chunk("A.2", "Finance")]})
        self.assertEqual(failure_analysis(same_document, departments)[0][0]["diagnosis"],
                         "wrong section in correct document")
        self.assertEqual(failure_analysis(other_policy, departments)[0][0]["diagnosis"],
                         "similar-policy confusion")


@unittest.skipUnless(RUN_MODEL_TESTS, "set RAGFORGE_MODEL_TESTS=1 to run tests that load the embedding model")
class ModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.embeddings import load_model
        from src.retrieve import build_index

        cls.model = load_model()
        cls.documents, cls.chunks, cls.embeddings = build_index(cls.model)

    def test_embedding_shapes_are_consistent_and_normalized(self):
        self.assertEqual(self.embeddings.ndim, 2)
        self.assertEqual(self.embeddings.shape[0], len(self.chunks))
        self.assertGreater(self.embeddings.shape[1], 0)
        self.assertTrue(np.all(np.isfinite(self.embeddings)))
        np.testing.assert_allclose(np.linalg.norm(self.embeddings, axis=1), 1.0, rtol=1e-4)

    def test_query_embedding_matches_chunk_dimension(self):
        from src.embeddings import embed_query

        self.assertEqual(embed_query(self.model, "refund deadline").shape, (self.embeddings.shape[1],))

    def test_retrieval_returns_valid_sorted_results(self):
        valid_chunk_ids = {chunk["chunk_id"] for chunk in self.chunks}
        valid_document_ids = {document["document_id"] for document in self.documents}
        results = retrieve("How long do overseas customers have to request a refund?",
                           self.model, self.chunks, self.embeddings, top_k=5)
        self.assertEqual(len(results), 5)
        scores = [r["similarity"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for result in results:
            self.assertIn(result["chunk_id"], valid_chunk_ids)
            self.assertIn(result["document_id"], valid_document_ids)
            self.assertTrue(np.isfinite(result["similarity"]))
            self.assertLessEqual(abs(result["similarity"]), 1.0 + 1e-5)

    def test_benchmark_runs_and_scores_are_bounded(self):
        from src.evaluate import rank_all_chunks

        questions = load_evaluation_questions(documents=self.documents)
        rankings = rank_all_chunks(questions, self.model, self.chunks, self.embeddings)
        for ranking in rankings.values():
            self.assertEqual(len(ranking), len(self.chunks))
        summary = summarize(questions, rankings)
        self.assertEqual(summary["overall"]["count"], len(questions))
        for row in [summary["overall"], *summary["by_category"].values()]:
            for name, value in row.items():
                if name != "count":
                    self.assertTrue(0.0 <= value <= 1.0, f"{name}={value}")


if __name__ == "__main__":
    unittest.main()
