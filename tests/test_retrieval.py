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
from src.retrieve import (
    cosine_similarities,
    cosine_similarity,
    load_evaluation_questions,
    rank_chunks,
    recall_at_k,
    reciprocal_rank_at_k,
    retrieve,
    summarize_evaluation,
    unique_document_ids,
)

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


class EvaluationTests(unittest.TestCase):
    def test_labels_reference_existing_documents(self):
        known = {document["document_id"] for document in load_documents()}
        questions = load_evaluation_questions(known_document_ids=known)
        self.assertGreaterEqual(len(questions), 20)
        for item in questions:
            self.assertTrue(set(item["relevant_document_ids"]) <= known, item["question_id"])

    def test_all_categories_present(self):
        categories = {item["category"] for item in load_evaluation_questions()}
        self.assertEqual(
            categories,
            {"exact_keyword", "semantic_paraphrase", "exact_identifier", "similar_policy_disambiguation"},
        )

    def test_unknown_document_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "q.json")
            path.write_text(json.dumps([
                {"question_id": "Q1", "question": "q?", "relevant_document_ids": ["NOPE-1"], "category": "c"}
            ]))
            with self.assertRaisesRegex(ValueError, "unknown documents"):
                load_evaluation_questions(path, known_document_ids={"DOC-1"})

    def test_unique_document_ids_preserve_rank_order(self):
        results = [{"document_id": d} for d in ["A", "B", "A", "C", "B"]]
        self.assertEqual(unique_document_ids(results), ["A", "B", "C"])

    def test_recall_at_k(self):
        self.assertEqual(recall_at_k(["A", "B", "C"], ["B"]), 1.0)
        self.assertEqual(recall_at_k(["A", "B", "C"], ["B", "Z"]), 0.5)
        self.assertEqual(recall_at_k(["A", "B", "C"], ["Z"]), 0.0)
        # Only the first k documents count.
        self.assertEqual(recall_at_k(["A", "B", "C", "D", "E", "F"], ["F"], k=5), 0.0)

    def test_reciprocal_rank_at_k(self):
        self.assertEqual(reciprocal_rank_at_k(["A", "B", "C"], ["A"]), 1.0)
        self.assertEqual(reciprocal_rank_at_k(["A", "B", "C"], ["C", "B"]), 0.5)
        self.assertEqual(reciprocal_rank_at_k(["A", "B", "C"], ["Z"]), 0.0)
        self.assertEqual(reciprocal_rank_at_k(["A", "B", "C", "D", "E", "F"], ["F"], k=5), 0.0)
        self.assertAlmostEqual(reciprocal_rank_at_k(["A", "B", "C", "D", "E"], ["E"], k=5), 0.2)

    def test_empty_relevant_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            recall_at_k(["A"], [])
        with self.assertRaises(ValueError):
            reciprocal_rank_at_k(["A"], [])

    def test_summary_averages_overall_and_by_category(self):
        questions = [
            {"question_id": "Q1", "category": "x", "relevant_document_ids": ["A"]},
            {"question_id": "Q2", "category": "x", "relevant_document_ids": ["B", "C"]},
            {"question_id": "Q3", "category": "y", "relevant_document_ids": ["Z"]},
        ]
        rankings = {"Q1": ["A", "B"], "Q2": ["A", "B"], "Q3": ["A", "B"]}
        summary = summarize_evaluation(questions, rankings)
        # Q1: R=1, RR=1. Q2: R=0.5, RR=0.5. Q3: R=0, RR=0.
        self.assertAlmostEqual(summary["by_category"]["x"]["recall"], 0.75)
        self.assertAlmostEqual(summary["by_category"]["x"]["mrr"], 0.75)
        self.assertAlmostEqual(summary["by_category"]["y"]["recall"], 0.0)
        self.assertAlmostEqual(summary["overall"]["recall"], 0.5)
        self.assertAlmostEqual(summary["overall"]["mrr"], 0.5)
        self.assertEqual(summary["overall"]["count"], 3)


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


if __name__ == "__main__":
    unittest.main()
