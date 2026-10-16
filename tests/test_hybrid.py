"""Tests for M2: BM25, Reciprocal Rank Fusion, and the dense / BM25 / hybrid interface.

Fast tests use a tiny deterministic fake embedding model, so they need no
download. Tests with the real MiniLM model run only with RAGFORGE_MODEL_TESTS=1.
"""

import json
import math
import os
import unittest
import zlib
from pathlib import Path

import numpy as np

from src.bm25 import (
    build_bm25_index,
    bm25_scores,
    chunk_index_text,
    inverse_document_frequency,
    minimal_stem,
    rank_bm25,
    tokenize,
)
from src.evaluate import filter_split, load_evaluation_questions, rank_questions, summarize
from src.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from src.ingest import load_documents
from src.retrieve import METHODS, build_search_index, format_result, retrieve, search

RUN_MODEL_TESTS = os.environ.get("RAGFORGE_MODEL_TESTS") == "1"
FIXTURE = Path(__file__).parent / "fixtures" / "dense_regression.json"


class FakeModel:
    """Deterministic stand-in for SentenceTransformer: hashed bag of words in 64 dimensions."""

    def encode(self, texts, **kwargs):
        vectors = np.zeros((len(texts), 64), dtype=np.float32)
        for row, text in enumerate(texts):
            vectors[row, 0] = 0.1  # never zero-norm
            for token in text.lower().split():
                vectors[row, zlib.crc32(token.encode()) % 64] += 1.0
        return vectors


def tiny_chunks(texts):
    return [{"chunk_id": f"D{i}.1#0", "document_id": f"D{i}", "section_id": f"D{i}.1", "title": "t",
             "section_title": "s", "department": "d", "text": text} for i, text in enumerate(texts)]


class TokenizerTests(unittest.TestCase):
    def test_words_are_lowercased_and_stopwords_removed(self):
        self.assertEqual(tokenize("What is the Refund Deadline?"), ["refund", "deadline"])

    def test_plurals_are_reduced(self):
        self.assertEqual(tokenize("refunds policies days business status"),
                         ["refund", "policy", "day", "business", "status"])
        self.assertEqual(minimal_stem("ss"), "ss")

    def test_numbers(self):
        self.assertEqual(tokenize("$25,000 or 25000, $0.65 per mile, 14 characters"),
                         ["25000", "25000", "0.65", "per", "mile", "14", "character"])

    def test_policy_identifiers_preserved(self):
        self.assertEqual(tokenize("What does FIN-205.3 say?"), ["fin-205.3", "fin-205", "say"])
        self.assertEqual(tokenize("HR-103.3 and SEC-301.4"), ["hr-103.3", "hr-103", "sec-301.4", "sec-301"])
        # A document ID at the end of a sentence keeps its ID and drops the period.
        self.assertEqual(tokenize("reported under SEC-304."), ["reported", "under", "sec-304"])

    def test_alphanumeric_tokens_kept_whole(self):
        self.assertEqual(tokenize("P1 cases need FIDO2 keys"), ["p1", "case", "need", "fido2", "key"])

    def test_tokenizer_is_deterministic(self):
        text = "Refunds for REF-729.2 within 45 days"
        self.assertEqual(tokenize(text), tokenize(text))

    def test_index_text_contains_identifier_title_and_text(self):
        chunk = tiny_chunks(["Some rule."])[0]
        self.assertEqual(chunk_index_text(chunk), "D0.1 t s Some rule.")


class BM25Tests(unittest.TestCase):
    def setUp(self):
        self.texts = ["apple banana apple", "banana cherry", "cherry durian elderberry fig grape", "kiwi"]
        self.index = build_bm25_index(self.texts)

    def test_term_frequency(self):
        self.assertEqual(self.index["term_frequencies"][0]["apple"], 2)
        self.assertEqual(self.index["term_frequencies"][0]["banana"], 1)
        self.assertEqual(self.index["lengths"].tolist(), [3, 2, 5, 1])

    def test_document_frequency(self):
        self.assertEqual(self.index["document_frequency"]["apple"], 1)
        self.assertEqual(self.index["document_frequency"]["banana"], 2)
        self.assertEqual(self.index["document_frequency"]["cherry"], 2)

    def test_idf_formula(self):
        self.assertAlmostEqual(self.index["idf"]["apple"], math.log(1 + (4 - 1 + 0.5) / (1 + 0.5)))
        self.assertGreater(self.index["idf"]["apple"], self.index["idf"]["banana"])
        # Non-negative even for a term in every document.
        self.assertGreater(inverse_document_frequency(4, 4), 0)

    def test_score_matches_hand_calculation(self):
        k1, b = 1.5, 0.75
        avgdl = (3 + 2 + 5 + 1) / 4
        idf = math.log(1 + 3.5 / 1.5)
        expected = idf * 2 * (k1 + 1) / (2 + k1 * (1 - b + b * 3 / avgdl))
        self.assertAlmostEqual(bm25_scores(self.index, "apple")[0], expected)

    def test_length_normalization(self):
        index = build_bm25_index(["term", "term padding padding padding padding"])
        short, long = bm25_scores(index, "term")
        self.assertGreater(short, long)
        index_no_norm = build_bm25_index(["term", "term padding padding padding padding"], b=0.0)
        short, long = bm25_scores(index_no_norm, "term")
        self.assertAlmostEqual(short, long)

    def test_term_frequency_saturates(self):
        # b = 0 removes length effects, isolating the tf part: idf * tf * (k1 + 1) / (tf + k1).
        texts = ["xx", "xx xx", "xx xx xx", "yy"]
        index = build_bm25_index(texts, b=0.0)
        one, two, three, _ = bm25_scores(index, "xx")
        self.assertLess(one, two)
        self.assertLess(two, three)
        self.assertLess(three - two, two - one)               # each extra occurrence adds less
        self.assertLess(three, index["idf"]["xx"] * (1.5 + 1))  # bounded by idf * (k1 + 1)
        # k1 = 0 ignores repetition entirely.
        flat = bm25_scores(build_bm25_index(texts, k1=0.0, b=0.0), "xx")
        self.assertAlmostEqual(flat[0], flat[2])

    def test_ranking_on_tiny_corpus(self):
        chunks = tiny_chunks(self.texts)
        results = rank_bm25("cherry banana", self.index, chunks, top_k=4)
        self.assertEqual([r["chunk_id"] for r in results], ["D1.1#0", "D0.1#0", "D2.1#0"])
        self.assertEqual([r["rank"] for r in results], [1, 2, 3])
        self.assertTrue(all(r["score_type"] == "bm25" and np.isfinite(r["score"]) for r in results))

    def test_top_k_limits_results(self):
        results = rank_bm25("cherry banana", self.index, tiny_chunks(self.texts), top_k=1)
        self.assertEqual(len(results), 1)

    def test_empty_query_rejected(self):
        for bad in ("", "   ", None):
            with self.assertRaises(ValueError):
                rank_bm25(bad, self.index, tiny_chunks(self.texts), top_k=3)

    def test_unknown_terms_score_zero(self):
        scores = bm25_scores(self.index, "zucchini quince")
        self.assertTrue(np.all(scores == 0))
        self.assertEqual(rank_bm25("zucchini quince", self.index, tiny_chunks(self.texts), top_k=3), [])
        # Stopword-only queries behave the same way.
        self.assertEqual(rank_bm25("what is the", self.index, tiny_chunks(self.texts), top_k=3), [])

    def test_ties_broken_by_chunk_order(self):
        texts = ["same words here", "other", "same words here"]
        results = rank_bm25("same words", build_bm25_index(texts), tiny_chunks(texts), top_k=3)
        self.assertEqual([r["chunk_id"] for r in results], ["D0.1#0", "D2.1#0"])
        self.assertEqual(results[0]["score"], results[1]["score"])

    def test_repeated_query_terms_count_once(self):
        np.testing.assert_allclose(bm25_scores(self.index, "apple apple"), bm25_scores(self.index, "apple"))

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            build_bm25_index([])
        with self.assertRaises(ValueError):
            build_bm25_index(["a b"], b=1.5)


class FusionTests(unittest.TestCase):
    def test_one_based_ranks(self):
        fused = reciprocal_rank_fusion({"a": ["x", "y"]})
        self.assertAlmostEqual(fused[0]["rrf_score"], 1 / 61)
        self.assertAlmostEqual(fused[1]["rrf_score"], 1 / 62)
        self.assertEqual(fused[0]["ranks"], {"a": 1})

    def test_overlapping_lists(self):
        fused = reciprocal_rank_fusion({"dense": ["a", "b", "c"], "bm25": ["c", "a"]})
        self.assertEqual([entry["id"] for entry in fused], ["a", "c", "b"])
        self.assertAlmostEqual(fused[0]["rrf_score"], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(fused[1]["rrf_score"], 1 / 63 + 1 / 61)
        self.assertEqual(fused[1]["ranks"], {"dense": 3, "bm25": 1})

    def test_candidate_in_only_one_list(self):
        fused = {entry["id"]: entry for entry in reciprocal_rank_fusion({"dense": ["a"], "bm25": ["b"]})}
        self.assertEqual(set(fused), {"a", "b"})
        self.assertEqual(fused["b"]["ranks"], {"dense": None, "bm25": 1})
        self.assertAlmostEqual(fused["b"]["rrf_score"], 1 / 61)

    def test_duplicates_within_a_list_count_once(self):
        fused = {entry["id"]: entry for entry in reciprocal_rank_fusion({"dense": ["a", "a", "b"]})}
        self.assertAlmostEqual(fused["a"]["rrf_score"], 1 / 61)
        self.assertEqual(fused["b"]["ranks"]["dense"], 2)

    def test_ties_are_deterministic(self):
        first = reciprocal_rank_fusion({"dense": ["b", "a"], "bm25": ["a", "b"]})
        second = reciprocal_rank_fusion({"bm25": ["a", "b"], "dense": ["b", "a"]})
        self.assertEqual(first[0]["rrf_score"], first[1]["rrf_score"])
        self.assertEqual([e["id"] for e in first], ["a", "b"])
        self.assertEqual([e["id"] for e in second], ["a", "b"])

    def test_ties_prefer_better_best_rank(self):
        # z: ranks 2 and 2 -> 2/62 = 0.032258; y: ranks 1 and 3 -> 1/61 + 1/63 = 0.032266 (slightly higher).
        fused = reciprocal_rank_fusion({"p": ["y", "z", "w"], "q": ["w", "z", "y"]})
        self.assertEqual([e["id"] for e in fused][:2], ["w", "y"])

    def test_configurable_constant(self):
        fused = reciprocal_rank_fusion({"a": ["x", "y"]}, k=1)
        self.assertAlmostEqual(fused[0]["rrf_score"], 1 / 2)
        self.assertAlmostEqual(fused[1]["rrf_score"], 1 / 3)
        self.assertEqual(DEFAULT_RRF_K, 60)
        for bad in (0, -5, "60", None):
            with self.assertRaises(ValueError):
                reciprocal_rank_fusion({"a": ["x"]}, k=bad)

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion({})


class RetrievalInterfaceTests(unittest.TestCase):
    """Integration over the real 92 chunks, using the fake embedding model."""

    @classmethod
    def setUpClass(cls):
        cls.index = build_search_index(METHODS, model=FakeModel())
        cls.chunk_ids = {chunk["chunk_id"] for chunk in cls.index["chunks"]}
        cls.section_ids = {chunk["section_id"] for chunk in cls.index["chunks"]}

    def assertValidResults(self, results, score_type):
        self.assertTrue(results)
        self.assertEqual([r["rank"] for r in results], list(range(1, len(results) + 1)))
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for result in results:
            self.assertIn(result["chunk_id"], self.chunk_ids)
            self.assertIn(result["section_id"], self.section_ids)
            for field in ("document_id", "title", "department", "section_title", "text"):
                self.assertTrue(result[field])
            self.assertEqual(result["score_type"], score_type)
            self.assertTrue(np.isfinite(result["score"]))

    def test_all_methods_share_one_chunk_universe(self):
        self.assertEqual(len(self.index["chunks"]), self.index["bm25"]["n_documents"])
        self.assertEqual(len(self.index["chunks"]), self.index["chunk_embeddings"].shape[0])

    def test_dense_search_matches_m1_retrieve(self):
        question = "How long do overseas customers have to request a refund?"
        via_search = search(question, self.index, "dense", top_k=10)
        via_retrieve = retrieve(question, self.index["model"], self.index["chunks"],
                                self.index["chunk_embeddings"], top_k=10)
        self.assertEqual([r["chunk_id"] for r in via_search], [r["chunk_id"] for r in via_retrieve])
        self.assertEqual([r["score"] for r in via_search], [r["similarity"] for r in via_retrieve])
        self.assertValidResults(via_search, "cosine")

    def test_bm25_search_returns_valid_metadata(self):
        results = search("What does FIN-205.3 say?", self.index, "bm25", top_k=5)
        self.assertValidResults(results, "bm25")
        self.assertEqual(results[0]["section_id"], "FIN-205.3")

    def test_hybrid_search_returns_valid_metadata(self):
        results = search("What does FIN-205.3 say?", self.index, "hybrid", top_k=5)
        self.assertValidResults(results, "rrf")
        for result in results:
            self.assertIn("dense_rank", result)
            self.assertIn("bm25_rank", result)
            expected = sum(1 / (DEFAULT_RRF_K + r) for r in (result["dense_rank"], result["bm25_rank"]) if r)
            self.assertAlmostEqual(result["score"], expected)
        self.assertIn("RRF score", format_result(results[0]))

    def test_hybrid_covers_every_chunk(self):
        results = search("refund deadline", self.index, "hybrid", top_k=len(self.index["chunks"]))
        self.assertEqual({r["chunk_id"] for r in results}, self.chunk_ids)

    def test_hybrid_with_no_lexical_match_follows_dense(self):
        question = "zzzz qqqq"
        hybrid = search(question, self.index, "hybrid", top_k=5)
        dense = search(question, self.index, "dense", top_k=5)
        self.assertEqual([r["chunk_id"] for r in hybrid], [r["chunk_id"] for r in dense])
        self.assertTrue(all(r["bm25_rank"] is None for r in hybrid))

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            search("question", self.index, "sparse")
        for method in METHODS:
            with self.assertRaises(ValueError):
                search("  ", self.index, method)
            with self.assertRaises(ValueError):
                search("question", self.index, method, top_k=0)

    def test_bm25_only_index_needs_no_model(self):
        index = build_search_index(("bm25",))
        self.assertIsNone(index["model"])
        self.assertIsNone(index["chunk_embeddings"])
        self.assertTrue(search("corporate card limit", index, "bm25"))

    def test_all_methods_work_with_evaluation(self):
        questions = filter_split(load_evaluation_questions(documents=load_documents()), "development")
        for method in METHODS:
            rankings, latencies = rank_questions(questions, self.index, method)
            self.assertEqual(len(latencies), len(questions))
            self.assertTrue(all(latency >= 0 for latency in latencies))
            summary = summarize(questions, rankings)
            self.assertEqual(summary["overall"]["count"], len(questions))
            for name, value in summary["overall"].items():
                if name != "count":
                    self.assertTrue(0.0 <= value <= 1.0, f"{method} {name}={value}")


@unittest.skipUnless(RUN_MODEL_TESTS, "set RAGFORGE_MODEL_TESTS=1 to run tests that load the embedding model")
class RealModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = build_search_index(METHODS)

    def test_dense_retrieval_unchanged_from_m1(self):
        fixture = json.loads(FIXTURE.read_text())
        self.assertEqual(len(self.index["chunks"]), fixture["n_chunks"])
        for item in fixture["queries"]:
            results = search(item["question"], self.index, "dense", top_k=5)
            self.assertEqual([r["chunk_id"] for r in results], [e["chunk_id"] for e in item["top5"]],
                             item["question"])
            for result, expected in zip(results, item["top5"]):
                self.assertAlmostEqual(result["similarity"], expected["similarity"], places=5)

    def test_hybrid_with_real_model(self):
        results = search("What does FIN-205.3 say?", self.index, "hybrid", top_k=5)
        self.assertEqual(len(results), 5)
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(r["dense_rank"] is not None for r in results))


if __name__ == "__main__":
    unittest.main()
