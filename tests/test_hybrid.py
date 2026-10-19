"""Tests for M2 / M2.5: BM25, (weighted) Reciprocal Rank Fusion, and the dense / BM25 / hybrid interface.

Fast tests use a tiny deterministic fake embedding model, so they need no
download. Tests with the real MiniLM model run only with RAGFORGE_MODEL_TESTS=1.
"""

import contextlib
import io
import json
import math
import os
import unittest
import zlib
from unittest import mock
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
from src.evaluate import (
    COMPARE_CONFIGS,
    FUSION_CONFIGS,
    change_report,
    compare_verdict,
    configuration_latency,
    filter_split,
    load_evaluation_questions,
    rank_configurations,
    rank_questions,
    summarize,
)
from src.evaluate import main as evaluate_main
from src.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion, validate_weights
from src.ingest import load_documents
from src.retrieve import METHODS, build_search_index, format_result, hybrid_candidates, retrieve, search
from src.retrieve import main as retrieve_main

RUN_MODEL_TESTS = os.environ.get("RAGFORGE_MODEL_TESTS") == "1"
FIXTURE = Path(__file__).parent / "fixtures" / "dense_regression.json"
HYBRID_FIXTURE = Path(__file__).parent / "fixtures" / "hybrid_regression.json"


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
        # A chunk without the term scores exactly 0 (not NaN) even when k1 = 0.
        self.assertTrue(np.all(np.isfinite(flat)))
        self.assertEqual(flat[3], 0.0)

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
            # Default weights are 0.5 / 0.5 (M2.5): each contribution is 0.5 / (k + rank).
            expected = sum(0.5 / (DEFAULT_RRF_K + r) for r in (result["dense_rank"], result["bm25_rank"]) if r)
            self.assertAlmostEqual(result["score"], expected)
        self.assertIn("Weighted RRF score", format_result(results[0]))

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


class WeightedFusionTests(unittest.TestCase):
    def test_weighted_score_calculation(self):
        fused = reciprocal_rank_fusion({"dense": ["a", "b"], "bm25": ["b", "a"]},
                                       weights={"dense": 0.7, "bm25": 0.3})
        scores = {entry["id"]: entry["rrf_score"] for entry in fused}
        self.assertAlmostEqual(scores["a"], 0.7 / 61 + 0.3 / 62)
        self.assertAlmostEqual(scores["b"], 0.7 / 62 + 0.3 / 61)
        self.assertEqual([entry["id"] for entry in fused], ["a", "b"])

    def test_equal_weights_match_unweighted_ranking_exactly(self):
        rankings = {"dense": ["a", "b", "c", "d", "e"], "bm25": ["e", "c", "x", "a"]}
        unweighted = reciprocal_rank_fusion(rankings)
        equal = reciprocal_rank_fusion(rankings, weights={"dense": 0.5, "bm25": 0.5})
        self.assertEqual([e["id"] for e in equal], [e["id"] for e in unweighted])
        for weighted_entry, plain_entry in zip(equal, unweighted):
            # Halving is exact in binary floating point, so this holds bit for bit.
            self.assertEqual(weighted_entry["rrf_score"], 0.5 * plain_entry["rrf_score"])
            self.assertEqual(weighted_entry["ranks"], plain_entry["ranks"])

    def test_one_based_ranks(self):
        fused = reciprocal_rank_fusion({"dense": ["x", "y"], "bm25": []}, weights={"dense": 0.8, "bm25": 0.2})
        self.assertAlmostEqual(fused[0]["rrf_score"], 0.8 / 61)
        self.assertAlmostEqual(fused[1]["rrf_score"], 0.8 / 62)

    def test_missing_candidate_gets_only_its_weighted_share(self):
        fused = {e["id"]: e for e in reciprocal_rank_fusion({"dense": ["a"], "bm25": ["b"]},
                                                            weights={"dense": 0.8, "bm25": 0.2})}
        self.assertAlmostEqual(fused["b"]["rrf_score"], 0.2 / 61)
        self.assertEqual(fused["b"]["ranks"], {"dense": None, "bm25": 1})
        self.assertAlmostEqual(fused["a"]["rrf_score"], 0.8 / 61)

    def test_duplicates_count_once_with_weights(self):
        fused = {e["id"]: e for e in reciprocal_rank_fusion({"dense": ["a", "a", "b"], "bm25": ["b"]},
                                                            weights={"dense": 0.6, "bm25": 0.4})}
        self.assertAlmostEqual(fused["a"]["rrf_score"], 0.6 / 61)
        self.assertAlmostEqual(fused["b"]["rrf_score"], 0.6 / 62 + 0.4 / 61)

    def test_ties_are_deterministic_with_weights(self):
        first = reciprocal_rank_fusion({"dense": ["b", "a"], "bm25": ["a", "b"]}, weights={"dense": 0.5, "bm25": 0.5})
        second = reciprocal_rank_fusion({"bm25": ["a", "b"], "dense": ["b", "a"]}, weights={"bm25": 0.5, "dense": 0.5})
        self.assertEqual([e["id"] for e in first], ["a", "b"])
        self.assertEqual([e["id"] for e in second], ["a", "b"])

    def test_zero_weight_keeps_candidates_but_ignores_their_ranking(self):
        fused = reciprocal_rank_fusion({"dense": ["a", "b"], "bm25": ["b", "z"]}, weights={"dense": 1.0, "bm25": 0.0})
        self.assertEqual([e["id"] for e in fused], ["a", "b", "z"])
        self.assertEqual(fused[2]["rrf_score"], 0.0)

    def test_valid_weights(self):
        self.assertEqual(validate_weights({"dense": 0.7, "bm25": 0.3}), {"dense": 0.7, "bm25": 0.3})
        validate_weights({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})  # sums to 1 within tolerance
        validate_weights({"dense": 1, "bm25": 0})

    def test_invalid_weights_rejected(self):
        bad = {
            "negative": {"dense": 1.2, "bm25": -0.2},
            "above one": {"dense": 1.5, "bm25": 0.5},
            "wrong sum": {"dense": 0.7, "bm25": 0.2},
            "nan": {"dense": float("nan"), "bm25": 0.5},
            "infinite": {"dense": float("inf"), "bm25": 0.5},
            "both zero": {"dense": 0.0, "bm25": 0.0},
            "not a number": {"dense": "0.5", "bm25": 0.5},
            "boolean": {"dense": True, "bm25": 0.0},
        }
        for label, weights in bad.items():
            with self.subTest(label), self.assertRaises(ValueError):
                validate_weights(weights)
        with self.assertRaisesRegex(ValueError, "exactly"):
            reciprocal_rank_fusion({"dense": ["a"], "bm25": ["a"]}, weights={"dense": 1.0})
        with self.assertRaises(ValueError):
            validate_weights({})

    def test_m25_configurations_are_the_four_specified(self):
        self.assertEqual(FUSION_CONFIGS, (("rrf_50_50", 0.5, 0.5), ("rrf_60_40", 0.6, 0.4),
                                          ("rrf_70_30", 0.7, 0.3), ("rrf_80_20", 0.8, 0.2)))
        for _, dense_weight, bm25_weight in FUSION_CONFIGS:
            validate_weights({"dense": dense_weight, "bm25": bm25_weight})
        self.assertEqual(COMPARE_CONFIGS, ("dense", "bm25", "rrf_50_50", "rrf_60_40", "rrf_70_30", "rrf_80_20"))


class WeightedRetrievalTests(unittest.TestCase):
    """Weighted hybrid retrieval and the six-configuration evaluation, with the fake embedding model."""

    @classmethod
    def setUpClass(cls):
        cls.index = build_search_index(METHODS, model=FakeModel())
        cls.questions = filter_split(load_evaluation_questions(documents=load_documents()), "development")

    def test_default_hybrid_ranking_equals_unweighted_m2_fusion(self):
        for question in ("What does FIN-205.3 say?", "refund deadline for overseas orders"):
            dense, lexical = hybrid_candidates(question, self.index)
            m2 = reciprocal_rank_fusion({"dense": [r["chunk_id"] for r in dense],
                                         "bm25": [r["chunk_id"] for r in lexical]})
            default = search(question, self.index, "hybrid", top_k=len(self.index["chunks"]))
            self.assertEqual([r["chunk_id"] for r in default], [e["id"] for e in m2])
            self.assertEqual([r["score"] for r in default], [0.5 * e["rrf_score"] for e in m2])

    def test_weights_change_hybrid_scores_and_are_reported(self):
        results = search("What does FIN-205.3 say?", self.index, "hybrid", top_k=3,
                         dense_weight=0.8, bm25_weight=0.2)
        for result in results:
            expected = (0.8 / (DEFAULT_RRF_K + result["dense_rank"])
                        + (0.2 / (DEFAULT_RRF_K + result["bm25_rank"]) if result["bm25_rank"] else 0.0))
            self.assertAlmostEqual(result["score"], expected)
            self.assertEqual(result["weights"], {"dense": 0.8, "bm25": 0.2})
        text = format_result(results[0])
        self.assertIn("Weighted RRF score", text)
        self.assertIn("dense 0.80 / BM25 0.20", text)

    def test_invalid_weights_rejected_by_search(self):
        with self.assertRaises(ValueError):
            search("question", self.index, "hybrid", dense_weight=0.9, bm25_weight=0.3)

    def test_bm25_unchanged_from_m2(self):
        fixture = json.loads(HYBRID_FIXTURE.read_text())
        for item in fixture["queries"]:
            results = search(item["question"], self.index, "bm25", top_k=10)
            self.assertEqual([r["chunk_id"] for r in results], [e["chunk_id"] for e in item["bm25_top10"]])
            for result, expected in zip(results, item["bm25_top10"]):
                self.assertAlmostEqual(result["score"], expected["score"], places=5)

    def test_configurations_reuse_one_pair_of_rankings(self):
        rankings, timing = rank_configurations(self.questions, self.index)
        self.assertEqual(tuple(rankings), COMPARE_CONFIGS)
        question_ids = [q["question_id"] for q in self.questions]
        for config in COMPARE_CONFIGS:
            self.assertEqual(list(rankings[config]), question_ids)
            parts = timing[config]
            for retrieval, fusion, total in zip(parts["retrieval_ms"], parts["fusion_ms"], parts["total_ms"]):
                self.assertAlmostEqual(total, retrieval + fusion)
        self.assertTrue(all(ms == 0.0 for ms in timing["dense"]["fusion_ms"]))
        self.assertTrue(all(ms > 0.0 for ms in timing["rrf_70_30"]["fusion_ms"]))
        # Hybrid configurations are charged the full dense + BM25 retrieval time.
        for i in range(len(self.questions)):
            self.assertAlmostEqual(timing["rrf_50_50"]["retrieval_ms"][i],
                                   timing["dense"]["retrieval_ms"][i] + timing["bm25"]["retrieval_ms"][i])

    def test_cached_configurations_match_direct_search(self):
        rankings, _ = rank_configurations(self.questions[:5], self.index)
        for question in self.questions[:5]:
            question_id = question["question_id"]
            for method, config in (("dense", "dense"), ("bm25", "bm25"), ("hybrid", "rrf_50_50")):
                direct = search(question["question"], self.index, method, top_k=len(self.index["chunks"]))
                self.assertEqual([r["chunk_id"] for r in rankings[config][question_id]],
                                 [r["chunk_id"] for r in direct])
            direct = search(question["question"], self.index, "hybrid", top_k=len(self.index["chunks"]),
                            dense_weight=0.7, bm25_weight=0.3)
            self.assertEqual([r["chunk_id"] for r in rankings["rrf_70_30"][question_id]],
                             [r["chunk_id"] for r in direct])

    def test_results_deterministic_for_fixed_rankings(self):
        first, _ = rank_configurations(self.questions, self.index)
        second, _ = rank_configurations(self.questions, self.index)
        for config in COMPARE_CONFIGS:
            a, b = summarize(self.questions, first[config]), summarize(self.questions, second[config])
            self.assertEqual(a["overall"], b["overall"])
            for question_id in first[config]:
                self.assertEqual([r["chunk_id"] for r in first[config][question_id]],
                                 [r["chunk_id"] for r in second[config][question_id]])

    def test_change_report_and_verdicts(self):
        base = {"section_hit@1": 1.0, "section_recall@5": 1.0, "section_mrr@5": 1.0}
        self.assertEqual(compare_verdict(base, dict(base)), "unchanged")
        self.assertEqual(compare_verdict(base, {**base, "section_mrr@5": 0.5}), "regressed")
        self.assertEqual(compare_verdict({**base, "section_hit@1": 0.0}, base), "improved")
        self.assertEqual(compare_verdict({**base, "section_hit@1": 0.0}, {**base, "section_recall@5": 0.5}), "mixed")
        rankings, timing = rank_configurations(self.questions, self.index)
        results = {c: (summarize(self.questions, rankings[c]), configuration_latency(timing[c]))
                   for c in COMPARE_CONFIGS}
        report = change_report(results, "dense", "rrf_60_40")
        self.assertEqual(sorted(sum(report.values(), [])), sorted(q["question_id"] for q in self.questions))
        self.assertEqual(change_report(results, "rrf_50_50", "rrf_50_50")["unchanged"],
                         [q["question_id"] for q in self.questions])


class CommandLineTests(unittest.TestCase):
    """CLI argument handling; none of these load the embedding model."""

    def run_cli(self, main, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                main(argv)
                code = 0
            except SystemExit as exit_:
                code = exit_.code
        return code, out.getvalue(), err.getvalue()

    def test_existing_bm25_command_still_works(self):
        code, out, _ = self.run_cli(retrieve_main, ["--question", "What does FIN-205.3 say?", "--method", "bm25"])
        self.assertEqual(code, 0)
        self.assertIn("RAGForge — BM25 Retrieval", out)
        self.assertIn("FIN-205.3#0", out)

    def test_weight_flags_require_hybrid(self):
        code, _, err = self.run_cli(retrieve_main, ["--question", "q", "--method", "bm25", "--dense-weight", "0.7"])
        self.assertEqual(code, 2)
        self.assertIn("apply only to --method hybrid", err)

    def test_invalid_weight_flags_rejected_before_model_load(self):
        for argv in (["--dense-weight", "0.9", "--bm25-weight", "0.3"],
                     ["--dense-weight", "-0.1", "--bm25-weight", "1.1"],
                     ["--dense-weight", "nan", "--bm25-weight", "0.5"],
                     ["--dense-weight", "0.7"]):  # bm25 defaults to 0.5: sums to 1.2
            code, _, err = self.run_cli(retrieve_main, ["--question", "q", "--method", "hybrid"] + argv)
            self.assertEqual(code, 2, argv)
            self.assertIn("error", err)

    def test_held_out_splits_need_explicit_confirmation(self):
        for argv in (["--split", "test"], ["--split", "test", "--compare"], ["--split", "all"]):
            code, _, err = self.run_cli(evaluate_main, argv)
            self.assertEqual(code, 2, argv)
            self.assertIn("--confirm-held-out", err)
        code, _, err = self.run_cli(evaluate_main, ["--split", "test", "--failures", "--confirm-held-out"])
        self.assertEqual(code, 2)
        self.assertIn("development split", err)

    def test_compare_defaults_to_development_and_reports_six_configurations(self):
        real_build = build_search_index
        with mock.patch("src.retrieve.build_search_index",
                        side_effect=lambda methods, **kw: real_build(methods, model=FakeModel())):
            code, out, _ = self.run_cli(evaluate_main, ["--compare"])
        self.assertEqual(code, 0)
        self.assertIn("Split: development", out)
        self.assertIn("Questions: 37", out)
        for name in ("Dense", "BM25", "RRF .5/.5", "RRF .6/.4", "RRF .7/.3", "RRF .8/.2"):
            self.assertIn(name, out)
        self.assertIn("Fusion-only ms", out)


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

    def test_default_hybrid_unchanged_from_m2(self):
        fixture = json.loads(HYBRID_FIXTURE.read_text())
        for item in fixture["queries"]:
            results = search(item["question"], self.index, "hybrid", top_k=10)
            self.assertEqual([r["chunk_id"] for r in results], [e["chunk_id"] for e in item["hybrid_top10"]],
                             item["question"])
            for result, expected in zip(results, item["hybrid_top10"]):
                # Default weights are 0.5 / 0.5: same ranking and ranks, every score halved.
                self.assertAlmostEqual(result["score"], 0.5 * expected["rrf_score_unweighted"], places=8)
                self.assertEqual(result["dense_rank"], expected["dense_rank"])
                self.assertEqual(result["bm25_rank"], expected["bm25_rank"])

    def test_hybrid_with_real_model(self):
        results = search("What does FIN-205.3 say?", self.index, "hybrid", top_k=5)
        self.assertEqual(len(results), 5)
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(r["dense_rank"] is not None for r in results))


if __name__ == "__main__":
    unittest.main()
