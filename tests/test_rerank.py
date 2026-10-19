"""Tests for M3: cross-encoder reranking.

Fast tests use a deterministic mock cross-encoder and the fake embedding model,
so nothing is downloaded. Tests with the real cross-encoder run only with
RAGFORGE_MODEL_TESTS=1.
"""

import contextlib
import copy
import io
import os
import unittest
from unittest import mock

import numpy as np

from src.embeddings import chunk_embedding_text
from src.evaluate import (
    RERANK_PIPELINES,
    candidate_recall,
    evidence_coverage,
    filter_split,
    load_evaluation_questions,
    rank_rerank_pipelines,
    rerank_latency,
    summarize,
)
from src.evaluate import main as evaluate_main
from src.ingest import load_documents
from src.rerank import (
    DEFAULT_CANDIDATES,
    DEFAULT_RERANK_MODEL,
    build_pairs,
    candidate_text,
    check_candidate_settings,
    rerank,
    score_pairs,
)
from src.retrieve import ALL_METHODS, build_search_index, format_result, search
from src.retrieve import main as retrieve_main
from test_hybrid import FakeModel

RUN_MODEL_TESTS = os.environ.get("RAGFORGE_MODEL_TESTS") == "1"


class MockCrossEncoder:
    """Deterministic stand-in for CrossEncoder.

    Default score: number of distinct lowercase words shared by question and
    passage. `scores` (passage text -> score) or `output` (a fixed array)
    override it. Every predict() call is recorded.
    """

    device = "mock"

    def __init__(self, scores=None, output=None):
        self.scores = scores or {}
        self.output = output
        self.calls = []

    def predict(self, pairs, batch_size=32, convert_to_numpy=True, show_progress_bar=False):
        self.calls.append({"pairs": list(pairs), "batch_size": batch_size})
        if self.output is not None:
            return self.output
        return np.array([
            self.scores.get(text, len(set(question.lower().split()) & set(text.lower().split())))
            for question, text in pairs
        ], dtype=np.float32)


def make_candidate(i, score=None, **extra):
    candidate = {"chunk_id": f"D{i}.1#0", "document_id": f"D{i}", "section_id": f"D{i}.1",
                 "title": f"Title {i}", "section_title": f"Section {i}", "department": "Dept",
                 "text": f"passage number {i}", "rank": i + 1,
                 "score": 1.0 - i / 10 if score is None else score, "score_type": "cosine"}
    candidate.update(extra)
    return candidate


def scored_model(candidates, values):
    """Mock that gives each candidate (by its passage text) the matching value."""
    return MockCrossEncoder(scores={candidate_text(c): v for c, v in zip(candidates, values)})


class RerankUnitTests(unittest.TestCase):
    def test_pair_construction_uses_chunk_fields_only(self):
        candidate = make_candidate(0)
        pairs = build_pairs("What is it?", [candidate])
        self.assertEqual(pairs, [("What is it?", "Title 0 - Section 0 (D0.1): passage number 0")])
        # Same representation as the dense retriever; no scores, ranks or labels.
        self.assertEqual(candidate_text(candidate), chunk_embedding_text(candidate))

    def test_scores_associated_with_their_chunks_and_sorted(self):
        candidates = [make_candidate(i) for i in range(4)]
        results = rerank("q", candidates, scored_model(candidates, [0.1, 3.0, -2.0, 1.5]), top_k=4)
        self.assertEqual([r["chunk_id"] for r in results], ["D1.1#0", "D3.1#0", "D0.1#0", "D2.1#0"])
        self.assertEqual([r["score"] for r in results], [3.0, 1.5, 0.10000000149011612, -2.0])
        self.assertEqual([r["rank"] for r in results], [1, 2, 3, 4])
        self.assertEqual([r["retrieval_rank"] for r in results], [2, 4, 1, 3])

    def test_ties_broken_by_retrieval_rank(self):
        candidates = [make_candidate(i) for i in range(4)]
        results = rerank("q", candidates, scored_model(candidates, [1.0, 2.0, 2.0, 1.0]), top_k=4)
        self.assertEqual([r["chunk_id"] for r in results], ["D1.1#0", "D2.1#0", "D0.1#0", "D3.1#0"])
        again = rerank("q", candidates, scored_model(candidates, [1.0, 2.0, 2.0, 1.0]), top_k=4)
        self.assertEqual(results, again)

    def test_metadata_preserved(self):
        candidates = [make_candidate(0, score=0.0321, score_type="rrf", dense_rank=3, bm25_rank=None,
                                     weights={"dense": 0.6, "bm25": 0.4})]
        result = rerank("q", candidates, MockCrossEncoder(output=np.array([4.2])), top_k=1)[0]
        for field in ("chunk_id", "document_id", "section_id", "title", "section_title", "department", "text",
                      "dense_rank", "bm25_rank", "weights"):
            self.assertEqual(result[field], candidates[0][field])
        self.assertEqual(result["retrieval_rank"], 1)
        self.assertEqual(result["retrieval_score"], 0.0321)
        self.assertEqual(result["retrieval_score_type"], "rrf")
        self.assertEqual(result["score_type"], "cross-encoder")
        self.assertAlmostEqual(result["score"], 4.2, places=5)

    def test_candidates_not_mutated(self):
        candidates = [make_candidate(i) for i in range(3)]
        before = copy.deepcopy(candidates)
        rerank("q", candidates, scored_model(candidates, [0.0, 5.0, 1.0]), top_k=3)
        self.assertEqual(candidates, before)

    def test_single_batched_call(self):
        candidates = [make_candidate(i) for i in range(20)]
        model = MockCrossEncoder()
        rerank("q", candidates, model, top_k=5, batch_size=32)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(len(model.calls[0]["pairs"]), 20)
        self.assertEqual(model.calls[0]["batch_size"], 32)

    def test_empty_candidates(self):
        model = MockCrossEncoder()
        self.assertEqual(rerank("q", [], model), [])
        self.assertEqual(model.calls, [])

    def test_fewer_candidates_than_top_k(self):
        candidates = [make_candidate(i) for i in range(3)]
        self.assertEqual(len(rerank("q", candidates, MockCrossEncoder(), top_k=5)), 3)

    def test_duplicate_chunk_ids_rejected(self):
        candidates = [make_candidate(0), make_candidate(1), make_candidate(0)]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            rerank("q", candidates, MockCrossEncoder())

    def test_invalid_arguments_rejected(self):
        candidates = [make_candidate(0)]
        for bad in (0, -1, 2.5, True, None):
            with self.assertRaises(ValueError):
                rerank("q", candidates, MockCrossEncoder(), top_k=bad)
        with self.assertRaises(ValueError):
            rerank("q", candidates, MockCrossEncoder(), batch_size=0)
        for question in ("", "  ", None):
            with self.assertRaises(ValueError):
                rerank(question, candidates, MockCrossEncoder())
        with self.assertRaises(ValueError):
            check_candidate_settings(5, 10)  # top_k above the candidate count
        with self.assertRaises(ValueError):
            check_candidate_settings(0, 1)
        self.assertEqual(check_candidate_settings(20, 5), (20, 5))

    def test_nonfinite_scores_rejected(self):
        candidates = [make_candidate(i) for i in range(2)]
        for bad in (np.array([1.0, np.nan]), np.array([np.inf, 0.0])):
            with self.assertRaisesRegex(ValueError, "NaN or infinite"):
                rerank("q", candidates, MockCrossEncoder(output=bad))

    def test_output_shapes(self):
        pairs = [("q", "a"), ("q", "b")]
        np.testing.assert_array_equal(score_pairs(MockCrossEncoder(output=np.array([[1.0], [2.0]])), pairs),
                                      [1.0, 2.0])
        for bad in (np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, 2.0, 3.0]), np.array(1.0)):
            with self.assertRaisesRegex(ValueError, "Expected 2"):
                score_pairs(MockCrossEncoder(output=bad), pairs)


class RerankPipelineTests(unittest.TestCase):
    """Rerank pipelines over the real 92 chunks with fake embeddings and a mock cross-encoder."""

    @classmethod
    def setUpClass(cls):
        cls.reranker = MockCrossEncoder()
        cls.index = build_search_index(ALL_METHODS, model=FakeModel(), reranker=cls.reranker)
        cls.questions = filter_split(load_evaluation_questions(documents=load_documents()), "development")

    def setUp(self):
        self.reranker.calls.clear()

    def test_dense_rerank_reranks_the_dense_top_20(self):
        question = "How long do overseas customers have to request a refund?"
        candidates = search(question, self.index, "dense", top_k=DEFAULT_CANDIDATES)
        results = search(question, self.index, "dense-rerank")
        self.assertEqual(len(results), 5)
        self.assertEqual(len(self.reranker.calls[-1]["pairs"]), DEFAULT_CANDIDATES)
        self.assertTrue({r["chunk_id"] for r in results} <= {c["chunk_id"] for c in candidates})
        self.assertTrue(all(r["retrieval_score_type"] == "cosine" for r in results))
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_hybrid_rerank_uses_06_04_candidates(self):
        question = "What does FIN-205.3 say?"
        candidates = search(question, self.index, "hybrid", top_k=DEFAULT_CANDIDATES, dense_weight=0.6,
                            bm25_weight=0.4)
        results = search(question, self.index, "hybrid-rerank")
        sent = [text for _, text in self.reranker.calls[-1]["pairs"]]
        self.assertEqual(sent, [candidate_text(c) for c in candidates])
        self.assertTrue(all(r["weights"] == {"dense": 0.6, "bm25": 0.4} for r in results))
        self.assertTrue(all(r["retrieval_score_type"] == "rrf" for r in results))

    def test_plain_hybrid_default_still_equal_weights(self):
        results = search("refund deadline", self.index, "hybrid", top_k=3)
        self.assertTrue(all(r["weights"] == {"dense": 0.5, "bm25": 0.5} for r in results))

    def test_candidate_count_and_top_k_enforced(self):
        results = search("corporate card limit", self.index, "dense-rerank", top_k=3, candidate_count=7)
        self.assertEqual(len(self.reranker.calls[-1]["pairs"]), 7)
        self.assertEqual(len(results), 3)
        with self.assertRaises(ValueError):
            search("corporate card limit", self.index, "dense-rerank", top_k=8, candidate_count=7)
        with self.assertRaises(ValueError):
            search("corporate card limit", self.index, "hybrid-rerank", candidate_count=0)

    def test_format_distinguishes_reranker_and_retrieval_scores(self):
        dense_text = format_result(search("refund deadline", self.index, "dense-rerank")[0])
        hybrid_text = format_result(search("refund deadline", self.index, "hybrid-rerank")[0])
        self.assertIn("Cross-encoder score", dense_text)
        self.assertIn("not a probability", dense_text)
        self.assertIn("First-stage retrieval: dense rank", dense_text)
        self.assertIn("cosine similarity", dense_text)
        self.assertIn("weighted RRF", hybrid_text)
        self.assertIn("Section: ", dense_text)

    def test_evaluation_pipelines(self):
        rankings, first_stage, timing = rank_rerank_pipelines(self.questions, self.index)
        self.assertEqual(tuple(rankings), RERANK_PIPELINES)
        for question in self.questions:
            question_id = question["question_id"]
            self.assertEqual(len(first_stage["dense"][question_id]), len(self.index["chunks"]))
            for pipeline, source in (("dense_rerank", "dense"), ("hybrid_rerank", "hybrid_60_40")):
                reranked = rankings[pipeline][question_id]
                self.assertEqual(len(reranked), DEFAULT_CANDIDATES)
                self.assertEqual({r["chunk_id"] for r in reranked},
                                 {r["chunk_id"] for r in first_stage[source][question_id][:DEFAULT_CANDIDATES]})
                direct = search(question["question"], self.index, pipeline.replace("_", "-"))
                self.assertEqual([r["chunk_id"] for r in direct], [r["chunk_id"] for r in reranked[:5]])
        for pipeline in RERANK_PIPELINES:
            summary = summarize(self.questions, rankings[pipeline])
            for name, value in summary["overall"].items():
                if name != "count":
                    self.assertTrue(0.0 <= value <= 1.0, f"{pipeline} {name}")
        latency = rerank_latency(timing)
        for key in ("dense_retrieval", "hybrid_retrieval", "rerank_dense", "rerank_hybrid",
                    "dense_rerank_total", "hybrid_rerank_total", "dense_rerank_direct", "hybrid_rerank_direct"):
            self.assertEqual(len(latency[key]), 2)
            self.assertGreaterEqual(latency[key][0], 0.0)
        self.assertAlmostEqual(latency["dense_rerank_total"][0],
                               latency["dense_retrieval"][0] + latency["rerank_dense"][0], places=6)


class CandidateMetricTests(unittest.TestCase):
    def ranking(self, sections):
        return [{"section_id": s, "chunk_id": f"{s}#0", "document_id": s.split(".")[0]} for s in sections]

    def test_candidate_recall_gives_partial_credit(self):
        questions = [
            {"question_id": "Q1", "category": "multi_section", "relevant_section_ids": ["A.1", "B.1"]},
            {"question_id": "Q2", "category": "exact_keyword", "relevant_section_ids": ["Z.1"]},
        ]
        filler = [f"F.{i}" for i in range(30)]
        rankings = {"Q1": self.ranking(["A.1"] + filler[:12] + ["B.1"] + filler[12:]),
                    "Q2": self.ranking(filler)}
        means, rows = candidate_recall(questions, rankings)
        self.assertEqual(rows[0]["recall@5"], 0.5)
        self.assertEqual(rows[0]["recall@10"], 0.5)
        self.assertEqual(rows[0]["recall@20"], 1.0)   # B.1 at rank 14
        self.assertEqual(rows[0]["hit@5"], 1.0)
        self.assertEqual(rows[1]["hit@20"], 0.0)
        self.assertAlmostEqual(means["recall@20"], 0.5)
        self.assertAlmostEqual(means["hit@20"], 0.5)

    def test_evidence_coverage_separates_missing_from_misranked(self):
        questions = [{"question_id": "Q1", "relevant_section_ids": ["A.1", "B.1", "C.1"]}]
        first_stage = {"Q1": self.ranking(["A.1", "X.1", "B.1", "Y.1", "C.1"])}
        reranked = {"Q1": self.ranking(["X.1", "Y.1", "A.1"])}  # B.1 lost from the top, C.1 not a candidate
        reranked["Q1"] += self.ranking(["P.1", "Q.1", "R.1", "B.1"])
        missing, misranked = evidence_coverage(questions, first_stage, reranked, candidate_count=4)
        self.assertEqual(missing, [("Q1", "C.1", 5)])
        self.assertEqual(misranked, [("Q1", "B.1", 7)])


class RerankCommandLineTests(unittest.TestCase):
    def run_cli(self, main, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                main(argv)
                code = 0
            except SystemExit as exit_:
                code = exit_.code
        return code, out.getvalue(), err.getvalue()

    def test_invalid_rerank_flags_rejected_before_loading_models(self):
        cases = (
            (["--method", "dense", "--candidates", "10"], "applies only"),
            (["--method", "dense-rerank", "--candidates", "3"], "cannot exceed"),
            (["--method", "dense-rerank", "--candidates", "0"], "positive integer"),
            (["--method", "dense-rerank", "--dense-weight", "0.6"], "apply only"),
            (["--method", "hybrid-rerank", "--dense-weight", "0.9", "--bm25-weight", "0.4"], "sum to 1"),
        )
        for argv, message in cases:
            code, _, err = self.run_cli(retrieve_main, ["--question", "q"] + argv)
            self.assertEqual(code, 2, argv)
            self.assertIn(message, err, argv)

    def test_evaluate_rerank_guards(self):
        code, _, err = self.run_cli(evaluate_main, ["--rerank", "--compare"])
        self.assertEqual(code, 2)
        self.assertIn("choose one", err)
        code, _, err = self.run_cli(evaluate_main, ["--rerank", "--split", "test"])
        self.assertEqual(code, 2)
        self.assertIn("--confirm-held-out", err)

    def test_evaluate_rerank_runs_on_development(self):
        real_build = build_search_index
        with mock.patch("src.retrieve.build_search_index",
                        side_effect=lambda methods, **kw: real_build(methods, model=FakeModel(),
                                                                      reranker=MockCrossEncoder())):
            code, out, _ = self.run_cli(evaluate_main, ["--rerank", "--failures"])
        self.assertEqual(code, 0)
        self.assertIn("Split: development", out)
        self.assertIn("Questions: 37", out)
        for name in ("Dense", "Hybrid .6/.4", "Dense→CE", "Hybrid→CE", "Recall@20", "Hit@20",
                     "Evidence coverage", "measured end to end"):
            self.assertIn(name, out)

    def test_existing_commands_unchanged(self):
        code, out, _ = self.run_cli(retrieve_main, ["--question", "What does FIN-205.3 say?", "--method", "bm25"])
        self.assertEqual(code, 0)
        self.assertIn("BM25 score", out)


@unittest.skipUnless(RUN_MODEL_TESTS, "set RAGFORGE_MODEL_TESTS=1 to run tests that load the cross-encoder")
class RealCrossEncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.rerank import load_cross_encoder

        cls.model = load_cross_encoder(DEFAULT_RERANK_MODEL)

    def test_loads_and_scores_are_finite_logits(self):
        pairs = [("How long do overseas customers have to request a refund?",
                  "International customers may request a refund within 45 days of delivery."),
                 ("How long do overseas customers have to request a refund?",
                  "All user account passwords must be at least 14 characters long.")]
        scores = score_pairs(self.model, pairs)
        self.assertEqual(scores.shape, (2,))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertGreater(scores[0], scores[1])
        self.assertTrue(np.any((scores < 0) | (scores > 1)))  # raw logits, not probabilities

    def test_real_pipelines(self):
        index = build_search_index(ALL_METHODS, reranker=self.model)
        for method in ("dense-rerank", "hybrid-rerank"):
            results = search("What does FIN-205.3 say?", index, method)
            self.assertEqual(len(results), 5)
            self.assertTrue(all(np.isfinite(r["score"]) for r in results))
            scores = [r["score"] for r in results]
            self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main()
