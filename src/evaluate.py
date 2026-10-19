"""Retrieval benchmark: document-level and section-level (evidence-level) metrics.

Usage:
    python -m src.evaluate --split development                      # dense (default method)
    python -m src.evaluate --split development --method bm25        # one method: dense | bm25 | hybrid
    python -m src.evaluate --split development --compare            # dense, BM25, four RRF weightings
    python -m src.evaluate --split development --failures           # plus per-question diagnostics
    python -m src.evaluate --split development --compare --failures # plus per-question changes and ranks
    python -m src.evaluate --split development --rerank             # dense / hybrid .6/.4, with and without
                                                                    # cross-encoder reranking (M3)
    python -m src.evaluate --split development --rerank --failures  # plus changes, coverage and ranks
    python -m src.evaluate --split test --compare --confirm-held-out   # held-out: final reporting only
    python -m src.evaluate --split all --confirm-held-out              # descriptive reporting only

Two ranked lists are scored for each question, both taken from the same top-k chunks:

  * Document level (M1 convention): the chunks are collapsed into unique
    document IDs in rank order, so a document's rank is its position among
    the distinct documents.
  * Section level: the section ID of every chunk, in chunk rank order, with
    no collapsing. A section split into two chunks can appear twice; recall
    counts it once, and reciprocal rank uses the rank of its first chunk.

The metric functions below are generic over IDs, so the same code computes
both levels. Only the ranked list passed in differs.
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from src.ingest import PROJECT_ROOT

DEFAULT_K = 5
EVALUATION_PATH = PROJECT_ROOT / "data" / "evaluations" / "retrieval_questions.json"

CATEGORIES = (
    "exact_keyword",
    "exact_identifier",
    "semantic_paraphrase",
    "similar_policy_disambiguation",
    "numerical_threshold",
    "policy_exception",
    "multi_section",
)
SPLITS = ("development", "test")
REQUIRED_QUESTION_FIELDS = (
    "question_id", "question", "relevant_document_ids", "relevant_section_ids", "category", "split",
)

# Deterministic split recipe; see assign_splits().
SPLIT_SEED = 1729
TEST_FRACTION = 0.3


# ---------------------------------------------------------------------------
# Generic ranking metrics
# ---------------------------------------------------------------------------

def _relevant_set(relevant_ids):
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("relevant IDs must not be empty.")
    return relevant


def recall_at_k(ranked_ids, relevant_ids, k=DEFAULT_K):
    """Fraction of the unique relevant IDs that appear in the first k ranked positions.

    Duplicates in `ranked_ids` (several chunks from one section) count once.
    """
    relevant = _relevant_set(relevant_ids)
    return len(relevant & set(ranked_ids[:k])) / len(relevant)


def hit_at_k(ranked_ids, relevant_ids, k=1):
    """1.0 if any relevant ID appears in the first k ranked positions, else 0.0."""
    relevant = _relevant_set(relevant_ids)
    return 1.0 if relevant & set(ranked_ids[:k]) else 0.0


def reciprocal_rank_at_k(ranked_ids, relevant_ids, k=DEFAULT_K):
    """1 / rank of the first relevant ID within the first k positions, or 0.0 if none."""
    relevant = _relevant_set(relevant_ids)
    for rank, item in enumerate(ranked_ids[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def unique_document_ids(results):
    """Collapse ranked chunks into ranked unique document IDs (first occurrence wins)."""
    ranked = []
    for result in results:
        if result["document_id"] not in ranked:
            ranked.append(result["document_id"])
    return ranked


def score_question(question, ranked_chunks, k=DEFAULT_K):
    """Score one question from its ranked chunk list (each chunk has document_id and section_id)."""
    top = ranked_chunks[:k]
    document_ranking = unique_document_ids(top)
    section_ranking = [chunk["section_id"] for chunk in top]
    sections = question["relevant_section_ids"]
    return {
        "document_recall@5": recall_at_k(document_ranking, question["relevant_document_ids"], k),
        "document_mrr@5": reciprocal_rank_at_k(document_ranking, question["relevant_document_ids"], k),
        "section_recall@1": recall_at_k(section_ranking, sections, 1),
        "section_hit@1": hit_at_k(section_ranking, sections, 1),
        "section_recall@5": recall_at_k(section_ranking, sections, k),
        "section_mrr@5": reciprocal_rank_at_k(section_ranking, sections, k),
    }


METRIC_NAMES = (
    "document_recall@5", "document_mrr@5",
    "section_recall@1", "section_hit@1", "section_recall@5", "section_mrr@5",
)


# ---------------------------------------------------------------------------
# Dataset loading, validation and splitting
# ---------------------------------------------------------------------------

def _check_label_list(value, field, question_id):
    if not isinstance(value, list) or not value:
        raise ValueError(f"Question {question_id}: '{field}' must be a non-empty list.")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"Question {question_id}: '{field}' contains an empty or non-string label.")
    if len(set(value)) != len(value):
        raise ValueError(f"Question {question_id}: '{field}' contains duplicate labels.")


def validate_questions(questions, documents=None):
    """Check structure, categories, splits and, if documents are given, every label reference."""
    if not isinstance(questions, list) or not questions:
        raise ValueError("Expected a non-empty list of questions.")

    section_to_document = None
    if documents is not None:
        section_to_document = {
            section["section_id"]: document["document_id"]
            for document in documents for section in document["sections"]
        }
        known_documents = {document["document_id"] for document in documents}

    seen_ids = set()
    for question in questions:
        question_id = question.get("question_id", "?") if isinstance(question, dict) else "?"
        if not isinstance(question, dict):
            raise ValueError(f"Question {question_id}: expected a JSON object.")
        missing = [field for field in REQUIRED_QUESTION_FIELDS if field not in question]
        if missing:
            raise ValueError(f"Question {question_id}: missing field(s) {', '.join(missing)}.")
        if question_id in seen_ids:
            raise ValueError(f"Duplicate question_id {question_id}.")
        seen_ids.add(question_id)
        if not isinstance(question["question"], str) or not question["question"].strip():
            raise ValueError(f"Question {question_id}: question text is empty.")
        if question["category"] not in CATEGORIES:
            raise ValueError(f"Question {question_id}: unknown category '{question['category']}'.")
        if question["split"] not in SPLITS:
            raise ValueError(f"Question {question_id}: split must be one of {SPLITS}.")
        _check_label_list(question["relevant_document_ids"], "relevant_document_ids", question_id)
        _check_label_list(question["relevant_section_ids"], "relevant_section_ids", question_id)

        if section_to_document is None:
            continue
        unknown = set(question["relevant_document_ids"]) - known_documents
        if unknown:
            raise ValueError(f"Question {question_id}: unknown documents {sorted(unknown)}.")
        unknown = set(question["relevant_section_ids"]) - set(section_to_document)
        if unknown:
            raise ValueError(f"Question {question_id}: unknown sections {sorted(unknown)}.")
        section_documents = {section_to_document[s] for s in question["relevant_section_ids"]}
        if not section_documents <= set(question["relevant_document_ids"]):
            raise ValueError(
                f"Question {question_id}: sections {question['relevant_section_ids']} belong to documents "
                f"not labeled relevant ({sorted(section_documents - set(question['relevant_document_ids']))})."
            )
        if section_documents != set(question["relevant_document_ids"]):
            raise ValueError(
                f"Question {question_id}: relevant documents "
                f"{sorted(set(question['relevant_document_ids']) - section_documents)} have no labeled section."
            )
    return questions


def load_evaluation_questions(path=EVALUATION_PATH, documents=None):
    """Load and validate the labeled questions."""
    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_questions(questions, documents)


def filter_split(questions, split):
    if split == "all":
        return list(questions)
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS + ('all',)}, got {split!r}.")
    return [question for question in questions if question["split"] == split]


def assign_splits(questions, seed=SPLIT_SEED, test_fraction=TEST_FRACTION):
    """Deterministic stratified split: question_id -> 'development' | 'test'.

    Within each category, question IDs are sorted, shuffled with a fixed seed,
    and the first round(test_fraction * n) become test questions. This keeps
    every category represented in both splits. The recipe uses only IDs and
    categories, never retrieval results.
    """
    by_category = {}
    for question in questions:
        by_category.setdefault(question["category"], []).append(question["question_id"])

    rng = random.Random(seed)
    splits = {}
    for category in sorted(by_category):
        ids = sorted(by_category[category])
        rng.shuffle(ids)
        n_test = round(test_fraction * len(ids))
        for index, question_id in enumerate(ids):
            splits[question_id] = "test" if index < n_test else "development"
    return splits


# ---------------------------------------------------------------------------
# Running and summarizing
# ---------------------------------------------------------------------------

def mean_metrics(rows):
    return {name: sum(row[name] for row in rows) / len(rows) for name in METRIC_NAMES}


def summarize(questions, rankings, k=DEFAULT_K):
    """Score every question. `rankings` maps question_id -> ranked chunk list (best first)."""
    per_question = []
    for question in questions:
        row = {"question": question, "ranking": rankings[question["question_id"]]}
        row.update(score_question(question, rankings[question["question_id"]], k))
        per_question.append(row)

    by_category = {}
    for category in CATEGORIES:
        rows = [row for row in per_question if row["question"]["category"] == category]
        if rows:
            by_category[category] = {"count": len(rows), **mean_metrics(rows)}

    return {
        "per_question": per_question,
        "by_category": by_category,
        "overall": {"count": len(per_question), **mean_metrics(per_question)},
    }


def rank_all_chunks(questions, model, chunks, chunk_embeddings):
    """Full dense chunk ranking per question (M1.5 interface). Metrics use the top k; failure analysis uses deeper ranks."""
    from src.retrieve import retrieve

    return {
        question["question_id"]: retrieve(question["question"], model, chunks, chunk_embeddings, top_k=len(chunks))
        for question in questions
    }


def rank_questions(questions, index, method):
    """Full ranking per question for one method, plus per-query latency in milliseconds.

    Latency covers query-time work only (query embedding, scoring, sorting,
    fusion), not model loading or index construction. One untimed warm-up
    query runs first so one-off initialization does not land on the first
    timed question. BM25 rankings contain only chunks that share a term with
    the query, so they can be shorter than the chunk list.
    """
    from src.retrieve import search

    n_chunks = len(index["chunks"])
    search("warm-up query", index, method, top_k=n_chunks)
    rankings, latencies_ms = {}, []
    for question in questions:
        start = time.perf_counter()
        rankings[question["question_id"]] = search(question["question"], index, method, top_k=n_chunks)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
    return rankings, latencies_ms


def latency_stats(latencies_ms):
    ordered = sorted(latencies_ms)
    return {
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "max_ms": ordered[-1],
    }


def section_rank(ranking, section_id):
    """1-based chunk rank of the first chunk from one section, or None if the ranking never contains it."""
    return first_relevant_rank(ranking, [section_id])


# The fixed M2.5 fusion configurations: (name, dense weight, BM25 weight).
# These four are the whole experiment; no other weights are searched.
FUSION_CONFIGS = (
    ("rrf_50_50", 0.5, 0.5),
    ("rrf_60_40", 0.6, 0.4),
    ("rrf_70_30", 0.7, 0.3),
    ("rrf_80_20", 0.8, 0.2),
)
COMPARE_CONFIGS = ("dense", "bm25") + tuple(name for name, _, _ in FUSION_CONFIGS)


def rank_configurations(questions, index, fusion_configs=FUSION_CONFIGS):
    """Full rankings for dense, BM25 and every fusion configuration, with latency parts in ms.

    Each question is retrieved once by dense and once by BM25 (complete
    rankings); every fusion configuration then reuses those two lists, so
    embeddings and BM25 scores are never recomputed per weighting. Latency
    parts per configuration and question:
        retrieval_ms  underlying retrieval (dense, BM25, or dense + BM25 for fusion)
        fusion_ms     weighted RRF over the cached rankings (0 for dense / BM25)
        total_ms      retrieval_ms + fusion_ms: the end-to-end query time
    """
    from src.retrieve import fuse_rankings, search

    n_chunks = len(index["chunks"])
    search("warm-up query", index, "dense", top_k=n_chunks)  # untimed one-off initialization
    names = ("dense", "bm25") + tuple(name for name, _, _ in fusion_configs)
    rankings = {name: {} for name in names}
    timing = {name: {"retrieval_ms": [], "fusion_ms": [], "total_ms": []} for name in names}

    def record(name, question_id, ranking, retrieval_ms, fusion_ms):
        rankings[name][question_id] = ranking
        timing[name]["retrieval_ms"].append(retrieval_ms)
        timing[name]["fusion_ms"].append(fusion_ms)
        timing[name]["total_ms"].append(retrieval_ms + fusion_ms)

    for question in questions:
        question_id, text = question["question_id"], question["question"]
        start = time.perf_counter()
        dense = search(text, index, "dense", top_k=n_chunks)
        dense_ms = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        lexical = search(text, index, "bm25", top_k=n_chunks)
        bm25_ms = (time.perf_counter() - start) * 1000.0
        record("dense", question_id, dense, dense_ms, 0.0)
        record("bm25", question_id, lexical, bm25_ms, 0.0)
        for name, dense_weight, bm25_weight in fusion_configs:
            start = time.perf_counter()
            fused = fuse_rankings(dense, lexical, index["chunks"], n_chunks,
                                  dense_weight=dense_weight, bm25_weight=bm25_weight)
            record(name, question_id, fused, dense_ms + bm25_ms, (time.perf_counter() - start) * 1000.0)
    return rankings, timing


def configuration_latency(parts):
    """Mean of each latency part, plus the median total, in ms."""
    return {
        "retrieval": statistics.fmean(parts["retrieval_ms"]),
        "fusion": statistics.fmean(parts["fusion_ms"]),
        "total": statistics.fmean(parts["total_ms"]),
        "total_median": statistics.median(parts["total_ms"]),
    }


# ---------------------------------------------------------------------------
# Failure analysis (development split only)
# ---------------------------------------------------------------------------

CATEGORY_DIAGNOSES = {
    "exact_identifier": "exact identifier mismatch",
    "similar_policy_disambiguation": "similar-policy confusion",
    "numerical_threshold": "numerical detail mismatch",
    "policy_exception": "exception retrieval failure",
    "multi_section": "multi-section coverage failure",
}


def first_relevant_rank(ranking, relevant_section_ids):
    """1-based chunk rank of the first chunk from a relevant section in a full ranking, or None."""
    relevant = set(relevant_section_ids)
    for rank, chunk in enumerate(ranking, start=1):
        if chunk["section_id"] in relevant:
            return rank
    return None


def diagnose(question, top_chunk, departments_by_document):
    """Heuristic diagnostic label for a top-1 miss. It describes the miss and does not establish its cause."""
    relevant_documents = set(question["relevant_document_ids"])
    if top_chunk["document_id"] in relevant_documents:
        relation = "same document, wrong section"
    elif top_chunk["department"] in {departments_by_document[d] for d in relevant_documents}:
        relation = "related policy, same department"
    else:
        relation = "different department"

    label = CATEGORY_DIAGNOSES.get(question["category"])
    if question["category"] == "similar_policy_disambiguation" and relation == "same document, wrong section":
        label = None  # the right policy was found, so this is not cross-policy confusion
    if label is None:  # exact_keyword / semantic_paraphrase: fall back to what was retrieved.
        if relation == "same document, wrong section":
            label = "wrong section in correct document"
        elif relation == "related policy, same department":
            label = "similar-policy confusion"
        else:
            label = "semantic mismatch"
    return label, relation


def failure_analysis(summary, departments_by_document, k=DEFAULT_K):
    """Top-1 misses, plus multi-section questions whose top-k misses some required section."""
    top1_misses = []
    partial_coverage = []
    for row in summary["per_question"]:
        question, ranking = row["question"], row["ranking"]
        if row["section_hit@1"] == 0.0:
            if ranking:
                label, relation = diagnose(question, ranking[0], departments_by_document)
            else:
                label, relation = "no results", "retriever returned nothing"
            top1_misses.append({
                "question": question,
                "top_chunk": ranking[0] if ranking else None,
                "first_relevant_rank": first_relevant_rank(ranking, question["relevant_section_ids"]),
                "diagnosis": label,
                "relation": relation,
            })
        if len(question["relevant_section_ids"]) > 1 and row["section_recall@5"] < 1.0:
            found = {chunk["section_id"] for chunk in ranking[:k]}
            partial_coverage.append({
                "question": question,
                "missing_sections": [s for s in question["relevant_section_ids"] if s not in found],
                "section_recall@5": row["section_recall@5"],
            })
    return top1_misses, partial_coverage


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

METRIC_LABELS = (
    ("document_recall@5", "Document Recall@5"),
    ("section_recall@1", "Section Recall@1"),
    ("section_hit@1", "Section Hit@1"),
    ("section_recall@5", "Section Recall@5"),
    ("section_mrr@5", "Section MRR@5"),
)
SMALL_SAMPLE = 5


METHOD_NAMES = {
    "dense": "Dense", "bm25": "BM25", "hybrid": "Hybrid (RRF)",
    "rrf_50_50": "RRF .5/.5", "rrf_60_40": "RRF .6/.4", "rrf_70_30": "RRF .7/.3", "rrf_80_20": "RRF .8/.2",
    "hybrid_60_40": "Hybrid .6/.4", "dense_rerank": "Dense→CE", "hybrid_rerank": "Hybrid→CE",
}


def print_summary(summary, split, show_per_question, method="dense", latency=None):
    overall = summary["overall"]
    print(f"\nRAGForge — {METHOD_NAMES[method]} Retrieval Benchmark")
    print(f"Split: {split}")
    print(f"Questions: {overall['count']}")
    for key, label in METRIC_LABELS:
        print(f"{label + ':':<20}{overall[key]:.3f}")
    print(f"{'Document MRR@5:':<20}{overall['document_mrr@5']:.3f}   (M1 metric, kept for comparison)")
    if latency:
        print(f"{'Query latency:':<20}mean {latency['mean_ms']:.2f} ms, median {latency['median_ms']:.2f} ms, "
              f"max {latency['max_ms']:.2f} ms")

    columns = ("document_recall@5", "section_recall@1", "section_hit@1", "section_recall@5", "section_mrr@5")
    print(f"\n{'Category':<30}{'N':>3}{'DocR@5':>8}{'SecR@1':>8}{'SecH@1':>8}{'SecR@5':>8}{'SecMRR':>8}")
    for category, scores in summary["by_category"].items():
        values = "".join(f"{scores[c]:>8.3f}" for c in columns)
        flag = "  (n<5: very noisy)" if scores["count"] < SMALL_SAMPLE else ""
        print(f"{category:<30}{scores['count']:>3}{values}{flag}")

    if show_per_question:
        print("\nPer-question section results (top-5 chunk sections, best first):")
        for row in summary["per_question"]:
            question = row["question"]
            sections = [chunk["section_id"] for chunk in row["ranking"][:DEFAULT_K]]
            print(
                f"  {question['question_id']} {question['category']:<30} "
                f"H@1={row['section_hit@1']:.0f} R@5={row['section_recall@5']:.2f} RR={row['section_mrr@5']:.2f}  "
                f"relevant={question['relevant_section_ids']} retrieved={sections}"
            )


def print_failures(top1_misses, partial_coverage):
    print(f"\nFailure analysis (development only): {len(top1_misses)} top-1 misses")
    print("Diagnoses are heuristic labels from the question category and the retrieved chunk, not proven causes.")
    for miss in top1_misses:
        question, top = miss["question"], miss["top_chunk"]
        rank = miss["first_relevant_rank"]
        if rank is None:
            rank_text = "not found"
        else:
            rank_text = f"{rank}" + ("" if rank <= DEFAULT_K else " (outside top 5)")
        print(f"\n  {question['question_id']} [{question['category']}] {question['question']}")
        print(f"    Relevant sections:   {question['relevant_section_ids']}")
        if top is None:
            print("    Top-ranked section:  (no results)")
        else:
            print(f"    Top-ranked section:  {top['section_id']} {top['title']} - {top['section_title']} "
                  f"({top['score_type']} score {top['score']:.4f})")
        print(f"    First relevant rank: {rank_text}")
        print(f"    Diagnosis:           {miss['diagnosis']} ({miss['relation']})")

    counts = {}
    for miss in top1_misses:
        counts[miss["diagnosis"]] = counts.get(miss["diagnosis"], 0) + 1
    if counts:
        print("\n  Misses by diagnosis:")
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"    {label:<36}{count}")

    print(f"\nMulti-section questions missing a required section in the top 5: {len(partial_coverage)}")
    for item in partial_coverage:
        question = item["question"]
        print(f"  {question['question_id']} R@5={item['section_recall@5']:.2f} missing={item['missing_sections']}  "
              f"{question['question']}")


def compare_question(question, rows_by_method):
    """Per-question cross-method view: rank of every relevant section under each method."""
    ranks = {
        section_id: {method: section_rank(row["ranking"], section_id) for method, row in rows_by_method.items()}
        for section_id in question["relevant_section_ids"]
    }
    return {"question": question, "section_ranks": ranks,
            "scores": {method: {name: row[name] for name in METRIC_NAMES} for method, row in rows_by_method.items()}}


VERDICT_METRICS = ("section_hit@1", "section_recall@5", "section_mrr@5")


def compare_verdict(base_scores, new_scores):
    """'improved' / 'regressed' / 'mixed' / 'unchanged' on Section Hit@1, Recall@5 and MRR@5."""
    better = any(new_scores[k] > base_scores[k] + 1e-12 for k in VERDICT_METRICS)
    worse = any(new_scores[k] < base_scores[k] - 1e-12 for k in VERDICT_METRICS)
    if better and worse:
        return "mixed"
    return "improved" if better else "regressed" if worse else "unchanged"


def change_report(results, base, new):
    """Question IDs grouped by how `new` compares with `base`, per question."""
    base_rows = {row["question"]["question_id"]: row for row in results[base][0]["per_question"]}
    report = {"improved": [], "regressed": [], "mixed": [], "unchanged": []}
    for row in results[new][0]["per_question"]:
        question_id = row["question"]["question_id"]
        report[compare_verdict(base_rows[question_id], row)].append(question_id)
    return report


def print_comparison(results, split, timings):
    configs = list(results)
    width = 11
    header = "".join(f"{METHOD_NAMES[c]:>{width}}" for c in configs)
    print("\nRAGForge — Retrieval Configuration Comparison")
    print(f"Split: {split}")
    print(f"Questions: {next(iter(results.values()))[0]['overall']['count']}")
    print(f"\n{'Metric':<24}{header}")
    for key, label in METRIC_LABELS + (("document_mrr@5", "Document MRR@5 (M1)"),):
        print(f"{label:<24}" + "".join(f"{results[c][0]['overall'][key]:>{width}.3f}" for c in configs))
    for part, label in (("retrieval", "Retrieval ms (mean)"), ("fusion", "Fusion-only ms (mean)"),
                        ("total", "Total query ms (mean)"), ("total_median", "Total query ms (median)")):
        print(f"{label:<24}" + "".join(f"{results[c][1][part]:>{width}.3f}" for c in configs))
    built = ", ".join(f"{name.replace('_', ' ')} {seconds:.2f} s" for name, seconds in timings.items())
    print(f"\nOne-time setup (not in query latency): {built}")
    print("Hybrid total = dense retrieval + BM25 retrieval + fusion, each timed separately per question.")

    for key, label in (("section_hit@1", "Section Hit@1"), ("section_recall@5", "Section Recall@5"),
                       ("section_mrr@5", "Section MRR@5")):
        print(f"\n{label} by category")
        print(f"{'Category':<30}{'N':>3}{header}")
        for category, scores in results[configs[0]][0]["by_category"].items():
            flag = "  (n<5)" if scores["count"] < SMALL_SAMPLE else ""
            values = "".join(f"{results[c][0]['by_category'][category][key]:>{width}.3f}" for c in configs)
            print(f"{category:<30}{scores['count']:>3}{values}{flag}")


def print_change_reports(results):
    """Per-question improvements and regressions of each fusion configuration."""
    fusion = [c for c in results if c.startswith("rrf_")]
    print("\nPer-question changes (Section Hit@1, Recall@5, MRR@5; 'mixed' = some up, some down)")
    for new in fusion:
        for base in ("dense", "rrf_50_50"):
            if new == base or base not in results:
                continue
            report = change_report(results, base, new)
            print(f"\n  {METHOD_NAMES[new]} vs {METHOD_NAMES[base]}: "
                  + ", ".join(f"{verdict} {len(ids)}" for verdict, ids in report.items()))
            for verdict in ("improved", "regressed", "mixed"):
                if report[verdict]:
                    print(f"    {verdict:<10}{' '.join(report[verdict])}")


def print_rank_table(results, missing_note="'-' = not retrieved at any depth"):
    """Development diagnostics: questions where any configuration misses rank 1 or a required section in the top 5."""
    configs = list(results)
    width = max(11, max(len(METHOD_NAMES[c]) for c in configs) + 2)
    rows_by_question = {}
    for config in configs:
        for row in results[config][0]["per_question"]:
            rows_by_question.setdefault(row["question"]["question_id"], {})[config] = row

    print(f"\nSection ranks across configurations (development only; {missing_note})")
    print("Shown when any configuration misses Section Hit@1 or Section Recall@5. Ranks are chunk ranks.")
    for question_id, rows in rows_by_question.items():
        if all(row["section_hit@1"] == 1.0 and row["section_recall@5"] == 1.0 for row in rows.values()):
            continue
        view = compare_question(rows[configs[0]]["question"], rows)
        question = view["question"]
        print(f"\n  {question_id} [{question['category']}] {question['question']}")
        print(f"    {'section':<12}" + "".join(f"{METHOD_NAMES[c]:>{width}}" for c in configs))
        for section_id, ranks in view["section_ranks"].items():
            print(f"    {section_id:<12}" + "".join(f"{'-' if ranks[c] is None else ranks[c]:>{width}}"
                                                    for c in configs))
        for key, label in (("section_hit@1", "Hit@1"), ("section_recall@5", "Recall@5"), ("section_mrr@5", "MRR@5")):
            print(f"    {label:<12}" + "".join(f"{view['scores'][c][key]:>{width}.2f}" for c in configs))
        if "dense" in configs:
            verdicts = [f"{METHOD_NAMES[c]} {compare_verdict(view['scores']['dense'], view['scores'][c])}"
                        for c in configs if c not in ("dense", "bm25")]
            print(f"    vs dense:   {', '.join(verdicts)}")


# ---------------------------------------------------------------------------
# M3: cross-encoder reranking pipelines
# ---------------------------------------------------------------------------

RERANK_PIPELINES = ("dense", "hybrid_60_40", "dense_rerank", "hybrid_rerank")
RERANK_HYBRID_WEIGHTS = (0.6, 0.4)  # the M2.5 comparison configuration
CANDIDATE_KS = (5, 10, 20)


def rank_rerank_pipelines(questions, index, candidate_count=None, hybrid_weights=RERANK_HYBRID_WEIGHTS):
    """Rankings for dense, hybrid 0.6/0.4, and both of them reranked by the cross-encoder.

    Per question, the dense and BM25 rankings are computed once, fused once,
    and the first `candidate_count` chunks of each first-stage ranking are
    reranked by the same loaded cross-encoder. Returns:
        rankings     pipeline -> question_id -> ranked results (rerank lists hold
                     the reranked candidates only)
        first_stage  "dense" / "hybrid_60_40" -> question_id -> full first-stage ranking
        timing       part -> list of ms per question; parts are timed separately,
                     and the *_direct entries time the full search() call end to end

    Timing protocol: one complete untimed pass over all questions runs first
    (every component, both pipelines), so model and device warm-up does not
    land on the timed pass. A single short warm-up was not enough on MPS:
    it left the first component's mean inflated by about 10 ms.
    """
    from src.rerank import DEFAULT_CANDIDATES, rerank
    from src.retrieve import fuse_rankings, search

    candidate_count = DEFAULT_CANDIDATES if candidate_count is None else candidate_count
    reranker, chunks = index["reranker"], index["chunks"]
    n_chunks = len(chunks)
    dense_weight, bm25_weight = hybrid_weights

    # Untimed warm-up pass over every question and component.
    for question in questions:
        text = question["question"]
        dense = search(text, index, "dense", top_k=n_chunks)
        lexical = search(text, index, "bm25", top_k=n_chunks)
        hybrid = fuse_rankings(dense, lexical, chunks, n_chunks, dense_weight=dense_weight, bm25_weight=bm25_weight)
        rerank(text, dense[:candidate_count], reranker, top_k=candidate_count)
        rerank(text, hybrid[:candidate_count], reranker, top_k=candidate_count)

    rankings = {name: {} for name in RERANK_PIPELINES}
    first_stage = {"dense": {}, "hybrid_60_40": {}}
    parts = ("dense_ms", "bm25_ms", "fusion_ms", "rerank_dense_ms", "rerank_hybrid_ms",
             "dense_rerank_direct_ms", "hybrid_rerank_direct_ms")
    timing = {part: [] for part in parts}

    def timed(part, function, *args, **kwargs):
        start = time.perf_counter()
        value = function(*args, **kwargs)
        timing[part].append((time.perf_counter() - start) * 1000.0)
        return value

    for question in questions:
        question_id, text = question["question_id"], question["question"]
        dense = timed("dense_ms", search, text, index, "dense", top_k=n_chunks)
        lexical = timed("bm25_ms", search, text, index, "bm25", top_k=n_chunks)
        hybrid = timed("fusion_ms", fuse_rankings, dense, lexical, chunks, n_chunks,
                       dense_weight=dense_weight, bm25_weight=bm25_weight)
        dense_reranked = timed("rerank_dense_ms", rerank, text, dense[:candidate_count], reranker,
                               top_k=candidate_count)
        hybrid_reranked = timed("rerank_hybrid_ms", rerank, text, hybrid[:candidate_count], reranker,
                                top_k=candidate_count)
        timed("dense_rerank_direct_ms", search, text, index, "dense-rerank", candidate_count=candidate_count)
        timed("hybrid_rerank_direct_ms", search, text, index, "hybrid-rerank", candidate_count=candidate_count,
              dense_weight=dense_weight, bm25_weight=bm25_weight)

        first_stage["dense"][question_id] = dense
        first_stage["hybrid_60_40"][question_id] = hybrid
        rankings["dense"][question_id] = dense
        rankings["hybrid_60_40"][question_id] = hybrid
        rankings["dense_rerank"][question_id] = dense_reranked
        rankings["hybrid_rerank"][question_id] = hybrid_reranked
    return rankings, first_stage, timing


def rerank_latency(timing):
    """(mean, median) ms per question for each component; '_total' rows are per-question sums of components."""
    hybrid = [d + b + f for d, b, f in zip(timing["dense_ms"], timing["bm25_ms"], timing["fusion_ms"])]
    series = {
        "dense_retrieval": timing["dense_ms"],
        "hybrid_retrieval": hybrid,
        "rerank_dense": timing["rerank_dense_ms"],
        "rerank_hybrid": timing["rerank_hybrid_ms"],
        "dense_rerank_total": [d + r for d, r in zip(timing["dense_ms"], timing["rerank_dense_ms"])],
        "hybrid_rerank_total": [h + r for h, r in zip(hybrid, timing["rerank_hybrid_ms"])],
        "dense_rerank_direct": timing["dense_rerank_direct_ms"],
        "hybrid_rerank_direct": timing["hybrid_rerank_direct_ms"],
    }
    return {name: (statistics.fmean(values), statistics.median(values)) for name, values in series.items()}


def candidate_recall(questions, first_stage_rankings, ks=CANDIDATE_KS):
    """Section Recall@K and Hit@K of first-stage rankings, i.e. before any reranking.

    Recall@K counts the fraction of a question's relevant sections among the
    first K chunks, so a multi-section question with one of two sections
    found scores 0.5, not 1.
    """
    rows = []
    for question in questions:
        sections = [chunk["section_id"] for chunk in first_stage_rankings[question["question_id"]]]
        row = {"question_id": question["question_id"], "category": question["category"]}
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(sections, question["relevant_section_ids"], k)
            row[f"hit@{k}"] = hit_at_k(sections, question["relevant_section_ids"], k)
        rows.append(row)
    keys = [key for key in rows[0] if key.startswith(("recall@", "hit@"))]
    return {key: statistics.fmean(row[key] for row in rows) for key in keys}, rows


def evidence_coverage(questions, first_stage_rankings, reranked, candidate_count, k=DEFAULT_K):
    """Relevant sections missing from the candidates, and those present but ranked outside the top k."""
    missing, misranked = [], []
    for question in questions:
        question_id = question["question_id"]
        candidates = first_stage_rankings[question_id][:candidate_count]
        final = reranked[question_id]
        for section_id in question["relevant_section_ids"]:
            if section_rank(candidates, section_id) is None:
                missing.append((question_id, section_id, section_rank(first_stage_rankings[question_id], section_id)))
            else:
                final_rank = section_rank(final, section_id)
                if final_rank > k:
                    misranked.append((question_id, section_id, final_rank))
    return missing, misranked


def print_rerank_comparison(results, candidate_stats, latency, info):
    pipelines = list(results)
    width = 14
    header = "".join(f"{METHOD_NAMES[p]:>{width}}" for p in pipelines)
    print("\nRAGForge — Cross-Encoder Reranking Comparison")
    print(f"Split: {info['split']}")
    print(f"Questions: {next(iter(results.values()))['overall']['count']}")
    print(f"Cross-encoder: {info['model']} on {info['device']}, batch size {info['batch_size']}, "
          f"{info['candidates']} candidates -> top {DEFAULT_K}; hybrid weights 0.6 / 0.4")
    print(f"\n{'Metric':<24}{header}")
    for key, label in METRIC_LABELS:
        print(f"{label:<24}" + "".join(f"{results[p]['overall'][key]:>{width}.3f}" for p in pipelines))

    print("\nCandidate recall before reranking (first-stage rankings; fraction of relevant sections found)")
    print(f"{'First stage':<24}" + "".join(f"{label:>{width}}" for label in
                                        ("Recall@5", "Recall@10", "Recall@20", "Hit@20")))
    for name, stats in candidate_stats.items():
        values = (stats["recall@5"], stats["recall@10"], stats["recall@20"], stats["hit@20"])
        print(f"{METHOD_NAMES[name]:<24}" + "".join(f"{v:>{width}.3f}" for v in values))

    print("\nQuery latency per question in ms, mean / median (after a full untimed warm-up pass; loading excluded)")
    rows = (
        ("Dense retrieval", "dense_retrieval", "measured"),
        ("Hybrid retrieval", "hybrid_retrieval", "sum: dense + BM25 + fusion"),
        ("Cross-encoder (dense)", "rerank_dense", "measured, 20 pairs"),
        ("Cross-encoder (hybrid)", "rerank_hybrid", "measured, 20 pairs"),
        ("Dense→CE total", "dense_rerank_total", "sum of components"),
        ("Dense→CE total", "dense_rerank_direct", "measured end to end"),
        ("Hybrid→CE total", "hybrid_rerank_total", "sum of components"),
        ("Hybrid→CE total", "hybrid_rerank_direct", "measured end to end"),
    )
    for label, key, how in rows:
        mean, median = latency[key]
        print(f"  {label:<24}{mean:>8.2f} /{median:>7.2f}   ({how})")
    built = ", ".join(f"{name.replace('_', ' ')} {seconds:.2f} s" for name, seconds in info["timings"].items())
    print(f"One-time setup (not in query latency): {built}")

    for key, label in (("section_hit@1", "Section Hit@1"), ("section_recall@5", "Section Recall@5"),
                       ("section_mrr@5", "Section MRR@5")):
        print(f"\n{label} by category (categories with n<5 are very noisy)")
        print(f"{'Category':<30}{'N':>3}{header}")
        for category, scores in results[pipelines[0]]["by_category"].items():
            values = "".join(f"{results[p]['by_category'][category][key]:>{width}.3f}" for p in pipelines)
            print(f"{category:<30}{scores['count']:>3}{values}")


def print_rerank_changes(results, questions):
    """Per-question and per-category improvements/regressions of reranking."""
    categories = {q["question_id"]: q["category"] for q in questions}
    wrapped = {name: (summary, None) for name, summary in results.items()}
    print("\nPer-question changes (Section Hit@1, Recall@5, MRR@5; 'mixed' = some up, some down)")
    for new, base in (("dense_rerank", "dense"), ("hybrid_rerank", "hybrid_60_40"),
                      ("hybrid_rerank", "dense"), ("hybrid_rerank", "dense_rerank")):
        report = change_report(wrapped, base, new)
        print(f"\n  {METHOD_NAMES[new]} vs {METHOD_NAMES[base]}: "
              + ", ".join(f"{verdict} {len(ids)}" for verdict, ids in report.items()))
        for verdict in ("improved", "regressed", "mixed"):
            if report[verdict]:
                labelled = " ".join(f"{qid}({categories[qid]})" for qid in report[verdict])
                print(f"    {verdict:<10}{labelled}")


def print_evidence_coverage(questions, first_stage, rankings, candidate_count):
    print(f"\nEvidence coverage of the {candidate_count} rerank candidates (development only)")
    for pipeline, source in (("dense_rerank", "dense"), ("hybrid_rerank", "hybrid_60_40")):
        missing, misranked = evidence_coverage(questions, first_stage[source], rankings[pipeline], candidate_count)
        print(f"\n  {METHOD_NAMES[pipeline]}")
        print("    Relevant sections absent from the candidates (first-stage rank in brackets):")
        print("      " + (", ".join(f"{q} {s} [{r}]" for q, s, r in missing) or "none"))
        print(f"    Relevant sections in the candidates but reranked outside the top {DEFAULT_K}:")
        print("      " + (", ".join(f"{q} {s} [final {r}]" for q, s, r in misranked) or "none"))


def main(argv=None):
    from src.retrieve import METHODS, build_search_index

    parser = argparse.ArgumentParser(description="RAGForge retrieval benchmark.")
    parser.add_argument("--split", choices=SPLITS + ("all",), default="development",
                        help="question split to evaluate (default: development)")
    parser.add_argument("--method", choices=METHODS, default="dense",
                        help="retrieval method to evaluate (default: dense)")
    parser.add_argument("--compare", action="store_true",
                        help="dense, BM25 and four RRF weightings side by side")
    parser.add_argument("--rerank", action="store_true",
                        help="dense and hybrid 0.6/0.4, each with and without cross-encoder reranking")
    parser.add_argument("--failures", action="store_true",
                        help="per-question diagnostics (development split only); with --compare, "
                             "section ranks and per-question changes across configurations")
    parser.add_argument("--confirm-held-out", action="store_true",
                        help="required for --split test or all, which include held-out questions")
    args = parser.parse_args(argv)
    if args.failures and args.split != "development":
        parser.error("--failures is only available for the development split; the test split stays held out.")
    if args.compare and args.rerank:
        parser.error("choose one of --compare (M2.5 fusion weights) or --rerank (M3 reranking).")
    if args.split != "development" and not args.confirm_held_out:
        parser.error(f"--split {args.split} includes held-out test questions. Run it only on frozen "
                     "configurations, and add --confirm-held-out to proceed.")

    from src.ingest import load_documents

    documents = load_documents()
    questions = filter_split(load_evaluation_questions(documents=documents), args.split)
    from src.retrieve import ALL_METHODS

    methods = ALL_METHODS if args.rerank else METHODS if args.compare else (args.method,)
    index = build_search_index(methods)
    details = [f"Loaded {len(documents)} documents", f"{len(index['chunks'])} chunks"]
    if index["chunk_embeddings"] is not None:
        details.append(f"embedding dimension {index['chunk_embeddings'].shape[1]}")
    if index["bm25"] is not None:
        details.append(f"BM25 vocabulary {len(index['bm25']['idf'])} terms")
    print(", ".join(details) + ".")

    if args.rerank:
        from src.rerank import DEFAULT_BATCH_SIZE, DEFAULT_CANDIDATES, DEFAULT_RERANK_MODEL

        rankings, first_stage, timing = rank_rerank_pipelines(questions, index)
        results = {name: summarize(questions, rankings[name]) for name in RERANK_PIPELINES}
        candidate_stats = {name: candidate_recall(questions, first_stage[name])[0] for name in first_stage}
        info = {"split": args.split, "model": DEFAULT_RERANK_MODEL, "device": index["reranker"].device,
                "batch_size": DEFAULT_BATCH_SIZE, "candidates": DEFAULT_CANDIDATES, "timings": index["timings"]}
        print_rerank_comparison(results, candidate_stats, rerank_latency(timing), info)
        if args.failures:
            print_rerank_changes(results, questions)
            print_evidence_coverage(questions, first_stage, rankings, DEFAULT_CANDIDATES)
            print_rank_table({name: (summary, None) for name, summary in results.items()},
                             missing_note=f"'-' in rerank columns = not among the {DEFAULT_CANDIDATES} candidates")
    elif args.compare:
        rankings, timing = rank_configurations(questions, index)
        results = {config: (summarize(questions, rankings[config]), configuration_latency(timing[config]))
                   for config in COMPARE_CONFIGS}
        print_comparison(results, args.split, index["timings"])
        if args.failures:
            print_change_reports(results)
            print_rank_table(results)
    else:
        rankings, latencies_ms = rank_questions(questions, index, args.method)
        summary, latency = summarize(questions, rankings), latency_stats(latencies_ms)
        print_summary(summary, args.split, show_per_question=args.failures, method=args.method, latency=latency)
        built = ", ".join(f"{name.replace('_', ' ')} {seconds:.2f} s" for name, seconds in index["timings"].items())
        print(f"One-time setup (not in query latency): {built}")
        if args.failures:
            departments = {document["document_id"]: document["department"] for document in documents}
            print_failures(*failure_analysis(summary, departments))

    print("\nNote: small synthetic benchmark; treat scores as a development baseline, not a reliable estimate.")
    if args.split != "development":
        print("Do not use these results to choose retrieval settings; select on the development split only.")


if __name__ == "__main__":
    main()
