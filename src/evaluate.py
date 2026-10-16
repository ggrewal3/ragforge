"""Retrieval benchmark: document-level and section-level (evidence-level) metrics.

Usage:
    python -m src.evaluate --split development                      # dense (default method)
    python -m src.evaluate --split development --method bm25        # one method: dense | bm25 | hybrid
    python -m src.evaluate --split development --compare            # all three methods side by side
    python -m src.evaluate --split development --failures           # plus per-question diagnostics
    python -m src.evaluate --split development --compare --failures # plus section ranks across methods
    python -m src.evaluate --split test --compare                   # held-out: final reporting only
    python -m src.evaluate --split all                              # descriptive reporting only

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


METHOD_NAMES = {"dense": "Dense", "bm25": "BM25", "hybrid": "Hybrid (RRF)"}


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


def hybrid_verdict(dense_scores, hybrid_scores):
    """'helped' / 'hurt' / 'mixed' / 'no change', judged on Section Hit@1, Recall@5 and MRR@5 versus dense."""
    keys = ("section_hit@1", "section_recall@5", "section_mrr@5")
    better = any(hybrid_scores[k] > dense_scores[k] + 1e-12 for k in keys)
    worse = any(hybrid_scores[k] < dense_scores[k] - 1e-12 for k in keys)
    if better and worse:
        return "mixed"
    return "helped" if better else "hurt" if worse else "no change"


def print_comparison(results, split, timings):
    methods = list(results)
    print("\nRAGForge — Retrieval Method Comparison")
    print(f"Split: {split}")
    print(f"Questions: {next(iter(results.values()))[0]['overall']['count']}")
    print(f"\n{'Metric':<24}" + "".join(f"{METHOD_NAMES[m]:>14}" for m in methods))
    for key, label in METRIC_LABELS + (("document_mrr@5", "Document MRR@5 (M1)"),):
        print(f"{label:<24}" + "".join(f"{results[m][0]['overall'][key]:>14.3f}" for m in methods))
    for key, label in (("mean_ms", "Query latency mean ms"), ("median_ms", "Query latency median ms")):
        print(f"{label:<24}" + "".join(f"{results[m][1][key]:>14.2f}" for m in methods))
    built = ", ".join(f"{name.replace('_', ' ')} {seconds:.2f} s" for name, seconds in timings.items())
    print(f"\nOne-time setup (not in query latency): {built}")

    for key, label in (("section_hit@1", "Section Hit@1"), ("section_mrr@5", "Section MRR@5")):
        print(f"\n{label} by category")
        print(f"{'Category':<30}{'N':>3}" + "".join(f"{METHOD_NAMES[m]:>14}" for m in methods))
        for category, scores in results[methods[0]][0]["by_category"].items():
            flag = "  (n<5)" if scores["count"] < SMALL_SAMPLE else ""
            values = "".join(f"{results[m][0]['by_category'][category][key]:>14.3f}" for m in methods)
            print(f"{category:<30}{scores['count']:>3}{values}{flag}")


def print_rank_table(results):
    """Development diagnostics: questions where any method misses rank 1 or a required section in the top 5."""
    methods = list(results)
    rows_by_question = {}
    for method in methods:
        for row in results[method][0]["per_question"]:
            rows_by_question.setdefault(row["question"]["question_id"], {})[method] = row

    print("\nSection ranks across methods (development only; '-' = not retrieved at any depth)")
    print("Shown when any method misses Section Hit@1 or Section Recall@5. Ranks are chunk ranks in the full ranking.")
    verdicts = {}
    for question_id, rows in rows_by_question.items():
        if all(row["section_hit@1"] == 1.0 and row["section_recall@5"] == 1.0 for row in rows.values()):
            continue
        view = compare_question(rows[methods[0]]["question"], rows)
        question = view["question"]
        print(f"\n  {question_id} [{question['category']}] {question['question']}")
        print(f"    {'section':<12}" + "".join(f"{METHOD_NAMES[m]:>14}" for m in methods))
        for section_id, ranks in view["section_ranks"].items():
            cells = "".join(f"{'-' if ranks[m] is None else ranks[m]:>14}" for m in methods)
            print(f"    {section_id:<12}{cells}")
        for key, label in (("section_hit@1", "Hit@1"), ("section_recall@5", "Recall@5"), ("section_mrr@5", "MRR@5")):
            print(f"    {label:<12}" + "".join(f"{view['scores'][m][key]:>14.2f}" for m in methods))
        if "dense" in methods and "hybrid" in methods:
            verdict = hybrid_verdict(view["scores"]["dense"], view["scores"]["hybrid"])
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
            print(f"    Hybrid vs dense: {verdict}")
    if verdicts:
        print("\n  Hybrid vs dense over the questions listed: "
              + ", ".join(f"{name} {count}" for name, count in sorted(verdicts.items())))


def main(argv=None):
    from src.retrieve import METHODS, build_search_index

    parser = argparse.ArgumentParser(description="RAGForge retrieval benchmark.")
    parser.add_argument("--split", choices=SPLITS + ("all",), default="development",
                        help="question split to evaluate (default: development)")
    parser.add_argument("--method", choices=METHODS, default="dense",
                        help="retrieval method to evaluate (default: dense)")
    parser.add_argument("--compare", action="store_true", help="evaluate dense, BM25 and hybrid side by side")
    parser.add_argument("--failures", action="store_true",
                        help="per-question diagnostics (development split only); with --compare, "
                             "section ranks across methods")
    args = parser.parse_args(argv)
    if args.failures and args.split != "development":
        parser.error("--failures is only available for the development split; the test split stays held out.")

    from src.ingest import load_documents

    documents = load_documents()
    questions = filter_split(load_evaluation_questions(documents=documents), args.split)
    methods = METHODS if args.compare else (args.method,)
    index = build_search_index(methods)
    details = [f"Loaded {len(documents)} documents", f"{len(index['chunks'])} chunks"]
    if index["chunk_embeddings"] is not None:
        details.append(f"embedding dimension {index['chunk_embeddings'].shape[1]}")
    if index["bm25"] is not None:
        details.append(f"BM25 vocabulary {len(index['bm25']['idf'])} terms")
    print(", ".join(details) + ".")

    results = {}
    for method in methods:
        rankings, latencies_ms = rank_questions(questions, index, method)
        results[method] = (summarize(questions, rankings), latency_stats(latencies_ms))

    if args.compare:
        print_comparison(results, args.split, index["timings"])
        if args.failures:
            print_rank_table(results)
    else:
        summary, latency = results[args.method]
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
