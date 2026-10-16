"""Dense retrieval: rank chunks by cosine similarity to a question, plus a baseline evaluation.

Usage:
    python -m src.retrieve                      # interactive question loop
    python -m src.retrieve --question "..."     # answer one or more questions and exit
    python -m src.retrieve --evaluate           # Recall@5 / MRR@5 on the labeled questions
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.chunking import DEFAULT_MAX_WORDS, chunk_documents
from src.embeddings import DEFAULT_MODEL_NAME, embed_chunks, embed_query, load_model
from src.ingest import DOCUMENTS_DIR, PROJECT_ROOT, load_documents

DEFAULT_TOP_K = 5
EVALUATION_PATH = PROJECT_ROOT / "data" / "evaluations" / "retrieval_questions.json"


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a, b):
    """cosine(a, b) = (a . b) / (||a|| * ||b||), for two 1-D vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(f"Expected two 1-D vectors, got shapes {a.shape} and {b.shape}.")
    if a.shape != b.shape:
        raise ValueError(f"Vector shapes differ: {a.shape} vs {b.shape}.")
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cosine similarity is undefined for a zero-norm vector.")
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarities(query_vector, matrix):
    """Cosine similarity between one query vector and every row of a matrix.

    Vectorized form of the same formula: the numerator for all rows at once is
    the matrix-vector product `matrix @ query`. For unit-normalized embeddings
    (which embeddings.py produces) every norm is 1, so the scores are exactly
    `matrix @ query`. The division is kept so the function stays correct for
    any input.
    """
    query_vector = np.asarray(query_vector, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    if query_vector.ndim != 1:
        raise ValueError(f"Query embedding must be 1-D, got shape {query_vector.shape}.")
    if matrix.ndim != 2:
        raise ValueError(f"Chunk embeddings must be 2-D, got shape {matrix.shape}.")
    if matrix.shape[1] != query_vector.shape[0]:
        raise ValueError(
            f"Dimension mismatch: query has {query_vector.shape[0]} dims, "
            f"chunk embeddings have {matrix.shape[1]}."
        )
    query_norm = np.linalg.norm(query_vector)
    row_norms = np.linalg.norm(matrix, axis=1)
    if query_norm == 0:
        raise ValueError("Query embedding has zero norm.")
    if np.any(row_norms == 0):
        raise ValueError(f"Chunk embedding(s) with zero norm at row(s) {np.flatnonzero(row_norms == 0).tolist()}.")
    return (matrix @ query_vector) / (row_norms * query_norm)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def validate_top_k(top_k):
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 1:
        raise ValueError(f"top_k must be a positive integer, got {top_k!r}.")
    return int(top_k)


def rank_chunks(query_embedding, chunk_embeddings, chunks, top_k=DEFAULT_TOP_K):
    """Return the top_k chunks by cosine similarity, highest first.

    If top_k exceeds the number of chunks, all chunks are returned.
    """
    top_k = validate_top_k(top_k)
    if not chunks:
        raise ValueError("Cannot retrieve from an empty chunk collection.")
    if len(chunks) != len(chunk_embeddings):
        raise ValueError(
            f"Got {len(chunks)} chunks but {len(chunk_embeddings)} chunk embeddings."
        )

    scores = cosine_similarities(query_embedding, chunk_embeddings)
    if not np.all(np.isfinite(scores)):
        raise ValueError("Similarity scores contain NaN or infinite values.")

    # Stable sort on the negated scores: highest similarity first, ties broken by chunk order.
    order = np.argsort(-scores, kind="stable")[:top_k]
    results = []
    for rank, index in enumerate(order, start=1):
        result = dict(chunks[index])
        result["rank"] = rank
        result["similarity"] = float(scores[index])
        results.append(result)
    return results


def build_index(model, documents_dir=DOCUMENTS_DIR, max_words=DEFAULT_MAX_WORDS):
    """Load, chunk, and embed the corpus once. Returns (documents, chunks, chunk_embeddings)."""
    documents = load_documents(documents_dir)
    chunks = chunk_documents(documents, max_words)
    if not chunks:
        raise ValueError("Chunking produced no chunks.")
    chunk_embeddings = embed_chunks(model, chunks)
    return documents, chunks, chunk_embeddings


def retrieve(question, model, chunks, chunk_embeddings, top_k=DEFAULT_TOP_K):
    """Embed a question and return its top_k most similar chunks."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")
    top_k = validate_top_k(top_k)
    query_embedding = embed_query(model, question)
    return rank_chunks(query_embedding, chunk_embeddings, chunks, top_k)


def format_result(result):
    return "\n".join([
        f"Rank: {result['rank']}",
        f"Similarity: {result['similarity']:.4f}",
        f"Document: {result['document_id']}",
        f"Title: {result['title']}",
        f"Department: {result['department']}",
        f"Section: {result['section_title']} ({result['chunk_id']})",
        "Text:",
        result["text"],
    ])


# ---------------------------------------------------------------------------
# Evaluation metrics (document level)
# ---------------------------------------------------------------------------

def unique_document_ids(results):
    """Collapse ranked chunks into ranked unique document IDs (first occurrence wins)."""
    ranked = []
    for result in results:
        if result["document_id"] not in ranked:
            ranked.append(result["document_id"])
    return ranked


def recall_at_k(ranked_document_ids, relevant_document_ids, k=DEFAULT_TOP_K):
    """Fraction of the relevant documents that appear in the first k ranked documents."""
    relevant = set(relevant_document_ids)
    if not relevant:
        raise ValueError("relevant_document_ids must not be empty.")
    retrieved = set(ranked_document_ids[:k])
    return len(relevant & retrieved) / len(relevant)


def reciprocal_rank_at_k(ranked_document_ids, relevant_document_ids, k=DEFAULT_TOP_K):
    """1 / rank of the first relevant document within the first k, or 0 if none appear."""
    relevant = set(relevant_document_ids)
    if not relevant:
        raise ValueError("relevant_document_ids must not be empty.")
    for rank, document_id in enumerate(ranked_document_ids[:k], start=1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def load_evaluation_questions(path=EVALUATION_PATH, known_document_ids=None):
    """Load labeled questions and check their structure and document references."""
    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"{path}: expected a non-empty JSON list of questions.")
    seen = set()
    for item in questions:
        for field in ("question_id", "question", "relevant_document_ids", "category"):
            if field not in item:
                raise ValueError(f"{path}: question {item.get('question_id', '?')} is missing '{field}'.")
        if item["question_id"] in seen:
            raise ValueError(f"{path}: duplicate question_id {item['question_id']}.")
        seen.add(item["question_id"])
        if not item["question"].strip() or not item["relevant_document_ids"]:
            raise ValueError(f"{path}: question {item['question_id']} has empty text or labels.")
        if known_document_ids is not None:
            unknown = set(item["relevant_document_ids"]) - set(known_document_ids)
            if unknown:
                raise ValueError(f"{path}: question {item['question_id']} references unknown documents {sorted(unknown)}.")
    return questions


def summarize_evaluation(questions, rankings, k=DEFAULT_TOP_K):
    """Compute per-question scores and mean Recall@k / MRR@k overall and per category.

    `rankings` maps question_id -> ranked unique document IDs.
    """
    per_question = []
    for item in questions:
        ranked = rankings[item["question_id"]]
        per_question.append({
            "question_id": item["question_id"],
            "category": item["category"],
            "ranked_document_ids": ranked[:k],
            "relevant_document_ids": item["relevant_document_ids"],
            "recall": recall_at_k(ranked, item["relevant_document_ids"], k),
            "reciprocal_rank": reciprocal_rank_at_k(ranked, item["relevant_document_ids"], k),
        })

    def mean_scores(rows):
        return {
            "count": len(rows),
            "recall": sum(row["recall"] for row in rows) / len(rows),
            "mrr": sum(row["reciprocal_rank"] for row in rows) / len(rows),
        }

    by_category = {}
    for category in sorted({row["category"] for row in per_question}):
        by_category[category] = mean_scores([row for row in per_question if row["category"] == category])

    return {"per_question": per_question, "by_category": by_category, "overall": mean_scores(per_question)}


def run_evaluation(model, chunks, chunk_embeddings, questions, k=DEFAULT_TOP_K):
    """Retrieve the top-k chunks per question and score them at the document level."""
    rankings = {}
    for item in questions:
        results = retrieve(item["question"], model, chunks, chunk_embeddings, top_k=k)
        rankings[item["question_id"]] = unique_document_ids(results)
    return summarize_evaluation(questions, rankings, k)


def print_evaluation(summary, k=DEFAULT_TOP_K):
    print(f"\nPer-question results (top {k} chunks -> unique documents):")
    for row in summary["per_question"]:
        print(
            f"  {row['question_id']}  {row['category']:<30} R={row['recall']:.2f}  RR={row['reciprocal_rank']:.2f}  "
            f"relevant={row['relevant_document_ids']}  retrieved={row['ranked_document_ids']}"
        )
    print(f"\n{'Category':<30} {'N':>3} {f'Recall@{k}':>10} {f'MRR@{k}':>8}")
    for category, scores in summary["by_category"].items():
        print(f"{category:<30} {scores['count']:>3} {scores['recall']:>10.3f} {scores['mrr']:>8.3f}")
    overall = summary["overall"]
    print(f"{'overall':<30} {overall['count']:>3} {overall['recall']:>10.3f} {overall['mrr']:>8.3f}")
    print(
        "\nNote: this is an initial development baseline on a small hand-labeled set, "
        "not a statistically reliable benchmark."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def answer(question, model, chunks, chunk_embeddings, top_k):
    for result in retrieve(question, model, chunks, chunk_embeddings, top_k):
        print()
        print(format_result(result))
    print()


def interactive_loop(model, chunks, chunk_embeddings, top_k):
    while True:
        print("Enter your question (or 'exit' / 'quit'):")
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            print("Please enter a non-empty question.\n")
            continue
        answer(question, model, chunks, chunk_embeddings, top_k)


def main(argv=None):
    parser = argparse.ArgumentParser(description="RAGForge M1 dense retrieval.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="number of chunks to return (default 5)")
    parser.add_argument("--question", action="append", help="answer this question and exit (repeatable)")
    parser.add_argument("--evaluate", action="store_true", help="run the baseline Recall@5 / MRR@5 evaluation")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Sentence Transformers model name")
    args = parser.parse_args(argv)
    try:
        top_k = validate_top_k(args.top_k)
    except ValueError as error:
        parser.error(str(error))

    print("RAGForge — Dense Retrieval")
    model = load_model(args.model)
    documents, chunks, chunk_embeddings = build_index(model)
    print(f"Loaded {len(documents)} documents.")
    print(f"Created {len(chunks)} chunks.")
    print(f"Embedding model: {args.model}")
    print(f"Embedding dimension: {chunk_embeddings.shape[1]}.")

    if args.evaluate:
        questions = load_evaluation_questions(known_document_ids={d["document_id"] for d in documents})
        print(f"Evaluating {len(questions)} labeled questions.")
        print_evaluation(run_evaluation(model, chunks, chunk_embeddings, questions, DEFAULT_TOP_K))
    elif args.question:
        for question in args.question:
            print(f"\n=== Question: {question}")
            answer(question, model, chunks, chunk_embeddings, top_k)
    else:
        print()
        interactive_loop(model, chunks, chunk_embeddings, top_k)


if __name__ == "__main__":
    main()
