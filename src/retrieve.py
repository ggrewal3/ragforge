"""Retrieval over the policy chunks: dense (cosine), BM25, or hybrid (RRF).

Usage:
    python -m src.retrieve                                   # interactive, dense retrieval
    python -m src.retrieve --method bm25                     # interactive, BM25 (no model needed)
    python -m src.retrieve --question "..." --method hybrid  # answer one or more questions and exit
    python -m src.retrieve --question "..." --method hybrid --dense-weight 0.7 --bm25-weight 0.3
    python -m src.retrieve --evaluate                        # alias for: python -m src.evaluate --split development

The dense functions (rank_chunks, retrieve, build_index) are unchanged from M1.
search() is the common entry point for all three methods. BM25 lives in
src/bm25.py, fusion in src/fusion.py, and evaluation in src/evaluate.py.
"""

import argparse
import time

import numpy as np

from src.bm25 import build_chunk_bm25_index, rank_bm25
from src.chunking import DEFAULT_MAX_WORDS, chunk_documents
from src.embeddings import DEFAULT_MODEL_NAME, embed_chunks, embed_query, load_model
from src.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion, validate_weights
from src.ingest import DOCUMENTS_DIR, load_documents

DEFAULT_TOP_K = 5
METHODS = ("dense", "bm25", "hybrid")
DEFAULT_DENSE_WEIGHT = 0.5
DEFAULT_BM25_WEIGHT = 0.5


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


# ---------------------------------------------------------------------------
# Common interface for dense, BM25 and hybrid retrieval
# ---------------------------------------------------------------------------

def validate_method(method):
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}.")
    return method


def build_search_index(methods=METHODS, model=None, model_name=DEFAULT_MODEL_NAME,
                       documents_dir=DOCUMENTS_DIR, max_words=DEFAULT_MAX_WORDS):
    """Build everything the requested methods need, once.

    All methods share the same chunk list, so they rank the same chunk universe.
    The embedding model is loaded only if dense or hybrid retrieval is requested.
    Returns a dict with documents, chunks, model, chunk_embeddings, bm25, and
    build timings in seconds.
    """
    for method in methods:
        validate_method(method)
    timings = {}
    index = {"model": None, "chunk_embeddings": None, "bm25": None, "timings": timings}

    if "dense" in methods or "hybrid" in methods:
        start = time.perf_counter()
        index["model"] = model if model is not None else load_model(model_name)
        timings["model_load"] = time.perf_counter() - start
        start = time.perf_counter()
        index["documents"], index["chunks"], index["chunk_embeddings"] = build_index(
            index["model"], documents_dir, max_words)
        timings["dense_index"] = time.perf_counter() - start
    else:
        index["documents"] = load_documents(documents_dir)
        index["chunks"] = chunk_documents(index["documents"], max_words)

    if "bm25" in methods or "hybrid" in methods:
        start = time.perf_counter()
        index["bm25"] = build_chunk_bm25_index(index["chunks"])
        timings["bm25_index"] = time.perf_counter() - start
    return index


def _dense_search(question, index, top_k):
    results = retrieve(question, index["model"], index["chunks"], index["chunk_embeddings"], top_k)
    for result in results:
        result["score"] = result["similarity"]
        result["score_type"] = "cosine"
    return results


def hybrid_candidates(question, index):
    """Complete dense and BM25 rankings for a question, the inputs to fusion.

    Fusion runs on the complete rankings so no relevant chunk is cut off
    before fusion. Computing them separately lets several fusion weightings
    reuse one pair of rankings.
    """
    n_chunks = len(index["chunks"])
    dense = _dense_search(question, index, n_chunks)
    lexical = rank_bm25(question, index["bm25"], index["chunks"], n_chunks)
    return dense, lexical


def fuse_rankings(dense, lexical, chunks, top_k, rrf_k=DEFAULT_RRF_K,
                  dense_weight=DEFAULT_DENSE_WEIGHT, bm25_weight=DEFAULT_BM25_WEIGHT):
    """Weighted RRF over dense and BM25 result lists, returned as chunk results."""
    weights = {"dense": dense_weight, "bm25": bm25_weight}
    fused = reciprocal_rank_fusion(
        {"dense": [r["chunk_id"] for r in dense], "bm25": [r["chunk_id"] for r in lexical]},
        k=rrf_k, weights=weights)

    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    dense_scores = {r["chunk_id"]: r["score"] for r in dense}
    bm25_scores = {r["chunk_id"]: r["score"] for r in lexical}
    results = []
    for rank, entry in enumerate(fused[:top_k], start=1):
        result = dict(chunks_by_id[entry["id"]])
        result.update(
            rank=rank,
            score=entry["rrf_score"],
            score_type="rrf",
            weights=weights,
            dense_rank=entry["ranks"]["dense"],
            bm25_rank=entry["ranks"]["bm25"],  # None if the chunk shares no term with the query
            dense_score=dense_scores.get(entry["id"]),
            bm25_score=bm25_scores.get(entry["id"]),
        )
        results.append(result)
    return results


def search(question, index, method="dense", top_k=DEFAULT_TOP_K, rrf_k=DEFAULT_RRF_K,
           dense_weight=DEFAULT_DENSE_WEIGHT, bm25_weight=DEFAULT_BM25_WEIGHT):
    """Rank chunks for a question with the chosen method. Every result has rank, score and score_type.

    dense_weight / bm25_weight apply to hybrid only. The 0.5 / 0.5 default ranks
    exactly like M2's unweighted RRF, with every score halved.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")
    validate_method(method)
    top_k = validate_top_k(top_k)
    if method == "dense":
        return _dense_search(question, index, top_k)
    if method == "bm25":
        return rank_bm25(question, index["bm25"], index["chunks"], top_k)
    validate_weights({"dense": dense_weight, "bm25": bm25_weight})  # fail before any retrieval work
    dense, lexical = hybrid_candidates(question, index)
    return fuse_rankings(dense, lexical, index["chunks"], top_k, rrf_k, dense_weight, bm25_weight)


def format_result(result):
    score_type = result.get("score_type", "cosine")
    if score_type == "cosine":
        score_line = f"Similarity: {result['similarity']:.4f}"
    elif score_type == "bm25":
        score_line = f"BM25 score: {result['score']:.4f}"
    else:
        weights = result["weights"]
        bm25_rank = result["bm25_rank"] if result["bm25_rank"] is not None else "none (no term overlap)"
        score_line = (f"Weighted RRF score: {result['score']:.6f} "
                      f"(weights dense {weights['dense']:.2f} / BM25 {weights['bm25']:.2f}; "
                      f"dense rank {result['dense_rank']}, BM25 rank {bm25_rank})")
    return "\n".join([
        f"Rank: {result['rank']}",
        score_line,
        f"Document: {result['document_id']}",
        f"Title: {result['title']}",
        f"Department: {result['department']}",
        f"Section: {result['section_title']} ({result['chunk_id']})",
        "Text:",
        result["text"],
    ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

HEADERS = {
    "dense": "RAGForge — Dense Retrieval",
    "bm25": "RAGForge — BM25 Retrieval",
    "hybrid": "RAGForge — Hybrid Retrieval (dense + BM25, weighted Reciprocal Rank Fusion)",
}


def answer(question, index, method, top_k, weights=None):
    weights = weights or {}
    results = search(question, index, method, top_k, **weights)
    if not results:
        print("\nNo chunk shares a term with this question (BM25 found no matches).")
    for result in results:
        print()
        print(format_result(result))
    print()


def interactive_loop(index, method, top_k, weights=None):
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
        answer(question, index, method, top_k, weights)


def main(argv=None):
    parser = argparse.ArgumentParser(description="RAGForge retrieval: dense, BM25, or hybrid.")
    parser.add_argument("--method", choices=METHODS, default="dense", help="retrieval method (default: dense)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="number of chunks to return (default 5)")
    parser.add_argument("--question", action="append", help="answer this question and exit (repeatable)")
    parser.add_argument("--dense-weight", type=float, default=None,
                        help=f"hybrid only: dense weight in weighted RRF (default {DEFAULT_DENSE_WEIGHT})")
    parser.add_argument("--bm25-weight", type=float, default=None,
                        help=f"hybrid only: BM25 weight in weighted RRF (default {DEFAULT_BM25_WEIGHT})")
    parser.add_argument("--evaluate", action="store_true",
                        help="run the development-split benchmark for --method (same as: python -m src.evaluate)")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Sentence Transformers model name")
    args = parser.parse_args(argv)
    try:
        top_k = validate_top_k(args.top_k)
    except ValueError as error:
        parser.error(str(error))

    weights = None
    if args.dense_weight is not None or args.bm25_weight is not None:
        if args.method != "hybrid":
            parser.error("--dense-weight and --bm25-weight apply only to --method hybrid.")
        if args.evaluate:
            parser.error("--evaluate uses the default weights; use 'python -m src.evaluate --compare' "
                         "for the weighted configurations.")
        weights = {
            "dense_weight": DEFAULT_DENSE_WEIGHT if args.dense_weight is None else args.dense_weight,
            "bm25_weight": DEFAULT_BM25_WEIGHT if args.bm25_weight is None else args.bm25_weight,
        }
        try:
            validate_weights({"dense": weights["dense_weight"], "bm25": weights["bm25_weight"]})
        except ValueError as error:
            parser.error(str(error))

    if args.evaluate:
        from src.evaluate import main as evaluate_main

        evaluate_main(["--split", "development", "--method", args.method])
        return

    print(HEADERS[args.method])
    index = build_search_index(methods=(args.method,), model_name=args.model)
    print(f"Loaded {len(index['documents'])} documents.")
    print(f"Created {len(index['chunks'])} chunks.")
    if index["chunk_embeddings"] is not None:
        print(f"Embedding model: {args.model}")
        print(f"Embedding dimension: {index['chunk_embeddings'].shape[1]}.")
    if index["bm25"] is not None:
        print(f"BM25 vocabulary: {len(index['bm25']['idf'])} terms.")

    if args.question:
        for question in args.question:
            print(f"\n=== Question: {question}")
            answer(question, index, args.method, top_k, weights)
    else:
        print()
        interactive_loop(index, args.method, top_k, weights)


if __name__ == "__main__":
    main()
