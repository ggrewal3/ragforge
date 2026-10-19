"""Second-stage reranking with a pretrained cross-encoder.

Bi-encoder (the dense retriever, all-MiniLM-L6-v2):
    question -> vector, chunk -> vector, independently; relevance = cosine.
    Chunk vectors are computed once in advance, so each query costs one
    forward pass plus a dot product with every chunk. Fast, but the model
    never sees the question and the chunk together.

Cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2):
    "[CLS] question [SEP] chunk [SEP]" -> transformer -> one relevance score.
    Every attention layer lets question tokens attend to chunk tokens, so it
    can judge, for example, whether a passage about refund *processing*
    actually answers a question about refund *deadlines*. Nothing can be
    precomputed: each question-chunk pair needs its own forward pass, which is
    why it reranks a short candidate list instead of searching all chunks.

The scores are raw logits from a model trained on web search (MS MARCO)
relevance. They are unbounded, comparable only within one question, and not
calibrated probabilities. They measure topical relevance, not whether a
passage is true or entails an answer.

A reranker can only reorder what it is given: a relevant chunk that is not
among the candidates cannot be recovered.
"""

import numpy as np

from src.embeddings import chunk_embedding_text

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CANDIDATES = 20
DEFAULT_RERANK_TOP_K = 5
DEFAULT_BATCH_SIZE = 32


def load_cross_encoder(model_name=DEFAULT_RERANK_MODEL):
    """Load the cross-encoder (downloaded to the Hugging Face cache on first use).

    The activation is fixed to identity so scores are raw logits regardless of
    library defaults.
    """
    import torch
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, activation_fn=torch.nn.Identity())


def candidate_text(chunk):
    """Text the cross-encoder sees for a chunk.

    It is the same representation the dense retriever embeds (document title,
    section title, section ID, chunk text), used for dense and hybrid
    candidates alike. Retrieval scores, ranks and labels are never included.
    """
    return chunk_embedding_text(chunk)


def build_pairs(question, candidates):
    """(question, chunk text) pairs, one per candidate, in candidate order."""
    return [(question, candidate_text(candidate)) for candidate in candidates]


def _validate_positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def score_pairs(model, pairs, batch_size=DEFAULT_BATCH_SIZE):
    """Run the cross-encoder on all pairs in batches; return a 1-D float array of finite scores."""
    raw = model.predict(pairs, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    scores = np.asarray(raw, dtype=np.float64)
    if scores.ndim == 2 and scores.shape[1] == 1:
        scores = scores[:, 0]  # some models return one column per label
    if scores.shape != (len(pairs),):
        raise ValueError(f"Expected {len(pairs)} cross-encoder scores, got output of shape {scores.shape}.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Cross-encoder returned NaN or infinite scores.")
    return scores


def rerank(question, candidates, model, top_k=DEFAULT_RERANK_TOP_K, batch_size=DEFAULT_BATCH_SIZE):
    """Rerank candidate chunks by cross-encoder score and return the best top_k.

    `candidates` is a ranked result list from a first-stage retriever; its
    order defines each candidate's original retrieval rank. Results are new
    dicts (the inputs are not modified) with:
        rank                  final rank after reranking (1-based)
        score, score_type     cross-encoder logit, "cross-encoder"
        retrieval_rank        rank in the first-stage list (1-based)
        retrieval_score       first-stage score, and retrieval_score_type
    plus all chunk metadata and any first-stage diagnostics (such as dense_rank
    and bm25_rank for hybrid). Ties are broken by retrieval rank, then chunk ID.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")
    top_k = _validate_positive_int(top_k, "top_k")
    _validate_positive_int(batch_size, "batch_size")
    if not candidates:
        return []
    chunk_ids = [candidate["chunk_id"] for candidate in candidates]
    if len(set(chunk_ids)) != len(chunk_ids):
        duplicates = sorted({c for c in chunk_ids if chunk_ids.count(c) > 1})
        raise ValueError(f"Duplicate chunk IDs among rerank candidates: {duplicates}.")

    scores = score_pairs(model, build_pairs(question, candidates), batch_size)

    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i, chunk_ids[i]))
    results = []
    for final_rank, i in enumerate(order[:top_k], start=1):
        candidate = candidates[i]
        result = dict(candidate)
        result.update(
            rank=final_rank,
            score=float(scores[i]),
            score_type="cross-encoder",
            retrieval_rank=i + 1,
            retrieval_score=candidate.get("score"),
            retrieval_score_type=candidate.get("score_type"),
        )
        results.append(result)
    return results


def check_candidate_settings(candidate_count, top_k):
    """Validate candidate count and final top_k for a retrieve-then-rerank pipeline."""
    candidate_count = _validate_positive_int(candidate_count, "candidate_count")
    top_k = _validate_positive_int(top_k, "top_k")
    if top_k > candidate_count:
        raise ValueError(f"top_k ({top_k}) cannot exceed the number of rerank candidates ({candidate_count}).")
    return candidate_count, top_k
