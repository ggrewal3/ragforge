"""Reciprocal Rank Fusion (RRF): combine several rankings using ranks only.

    RRF(d) = sum over rankings r that contain d of  1 / (k + rank_r(d))

rank_r(d) is 1-based, and a ranking that does not contain d contributes
nothing. The constant k (default 60, from Cormack et al., 2009) flattens the
curve so that rank 1 vs rank 2 matters, but not overwhelmingly: with k = 60,
rank 1 scores 1/61 ≈ 0.0164 and rank 10 scores 1/70 ≈ 0.0143.

Why ranks and not scores? Cosine similarities (about -1..1, here mostly
0.2-0.7) and BM25 scores (0 to about 30, depending on the query) are on
unrelated scales. Adding them would let whichever scale is larger dominate,
and the balance would change from query to query. Ranks are comparable across
retrievers without any calibration.
"""

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(rankings, k=DEFAULT_RRF_K):
    """Fuse ranked ID lists.

    Args:
        rankings: dict mapping a retriever name to its ranked list of IDs (best first).
        k: smoothing constant (> 0).

    Returns:
        A list of dicts sorted best first:
            {"id", "rrf_score", "ranks": {retriever name: 1-based rank or None}, "best_rank"}

    Ties on the RRF score are broken by the best (smallest) rank in any
    list, then by ID string, so the output does not depend on dict or list order.
    Within one list only the first occurrence of an ID counts; later duplicates
    are ignored.
    """
    if not isinstance(k, (int, float)) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"RRF constant k must be a positive number, got {k!r}.")
    if not rankings:
        raise ValueError("Need at least one ranking to fuse.")

    ranks = {}  # id -> {retriever: rank}
    for name, ranked_ids in rankings.items():
        rank = 0
        for item in ranked_ids:
            if name in ranks.get(item, {}):
                continue  # duplicate within this list; it does not advance the rank
            rank += 1
            ranks.setdefault(item, {})[name] = rank

    fused = []
    for item, item_ranks in ranks.items():
        # Sum in a fixed (sorted) retriever order so equal inputs give bit-identical floats.
        score = sum(1.0 / (k + item_ranks[name]) for name in sorted(item_ranks))
        fused.append({
            "id": item,
            "rrf_score": score,
            "ranks": {name: item_ranks.get(name) for name in rankings},
            "best_rank": min(item_ranks.values()),
        })
    fused.sort(key=lambda entry: (-entry["rrf_score"], entry["best_rank"], str(entry["id"])))
    return fused
