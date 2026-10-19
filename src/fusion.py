"""Reciprocal Rank Fusion (RRF): combine several rankings using ranks only.

    RRF(d) = sum over rankings r that contain d of  1 / (k + rank_r(d))

rank_r(d) is 1-based, and a ranking that does not contain d contributes
nothing. The constant k (default 60, from Cormack et al., 2009) flattens the
curve so that rank 1 vs rank 2 matters, but not overwhelmingly: with k = 60,
rank 1 scores 1/61 ≈ 0.0164 and rank 10 scores 1/70 ≈ 0.0143.

Weighted RRF (M2.5) multiplies each ranking's contribution by a weight:

    WeightedRRF(d) = sum over rankings r that contain d of  w_r / (k + rank_r(d))

The weights are non-negative and sum to 1. Equal weights rescale every score
by the same constant, so they give exactly the same ranking as unweighted RRF.

Why ranks and not scores? Cosine similarities (about -1..1, here mostly
0.2-0.7) and BM25 scores (0 to about 30, depending on the query) are on
unrelated scales. Adding them would let whichever scale is larger dominate,
and the balance would change from query to query. Ranks are comparable across
retrievers without any calibration. The price is that RRF cannot see how
confident a retriever was: a BM25 score of 6.9 against 3.0 for the runner-up
becomes just "rank 1 vs rank 2".
"""

import math

DEFAULT_RRF_K = 60
WEIGHT_TOLERANCE = 1e-9


def validate_weights(weights, names=None):
    """Check retriever weights: finite, in [0, 1], summing to 1, not all zero.

    `names`, if given, are the retriever names that must each have exactly one weight.
    Returns the weights as a dict of floats.
    """
    if not isinstance(weights, dict) or not weights:
        raise ValueError("weights must be a non-empty dict of retriever name -> weight.")
    if names is not None and set(weights) != set(names):
        raise ValueError(f"weights must be given for exactly {sorted(names)}, got {sorted(weights)}.")

    clean = {}
    for name, weight in weights.items():
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(f"Weight for '{name}' must be a number, got {weight!r}.")
        weight = float(weight)
        if not math.isfinite(weight):
            raise ValueError(f"Weight for '{name}' must be finite, got {weight}.")
        if weight < 0:
            raise ValueError(f"Weight for '{name}' must be non-negative, got {weight}.")
        if weight > 1:
            raise ValueError(f"Weight for '{name}' must be at most 1, got {weight}.")
        clean[name] = weight

    total = sum(clean.values())
    if total == 0:
        raise ValueError("At least one weight must be positive.")
    if abs(total - 1.0) > WEIGHT_TOLERANCE:
        raise ValueError(f"Weights must sum to 1, got {total:.6g} ({clean}).")
    return clean


def reciprocal_rank_fusion(rankings, k=DEFAULT_RRF_K, weights=None):
    """Fuse ranked ID lists, optionally weighting each ranking.

    Args:
        rankings: dict mapping a retriever name to its ranked list of IDs (best first).
        k: smoothing constant (> 0).
        weights: None for unweighted RRF (every ranking counts 1, the M2 behavior),
            or a dict with one weight per ranking, validated by validate_weights().

    Returns:
        A list of dicts sorted best first:
            {"id", "rrf_score", "ranks": {retriever name: 1-based rank or None}, "best_rank"}

    Ties on the score are broken by the best (smallest) rank in any list, then
    by ID string, so the output does not depend on dict or list order. Within
    one list only the first occurrence of an ID counts; later duplicates are
    ignored. Every ID found in any list is returned, including IDs whose only
    ranking has weight 0 (they score 0).
    """
    if not isinstance(k, (int, float)) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"RRF constant k must be a positive number, got {k!r}.")
    if not rankings:
        raise ValueError("Need at least one ranking to fuse.")
    if weights is None:
        weights = {name: 1.0 for name in rankings}
    else:
        weights = validate_weights(weights, names=rankings)

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
        score = sum(weights[name] / (k + item_ranks[name]) for name in sorted(item_ranks))
        fused.append({
            "id": item,
            "rrf_score": score,
            "ranks": {name: item_ranks.get(name) for name in rankings},
            "best_rank": min(item_ranks.values()),
        })
    fused.sort(key=lambda entry: (-entry["rrf_score"], entry["best_rank"], str(entry["id"])))
    return fused
