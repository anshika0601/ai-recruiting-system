"""
evaluation/metrics.py

Ranking metrics for comparing pipeline output to a human baseline.

Why these three metrics:
  - MAE (Mean Absolute Error): tells you "on average, how many rank
    positions is the pipeline off?" Robust and intuitive.
  - Top-K overlap: tells you "did the pipeline surface the same top
    candidates as the human?" This is what actually matters in screening.
  - Spearman correlation: tells you "is the overall ordering similar?"
    Handles ties gracefully, standard rank-correlation stat.

Strict rank-position matching (used previously) is discouraged — it
flips wildly on small candidate pools and hides real improvements.
"""
from typing import Dict, List


def evaluate_ranking(
    human_ranks:    List[int],
    pipeline_ranks: List[int],
    top_k:          int = 3,
) -> Dict[str, float]:
    """
    Compare a pipeline ranking to a human ranking.

    Args:
        human_ranks:    Rank assigned by human, index-aligned with pipeline_ranks.
                        e.g. [1, 2, 3, 4, 5, 6, 7, 8, 9] means candidate at
                        index 0 was ranked 1st by the human, etc.
        pipeline_ranks: Rank assigned by the pipeline for the SAME candidates
                        in the SAME order.
        top_k:          How many top candidates to compare for overlap metric.

    Returns:
        {
            "mae":            mean absolute rank error (lower is better; 0 = perfect),
            "top_k_overlap":  fraction of human's top-K also in pipeline's top-K,
            "spearman":       rank correlation (-1 to +1; higher is better),
            "exact_match":    fraction of candidates in the exact same rank,
        }

    Raises:
        ValueError if the two lists have different lengths.
    """
    if len(human_ranks) != len(pipeline_ranks):
        raise ValueError(
            f"Rank list length mismatch: "
            f"human={len(human_ranks)}, pipeline={len(pipeline_ranks)}"
        )

    n = len(human_ranks)
    if n == 0:
        return {"mae": 0.0, "top_k_overlap": 0.0, "spearman": 0.0, "exact_match": 0.0}

    # 1. Mean absolute rank error
    mae = sum(abs(h - p) for h, p in zip(human_ranks, pipeline_ranks)) / n

    # 2. Top-K overlap
    human_top    = {i for i, r in enumerate(human_ranks)    if r <= top_k}
    pipeline_top = {i for i, r in enumerate(pipeline_ranks) if r <= top_k}
    top_k_overlap = len(human_top & pipeline_top) / top_k

    # 3. Exact position match (the old strict metric — kept for comparison)
    exact_match = sum(1 for h, p in zip(human_ranks, pipeline_ranks) if h == p) / n

    # 4. Spearman rank correlation
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(human_ranks, pipeline_ranks)
        # spearmanr returns nan when input has no variance; treat as 0
        spearman = float(rho) if rho == rho else 0.0
    except ImportError:
        # Fallback if scipy not installed — compute Spearman manually
        spearman = _spearman_manual(human_ranks, pipeline_ranks)

    return {
        "mae":            round(mae, 3),
        "top_k_overlap":  round(top_k_overlap, 3),
        "spearman":       round(spearman, 3),
        "exact_match":    round(exact_match, 3),
    }


def _spearman_manual(x: List[int], y: List[int]) -> float:
    """
    Compute Spearman rank correlation without scipy.
    Uses the standard formula: 1 - (6 * sum(d²) / (n * (n² - 1)))
    where d is the difference in ranks. Assumes no ties.
    """
    n = len(x)
    if n < 2:
        return 0.0
    d_squared_sum = sum((xi - yi) ** 2 for xi, yi in zip(x, y))
    return 1 - (6 * d_squared_sum) / (n * (n ** 2 - 1))


def format_metrics(metrics: Dict[str, float]) -> str:
    """Pretty-print metrics for terminal output."""
    return (
        f"  MAE (rank error)       : {metrics['mae']:.2f}   "
            f"(0 = perfect, lower is better)\n"
        f"  Top-3 overlap          : {metrics['top_k_overlap']:.1%}  "
            f"(fraction of human's top-3 also in pipeline's top-3)\n"
        f"  Spearman correlation   : {metrics['spearman']:+.2f}  "
            f"(-1 to +1, higher is better)\n"
        f"  Exact-position match   : {metrics['exact_match']:.1%}  "
            f"(strict metric — noisy on small samples)"
    )