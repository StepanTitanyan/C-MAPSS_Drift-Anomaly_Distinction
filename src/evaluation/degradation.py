"""
src/evaluation/degradation.py
=============================
Analyze whether anomaly scores naturally rise as engines degrade.

This is the critical sanity check BEFORE synthetic anomaly evaluation:
if the model can't detect the natural degradation signal in FD001,
it won't reliably detect injected anomalies either.

Analyses:
1. Per-engine Spearman correlation: score vs life_fraction
2. Bucketed score comparison: early vs middle vs late life
3. Kruskal-Wallis test: are the buckets statistically different?
4. Uncertainty behavior: does predicted σ increase with degradation?
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple


def per_engine_score_correlation(engine_scores: Dict[int, dict]) -> Dict[str, float]:
    """
    Compute Spearman correlation between anomaly score and life_fraction
    for each engine, then aggregate.

    Parameters
    ----------
    engine_scores : dict
        Mapping engine_id → {"scores": (T,), "life_fracs": (T,)}

    Returns
    -------
    dict with:
      - "mean_spearman": average Spearman ρ across engines
      - "median_spearman": median
      - "std_spearman": standard deviation
      - "pct_positive": fraction of engines with positive correlation
      - "per_engine": dict of engine_id → (ρ, p-value)
    """
    correlations = {}

    for engine_id, data in engine_scores.items():
        scores = data["scores"]
        life_fracs = data["life_fracs"]

        if len(scores) < 5:
            continue

        rho, pval = stats.spearmanr(life_fracs, scores)
        correlations[engine_id] = (float(rho), float(pval))

    if not correlations:
        return {"mean_spearman": 0.0, "median_spearman": 0.0,
                "std_spearman": 0.0, "pct_positive": 0.0,
                "per_engine": {}}

    rhos = [v[0] for v in correlations.values()]

    return {
        "mean_spearman": float(np.mean(rhos)),
        "median_spearman": float(np.median(rhos)),
        "std_spearman": float(np.std(rhos)),
        "pct_positive": float(np.mean([r > 0 for r in rhos])),
        "per_engine": correlations}


def bucketed_score_analysis(engine_scores: Dict[int, dict], buckets: Dict[str, Tuple[float, float]] = None) -> Dict:
    """
    Compare anomaly scores across life-fraction buckets.

    Parameters
    ----------
    engine_scores : dict
        Mapping engine_id → {"scores": (T,), "life_fracs": (T,)}
    buckets : dict
        Mapping bucket name → (low, high) life_fraction range.
        Default: early=[0, 0.3), middle=[0.3, 0.7), late=[0.7, 1.0]

    Returns
    -------
    dict with per-bucket statistics and Kruskal-Wallis test results.
    """
    if buckets is None:
        buckets = {
            "early": (0.0, 0.3),
            "middle": (0.3, 0.7),
            "late": (0.7, 1.0)}

    #Collect scores into buckets
    bucket_scores = {name: [] for name in buckets}

    for engine_id, data in engine_scores.items():
        scores = data["scores"]
        life_fracs = data["life_fracs"]

        for name, (low, high) in buckets.items():
            mask = (life_fracs >= low) & (life_fracs < high)
            bucket_scores[name].extend(scores[mask].tolist())

    #Compute statistics per bucket
    bucket_stats = {}
    for name, scores_list in bucket_scores.items():
        arr = np.array(scores_list)
        if len(arr) == 0:
            bucket_stats[name] = {"mean": 0, "median": 0, "std": 0, "n": 0}
        else:
            bucket_stats[name] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "n": len(arr),
                "q25": float(np.percentile(arr, 25)),
                "q75": float(np.percentile(arr, 75))}

    #Kruskal-Wallis test: are the buckets statistically different?
    groups = [np.array(bucket_scores[name]) for name in buckets
              if len(bucket_scores[name]) > 0]

    kw_result = {}
    if len(groups) >= 2:
        stat, pval = stats.kruskal(*groups)
        kw_result = {"statistic": float(stat), "p_value": float(pval)}
    else:
        kw_result = {"statistic": float("nan"), "p_value": float("nan")}

    return {
        "bucket_stats": bucket_stats,
        "kruskal_wallis": kw_result}


def uncertainty_vs_degradation(engine_data: Dict[int, dict]) -> Dict[str, float]:
    """
    Analyze whether predicted σ increases with degradation.

    This is unique to the probabilistic approach: if the model becomes
    more uncertain as the engine degrades, that's evidence it "knows"
    it's seeing unfamiliar patterns.

    Parameters
    ----------
    engine_data : dict
        Mapping engine_id → {"sigmas": (T, d), "life_fracs": (T,)}
        where sigmas are the predicted standard deviations.

    Returns
    -------
    dict with mean Spearman correlation between mean_σ and life_fraction.
    """
    correlations = []

    for engine_id, data in engine_data.items():
        if "sigmas" not in data:
            continue

        sigmas = data["sigmas"]
        life_fracs = data["life_fracs"]

        if len(life_fracs) < 5:
            continue

        #Mean sigma across sensors at each step
        mean_sigma = np.mean(sigmas, axis=1) if sigmas.ndim > 1 else sigmas

        rho, pval = stats.spearmanr(life_fracs, mean_sigma)
        correlations.append(float(rho))

    if not correlations:
        return {"mean_sigma_life_corr": 0.0, "pct_positive": 0.0}

    return {
        "mean_sigma_life_corr": float(np.mean(correlations)),
        "median_sigma_life_corr": float(np.median(correlations)),
        "pct_positive": float(np.mean([r > 0 for r in correlations]))}


def full_degradation_report(engine_scores: Dict[int, dict], engine_sigmas: Dict[int, dict] = None) -> Dict:
    """
    Run all degradation analyses and return a comprehensive report.

    Parameters
    ----------
    engine_scores : per-engine score data
    engine_sigmas : per-engine sigma data (optional, for probabilistic models)

    Returns
    -------
    Complete degradation analysis report.
    """
    report = {
        "score_correlation": per_engine_score_correlation(engine_scores),
        "bucketed_analysis": bucketed_score_analysis(engine_scores)}

    if engine_sigmas is not None:
        report["uncertainty_analysis"] = uncertainty_vs_degradation(engine_sigmas)

    return report
