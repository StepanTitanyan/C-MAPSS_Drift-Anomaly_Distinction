"""
src/anomaly/smoothing.py
========================
Smooth anomaly scores to reduce false positives and reveal temporal patterns.

Raw anomaly scores can be noisy — a single high-noise observation may produce
a score spike that isn't meaningful. Smoothing helps in two ways:
1. Reduces false positives from random noise.
2. Reveals temporal structure: drift produces a gradually rising smoothed
   score, while a transient anomaly produces a brief spike.

This temporal signature is critical for the drift-vs-anomaly distinction.
"""

import numpy as np
from typing import Optional


def exponential_moving_average(scores: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """
    Apply exponential moving average (EMA) to a score sequence.

    EMA_t = α · S_t + (1 - α) · EMA_{t-1}

    Higher α = more weight on current score (less smoothing).
    Lower α = more weight on history (more smoothing).

    Parameters
    ----------
    scores : (T,) — raw anomaly scores in time order
    alpha : float in (0, 1) — smoothing factor

    Returns
    -------
    smoothed : (T,) — EMA-smoothed scores
    """
    smoothed = np.zeros_like(scores)
    smoothed[0] = scores[0]

    for t in range(1, len(scores)):
        smoothed[t] = alpha * scores[t] + (1 - alpha) * smoothed[t - 1]

    return smoothed


def moving_average(scores: np.ndarray, window: int = 5,) -> np.ndarray:
    """
    Simple moving average of scores.

    Parameters
    ----------
    scores : (T,)
    window : int — number of steps to average over

    Returns
    -------
    smoothed : (T,) — same length as input (front-padded with partial averages)
    """
    smoothed = np.zeros_like(scores)
    for t in range(len(scores)):
        start = max(0, t - window + 1)
        smoothed[t] = scores[start:t + 1].mean()
    return smoothed


def smooth_engine_scores(engine_scores: dict, method: str = "ema", alpha: float = 0.2, window: int = 5) -> dict:
    """
    Apply smoothing to per-engine score dictionaries.

    Parameters
    ----------
    engine_scores : dict
        Mapping engine_id → {"scores": (T,), "cycles": (T,), "life_fracs": (T,)}
    method : str
        "ema" or "ma" (moving average)
    alpha : float
        EMA parameter (used if method="ema")
    window : int
        Moving average window (used if method="ma")

    Returns
    -------
    dict with same structure, scores replaced by smoothed versions.
    """
    result = {}
    for engine_id, data in engine_scores.items():
        scores = data["scores"]

        if method == "ema":
            smoothed = exponential_moving_average(scores, alpha=alpha)
        elif method == "ma":
            smoothed = moving_average(scores, window=window)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

        result[engine_id] = {
            "scores": smoothed,
            "raw_scores": scores,  #Keep originals for analysis
            "cycles": data["cycles"],
            "life_fracs": data["life_fracs"]}
        #Preserve any extra keys
        for key in data:
            if key not in result[engine_id]:
                result[engine_id][key] = data[key]

    return result
