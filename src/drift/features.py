"""
src/drift/features.py
=====================
Extract features from anomaly score trajectories and residual patterns
to distinguish drift from anomaly.

The core insight: anomalies and drift produce residuals with DIFFERENT
temporal and spatial signatures:
  - Anomalies: sharp score spike, concentrated in few sensors, transient
  - Drift: gradual score rise, distributed across sensors, persistent

We extract 12 features per flagged event, computed over a sliding analysis
window around the flag:

 1. max_score        — peak anomaly score in window
 2. mean_score       — average score in window
 3. score_slope      — linear slope of score over window (drift → positive sustained)
 4. score_curvature  — mean |d²S/dt²| (anomaly → high curvature at spike)
 5. score_volatility — std of dS/dt (anomaly → high volatility)
 6. duration         — consecutive steps above threshold
 7. sensor_concentration — Gini coefficient of per-sensor |residual|
 8. num_sensors_flagged  — sensors with |residual| > 95th percentile
 9. max_single_sensor    — largest per-sensor |residual|
10. mean_uncertainty     — mean predicted σ in window (PROBABILISTIC ONLY)
11. uncertainty_change   — slope of mean σ (drift → σ may increase)
12. residual_autocorrelation — lag-1 autocorr of scores (drift → high)

Features 10-12 are UNIQUE to the probabilistic approach. If these improve
drift/anomaly separation, that's the paper's strongest argument for why
probabilistic modeling matters beyond just better point predictions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def _gini_coefficient(values: np.ndarray) -> float:
    """
    Compute Gini coefficient of an array.

    Gini = 0 means perfectly equal distribution across sensors.
    Gini → 1 means all residual is concentrated in one sensor.

    Anomalies (especially single-sensor) → high Gini.
    Drift (multi-sensor correlated) → lower Gini.
    """
    values = np.abs(values)
    if values.sum() < 1e-10:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_vals) / (n * np.sum(sorted_vals))) - (n + 1) / n)


def _linear_slope(y: np.ndarray) -> float:
    """Compute slope of a linear fit to array y."""
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=np.float64)
    #Simple least-squares slope
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom < 1e-10:
        return 0.0
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def _autocorrelation_lag1(y: np.ndarray) -> float:
    """Compute lag-1 autocorrelation of a sequence."""
    if len(y) < 3:
        return 0.0
    y_mean = y.mean()
    y_centered = y - y_mean
    var = np.sum(y_centered ** 2)
    if var < 1e-10:
        return 0.0
    cov = np.sum(y_centered[:-1] * y_centered[1:])
    return float(cov / var)


def extract_event_features(scores: np.ndarray, per_sensor_residuals: np.ndarray, predicted_sigmas: Optional[np.ndarray], event_center_idx: int, analysis_window: int = 15, threshold: float = 2.0, sensor_percentile_95: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Extract the 12 features for one flagged event.

    Parameters
    ----------
    scores : (T,)
        Anomaly scores over the engine trajectory.
    per_sensor_residuals : (T, d)
        Per-sensor absolute residuals |x - μ| over the trajectory.
    predicted_sigmas : (T, d) or None
        Predicted standard deviations from the probabilistic model.
        None if using a deterministic model (features 10-12 will be 0).
    event_center_idx : int
        Index of the flagged point (peak score or threshold crossing).
    analysis_window : int
        Number of steps around the event to analyze.
    threshold : float
        Score threshold for "elevated" (used to compute duration).
    sensor_percentile_95 : (d,) or None
        95th percentile of per-sensor residuals on normal data.
        Used to determine which sensors are "flagged".

    Returns
    -------
    dict mapping feature name → float value.
    """
    T = len(scores)
    d = per_sensor_residuals.shape[1] if per_sensor_residuals.ndim > 1 else 1
    half_w = analysis_window // 2

    #Window boundaries (clamped to valid range)
    w_start = max(0, event_center_idx - half_w)
    w_end = min(T, event_center_idx + half_w + 1)

    #Extract windowed data
    w_scores = scores[w_start:w_end]
    w_residuals = per_sensor_residuals[w_start:w_end]

    features = {}

    #--- Feature 1: max_score ---
    features["max_score"] = float(np.max(w_scores))

    #--- Feature 2: mean_score ---
    features["mean_score"] = float(np.mean(w_scores))

    #--- Feature 3: score_slope ---
    # Positive sustained slope → drift-like. Sharp spike → near-zero average slope.
    features["score_slope"] = _linear_slope(w_scores)

    #--- Feature 4: score_curvature ---
    # Second derivative: high at spike peaks, low for gradual changes.
    if len(w_scores) >= 3:
        d2 = np.diff(w_scores, n=2)
        features["score_curvature"] = float(np.mean(np.abs(d2)))
    else:
        features["score_curvature"] = 0.0

    #--- Feature 5: score_volatility ---
    #Std of first differences: anomalies have volatile score changes.
    if len(w_scores) >= 2:
        d1 = np.diff(w_scores)
        features["score_volatility"] = float(np.std(d1))
    else:
        features["score_volatility"] = 0.0

    #--- Feature 6: duration ---
    #Count consecutive steps above threshold around the event center.
    count = 0
    for idx in range(event_center_idx, T):
        if scores[idx] > threshold:
            count += 1
        else:
            break
    #Also check backwards
    for idx in range(event_center_idx - 1, -1, -1):
        if scores[idx] > threshold:
            count += 1
        else:
            break
    features["duration"] = float(count)

    #--- Feature 7: sensor_concentration (Gini) ---
    #At the event center, how concentrated are residuals across sensors?
    center_residuals = per_sensor_residuals[event_center_idx]
    features["sensor_concentration"] = _gini_coefficient(center_residuals)

    #--- Feature 8: num_sensors_flagged ---
    if sensor_percentile_95 is not None:
        flagged = np.sum(np.abs(center_residuals) > sensor_percentile_95)
    else:
        #Fallback: flag sensors with residual > 2.0 (since data is standardized)
        flagged = np.sum(np.abs(center_residuals) > 2.0)
    features["num_sensors_flagged"] = float(flagged)

    #--- Feature 9: max_single_sensor ---
    features["max_single_sensor"] = float(np.max(np.abs(center_residuals)))

    #--- Features 10-12: PROBABILISTIC FEATURES ---
    if predicted_sigmas is not None:
        w_sigmas = predicted_sigmas[w_start:w_end]

        #Feature 10: mean_uncertainty
        features["mean_uncertainty"] = float(np.mean(w_sigmas))

        #Feature 11: uncertainty_change (slope of mean sigma)
        mean_sigma_per_step = np.mean(w_sigmas, axis=1) if w_sigmas.ndim > 1 else w_sigmas
        features["uncertainty_change"] = _linear_slope(mean_sigma_per_step)

        #Feature 12: residual_autocorrelation
        features["residual_autocorrelation"] = _autocorrelation_lag1(w_scores)
    else:
        features["mean_uncertainty"] = 0.0
        features["uncertainty_change"] = 0.0
        features["residual_autocorrelation"] = _autocorrelation_lag1(w_scores)

    return features


def extract_features_for_trajectory(scores: np.ndarray, per_sensor_residuals: np.ndarray, predicted_sigmas: Optional[np.ndarray], labels: np.ndarray, threshold: float = 2.0, analysis_window: int = 15, sensor_percentile_95: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features for all flagged events in a trajectory.

    An "event" is defined as a contiguous block of non-zero labels.
    We extract features at the center (or peak score) of each event.

    Parameters
    ----------
    scores : (T,)
    per_sensor_residuals : (T, d)
    predicted_sigmas : (T, d) or None
    labels : (T,) — ground truth: 0=normal, 1=anomaly, 2=drift
    threshold : float
    analysis_window : int
    sensor_percentile_95 : (d,) or None

    Returns
    -------
    feature_matrix : (num_events, 12)
    event_labels : (num_events,) — 1 for anomaly events, 2 for drift events
    """
    feature_names = [
        "max_score", "mean_score", "score_slope", "score_curvature",
        "score_volatility", "duration", "sensor_concentration",
        "num_sensors_flagged", "max_single_sensor",
        "mean_uncertainty", "uncertainty_change", "residual_autocorrelation"]

    #Find contiguous blocks of non-zero labels
    events = []
    in_event = False
    event_start = 0
    event_label = 0

    for i in range(len(labels)):
        if labels[i] > 0 and not in_event:
            in_event = True
            event_start = i
            event_label = labels[i]
        elif (labels[i] == 0 or labels[i] != event_label) and in_event:
            events.append((event_start, i, event_label))
            in_event = False
            if labels[i] > 0:
                in_event = True
                event_start = i
                event_label = labels[i]

    if in_event:
        events.append((event_start, len(labels), event_label))

    if not events:
        return np.empty((0, 12)), np.empty((0,), dtype=np.int32)

    feature_list = []
    label_list = []

    for start, end, label in events:
        # Use the point with highest score as event center
        event_scores = scores[start:end]
        center_idx = start + np.argmax(event_scores)

        feats = extract_event_features(
            scores=scores,
            per_sensor_residuals=per_sensor_residuals,
            predicted_sigmas=predicted_sigmas,
            event_center_idx=center_idx,
            analysis_window=analysis_window,
            threshold=threshold,
            sensor_percentile_95=sensor_percentile_95)

        feature_list.append([feats[name] for name in feature_names])
        label_list.append(label)

    return np.array(feature_list), np.array(label_list, dtype=np.int32)


#For convenience: the canonical feature name list
FEATURE_NAMES = [
    "max_score", "mean_score", "score_slope", "score_curvature",
    "score_volatility", "duration", "sensor_concentration",
    "num_sensors_flagged", "max_single_sensor",
    "mean_uncertainty", "uncertainty_change", "residual_autocorrelation"]


def extract_urd_features_for_trajectory(scores: np.ndarray, per_sensor_residuals: np.ndarray, predicted_sigmas: Optional[np.ndarray], labels: np.ndarray, urd_result: Optional[Dict] = None, threshold: float = 2.0, analysis_window: int = 15, sensor_percentile_95: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the EXTENDED 16-feature set using URD v2 decomposition.

    Combines the 9 standard score-shape features with 7 URD-specific features
    that capture the three-channel anomaly signature.

    Returns
    -------
    feature_matrix : (num_events, 16)
    event_labels : (num_events,)
    """
    from src.anomaly.urd import extract_urd_features, URD_FEATURE_NAMES

    n_urd_features = len(URD_FEATURE_NAMES)  # 16

    #Find events
    events = []
    in_event = False
    event_start = 0
    event_label = 0

    for i in range(len(labels)):
        if labels[i] > 0 and not in_event:
            in_event = True
            event_start = i
            event_label = labels[i]
        elif (labels[i] == 0 or labels[i] != event_label) and in_event:
            events.append((event_start, i, event_label))
            in_event = False
            if labels[i] > 0:
                in_event = True
                event_start = i
                event_label = labels[i]

    if in_event:
        events.append((event_start, len(labels), event_label))

    if not events:
        return np.empty((0, n_urd_features)), np.empty((0,), dtype=np.int32)

    #Standard feature names (first 9)
    std_feature_names = [
        "max_score", "mean_score", "score_slope", "score_curvature",
        "score_volatility", "duration", "sensor_concentration",
        "num_sensors_flagged", "max_single_sensor"]

    feature_list = []
    label_list = []

    for start, end, label in events:
        event_scores = scores[start:end]
        center_idx = start + np.argmax(event_scores)

        #Extract standard 9 features
        std_feats = extract_event_features(
            scores=scores,
            per_sensor_residuals=per_sensor_residuals,
            predicted_sigmas=predicted_sigmas,
            event_center_idx=center_idx,
            analysis_window=analysis_window,
            threshold=threshold,
            sensor_percentile_95=sensor_percentile_95)

        row = [std_feats[name] for name in std_feature_names]

        #Extract 7 URD features if available
        if urd_result is not None:
            urd_feats = extract_urd_features(
                urd_result, center_idx, analysis_window,
            )
            row.extend([
                urd_feats["deviation_at_peak"],
                urd_feats["uncertainty_at_peak"],
                urd_feats["stationarity_at_peak"],
                urd_feats["uncertainty_slope"],
                urd_feats["stationarity_max"],
                urd_feats["du_ratio"],
                urd_feats["signed_deviation_mean"]])
        else:
            row.extend([0.0] * 7)

        feature_list.append(row)
        label_list.append(label)

    return np.array(feature_list), np.array(label_list, dtype=np.int32)

