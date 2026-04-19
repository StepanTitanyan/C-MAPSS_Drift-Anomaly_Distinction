"""
src/evaluation/metrics.py
=========================
Evaluation metrics for anomaly detection.

Two levels of evaluation:

1. POINT-LEVEL: standard classification metrics at each time step.
   - Precision, Recall, F1 at a given threshold
   - ROC-AUC (threshold-independent)
   - PR-AUC (better than ROC for imbalanced data — anomalies are rare)

2. EVENT-LEVEL: more meaningful for time-series anomaly detection.
   - Event recall: what fraction of injected anomaly events are detected?
   - Detection delay: how many steps after onset until first detection?
   - Event precision: what fraction of detector alarms correspond to real events?

Event-level metrics are what reviewers value in time-series AD papers because:
  - A single anomaly event spanning 10 steps shouldn't count as 10 true positives.
  - Detecting an event 1 step late vs 8 steps late is very different operationally.
"""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    confusion_matrix)
from typing import Dict, List, Tuple, Optional


# =============================================================================
#Point-Level Metrics
# =============================================================================

def point_level_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    """
    Compute point-level detection metrics at a given threshold.

    Parameters
    ----------
    y_true : (N,) — binary ground truth (0=normal, 1=anomalous)
    scores : (N,) — anomaly scores (higher = more anomalous)
    threshold : float — score threshold for flagging

    Returns
    -------
    dict with precision, recall, f1, threshold, num_true_pos, num_false_pos, etc.
    """
    y_pred = (scores >= threshold).astype(int)

    #Handle edge cases where all predictions are same class
    n_pos = y_pred.sum()
    n_true = y_true.sum()

    metrics = {
        "threshold": threshold,
        "n_true_anomalies": int(n_true),
        "n_flagged": int(n_pos)}

    if n_pos == 0 or n_true == 0:
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["f1"] = 0.0
    else:
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    return metrics


def threshold_independent_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """
    Compute threshold-independent metrics (AUC-based).

    Parameters
    ----------
    y_true : (N,) — binary ground truth
    scores : (N,) — anomaly scores

    Returns
    -------
    dict with roc_auc, pr_auc
    """
    metrics = {}

    #Need both classes present
    if len(np.unique(y_true)) < 2:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
        return metrics

    metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
    metrics["pr_auc"] = float(average_precision_score(y_true, scores))

    return metrics


def compute_curves(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute ROC and PR curves for plotting.

    Returns
    -------
    dict with "roc": (fpr, tpr) and "pr": (precision, recall)
    """
    if len(np.unique(y_true)) < 2:
        return {"roc": (np.array([]), np.array([])),
                "pr": (np.array([]), np.array([]))}

    fpr, tpr, _ = roc_curve(y_true, scores)
    prec, rec, _ = precision_recall_curve(y_true, scores)

    return {
        "roc": (fpr, tpr),
        "pr": (prec, rec)}


# =============================================================================
#Event-Level Metrics
# =============================================================================

def _find_events(labels: np.ndarray, target_label: int = 1) -> List[Tuple[int, int]]:
    """
    Find contiguous blocks of a specific label.

    Returns list of (start_idx, end_idx) tuples.
    """
    events = []
    in_event = False
    start = 0

    for i in range(len(labels)):
        if labels[i] == target_label and not in_event:
            in_event = True
            start = i
        elif labels[i] != target_label and in_event:
            events.append((start, i))
            in_event = False

    if in_event:
        events.append((start, len(labels)))

    return events


def event_level_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float, tolerance: int = 0) -> Dict[str, float]:
    """
    Compute event-level detection metrics.

    An anomaly EVENT is a contiguous block of true anomaly labels.
    An event is "detected" if at least one point within it (or within
    'tolerance' steps after its start) is flagged.

    Parameters
    ----------
    y_true : (N,) — ground truth labels (0=normal, 1=anomaly)
    scores : (N,) — anomaly scores
    threshold : float — detection threshold
    tolerance : int — extra steps after event start to count as "detected"

    Returns
    -------
    dict with:
      - event_recall: fraction of true events detected
      - mean_detection_delay: average steps from event start to first detection
      - median_detection_delay: median detection delay
      - event_precision: fraction of detector alarm events that overlap a true event
      - n_true_events: number of ground truth events
      - n_detected_events: number detected
    """
    y_pred = (scores >= threshold).astype(int)

    #Find true events
    true_events = _find_events(y_true, target_label=1)
    n_true_events = len(true_events)

    if n_true_events == 0:
        return {
            "event_recall": float("nan"),
            "mean_detection_delay": float("nan"),
            "median_detection_delay": float("nan"),
            "event_precision": float("nan"),
            "n_true_events": 0,
            "n_detected_events": 0}

    #Check each true event for detection
    detected = 0
    delays = []

    for start, end in true_events:
        search_end = min(end + tolerance, len(y_pred))
        event_preds = y_pred[start:search_end]

        if event_preds.any():
            detected += 1
            #Detection delay = first flagged step - event start
            first_flag = np.argmax(event_preds)
            delays.append(first_flag)

    event_recall = detected / n_true_events

    mean_delay = float(np.mean(delays)) if delays else float("nan")
    median_delay = float(np.median(delays)) if delays else float("nan")

    #Event precision: fraction of predicted alarm blocks overlapping a true event
    pred_events = _find_events(y_pred, target_label=1)
    n_pred_events = len(pred_events)

    true_positive_alarms = 0
    for p_start, p_end in pred_events:
        # Check if this alarm overlaps any true event
        for t_start, t_end in true_events:
            if p_start < t_end and p_end > t_start:  # Overlap check
                true_positive_alarms += 1
                break

    event_precision = true_positive_alarms / n_pred_events if n_pred_events > 0 else 0.0

    return {
        "event_recall": float(event_recall),
        "mean_detection_delay": mean_delay,
        "median_detection_delay": median_delay,
        "event_precision": float(event_precision),
        "n_true_events": n_true_events,
        "n_detected_events": detected,
        "n_alarm_events": n_pred_events}


def false_positive_rate(scores: np.ndarray, threshold: float) -> float:
    """
    Compute false positive rate on known-normal data.

    Parameters
    ----------
    scores : (N,) — scores on data known to be normal
    threshold : float

    Returns
    -------
    FPR : fraction of normal points flagged as anomalous.
    """
    return float(np.mean(scores >= threshold))


# =============================================================================
#Full Evaluation Report
# =============================================================================

def full_evaluation(y_true: np.ndarray, scores: np.ndarray, thresholds: Dict[float, float], clean_scores: Optional[np.ndarray] = None,) -> Dict:
    """
    Run complete evaluation at multiple thresholds.

    Parameters
    ----------
    y_true : (N,) — ground truth
    scores : (N,) — anomaly scores
    thresholds : dict mapping percentile → threshold value
    clean_scores : (M,) or None — scores on known-clean data for FPR

    Returns
    -------
    Comprehensive evaluation dict.
    """
    results = {"threshold_independent": threshold_independent_metrics(y_true, scores), "per_threshold": {}}

    for pct, thr in thresholds.items():
        key = f"p{pct}"
        results["per_threshold"][key] = {
            "point_level": point_level_metrics(y_true, scores, thr),
            "event_level": event_level_metrics(y_true, scores, thr)}
        if clean_scores is not None:
            results["per_threshold"][key]["false_positive_rate"] = false_positive_rate(clean_scores, thr)

    return results
