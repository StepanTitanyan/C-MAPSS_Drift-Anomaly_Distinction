"""
src/visualization/plots.py
==========================
Publication-quality visualization functions.

Each function produces one clear figure making one point.
Designed for the paper, with consistent styling and proper labeling.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Optional, Tuple

# Use non-interactive backend for saving figures
matplotlib.use("Agg")


def set_style():
    """Set consistent plot style for all figures."""
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "lines.linewidth": 1.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def plot_training_curves(
    history: Dict[str, list],
    title: str = "Training and Validation Loss",
    save_path: Optional[str] = None,
):
    """Plot training and validation loss curves."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="#2196F3")
    ax.plot(epochs, history["val_loss"], label="Val Loss", color="#FF5722")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_prediction_bands(
    true_values: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    cycles: np.ndarray,
    sensor_names: List[str],
    engine_id: int,
    n_sensors: int = 4,
    save_path: Optional[str] = None,
):
    """
    Plot true values vs predicted μ ± 2σ bands for one engine.

    This is Figure 3 of the paper: shows the model's probabilistic
    predictions with uncertainty bands. Uncertainty should grow in late life.
    """
    set_style()
    n_sensors = min(n_sensors, len(sensor_names))
    fig, axes = plt.subplots(n_sensors, 1, figsize=(12, 3 * n_sensors), sharex=True)
    if n_sensors == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(cycles, true_values[:, i], "k-", alpha=0.7, label="True", linewidth=1)
        ax.plot(cycles, mu[:, i], color="#2196F3", label="Predicted μ", linewidth=1)

        lower = mu[:, i] - 2 * sigma[:, i]
        upper = mu[:, i] + 2 * sigma[:, i]
        ax.fill_between(cycles, lower, upper, alpha=0.25, color="#2196F3", label="μ ± 2σ")

        ax.set_ylabel(sensor_names[i])
        if i == 0:
            ax.legend(loc="upper left", fontsize=9)

    axes[-1].set_xlabel("Cycle")
    fig.suptitle(f"Probabilistic Prediction — Engine {engine_id}", fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_anomaly_score_trajectory(
    scores: np.ndarray,
    life_fracs: np.ndarray,
    threshold: float = None,
    labels: Optional[np.ndarray] = None,
    title: str = "Anomaly Score Trajectory",
    save_path: Optional[str] = None,
):
    """
    Plot anomaly score over engine life with optional threshold line and labels.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(life_fracs, scores, color="#2196F3", linewidth=1, alpha=0.8)

    if threshold is not None:
        ax.axhline(y=threshold, color="#FF5722", linestyle="--", label=f"Threshold = {threshold:.2f}")

    if labels is not None:
        anomaly_mask = labels == 1
        drift_mask = labels == 2
        if anomaly_mask.any():
            ax.scatter(life_fracs[anomaly_mask], scores[anomaly_mask],
                       c="red", s=15, zorder=5, label="Anomaly", alpha=0.7)
        if drift_mask.any():
            ax.scatter(life_fracs[drift_mask], scores[drift_mask],
                       c="orange", s=15, zorder=5, label="Drift", alpha=0.7)

    ax.set_xlabel("Life Fraction")
    ax.set_ylabel("Anomaly Score (normalized)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_score_vs_life_fraction_aggregated(
    engine_scores: Dict[int, dict],
    n_buckets: int = 20,
    title: str = "Mean Anomaly Score vs Life Fraction",
    save_path: Optional[str] = None,
):
    """
    Aggregated score vs life_fraction across all engines.

    This is Figure 4 of the paper: shows the natural rise of scores near failure.
    """
    set_style()

    # Collect all (life_frac, score) pairs
    all_lf = []
    all_scores = []
    for engine_id, data in engine_scores.items():
        all_lf.extend(data["life_fracs"].tolist())
        all_scores.extend(data["scores"].tolist())

    all_lf = np.array(all_lf)
    all_scores = np.array(all_scores)

    # Bucket and compute statistics
    bucket_edges = np.linspace(0, 1, n_buckets + 1)
    bucket_centers = (bucket_edges[:-1] + bucket_edges[1:]) / 2
    means = []
    stds = []
    q25s = []
    q75s = []

    for i in range(n_buckets):
        mask = (all_lf >= bucket_edges[i]) & (all_lf < bucket_edges[i + 1])
        if mask.sum() > 0:
            bucket_vals = all_scores[mask]
            means.append(np.mean(bucket_vals))
            stds.append(np.std(bucket_vals))
            q25s.append(np.percentile(bucket_vals, 25))
            q75s.append(np.percentile(bucket_vals, 75))
        else:
            means.append(np.nan)
            stds.append(0)
            q25s.append(np.nan)
            q75s.append(np.nan)

    means = np.array(means)
    q25s = np.array(q25s)
    q75s = np.array(q75s)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bucket_centers, means, "o-", color="#2196F3", label="Mean Score")
    ax.fill_between(bucket_centers, q25s, q75s, alpha=0.2, color="#2196F3", label="IQR")
    ax.set_xlabel("Life Fraction")
    ax.set_ylabel("Anomaly Score (normalized)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_roc_pr_curves(
    results: Dict[str, Dict],
    title_prefix: str = "",
    save_path: Optional[str] = None,
):
    """
    Plot ROC and PR curves for multiple methods side by side.

    Parameters
    ----------
    results : dict mapping method_name → {"roc": (fpr, tpr), "pr": (prec, rec), "roc_auc": float, "pr_auc": float}
    """
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#607D8B"]

    for idx, (name, data) in enumerate(results.items()):
        color = colors[idx % len(colors)]

        fpr, tpr = data["roc"]
        if len(fpr) > 0:
            auc_val = data.get("roc_auc", 0)
            ax1.plot(fpr, tpr, color=color, label=f"{name} (AUC={auc_val:.3f})")

        prec, rec = data["pr"]
        if len(prec) > 0:
            pr_auc = data.get("pr_auc", 0)
            ax2.plot(rec, prec, color=color, label=f"{name} (AUC={pr_auc:.3f})")

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title(f"{title_prefix}ROC Curve")
    ax1.legend(fontsize=9)

    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"{title_prefix}Precision-Recall Curve")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix_3way(
    cm: np.ndarray,
    class_names: List[str] = ["Normal", "Anomaly", "Drift"],
    title: str = "Three-Way Confusion Matrix",
    save_path: Optional[str] = None,
):
    """Plot a 3×3 confusion matrix for the full Normal/Anomaly/Drift classification."""
    set_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = f"{cm[i, j]}"
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=13)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_feature_importance(
    importances: Dict[str, float],
    title: str = "Feature Importance for Drift/Anomaly Classification",
    save_path: Optional[str] = None,
):
    """Bar chart of feature importances from the second-stage classifier."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(importances.keys())
    values = list(importances.values())

    # Sort by importance
    sorted_idx = np.argsort(values)
    names = [names[i] for i in sorted_idx]
    values = [values[i] for i in sorted_idx]

    # Color probabilistic features differently
    prob_features = {"mean_uncertainty", "uncertainty_change", "residual_autocorrelation"}
    colors = ["#FF9800" if n in prob_features else "#2196F3" for n in names]

    ax.barh(names, values, color=colors)
    ax.set_xlabel("Importance")
    ax.set_title(title)

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="Standard features"),
        Patch(facecolor="#FF9800", label="Probabilistic features"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()
