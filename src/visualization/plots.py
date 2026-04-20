"""
src/visualization/plots.py
===========================
Shared plotting utilities used by experiment scripts.

All figures use the same design language as the paper outputs:
  - Steel blue (#1565C0) for URD channel features (D, U, S)
  - Dark gray  (#546E7A) for standard score-shape features
  - Value labels on every bar  (f"{v:.3f}")
  - No top/right spines
  - #F9F9F9 axes background
  - Consistent grid, fontsize, and legend style
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional


# ── shared palette (must match PAL in 05_generate_paper_outputs.py) ──────────
_PAL = {
    "blue":   "#1565C0",
    "orange": "#E65100",
    "green":  "#2E7D32",
    "red":    "#C62828",
    "purple": "#6A1B9A",
    "gray":   "#546E7A",
    "teal":   "#00695C",
    "amber":  "#F57F17",
}

# URD-derived feature names → coloured steel blue in importance charts
_URD_FEATS = {
    "deviation_at_peak",
    "uncertainty_at_peak",
    "stationarity_at_peak",
    "uncertainty_slope",
    "stationarity_max",
    "du_ratio",
    "signed_deviation_mean",
}


def _style():
    """Apply the paper's shared rcParams — call at the top of every function."""
    plt.rcParams.update({
        "figure.facecolor":    "white",
        "axes.facecolor":      "#F9F9F9",
        "savefig.facecolor":   "white",
        "font.family":         "DejaVu Sans",
        "font.size":           11,
        "axes.labelsize":      12,
        "axes.titlesize":      13,
        "xtick.labelsize":     10,
        "ytick.labelsize":     10,
        "legend.fontsize":     9,
        "legend.framealpha":   0.92,
        "lines.linewidth":     1.8,
        "axes.linewidth":      0.8,
        "axes.grid":           True,
        "grid.alpha":          0.22,
        "grid.linewidth":      0.55,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
    })


def plot_feature_importance(
    importances: Dict[str, float],
    title: str = "Feature Importance",
    save_path: Optional[str] = None,
):
    """
    Horizontal bar chart of feature importances.

    Design matches fig7_feature_importance / fig10_xgboost_importance:
      - URD channel features (D, U, S)  -> steel blue  #1565C0
      - Standard score-shape features   -> dark gray   #546E7A
      - Value label f"{v:.3f}" on every bar
      - Bold title, no top/right spines, #F9F9F9 axes background
      - dpi=200, pad_inches=0.15
    """
    _style()
    items  = sorted(importances.items(), key=lambda kv: kv[1])
    names  = [k for k, _ in items]
    vals   = [v for _, v in items]
    colors = [_PAL["blue"] if n in _URD_FEATS else _PAL["gray"] for n in names]

    fig, ax = plt.subplots(figsize=(10.5, max(5.5, len(names) * 0.52)))
    bars = ax.barh(range(len(names)), vals, color=colors, edgecolor="white", height=0.72)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10.5)
    ax.set_xlabel("Feature Importance (Gini impurity reduction)", fontweight="bold", fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=12)

    mx = max(vals) if vals else 1.0
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_width() + mx * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}",
            va="center", fontsize=8.5, color="#37474F",
        )

    patches = [
        mpatches.Patch(color=_PAL["blue"], label="URD channel features (D, U, S)"),
        mpatches.Patch(color=_PAL["gray"], label="Standard score-shape features"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=10)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
    else:
        plt.show()


def plot_confusion_matrix_3way(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
):
    _style()
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(5, n * 1.1), max(4, n * 0.9)))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontweight="bold", fontsize=11)
    ax.set_ylabel("True",      fontweight="bold", fontsize=11)
    ax.set_title(title,        fontweight="bold", fontsize=12)
    for i in range(n):
        for j in range(n):
            clr = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=clr, fontsize=10, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
    else:
        plt.show()


def plot_training_curves(
    history: Dict[str, list],
    title: str = "Training and Validation Loss",
    save_path: Optional[str] = None,
):
    _style()
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], color=_PAL["blue"],   lw=1.6, label="Train Loss")
    ax.plot(epochs, history["val_loss"],   color=_PAL["orange"], lw=1.6, label="Val Loss")
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Loss",  fontweight="bold")
    ax.set_title(title,    fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
    else:
        plt.show()


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
    _style()
    n_sensors = min(n_sensors, len(sensor_names))
    fig, axes = plt.subplots(n_sensors, 1, figsize=(12, 3 * n_sensors), sharex=True)
    if n_sensors == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(cycles, true_values[:, i], color=_PAL["red"],  lw=0.9, alpha=0.7, label="True")
        ax.plot(cycles, mu[:, i],           color=_PAL["blue"], lw=1.2,            label="Pred mu")
        ax.fill_between(cycles,
                        mu[:, i] - 2 * sigma[:, i],
                        mu[:, i] + 2 * sigma[:, i],
                        alpha=0.18, color=_PAL["blue"], label="mu +/- 2sigma")
        ax.set_ylabel(sensor_names[i], fontweight="bold")
        if i == 0:
            ax.legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("Cycle", fontweight="bold")
    fig.suptitle(f"Probabilistic Prediction Bands - Engine {engine_id}",
                 fontweight="bold", fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
    else:
        plt.show()


def plot_score_vs_life_fraction_aggregated(
    engine_scores: Dict[int, dict],
    n_buckets: int = 20,
    title: str = "Mean Anomaly Score vs Life Fraction",
    save_path: Optional[str] = None,
):
    _style()
    all_lf, all_sc = [], []
    for data in engine_scores.values():
        all_lf.extend(data["life_fracs"].tolist())
        all_sc.extend(data["scores"].tolist())
    all_lf  = np.array(all_lf)
    all_sc  = np.array(all_sc)
    edges   = np.linspace(0, 1, n_buckets + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    means, q25s, q75s = [], [], []
    for i in range(n_buckets):
        mask = (all_lf >= edges[i]) & (all_lf < edges[i + 1])
        if mask.sum() > 0:
            means.append(np.mean(all_sc[mask]))
            q25s.append(np.percentile(all_sc[mask], 25))
            q75s.append(np.percentile(all_sc[mask], 75))
        else:
            means.append(np.nan); q25s.append(np.nan); q75s.append(np.nan)
    means = np.array(means); q25s = np.array(q25s); q75s = np.array(q75s)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(centers, means, "o-", color=_PAL["blue"], lw=1.6, label="Mean Score")
    ax.fill_between(centers, q25s, q75s, alpha=0.2, color=_PAL["blue"], label="IQR")
    ax.set_xlabel("Life Fraction",              fontweight="bold")
    ax.set_ylabel("Anomaly Score (normalised)", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
    else:
        plt.show()


def plot_roc_pr_curves(
    results: Dict[str, Dict],
    title_prefix: str = "",
    save_path: Optional[str] = None,
):
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    method_colors = [
        _PAL["blue"], _PAL["orange"], _PAL["green"],
        _PAL["red"],  _PAL["purple"], _PAL["gray"], _PAL["teal"],
    ]
    for idx, (name, data) in enumerate(results.items()):
        col = method_colors[idx % len(method_colors)]
        fpr, tpr = data.get("roc", ([], []))
        if len(fpr) > 0:
            ax1.plot(fpr, tpr, color=col, lw=1.6,
                     label=f"{name} ({data.get('roc_auc', 0):.3f})")
        prec, rec = data.get("pr", ([], []))
        if len(prec) > 0:
            ax2.plot(rec, prec, color=col, lw=1.6,
                     label=f"{name} ({data.get('pr_auc', 0):.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.25)
    ax1.set_xlabel("False Positive Rate", fontweight="bold")
    ax1.set_ylabel("True Positive Rate",  fontweight="bold")
    ax1.set_title(f"{title_prefix}ROC Curve", fontweight="bold")
    ax1.legend(fontsize=8, loc="lower right")
    ax2.set_xlabel("Recall",    fontweight="bold")
    ax2.set_ylabel("Precision", fontweight="bold")
    ax2.set_title(f"{title_prefix}Precision-Recall", fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
    else:
        plt.show()