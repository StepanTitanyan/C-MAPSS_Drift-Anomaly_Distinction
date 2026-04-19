"""
src/anomaly/scoring.py
======================
Convert model predictions into anomaly scores.

The anomaly score quantifies how "surprising" an observation is given the
model's learned normal behavior. Higher score = more anomalous.

Score types:
1. NLL (Gaussian Negative Log-Likelihood) — PRIMARY
   Uses both prediction error AND predicted uncertainty.
   A deviation in a high-confidence region scores higher than the same
   deviation in a low-confidence region.

2. MSE — for deterministic model baselines.

3. MAE — alternative to MSE, more robust to outliers.

4. Mahalanobis — accounts for cross-sensor correlation in residuals.
   If sensors 3 and 7 always deviate together under normal conditions,
   a joint deviation is less alarming than an uncorrelated one.
"""

import numpy as np
import math
from typing import Optional, Tuple


def compute_nll_scores(targets: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Compute per-sample Gaussian NLL anomaly scores.

    NLL_j = 0.5 * [log(σ_j²) + (x_j - μ_j)² / σ_j² + log(2π)]

    Total score = sum over sensors.

    Parameters
    ----------
    targets : (N, d) — true sensor values
    mu : (N, d) — predicted means
    sigma : (N, d) — predicted standard deviations

    Returns
    -------
    scores : (N,) — total NLL per sample (sum over sensors)
    per_sensor_scores : (N, d) — NLL per sensor per sample
    """
    variance = sigma ** 2
    per_sensor = 0.5 * (np.log(variance) + (targets - mu) ** 2 / variance + np.log(2 * np.pi))
    scores = per_sensor.sum(axis=1)
    return scores, per_sensor


def compute_mse_scores(targets: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    Compute per-sample MSE anomaly scores.

    Parameters
    ----------
    targets : (N, d)
    mu : (N, d)

    Returns
    -------
    scores : (N,) — mean squared error per sample (averaged over sensors)
    per_sensor_scores : (N, d)
    """
    per_sensor = (targets - mu) ** 2
    scores = per_sensor.mean(axis=1)
    return scores, per_sensor


def compute_mae_scores(targets: np.ndarray,mu: np.ndarray) -> np.ndarray:
    """
    Compute per-sample MAE anomaly scores.

    Parameters
    ----------
    targets : (N, d)
    mu : (N, d)

    Returns
    -------
    scores : (N,) — mean absolute error per sample
    per_sensor_scores : (N, d)
    """
    per_sensor = np.abs(targets - mu)
    scores = per_sensor.mean(axis=1)
    return scores, per_sensor


def compute_mahalanobis_scores(targets: np.ndarray, mu: np.ndarray, cov_matrix: np.ndarray,) -> np.ndarray:
    """
    Compute Mahalanobis distance scores.

    Accounts for cross-sensor correlations in residuals.
    If residuals r = x - μ are normally distributed with covariance Σ,
    then r^T Σ^{-1} r follows a chi-squared distribution.

    Parameters
    ----------
    targets : (N, d)
    mu : (N, d)
    cov_matrix : (d, d) — empirical covariance of residuals on normal data

    Returns
    -------
    scores : (N,) — Mahalanobis distance per sample
    """
    residuals = targets - mu  # (N, d)
    cov_inv = np.linalg.pinv(cov_matrix)  #pseudo-inverse for robustness

    #Mahalanobis: r^T Σ^{-1} r for each sample
    #Vectorized: (N, d) @ (d, d) → (N, d), then element-wise * (N, d), sum over d
    scores = np.sum(residuals @ cov_inv * residuals, axis=1)
    return scores


def estimate_normal_covariance(targets: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    Estimate the covariance matrix of residuals on normal data.

    This should be computed on the validation set (normal data only)
    and then used for Mahalanobis scoring on test data.

    Parameters
    ----------
    targets : (N, d) — true values from normal validation set
    mu : (N, d) — predicted means on normal validation set

    Returns
    -------
    cov : (d, d) — empirical covariance matrix
    """
    residuals = targets - mu
    cov = np.cov(residuals, rowvar=False)  # (d, d)
    return cov


class AnomalyScorer:
    """
    High-level anomaly scorer that wraps score computation and normalization.

    Usage:
        scorer = AnomalyScorer(score_type="nll")
        scorer.fit_normalization(val_targets, val_mu, val_sigma)
        scores = scorer.score(test_targets, test_mu, test_sigma)

    After fit_normalization, scores are z-normalized so that:
        - Normal data has mean ≈ 0 and std ≈ 1
        - Anomalies have high positive scores

    Parameters
    ----------
    score_type : str
        "nll", "mse", "mae", or "mahalanobis"
    """

    def __init__(self, score_type: str = "nll"):
        self.score_type = score_type
        self.normal_mean = None
        self.normal_std = None
        self.cov_matrix = None  #For Mahalanobis
        self.is_fitted = False

    def _raw_score(self, targets: np.ndarray, mu: np.ndarray, sigma: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Compute raw (un-normalized) scores."""
        if self.score_type == "nll":
            assert sigma is not None, "NLL scoring requires sigma"
            return compute_nll_scores(targets, mu, sigma)
        elif self.score_type == "mse":
            return compute_mse_scores(targets, mu)
        elif self.score_type == "mae":
            return compute_mae_scores(targets, mu)
        elif self.score_type == "mahalanobis":
            assert self.cov_matrix is not None, \
                "Mahalanobis requires fit_normalization first"
            scores = compute_mahalanobis_scores(targets, mu, self.cov_matrix)
            return scores, None
        else:
            raise ValueError(f"Unknown score type: {self.score_type}")

    def fit_normalization(self, targets: np.ndarray, mu: np.ndarray, sigma: Optional[np.ndarray] = None):
        """
        Compute normalization statistics from normal validation data.

        After this, score() returns z-normalized scores.

        Parameters
        ----------
        targets : (N, d) — normal validation targets
        mu : (N, d) — predicted means on normal validation data
        sigma : (N, d) — predicted sigmas (needed for NLL scoring)
        """
        #For Mahalanobis, also estimate covariance
        if self.score_type == "mahalanobis":
            self.cov_matrix = estimate_normal_covariance(targets, mu)

        raw_scores, _ = self._raw_score(targets, mu, sigma)
        self.normal_mean = np.mean(raw_scores)
        self.normal_std = np.std(raw_scores)
        if self.normal_std < 1e-8:
            self.normal_std = 1.0  # Prevent division by zero
        self.is_fitted = True

    def score(self, targets: np.ndarray, mu: np.ndarray, sigma: Optional[np.ndarray] = None, normalize: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Compute anomaly scores.

        Parameters
        ----------
        targets : (N, d)
        mu : (N, d)
        sigma : (N, d) or None
        normalize : bool
            If True, z-normalize using normal validation statistics.

        Returns
        -------
        scores : (N,) — anomaly scores (higher = more anomalous)
        per_sensor : (N, d) or None — per-sensor breakdown
        """
        raw_scores, per_sensor = self._raw_score(targets, mu, sigma)

        if normalize and self.is_fitted:
            scores = (raw_scores - self.normal_mean) / self.normal_std
        else:
            scores = raw_scores

        return scores, per_sensor

    def compute_thresholds(self, targets: np.ndarray, mu: np.ndarray, sigma: Optional[np.ndarray] = None, percentiles: list = [95.0, 97.5, 99.0]) -> dict:
        """
        Compute anomaly thresholds from normal validation data.

        Parameters
        ----------
        targets, mu, sigma : normal validation data
        percentiles : list of float
            Which percentiles to compute.

        Returns
        -------
        dict mapping percentile → threshold value
        """
        scores, _ = self.score(targets, mu, sigma, normalize=True)
        thresholds = {}
        for p in percentiles:
            thresholds[p] = float(np.percentile(scores, p))
        return thresholds
