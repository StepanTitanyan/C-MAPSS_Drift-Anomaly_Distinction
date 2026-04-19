"""
src/anomaly/urd.py — FINAL
============================
Uncertainty-Residual Decomposition (URD)

Three orthogonal channels for anomaly detection:

Channel 1: DEVIATION (D) — "How wrong is the prediction?"
    D_t = (1/d) Σ_j [(x_{t,j} - μ_{t,j})² / σ_{t,j}²]

Channel 2: UNCERTAINTY (U) — "How confident is the model?"
    U_t = (1/d) Σ_j [σ_{t,j} / σ_ref,j]

Channel 3: STATIONARITY (S) — "Has sensor variability collapsed?"
    Two-component additive approach (FDE + Run):
    
    Component A: First-Difference Energy (FDE)
        FDE(t,j,w) = (1/w) Σ (Δx_{i,j})²
        S_fde(t) = max_j [-log(FDE/FDE_ref + ε)]
    
    Component B: Run-length bonus (additive, not max)
        S(t) = S_fde(t) + γ · max(0, run_length(t) - 2)
    
    Why additive beats max: the run-length ADDS to FDE rather than
    competing with it. This means run-length alone cannot trigger
    false positives on normal data where variance occasionally dips.

Combined score: max(D_normalized, S_normalized)
"""

import numpy as np
from typing import Dict


class URDScorer:
    def __init__(self, fde_window: int = 5, run_delta: float = 1e-4, run_bonus: float = 2.0, run_threshold: int = 2, epsilon: float = 1e-10):
        self.fde_window = fde_window
        self.run_delta = run_delta
        self.run_bonus = run_bonus
        self.run_threshold = run_threshold
        self.epsilon = epsilon
        self.sigma_ref = None
        self.fde_ref = None
        self.deviation_mean = self.deviation_std = None
        self.stationarity_mean = self.stationarity_std = None
        self.is_fitted = False

    def fit(self, targets, mu, sigma):
        N, d = targets.shape
        self.sigma_ref = np.median(sigma, axis=0)
        diffs = np.diff(targets, axis=0)
        self.fde_ref = np.maximum(np.mean(diffs ** 2, axis=0), self.epsilon)
        dev = np.mean(((targets - mu) / sigma) ** 2, axis=1)
        self.deviation_mean, self.deviation_std = dev.mean(), max(dev.std(), 1e-8)
        stat = self._compute_stationarity(targets)
        valid = stat[~np.isnan(stat)]
        self.stationarity_mean = valid.mean() if len(valid) > 0 else 0.0
        self.stationarity_std = max(valid.std(), 1e-8) if len(valid) > 0 else 1.0
        self.is_fitted = True

    def _compute_fde_score(self, raw):
        T, d = raw.shape
        w = self.fde_window
        scores = np.full(T, np.nan)
        diffs = np.zeros_like(raw)
        diffs[1:] = raw[1:] - raw[:-1]
        sq = diffs ** 2
        for t in range(w, T):
            fde = np.mean(sq[t - w + 1:t + 1], axis=0)
            ratios = fde / self.fde_ref
            scores[t] = np.max(np.maximum(-np.log(ratios + self.epsilon), 0.0))
        return scores

    def _compute_run_length(self, raw):
        T, d = raw.shape
        runs = np.zeros((T, d))
        for t in range(1, T):
            for j in range(d):
                if abs(raw[t, j] - raw[t - 1, j]) < self.run_delta:
                    runs[t, j] = runs[t - 1, j] + 1.0
        return np.max(runs, axis=1)

    def _compute_stationarity(self, raw):
        T = len(raw)
        fde = self._compute_fde_score(raw)
        run_lengths = self._compute_run_length(raw)
        run_add = self.run_bonus * np.maximum(run_lengths - self.run_threshold, 0)
        combined = np.where(np.isnan(fde), run_add, fde + run_add)
        return combined

    def score(self, targets, mu, sigma, normalize=True):
        if not self.is_fitted:
            raise RuntimeError("URDScorer must be fit before scoring.")
        N, d = targets.shape
        norm_res = (targets - mu) / sigma

        deviation = np.mean(norm_res ** 2, axis=1)
        if normalize:
            deviation = (deviation - self.deviation_mean) / self.deviation_std

        uncertainty = np.mean(sigma / self.sigma_ref[np.newaxis, :], axis=1)

        fde_scores = self._compute_fde_score(targets)
        run_scores = self._compute_run_length(targets)
        stat_raw = self._compute_stationarity(targets)
        stationarity = stat_raw.copy()
        if normalize:
            stationarity = np.nan_to_num(
                (stationarity - self.stationarity_mean) / self.stationarity_std, nan=0.0)

        combined = np.maximum(deviation, stationarity)

        return {
            "deviation": deviation,
            "uncertainty": uncertainty,
            "stationarity": stationarity,
            "combined": combined,
            "norm_residuals": norm_res,
            "signed_residuals": norm_res,
            "signature": np.column_stack([deviation, uncertainty, stationarity]),
            "fde_scores": np.nan_to_num(fde_scores, nan=0.0),
            "run_scores": run_scores}

    def compute_thresholds(self, targets, mu, sigma, percentiles=None):
        if percentiles is None:
            percentiles = [95.0, 97.5, 99.0]
        result = self.score(targets, mu, sigma, normalize=True)
        thresholds = {}
        for ch in ["deviation", "uncertainty", "stationarity", "combined"]:
            valid = result[ch][~np.isnan(result[ch])]
            thresholds[ch] = {p: float(np.percentile(valid, p)) for p in percentiles}
        return thresholds


def extract_urd_features(urd_result, event_center_idx, analysis_window=15):
    D = urd_result["deviation"]
    U = urd_result["uncertainty"]
    S = urd_result["stationarity"]
    signed = urd_result["signed_residuals"]
    T = len(D)
    hw = analysis_window // 2
    ws, we = max(0, event_center_idx - hw), min(T, event_center_idx + hw + 1)

    features = {}
    features["deviation_at_peak"] = float(D[event_center_idx])
    features["uncertainty_at_peak"] = float(U[event_center_idx])
    features["stationarity_at_peak"] = float(S[event_center_idx])

    Uw = U[ws:we]
    if len(Uw) >= 2:
        x = np.arange(len(Uw), dtype=np.float64)
        xm, um = x.mean(), Uw.mean()
        den = np.sum((x - xm) ** 2)
        features["uncertainty_slope"] = float(
            np.sum((x - xm) * (Uw - um)) / den) if den > 1e-10 else 0.0
    else:
        features["uncertainty_slope"] = 0.0

    features["stationarity_max"] = float(np.max(S[ws:we]))
    u_val, d_val = float(U[event_center_idx]), float(D[event_center_idx])
    features["du_ratio"] = d_val / u_val if u_val > 0.01 else d_val * 100.0
    features["signed_deviation_mean"] = float(np.mean(signed[event_center_idx]))
    return features


URD_FEATURE_NAMES = [
    "max_score", "mean_score", "score_slope", "score_curvature",
    "score_volatility", "duration", "sensor_concentration",
    "num_sensors_flagged", "max_single_sensor",
    "deviation_at_peak", "uncertainty_at_peak", "stationarity_at_peak",
    "uncertainty_slope", "stationarity_max", "du_ratio",
    "signed_deviation_mean"]
