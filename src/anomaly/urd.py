r"""
src/anomaly/urd.py
===================
URD scorer with the CURRENT project baseline as the default configuration.

Baseline URD used throughout the codebase:
  - calibrated sigma on healthy validation windows
  - deviation channel D from Mahalanobis energy on normalised residuals
  - uncertainty channel U as relative sigma inflation
  - stationarity channel S from tuned FDE + additive run-length bonus
  - weighted fusion: combined = 0.35 * D + 0.65 * S

Residual definition:
    r_t = (x_t - \mu_t) / (\tau \odot \sigma_t)
where tau is a per-sensor calibration temperature fit on healthy validation data.

Deviation channel:
    D_t = r_t^T \Sigma_r^{-1} r_t

Uncertainty channel:
    U_t = (1/d) \sum_j \sigma_{t,j}^{eff} / \sigma^{ref}_j

Stationarity channel:
    S_t = FDE_t + \gamma \max(0, run_t - r_0)
with the tuned default parameters gamma=3 and r_0=1.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


class URDScorer:
    def __init__(
        self,
        fde_window: int = 5,
        run_delta: float = 1e-4,
        run_bonus: float = 3.0,
        run_threshold: int = 1,
        epsilon: float = 1e-10,
        calibrate_sigma: bool = True,
        d_mode: str = "mahal_norm",
        fusion_mode: str = "weighted",
        fusion_weights: Tuple[float, float] = (0.35, 0.65),
    ):
        self.fde_window = int(fde_window)
        self.run_delta = float(run_delta)
        self.run_bonus = float(run_bonus)
        self.run_threshold = int(run_threshold)
        self.epsilon = float(epsilon)

        self.calibrate_sigma = bool(calibrate_sigma)
        self.d_mode = str(d_mode)
        self.fusion_mode = str(fusion_mode)
        self.fusion_weights = tuple(float(x) for x in fusion_weights)

        self.sigma_ref = None
        self.sigma_temp = None
        self.fde_ref = None
        self.mahal_inv_cov = None
        self.deviation_mean = None
        self.deviation_std = None
        self.stationarity_mean = None
        self.stationarity_std = None
        self.is_fitted = False

    @classmethod
    def original_baseline(cls) -> "URDScorer":
        return cls(
            fde_window=5,
            run_delta=1e-4,
            run_bonus=2.0,
            run_threshold=2,
            calibrate_sigma=False,
            d_mode="classic",
            fusion_mode="max",
            fusion_weights=(0.7, 0.3),
        )

    def fit(self, targets, mu, sigma):
        targets = np.asarray(targets, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.maximum(np.asarray(sigma, dtype=np.float64), self.epsilon)

        raw_norm_res = (targets - mu) / sigma
        if self.calibrate_sigma:
            temp = np.sqrt(np.mean(raw_norm_res ** 2, axis=0))
            self.sigma_temp = np.clip(temp, 0.25, 4.0)
        else:
            self.sigma_temp = np.ones(targets.shape[1], dtype=np.float64)

        sigma_eff = np.maximum(sigma * self.sigma_temp[np.newaxis, :], self.epsilon)
        self.sigma_ref = np.median(sigma_eff, axis=0)

        diffs = np.diff(targets, axis=0)
        self.fde_ref = np.maximum(np.mean(diffs ** 2, axis=0), self.epsilon)

        norm_res = (targets - mu) / sigma_eff
        if self.d_mode == "mahal_norm":
            cov = np.cov(norm_res, rowvar=False)
            if cov.ndim == 0:
                cov = np.array([[float(cov)]], dtype=np.float64)
            reg = 1e-6 * np.eye(cov.shape[0], dtype=np.float64)
            self.mahal_inv_cov = np.linalg.pinv(cov + reg)
        else:
            self.mahal_inv_cov = None

        dev = self._compute_deviation_raw(norm_res)
        self.deviation_mean = float(np.mean(dev))
        self.deviation_std = max(float(np.std(dev)), 1e-8)

        stat = self._compute_stationarity(targets)
        valid = stat[~np.isnan(stat)]
        self.stationarity_mean = float(valid.mean()) if len(valid) > 0 else 0.0
        self.stationarity_std = max(float(valid.std()), 1e-8) if len(valid) > 0 else 1.0
        self.is_fitted = True

    def _compute_deviation_raw(self, norm_res):
        if self.d_mode == "mahal_norm":
            return np.einsum("ij,jk,ik->i", norm_res, self.mahal_inv_cov, norm_res)
        return np.mean(norm_res ** 2, axis=1)

    def _compute_fde_score(self, raw):
        raw = np.asarray(raw, dtype=np.float64)
        T, _ = raw.shape
        w = self.fde_window
        scores = np.full(T, np.nan, dtype=np.float64)
        diffs = np.zeros_like(raw)
        diffs[1:] = raw[1:] - raw[:-1]
        sq = diffs ** 2
        for t in range(w, T):
            fde = np.mean(sq[t - w + 1:t + 1], axis=0)
            ratios = fde / self.fde_ref
            scores[t] = np.max(np.maximum(-np.log(ratios + self.epsilon), 0.0))
        return scores

    def _compute_run_length(self, raw):
        raw = np.asarray(raw, dtype=np.float64)
        T, d = raw.shape
        runs = np.zeros((T, d), dtype=np.float64)
        for t in range(1, T):
            for j in range(d):
                if abs(raw[t, j] - raw[t - 1, j]) < self.run_delta:
                    runs[t, j] = runs[t - 1, j] + 1.0
        return np.max(runs, axis=1)

    def _compute_stationarity(self, raw):
        fde = self._compute_fde_score(raw)
        run_lengths = self._compute_run_length(raw)
        run_add = self.run_bonus * np.maximum(run_lengths - self.run_threshold, 0.0)
        return np.where(np.isnan(fde), run_add, fde + run_add)

    def _combine(self, deviation, stationarity):
        if self.fusion_mode == "weighted":
            wd, ws = self.fusion_weights
            return wd * deviation + ws * stationarity
        return np.maximum(deviation, stationarity)

    def score(self, targets, mu, sigma, normalize=True):
        if not self.is_fitted:
            raise RuntimeError("URDScorer must be fit before scoring.")
        targets = np.asarray(targets, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.maximum(np.asarray(sigma, dtype=np.float64) * self.sigma_temp[np.newaxis, :], self.epsilon)

        norm_res = (targets - mu) / sigma
        deviation_raw = self._compute_deviation_raw(norm_res)
        deviation = deviation_raw.copy()
        if normalize:
            deviation = (deviation - self.deviation_mean) / self.deviation_std

        uncertainty = np.mean(sigma / self.sigma_ref[np.newaxis, :], axis=1)
        fde_scores = self._compute_fde_score(targets)
        run_scores = self._compute_run_length(targets)
        stationarity_raw = self._compute_stationarity(targets)
        stationarity = stationarity_raw.copy()
        if normalize:
            stationarity = np.nan_to_num((stationarity - self.stationarity_mean) / self.stationarity_std, nan=0.0)

        combined = self._combine(deviation, stationarity)
        return {
            "deviation": deviation,
            "uncertainty": uncertainty,
            "stationarity": stationarity,
            "combined": combined,
            "norm_residuals": norm_res,
            "signed_residuals": norm_res,
            "signature": np.column_stack([deviation, uncertainty, stationarity]),
            "fde_scores": np.nan_to_num(fde_scores, nan=0.0),
            "run_scores": run_scores,
        }

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
        features["uncertainty_slope"] = float(np.sum((x - xm) * (Uw - um)) / den) if den > 1e-10 else 0.0
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
    "signed_deviation_mean",
]
