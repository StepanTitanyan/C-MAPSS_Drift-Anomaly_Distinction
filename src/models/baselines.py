"""
src/models/baselines.py
=======================
Non-neural baseline models for comparison.

These answer the critical reviewer question:
"Why do you need a probabilistic LSTM? Wouldn't a simpler method work?"

Baselines included:
1. NaivePersistence: predicts x_{t+1} = x_t (last observed value)
2. RidgeBaseline: flattened window → Ridge regression → next step
3. IsolationForestBaseline: standard anomaly detection baseline
4. OneClassSVMBaseline: another standard anomaly detection baseline

The first two are forecasting baselines (comparable to LSTM/GRU).
The last two are direct anomaly detection baselines (comparable to our
full anomaly scoring pipeline).
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from typing import Tuple, Optional


class NaivePersistence:
    """
    Predicts the next step as identical to the last observed step.

    x̂_{t+1} = x_t

    This is the absolute minimum benchmark. If an advanced model cannot
    beat this, something is fundamentally wrong.
    """

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        X : np.ndarray of shape (N, window_size, num_sensors)

        Returns
        -------
        mu : (N, num_sensors) — last observed values
        sigma : (N, num_sensors) — dummy ones
        """
        # Last time step of each window
        mu = X[:, -1, :]
        sigma = np.ones_like(mu)
        return mu, sigma


class RidgeBaseline:
    """
    Ridge regression on flattened windows.

    Flattens each (window_size, num_sensors) window into a single vector,
    then fits a linear model to predict the next step.

    This tests whether the temporal patterns are mostly linear.
    If Ridge performs close to LSTM, the temporal structure is simple.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Parameters
        ----------
        X : (N, window_size, num_sensors)
        y : (N, num_sensors)
        """
        # Flatten: (N, window_size * num_sensors)
        X_flat = X.reshape(X.shape[0], -1)
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_flat, y)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        mu : (N, num_sensors)
        sigma : (N, num_sensors) — dummy ones
        """
        X_flat = X.reshape(X.shape[0], -1)
        mu = self.model.predict(X_flat)
        sigma = np.ones_like(mu)
        return mu, sigma


class IsolationForestBaseline:
    """
    Isolation Forest for direct anomaly detection.

    Operates on flattened windows (no temporal structure awareness).
    This is a standard unsupervised anomaly detection method.

    The comparison shows whether our temporal probabilistic approach
    adds value over a structure-agnostic method.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray):
        """
        Fit on normal training windows.

        Parameters
        ----------
        X : (N, window_size, num_sensors)
        """
        X_flat = X.reshape(X.shape[0], -1)
        self.model.fit(X_flat)

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores (lower = more anomalous for IF).

        We negate so that higher = more anomalous (consistent with our framework).

        Parameters
        ----------
        X : (N, window_size, num_sensors)

        Returns
        -------
        scores : (N,) — anomaly scores, higher = more anomalous.
        """
        X_flat = X.reshape(X.shape[0], -1)
        # IF returns negative scores for anomalies; negate for consistency
        return -self.model.score_samples(X_flat)


class OneClassSVMBaseline:
    """
    One-Class SVM for direct anomaly detection.

    Another standard baseline. Learns a boundary around normal data;
    points outside the boundary are flagged as anomalous.
    """

    def __init__(self, kernel: str = "rbf", nu: float = 0.05, gamma: str = "scale"):
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)

    def fit(self, X: np.ndarray):
        """Fit on normal training windows."""
        X_flat = X.reshape(X.shape[0], -1)
        # OC-SVM can be slow on large datasets; subsample if needed
        if X_flat.shape[0] > 5000:
            idx = np.random.choice(X_flat.shape[0], 5000, replace=False)
            X_flat = X_flat[idx]
        self.model.fit(X_flat)

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores (higher = more anomalous).

        Parameters
        ----------
        X : (N, window_size, num_sensors)

        Returns
        -------
        scores : (N,)
        """
        X_flat = X.reshape(X.shape[0], -1)
        # OC-SVM: negative = anomalous; negate for consistency
        return -self.model.score_samples(X_flat)
