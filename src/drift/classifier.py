"""
src/drift/classifier.py
=======================
Second-stage classifier: given that an event has been flagged as abnormal
by the base anomaly detector, classify it as {anomaly, drift}.

This sits ON TOP of the base detector in a two-stage pipeline:
  Stage 1: Probabilistic forecaster → NLL-based anomaly score → flag events
  Stage 2: Extract residual features → classify flagged events as anomaly or drift

Why a lightweight classifier?
  - If Logistic Regression works well, it proves the 12 extracted features
    are doing the real discrimination work — stronger paper claim.
  - A complex black-box classifier would undermine interpretability.
  - The second stage has limited training data (synthetic events only).

Models:
  1. Logistic Regression — most interpretable, shows feature importance via coefficients
  2. Random Forest — handles nonlinear interactions
  3. XGBoost — gradient boosting baseline (optional)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, Tuple, Optional
import warnings


class DriftAnomalyClassifier:
    """
    Classifies flagged events as anomaly (1) or drift (2).

    Wraps sklearn classifiers with scaling and evaluation utilities.

    Parameters
    ----------
    model_type : str
        "logistic_regression", "random_forest", or "xgboost"
    random_seed : int
    **kwargs : additional model parameters
    """

    def __init__(self, model_type: str = "logistic_regression", random_seed: int = 789, **kwargs):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = self._create_model(model_type, random_seed, **kwargs)
        self.is_fitted = False
        self.feature_names = None

    def _create_model(self, model_type: str, seed: int, **kwargs):
        if model_type == "logistic_regression":
            return LogisticRegression(
                random_state=seed, max_iter=1000,
                class_weight="balanced",  # Handle imbalanced classes
                **kwargs)
        elif model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 6),
                random_state=seed,
                class_weight="balanced",
                n_jobs=-1)
        elif model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(
                    n_estimators=kwargs.get("n_estimators", 100),
                    max_depth=kwargs.get("max_depth", 4),
                    random_state=seed,
                    use_label_encoder=False,
                    eval_metric="logloss")
            except ImportError:
                warnings.warn("XGBoost not installed, falling back to Random Forest")
                return RandomForestClassifier(
                    n_estimators=100, max_depth=6,
                    random_state=seed, class_weight="balanced")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list = None):
        """
        Train the classifier.

        Parameters
        ----------
        X : (N, 12) — feature matrix from extract_features_for_trajectory
        y : (N,) — labels: 1=anomaly, 2=drift
        feature_names : optional list of feature names for interpretation
        """
        #Scale features
        X_scaled = self.scaler.fit_transform(X)

        #Map labels to 0/1 for binary classification
        #1 (anomaly) → 0, 2 (drift) → 1
        y_binary = (y == 2).astype(int)

        self.model.fit(X_scaled, y_binary)
        self.feature_names = feature_names
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict event type.

        Returns
        -------
        predictions : (N,) — 1=anomaly, 2=drift (original label space)
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted first.")

        X_scaled = self.scaler.transform(X)
        pred_binary = self.model.predict(X_scaled)
        #Map back: 0 → 1 (anomaly), 1 → 2 (drift)
        return np.where(pred_binary == 0, 1, 2)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Returns
        -------
        proba : (N, 2) — columns are [P(anomaly), P(drift)]
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted first.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict:
        """
        Evaluate classifier performance.

        Parameters
        ----------
        X : (N, 12) — test features
        y_true : (N,) — true labels (1=anomaly, 2=drift)

        Returns
        -------
        dict with:
          - "predictions": predicted labels
          - "confusion_matrix": 2×2 matrix
          - "classification_report": sklearn report string
          - "accuracy": overall accuracy
          - "drift_as_anomaly_rate": how often drift is misclassified as anomaly
          - "anomaly_as_drift_rate": how often anomaly is misclassified as drift
        """
        predictions = self.predict(X)

        #Map to binary for sklearn metrics
        y_binary = (y_true == 2).astype(int)
        pred_binary = (predictions == 2).astype(int)

        cm = confusion_matrix(y_binary, pred_binary)
        report = classification_report(
            y_binary, pred_binary,
            target_names=["anomaly", "drift"],
            output_dict=True)

        #Key error rates
        anomaly_mask = y_true == 1
        drift_mask = y_true == 2
        drift_as_anomaly = 0.0
        anomaly_as_drift = 0.0

        if drift_mask.sum() > 0:
            drift_as_anomaly = float((predictions[drift_mask] == 1).mean())
        if anomaly_mask.sum() > 0:
            anomaly_as_drift = float((predictions[anomaly_mask] == 2).mean())

        return {
            "predictions": predictions,
            "confusion_matrix": cm,
            "classification_report": report,
            "accuracy": float((predictions == y_true).mean()),
            "drift_as_anomaly_rate": drift_as_anomaly,
            "anomaly_as_drift_rate": anomaly_as_drift}

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores (if available).

        For Logistic Regression: absolute coefficient values.
        For Random Forest / XGBoost: feature importance attribute.
        """
        if not self.is_fitted:
            return None

        names = self.feature_names or [f"f{i}" for i in range(12)]

        if self.model_type == "logistic_regression":
            importances = np.abs(self.model.coef_[0])
        elif hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        else:
            return None

        return dict(zip(names, importances.tolist()))
