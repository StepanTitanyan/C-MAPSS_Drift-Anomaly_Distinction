"""
tests/test_pipeline.py
======================
End-to-end smoke tests for the full pipeline.

Creates synthetic data (no dependency on real C-MAPSS files) and verifies
that every component works correctly: preprocessing, windowing, model
forward pass, loss computation, scoring, and feature extraction.

Run with: python -m pytest tests/ -v
"""

import numpy as np
import pandas as pd
import torch
import pytest

# Make sure project is importable
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_synthetic_df(n_engines=5, cycles_range=(50, 100)):
    """Create a synthetic DataFrame mimicking C-MAPSS format."""
    rng = np.random.RandomState(42)
    rows = []
    sensor_cols = [f"s_{i}" for i in range(1, 22)]

    for engine in range(1, n_engines + 1):
        n_cycles = rng.randint(cycles_range[0], cycles_range[1])
        for cycle in range(1, n_cycles + 1):
            row = {"unit_nr": engine, "time_cycles": cycle}
            for col in ["setting_1", "setting_2", "setting_3"]:
                row[col] = rng.normal(0, 0.01)
            for col in sensor_cols:
                row[col] = rng.normal(500, 50) + cycle * 0.1  # Slight trend
            rows.append(row)

    return pd.DataFrame(rows)


# --- Preprocessing Tests ---

class TestPreprocessing:
    def test_life_fraction(self):
        from src.data.preprocessing import compute_life_fraction
        df = make_synthetic_df(n_engines=3)
        df = compute_life_fraction(df)

        assert "life_fraction" in df.columns
        assert df["life_fraction"].min() > 0
        assert abs(df["life_fraction"].max() - 1.0) < 1e-6

        # Last cycle of each engine should be 1.0
        for eid in df["unit_nr"].unique():
            edf = df[df["unit_nr"] == eid]
            assert abs(edf["life_fraction"].max() - 1.0) < 1e-6

    def test_sensor_selection(self):
        from src.data.preprocessing import select_sensors
        df = make_synthetic_df()
        selected = select_sensors(df, ["s_3", "s_4", "s_11"], keep_meta=True)

        assert "s_3" in selected.columns
        assert "s_4" in selected.columns
        assert "unit_nr" in selected.columns
        assert "s_1" not in selected.columns

    def test_scaler(self):
        from src.data.preprocessing import SensorScaler
        df = make_synthetic_df()
        sensor_cols = ["s_3", "s_4", "s_7"]

        scaler = SensorScaler(sensor_cols)
        df_scaled = scaler.fit_transform(df)

        # After scaling, mean should be ~0 and std ~1 on training data
        for col in sensor_cols:
            assert abs(df_scaled[col].mean()) < 0.1
            assert abs(df_scaled[col].std() - 1.0) < 0.1


# --- Windowing Tests ---

class TestWindowing:
    def test_create_windows_shape(self):
        from src.data.preprocessing import compute_life_fraction, select_sensors
        from src.data.windowing import create_windows

        df = make_synthetic_df(n_engines=3)
        df = compute_life_fraction(df)
        sensors = ["s_3", "s_4", "s_7"]
        df = select_sensors(df, sensors)

        X, y, meta = create_windows(df, sensors, window_size=10)

        assert X.ndim == 3
        assert X.shape[1] == 10  # window size
        assert X.shape[2] == 3   # num sensors
        assert y.shape[1] == 3
        assert X.shape[0] == y.shape[0]
        assert meta.shape[0] == X.shape[0]
        assert meta.shape[1] == 3  # engine_id, cycle, life_frac

    def test_normal_filtering(self):
        from src.data.preprocessing import compute_life_fraction, select_sensors
        from src.data.windowing import create_windows

        df = make_synthetic_df(n_engines=3)
        df = compute_life_fraction(df)
        sensors = ["s_3", "s_4"]
        df = select_sensors(df, sensors)

        X_all, _, _ = create_windows(df, sensors, window_size=10)
        X_normal, _, meta = create_windows(df, sensors, window_size=10, max_life_fraction=0.5)

        assert X_normal.shape[0] < X_all.shape[0]
        # All targets should be in first half of life
        assert np.all(meta[:, 2] <= 0.5)


# --- Model Tests ---

class TestModels:
    def test_gaussian_lstm_forward(self):
        from src.models.gaussian_lstm import GaussianLSTM
        model = GaussianLSTM(input_size=7, hidden_size=32, num_layers=1)

        x = torch.randn(4, 30, 7)  # batch=4, window=30, sensors=7
        mu, sigma = model(x)

        assert mu.shape == (4, 7)
        assert sigma.shape == (4, 7)
        assert torch.all(sigma > 0)  # Sigma must be positive

    def test_gaussian_gru_forward(self):
        from src.models.gaussian_gru import GaussianGRU
        model = GaussianGRU(input_size=7, hidden_size=32, num_layers=1)

        x = torch.randn(4, 30, 7)
        mu, sigma = model(x)

        assert mu.shape == (4, 7)
        assert sigma.shape == (4, 7)
        assert torch.all(sigma > 0)

    def test_deterministic_lstm_interface(self):
        from src.models.gaussian_lstm import DeterministicLSTM
        model = DeterministicLSTM(input_size=7, hidden_size=32, num_layers=1)

        x = torch.randn(4, 30, 7)
        mu, sigma = model(x)

        assert mu.shape == (4, 7)
        assert sigma.shape == (4, 7)
        # Sigma should be all ones (dummy)
        assert torch.allclose(sigma, torch.ones(4, 7))


# --- Loss Tests ---

class TestLosses:
    def test_gaussian_nll(self):
        from src.training.losses import GaussianNLLLoss
        loss_fn = GaussianNLLLoss()

        mu = torch.zeros(4, 7)
        sigma = torch.ones(4, 7)
        target = torch.zeros(4, 7)

        loss = loss_fn(mu, sigma, target)
        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0  # NLL is positive when data matches

    def test_nll_penalizes_overconfidence(self):
        from src.training.losses import GaussianNLLLoss
        loss_fn = GaussianNLLLoss()

        target = torch.tensor([[1.0]])
        mu = torch.tensor([[0.0]])  # Wrong prediction

        # Large sigma (honest uncertainty) → lower penalty
        sigma_large = torch.tensor([[5.0]])
        loss_large = loss_fn(mu, sigma_large, target).item()

        # Small sigma (overconfident) → higher penalty
        sigma_small = torch.tensor([[0.1]])
        loss_small = loss_fn(mu, sigma_small, target).item()

        assert loss_small > loss_large  # Overconfidence is penalized more


# --- Scoring Tests ---

class TestScoring:
    def test_nll_scoring(self):
        from src.anomaly.scoring import compute_nll_scores

        targets = np.array([[0.0, 0.0], [5.0, 5.0]])  # Second is far from 0
        mu = np.zeros((2, 2))
        sigma = np.ones((2, 2))

        scores, per_sensor = compute_nll_scores(targets, mu, sigma)

        assert scores.shape == (2,)
        assert scores[1] > scores[0]  # Outlier should have higher score

    def test_scorer_normalization(self):
        from src.anomaly.scoring import AnomalyScorer

        rng = np.random.RandomState(42)
        n, d = 1000, 7
        targets = rng.normal(0, 1, (n, d))
        mu = rng.normal(0, 0.5, (n, d))
        sigma = np.ones((n, d)) * 0.5

        scorer = AnomalyScorer(score_type="nll")
        scorer.fit_normalization(targets, mu, sigma)

        scores, _ = scorer.score(targets, mu, sigma, normalize=True)
        # After normalization, mean should be ~0, std ~1
        assert abs(np.mean(scores)) < 0.5
        assert abs(np.std(scores) - 1.0) < 0.5


# --- Synthetic Generator Tests ---

class TestSyntheticGenerators:
    def test_anomaly_injection(self):
        from src.synthetic.anomaly_generator import AnomalyGenerator

        sensors = ["s_3", "s_4", "s_7"]
        gen = AnomalyGenerator(sensors, random_seed=42)

        T, d = 100, 3
        engine_data = {
            "engine_id": 1,
            "sensor_values": np.zeros((T, d)),
            "cycles": np.arange(T),
            "life_fracs": np.linspace(0, 1, T),
        }

        traj = gen.create_injected_trajectory(
            engine_data, anomaly_type="spike",
            injection_life_frac=0.5, magnitude=5.0,
        )

        assert traj.sensor_values.shape == (T, d)
        assert traj.labels.shape == (T,)
        assert np.sum(traj.labels == 1) > 0  # At least one anomaly label
        assert len(traj.events) == 1

    def test_drift_injection(self):
        from src.synthetic.drift_generator import DriftGenerator

        sensors = ["s_3", "s_4", "s_7"]
        gen = DriftGenerator(sensors, random_seed=42)

        T, d = 200, 3
        engine_data = {
            "engine_id": 1,
            "sensor_values": np.zeros((T, d)),
            "cycles": np.arange(T),
            "life_fracs": np.linspace(0, 1, T),
        }

        traj = gen.create_drifted_trajectory(
            engine_data, drift_type="gradual_shift",
            injection_life_frac=0.3, rate=0.05, duration=50,
        )

        assert np.sum(traj.labels == 2) > 0  # Drift labels present
        assert np.sum(traj.labels == 1) == 0  # No anomaly labels


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
