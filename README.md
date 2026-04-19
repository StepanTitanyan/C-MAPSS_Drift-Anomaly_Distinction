# Drift-Aware Probabilistic Anomaly Detection in Multivariate Time Series

Using the NASA C-MAPSS turbofan engine dataset (FD001).

## Project Structure

```
project/
├── config/
│   └── config.yaml              # All hyperparameters and settings
│
├── data/
│   ├── raw/CMAPSSData/          # Put original .txt files here
│   └── processed/               # Cached processed data
│
├── src/
│   ├── data/
│   │   ├── loader.py            # Load raw C-MAPSS files
│   │   ├── preprocessing.py     # Sensor selection, scaling, life_fraction
│   │   ├── splits.py            # Engine-level train/val/test splitting
│   │   └── windowing.py         # Rolling window creation + DataLoaders
│   │
│   ├── models/
│   │   ├── gaussian_lstm.py     # PRIMARY: Probabilistic LSTM (μ + σ output)
│   │   ├── gaussian_gru.py      # Probabilistic GRU alternative
│   │   ├── lstm_autoencoder.py  # Reconstruction-based baseline
│   │   └── baselines.py         # Naive, Ridge, Isolation Forest, OC-SVM
│   │
│   ├── training/
│   │   ├── losses.py            # Gaussian NLL loss + MSE wrapper
│   │   └── trainer.py           # Training loop with early stopping
│   │
│   ├── anomaly/
│   │   ├── scoring.py           # NLL/MSE/Mahalanobis anomaly scores
│   │   └── smoothing.py         # EMA and moving average smoothing
│   │
│   ├── synthetic/
│   │   ├── anomaly_generator.py # 6 types of synthetic anomalies
│   │   └── drift_generator.py   # 5 types of synthetic drift
│   │
│   ├── drift/
│   │   ├── features.py          # 12 residual-pattern features
│   │   └── classifier.py        # Second-stage drift/anomaly classifier
│   │
│   ├── evaluation/
│   │   ├── metrics.py           # Point-level + event-level metrics
│   │   └── degradation.py       # Score vs life_fraction analysis
│   │
│   └── visualization/
│       └── plots.py             # All paper figures
│
├── experiments/
│   └── 01_train_baselines.py    # Stage A: train all models
│
├── outputs/
│   ├── models/                  # Saved checkpoints
│   ├── figures/                 # Generated plots
│   ├── results/                 # Metrics, tables, scores
│   └── logs/                    # Training logs
│
└── tests/                       # Unit tests
```

## Two-Stage Architecture

```
Stage 1: Probabilistic Forecaster → Anomaly Scoring
─────────────────────────────────────────────────────
  Input window (30 cycles × 7 sensors)
    → Gaussian LSTM
    → Predicted μ (mean) + σ (uncertainty)
    → NLL-based anomaly score
    → EMA smoothing
    → Threshold → Flag suspicious events

Stage 2: Drift vs Anomaly Classification
─────────────────────────────────────────
  Flagged event
    → Extract 12 residual-pattern features
      (score shape, sensor concentration, uncertainty behavior)
    → Lightweight classifier (Logistic Regression / Random Forest)
    → {Anomaly, Drift} prediction
```

## Quick Start

1. Place C-MAPSS data files in `data/raw/CMAPSSData/`
2. Install dependencies: `pip install -r requirements.txt`
3. Run Stage A: `python -m experiments.01_train_baselines`

## Key Design Decisions

- **Probabilistic from the start**: The model outputs μ AND σ (not just point predictions).
  The NLL anomaly score accounts for the model's own uncertainty.
- **Split by engine**: Train/val/test are split by engine ID, not by row,
  preventing temporal information leakage.
- **Normal-only training**: The model trains only on the first 50% of each engine's life,
  learning "what healthy looks like" without seeing degradation.
- **Synthetic evaluation**: Controlled anomaly and drift injection with full metadata
  enables rigorous breakdown analysis by type, severity, and life stage.
