# Drift-Aware Probabilistic Anomaly Detection in Multivariate Time Series

Using the NASA C-MAPSS turbofan engine dataset (FD001).  
**Primary model: Gaussian GRU** — same validation NLL as LSTM but trains in 42 vs 49 epochs.

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
│   │   ├── loader.py            # Load raw C-MAPSS .txt files into DataFrames
│   │   ├── preprocessing.py     # Sensor selection, Z-score scaling, life_fraction
│   │   ├── splits.py            # Engine-level train/val/test splitting
│   │   └── windowing.py         # Rolling window creation + DataLoaders
│   │
│   ├── models/
│   │   ├── gaussian_gru.py      # ★ PRIMARY: Probabilistic GRU (μ + σ output)
│   │   ├── gaussian_lstm.py     # Probabilistic LSTM (same dual-head design)
│   │   ├── lstm_autoencoder.py  # Reconstruction-based baseline
│   │   └── baselines.py         # Naive, Ridge, Isolation Forest, OC-SVM
│   │
│   ├── training/
│   │   ├── losses.py            # Gaussian NLL loss + MSE wrapper
│   │   └── trainer.py           # Training loop with early stopping
│   │
│   ├── anomaly/
│   │   ├── scoring.py           # NLL/MSE/Mahalanobis anomaly scores
│   │   ├── urd.py               # ★ URD: D + U + S channels (core contribution)
│   │   └── smoothing.py         # EMA score smoothing
│   │
│   ├── synthetic/
│   │   ├── anomaly_generator.py # 5 types of synthetic anomalies
│   │   └── drift_generator.py   # 4 types of synthetic drift
│   │
│   ├── drift/
│   │   ├── features.py          # 16-feature URD extraction per event
│   │   └── classifier.py        # Stage D/E drift-vs-anomaly classifier
│   │
│   ├── evaluation/
│   │   ├── metrics.py           # Point-level + event-level metrics
│   │   └── degradation.py       # Score vs life_fraction analysis
│   │
│   └── visualization/
│       └── plots.py             # Standard pipeline figures
│
├── experiments/
│   ├── 01_train_baselines.py    # Stage A: train all 4 models
│   ├── 02_synthetic_evaluation.py   # Stage C: NLL vs URD head-to-head
│   ├── 02b_method_comparison.py     # Stage C+: 6-method comparison table
│   ├── 03_drift_classification.py   # Stage D: drift vs anomaly, 16-feat ablation
│   ├── 04_urd_fingerprinting.py     # Stage E: 5-class + 9-class fingerprinting
│   └── 05_generate_paper_outputs.py # ★ ALL paper figures + tables → outputs/for_paper/
│
├── outputs/
│   ├── models/                  # Saved checkpoints (.pt files)
│   ├── figures/                 # Per-stage experiment figures
│   ├── results/                 # Metrics, tables, JSON results
│   ├── logs/                    # Training logs
│   └── for_paper/               # ★ All 7 figures + 4 tables for the paper
│
└── tests/                       # Unit tests
```

## How the Pipeline Works

```
Step 1  train_FD001.txt (100 engines, 26 cols, whitespace-separated)
            ↓ loader.py
Step 2  DataFrame  →  select 7 sensors  →  Z-score  →  split 70/15/15 by ENGINE
            ↓ windowing.py
Step 3  Rolling windows (30, 7) → target (7,)  [training: life_frac ≤ 0.5 only]
            ↓ gaussian_gru.py + trainer.py
Step 4  Gaussian GRU  →  (μ_t, σ_t) per sensor
        Loss: NLL = (1/d) Σ_j [ log(σ_j) + (x_j - μ_j)² / (2σ_j²) ]
            ↓ urd.py
Step 5  URD Decomposition:
        D_t = (1/d) Σ_j r²_{t,j}            r = (x-μ)/σ  ~  N(0,1) under normal
        U_t = (1/d) Σ_j σ_{t,j}/σ_ref,j
        S_t = FDE(t) + γ·max(0, run-2)      (stationarity — novel channel)
        combined = max(D_normalised, S_normalised)
            ↓ features.py + classifier.py
Step 6  16-dim feature vector per event  →  RF/LR  →  "anomaly" or "drift"
        Further:  →  5-class fingerprint (specific type + operational response)
```

## Quick Start

```bash
# 1. Place C-MAPSS data files in data/raw/CMAPSSData/
# 2. Install dependencies
pip install -r requirements.txt

# 3. Train all models (~2-5 min on GPU)
python -m experiments.01_train_baselines

# 4-7. Run evaluation stages
python -m experiments.02_synthetic_evaluation
python -m experiments.02b_method_comparison
python -m experiments.03_drift_classification
python -m experiments.04_urd_fingerprinting

# 8. Generate ALL paper figures and tables
python -m experiments.05_generate_paper_outputs
# → outputs/for_paper/ (7 figures + 4 tables, publication-ready)
```

## Paper Outputs (outputs/for_paper/)

| File | Contents |
|------|----------|
| fig1_pipeline_overview.png | Full pipeline with math annotation at every step |
| fig2_urd_channels.png | D/U/S trajectories for spike / drift / freeze |
| fig3_sensor_freeze_blind_spot.png | NLL vs Stationarity — the "aha" figure |
| fig4_roc_curves_by_type.png | ROC per anomaly type, 6 methods |
| fig5_signature_heatmap.png | (D,U,S) mean per event type — heatmap |
| fig6_feature_importance.png | RF feature importances, Stage D |
| fig7_prediction_bands.png | μ ± 2σ bands on a test engine |
| table1_anomaly_detection.csv | ROC-AUC per type × method |
| table2_drift_ablation.csv | 9/12/16-feat × LR/RF accuracy |
| table3_fingerprint.csv | 5-class Precision/Recall/F1 |
| table4_model_comparison.csv | MSE/NLL/epochs/time per model |

## Key Design Decisions

**Gaussian GRU over LSTM:** Both achieve validation NLL = 0.721.
GRU trains in 42 epochs vs 49, ~25s vs ~32s. Fewer parameters, identical quality.

**Probabilistic from the start:** The model outputs μ AND σ (not just μ).
Without σ, the U channel (uncertainty) and S channel (stationarity) cannot exist.
A deterministic model can only use Channel D — in a degraded form.

**Normal-only training (semi-supervised):** The model trains on life_fraction ≤ 0.5.
It learns what HEALTHY looks like. Any departure from that is anomalous by definition.

**Split by engine (not by row):** If we split rows, the model sees cycle 50 of Engine 5
in training and cycle 60 in testing — it already knows that engine. Engine-level splits
prevent temporal leakage entirely.

**Stationarity channel (S) — the novel contribution:**
Standard NLL scores frozen sensors as LOWER than normal (model predicts ≈ frozen value).
ROC-AUC = 0.44 (sub-random!). S_t = FDE(t) + γ·max(0, run-2) applied to RAW sensor
values (not residuals) lifts sensor freeze ROC-AUC to 0.71 (+0.27).
