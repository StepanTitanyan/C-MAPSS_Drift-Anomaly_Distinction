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
│   │   ├── urd.py               # ★ URD baseline: calibrated D + U + tuned S
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
│   ├── 01_train_baselines.py    # Stage A: train 5 models incl. TranAD
│   ├── 02_synthetic_evaluation.py   # Stage C: URD baseline vs TranAD
│   ├── 02b_method_comparison.py     # Stage C+: multi-method table incl. TranAD
│   ├── 03_drift_classification.py   # Stage D: drift vs anomaly, 16-feat ablation
│   ├── 04_urd_fingerprinting.py     # Stage E: 5-class + 9-class fingerprinting
│   └── 05_generate_paper_outputs.py # ★ ALL paper figures + tables → outputs/for_paper/
│
├── outputs/
│   ├── models/                  # Saved checkpoints (.pt files)
│   ├── figures/                 # Per-stage experiment figures
│   ├── results/                 # Metrics, tables, JSON results
│   ├── logs/                    # Training logs
│   └── for_paper/               # ★ Updated paper pack (10 figures + 4 tables)
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
        D_t = r_t^T Σ_r^{-1} r_t           r = (x-μ)/(τ·σ) after sigma calibration
        U_t = (1/d) Σ_j σ_eff,t,j/σ_ref,j
        S_t = FDE(t) + 3·max(0, run-1)      (tuned stationarity channel)
        combined = 0.35·D_normalised + 0.65·S_normalised
            ↓ features.py + classifier.py
Step 6  16-dim feature vector per event  →  XGBoost/RF/LR  →  "anomaly" or "drift"
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
# → outputs/for_paper/ (updated paper pack with TranAD comparison and calibration plots)
```

## Paper Outputs (outputs/for_paper/)

| File | Contents |
|------|----------|
| fig1_pipeline_overview.png | Updated pipeline overview with URD and TranAD branches |
| fig2_roc_pr_urd_vs_tranad.png | Direct ROC + PR comparison: URD baseline vs TranAD |
| fig3_threshold_sweep.png | Precision / recall / F1 threshold sweeps |
| fig4_per_type_pr.png | Per-anomaly-type PR-AUC bar chart |
| fig5_case_timeline_freeze.png | Raw signal + GRU band + D/U/S timeline on a freeze case |
| fig6_dus_distributions.png | D/U/S distribution plots by anomaly category |
| fig7_feature_importance.png | Stage D feature importance (XGBoost, URD 16 features) |
| fig8_probabilistic_calibration.png | Coverage and residual calibration for the URD backbone |
| fig9_fingerprint_5class_confusion.png | 5-class fingerprinting confusion matrix |
| fig10_stage_d_feature_importance.png | Copied stage-level feature-importance figure |
| table1_anomaly_detection.csv | URD vs TranAD ROC/PR summary |
| table2_drift_ablation.csv | Drift-vs-anomaly classifier comparison |
| table3_fingerprint.csv | 5-class Precision/Recall/F1 |
| table4_model_comparison.csv | Stage A model comparison including TranAD |

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

**Current URD baseline:**
The deployed baseline combines calibrated Mahalanobis deviation, uncertainty tracking, and a tuned stationarity channel with weighted fusion. In the latest internal comparison it reaches 0.8636 overall ROC-AUC and 0.8230 sensor-freeze ROC-AUC. The repo now also includes a practical TranAD-style transformer baseline trained on the exact same FD001 split, sensors, window length, and healthy-only assumption so the comparison is apples-to-apples.
