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
| fig1_pipeline_overview.png | Pipeline overview with updated URD baseline, TranAD branch, and downstream diagnosis blocks |
| fig2_urd_channels.png | Channel-level demonstration of D/U/S behavior on representative spike, drift, and freeze cases |
| fig3_sensor_freeze_blind_spot.png | Shows why residual-only scoring misses freeze while stationarity rescues it |
| fig2_roc_pr_urd_vs_tranad.png | Direct ROC + PR comparison: current URD baseline vs TranAD under matched FD001 protocol |
| fig3_threshold_sweep.png | Threshold sweep for precision / recall / F1 / false alarms for URD and TranAD |
| fig4_roc_curves_by_type.png | One-vs-rest ROC curves by anomaly type for all major scoring methods |
| fig4_per_type_pr.png | Per-anomaly-type PR-AUC comparison including URD and TranAD |
| fig5_case_timeline_freeze.png | Raw signal + GRU prediction band + D/U/S timeline on a freeze case |
| fig6_dus_distributions.png | Category-wise distributions of D, U, and S from the deployed URD baseline |
| fig7_feature_importance.png | Stage D feature importance with URD-derived vs standard features visually separated |
| fig8_probabilistic_calibration.png | Coverage and residual calibration for the Gaussian GRU backbone |
| fig9_fingerprint_5class_confusion.png | 5-class fingerprinting confusion matrix |
| fig10_stage_d_feature_importance.png | Stage D feature-importance figure exported for the paper pack |
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


## Latest Baseline Update (Detailed)

The current deployed detector is **not** the older `max(D,S)` URD variant. The codebase now uses a refined baseline that keeps the same three-channel idea but changes the internal mathematics of the deviation channel and the final fusion rule. The current detector is:


\[
	ext{URD baseline score}_t = 0.35\,\widetilde D_t + 0.65\,\widetilde S_t
\]

where \(\widetilde D_t\) and \(\widetilde S_t\) are the validation-normalised deviation and stationarity channels. The uncertainty channel \(U_t\) is still computed and used downstream for diagnosis and fingerprinting, but the deployed Stage C detector does **not** add it directly into the final anomaly score.

### 1. Sigma calibration

The Gaussian GRU predicts a mean and standard deviation for every sensor at the next step:

\[
\mu_t \in \mathbb{R}^7, \qquad \sigma_t \in \mathbb{R}^7
\]

On healthy validation windows, raw normalised residuals are first computed as

\[
 r^{raw}_{t,j} = rac{x_{t,j} - \mu_{t,j}}{\sigma_{t,j}}
\]

If the predicted \(\sigma\) values are perfectly calibrated, then healthy residuals should have variance close to 1. In practice they do not, so the code fits one per-sensor temperature

\[
	au_j = \sqrt{rac{1}{N}\sum_{t=1}^{N} \left(r^{raw}_{t,j}ight)^2}
\]

clipped to the interval \([0.25, 4.0]\). The effective standard deviation then becomes

\[
\sigma^{eff}_{t,j} = 	au_j\,\sigma_{t,j}
\]

This makes residual scaling more trustworthy before any channel is computed. Calibration is fit once on healthy validation windows for a fixed trained model, and then reused at scoring time.

### 2. Deviation channel \(D\): calibrated Mahalanobis energy on normalised residuals

The deployed baseline no longer uses the older mean-squared normalised residual

\[
rac{1}{d}\sum_j r_{t,j}^2
\]

as its main deviation channel. Instead it uses a multivariate residual energy. After sigma calibration:

\[
 r_t = \left(rac{x_{t,1}-\mu_{t,1}}{\sigma^{eff}_{t,1}},\ldots,rac{x_{t,7}-\mu_{t,7}}{\sigma^{eff}_{t,7}}ight)^	op
\]

Healthy validation residuals are used to estimate a covariance matrix

\[
\Sigma_r = \operatorname{Cov}(r_t)
\]

with a small ridge term before inversion. The deployed deviation score is then

\[
D_t = r_t^	op \Sigma_r^{-1} r_t
\]

This is a Mahalanobis-style energy on **normalised** residuals, so it captures both per-sensor surprise and cross-sensor surprise. If several sensors move in an unusual joint pattern, \(D_t\) rises even if no single sensor is extremely large on its own.

The raw \(D_t\) values are then standardised using healthy validation statistics:

\[
\widetilde D_t = rac{D_t - \mu_D^{val}}{\sigma_D^{val}}
\]

### 3. Uncertainty channel \(U\): sigma inflation relative to healthy reference

After calibration, the code stores a per-sensor healthy reference uncertainty

\[
\sigma^{ref}_j = \operatorname{median}_{t\in val}(\sigma^{eff}_{t,j})
\]

and computes

\[
U_t = rac{1}{d}\sum_{j=1}^{d} rac{\sigma^{eff}_{t,j}}{\sigma^{ref}_j}
\]

Interpretation:
- \(U_t pprox 1\): uncertainty is typical for healthy behavior
- \(U_t > 1\): the model is less confident than normal
- \(U_t < 1\): the model is more confident than normal

In the current codebase, \(U_t\) is most valuable in the Stage D/Stage E feature layer and in signature interpretation, not as a dominant term in the final Stage C score.

### 4. Stationarity channel \(S\): tuned FDE + run bonus

The current baseline kept the **raw-signal** philosophy that proved crucial for freeze detection. First-difference energy (FDE) is computed from raw target values, not from residuals:

\[
\Delta x_{t,j} = x_{t,j} - x_{t-1,j}
\]

For a window length \(w=5\), the code averages squared differences over the latest window:

\[
\operatorname{FDE}_{t,j} = rac{1}{w}\sum_{k=t-w+1}^{t} (\Delta x_{k,j})^2
\]

A healthy reference is estimated from validation data:

\[
\operatorname{FDE}^{ref}_j = \mathbb{E}_{t\in val}\left[(\Delta x_{t,j})^2ight]
\]

The stationarity evidence for FDE is

\[
S^{fde}_t = \max_j \max\left(0, -\lograc{\operatorname{FDE}_{t,j}}{\operatorname{FDE}^{ref}_j + arepsilon}ight)
\]

So if motion collapses, the ratio becomes very small and the score rises. The code then adds a tuned run-length bonus. Let \(run_t\) be the longest consecutive near-constant run across sensors, using `run_delta = 1e-4`, `run_threshold = 1`, and `run_bonus = 3.0`. Then:

\[
S_t = S^{fde}_t + 3\cdot \max(0, run_t - 1)
\]

As with \(D_t\), the stationarity channel is normalised on healthy validation windows:

\[
\widetilde S_t = rac{S_t - \mu_S^{val}}{\sigma_S^{val}}
\]

### 5. Final fusion and thresholding

The old baseline used a hard `max(D,S)` rule. The current one uses weighted fusion:

\[
A_t = 0.35\,\widetilde D_t + 0.65\,\widetilde S_t
\]

This means the detector leans more heavily on stationarity than on deviation. That matches the empirical findings: the largest gains over NLL come from recovering freeze/stuck faults. Thresholds are then fit on **healthy validation windows** at the 95th, 97.5th, and 99th percentiles of \(A_t\).

### 6. Latest matched comparison snapshot

Under the exact same FD001 split, same 7 sensors, same window length 30, and same healthy-only training rule, the current URD baseline achieved:

- **Overall ROC-AUC = 0.8636**
- **Overall PR-AUC = 0.4250**
- **Sensor-freeze ROC-AUC = 0.8230**
- **Sensor-freeze PR-AUC = 0.4467**
- At the 99th-percentile threshold: **P = 0.356, R = 0.542, F1 = 0.430**
- Event-level at p99: **EventP = 0.238, EventR = 0.862, EventF1 = 0.373**
- False alarms at p99: **33.8 per 1000 windows** and **5.69 per engine**

The matched TranAD baseline achieved:

- **Overall ROC-AUC = 0.7379**
- **Overall PR-AUC = 0.2475**
- **Sensor-freeze ROC-AUC = 0.4621**
- **Sensor-freeze PR-AUC = 0.0362**

### 7. TranAD baseline used in this repo

The TranAD comparator in this repository is a **practical TranAD-style transformer baseline adapted to the same protocol**. Instead of reconstructing the whole input window, it is adapted to your project’s next-step setup. Given a window \(X_t \in \mathbb{R}^{30	imes 7}\), it produces two predictions:

\[
\hat y_t^{(1)} = f_1(X_t), \qquad \hat y_t^{(2)} = f_2(X_t, focus_t)
\]

The first pass makes a coarse next-step prediction. A causal focus term is then formed from disagreement between the phase-1 prediction and the most recent observed step:

\[
focus_t = \sigma\left((\hat y_t^{(1)} - x_t)^2ight)
\]

This focus term is projected back into the transformer and the second pass produces the refined prediction \(\hat y_t^{(2)}\). The training loss is a weighted two-phase MSE objective with a larger weight on the second phase. The anomaly score used for comparison is simply the normalised next-step MSE:

\[
score^{TranAD}_t = rac{rac{1}{d}\sum_j (y_{t,j} - \hat y_{t,j}^{(2)})^2 - \mu_{val}}{\sigma_{val}}
\]

This makes the comparison apples-to-apples: same split, same windows, same sensors, same healthy-only training, but a very different TSAD philosophy.
