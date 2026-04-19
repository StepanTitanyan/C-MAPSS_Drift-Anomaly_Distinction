# C-MAPSS URD Framework — Complete Codebase Guide
## Every File, Every Function, Every Design Decision

**Primary model: Gaussian GRU** (gaussian_gru_best.pt).  
**Novel channel: Stationarity (S)** using FDE + run-length on raw sensor values.

---

## 1. Project Structure

```
project/
├── config/config.yaml                   ← Every hyperparameter in one place
├── experiments/
│   ├── 01_train_baselines.py            ← Stage A: train all 4 models
│   ├── 02_synthetic_evaluation.py       ← Stage C: NLL vs URD head-to-head
│   ├── 02b_method_comparison.py         ← Stage C+: 6-method comparison
│   ├── 03_drift_classification.py       ← Stage D: drift vs anomaly, ablation
│   ├── 04_urd_fingerprinting.py         ← Stage E: 5-class fingerprinting
│   └── 05_generate_paper_outputs.py     ← ★ ALL paper figures + tables
├── src/
│   ├── data/                            ← loader, preprocessing, splits, windowing
│   ├── models/                          ← gaussian_gru, gaussian_lstm, baselines
│   ├── training/                        ← losses.py, trainer.py
│   ├── anomaly/                         ← scoring.py, urd.py, smoothing.py
│   ├── synthetic/                       ← anomaly_generator.py, drift_generator.py
│   ├── drift/                           ← features.py, classifier.py
│   ├── evaluation/                      ← metrics.py, degradation.py
│   └── visualization/                   ← plots.py
├── outputs/
│   ├── models/                          ← .pt checkpoints
│   ├── figures/                         ← per-stage figures
│   ├── results/                         ← CSV + JSON results
│   ├── logs/                            ← training logs
│   └── for_paper/                       ← ★ 7 figures + 4 tables (run script 05)
└── tests/test_pipeline.py
```

---

## 2. The Data Flow (Step by Step)

### Step 1 — Raw .txt Files → DataFrame

```
train_FD001.txt
  100 engines × (128-362 cycles each)
  26 whitespace-separated columns, NO header
  unit_nr | time_cycles | setting_1-3 | s_1 ... s_21
    ↓ loader.py: load_train_data()
pandas DataFrame  (20,631 rows × 26 columns)
```

**Math:** `x_t ∈ R^21` — raw sensor reading vector at time t.

### Step 2 — Preprocessing → Normalised 7-sensor DataFrame

```
21 sensors
  → drop 7 zero-variance (s_1,s_5,s_6,s_10,s_16,s_18,s_19)
  → drop 2 weak-corr (s_9, s_14)
  → keep 7: s_3,s_4,s_11,s_17,s_7,s_12,s_20
  → compute life_fraction = time_cycles / max_cycles_per_engine
  → split 70/15/15 by ENGINE ID (not by row!)
  → Z-score: z_j = (x_j - μ^tr_j) / σ^tr_j  (fit on train ONLY)
```

**Math:** After scaling every sensor has mean≈0, std≈1. Correlation with life_fraction:
Pearson r = Σ(x_i - x̄)(y_i - ȳ) / √[Σ(x_i-x̄)² × Σ(y_i-ȳ)²]

### Step 3 — Rolling Windows → (X, y) Arrays

```
For each engine, slide window of size W=30:
  X[t] = sensor values at cycles [t-29, ..., t]     shape (30, 7)
  y[t] = sensor values at cycle t+1                  shape (7,)
Training windows: only from life_fraction ≤ 0.5 (healthy half)
```

### Step 4 — Gaussian GRU Training

```
Input: (batch, 30, 7)
  ↓ GRU(input=7, hidden=64, layers=2, dropout=0.2)
  ↓ h_n[-1]: (batch, 64)
  ↓ Linear(64→7)       → μ (batch, 7)
  ↓ Linear(64→7)       → softplus(·) + 1e-4  → σ (batch, 7)

Loss: NLL = (1/d) Σ_j [ log(σ_j) + (x_j - μ_j)² / (2σ_j²) ]
```

**Why NLL over MSE:** NLL forces calibrated uncertainty.
`log(σ_j)` penalises large σ. `(x-μ)²/(2σ²)` penalises errors, but rewards large σ.
Optimal: σ small when prediction is accurate, large when uncertain.

### Step 5 — URD Decomposition

```
Normalised residual: r_{t,j} = (x_{t,j} - μ_{t,j}) / σ_{t,j}
Under normal conditions: r ~ N(0,1)

Channel D (Deviation):
  D_t = (1/d) Σ_j r²_{t,j}
  Each r² ~ χ²(1), E[D_t]=1, SD[D_t]=√(2/7)≈0.53

Channel U (Uncertainty):
  U_t = (1/d) Σ_j σ_{t,j} / σ_ref,j
  σ_ref,j = median σ on normal validation data

Channel S (Stationarity — novel):
  Δx_{t,j} = x_{t,j} - x_{t-1,j}
  FDE(t,j,w) = (1/w) Σ (Δx)²                 ← near-zero for frozen sensor
  S_fde(t) = max_j { max(0, -log(FDE/FDE_ref + ε)) }
  run(t) = max run of consecutive identical values
  S_t = S_fde(t) + γ · max(0, run(t) - 2)    ← additive (not max)

Combined: max(D_normalised, S_normalised)
```

### Step 6 — Classification and Fingerprinting

```
For each flagged event:
  Extract 16 features from (D, U, S) around event center
  Random Forest / LR → "anomaly" or "drift"
  Further → 5-class type (spike, persistent, noise, malfunction, drift)
```

---

## 3. Source Files Reference

### config/config.yaml

Every hyperparameter lives here. Primary model setting:
```yaml
model:
  primary: "gaussian_gru"     # ← uses gaussian_gru_best.pt
  hidden_size: 64
  num_layers: 2
  dropout: 0.2
  sigma_min: 1.0e-4
```

### src/data/loader.py

**load_train_data(data_dir, subset)** → DataFrame (20,631 rows × 26 cols)

Reads whitespace-delimited .txt with no header. Assigns column names:
`unit_nr, time_cycles, setting_1-3, s_1 through s_21`. Sort by (engine, cycle).

### src/data/preprocessing.py

**compute_life_fraction(df):** Adds `life_fraction = time_cycles / max_cycles_per_engine`.
Goes from ~0 (new) to 1.0 (failure).

**select_sensors(df, sensor_list, keep_meta=True):** Keeps chosen sensors + metadata.

**SensorScaler(sensor_cols):** Wraps sklearn StandardScaler.
- `fit_transform(train_df)` — computes μ,σ from train, applies
- `transform(val_df)` / `transform(test_df)` — applies stored μ,σ
- Stores `means_` and `stds_` for interpretability

**filter_normal_region(df, threshold=0.5):** Keeps life_fraction ≤ threshold.
Used to create healthy-only training data.

**Critical rule:** Fit scaler on training data ONLY. Applying to all splits prevents data leakage.

### src/data/splits.py

**split_engines(df, train_ratio, val_ratio, test_ratio, random_seed):**
Splits by ENGINE ID — returns (train_ids, val_ids, test_ids) numpy arrays.
Splitting by row would let the model see part of an engine in training and
part in testing — temporal leakage inflates all metrics.

**apply_split(df, train_ids, val_ids, test_ids):** Returns `{"train": df, "val": df, "test": df}`.

### src/data/windowing.py

**create_windows(df, sensor_cols, window_size=30, max_life_fraction=None):**
Slides a window of W steps over each engine's trajectory.
- X: (N, 30, 7) — input windows
- y: (N, 7) — next-step targets
- meta: (N, 3) — [engine_id, target_cycle, target_life_frac]
- max_life_fraction=0.5 creates healthy-only training set

**create_full_sequence_windows(df, sensor_cols, window_size=30):**
Organises by engine: returns `{engine_id: {X, y, cycles, life_fracs}}`.
Used for trajectory-level scoring during degradation analysis.

---

## 4. Models (src/models/)

### gaussian_gru.py — ★ PRIMARY MODEL

```
Input: (batch, 30, 7)
  GRU(input=7, hidden=64, layers=2, dropout=0.2, batch_first=True)
  h_n[-1]: (batch, 64)         ← last layer's final hidden state
  Linear(64→7) → μ
  Linear(64→7) → softplus(·) + 1e-4 → σ
```

**Why GRU over LSTM:** GRU uses 2 gates (reset, update) vs LSTM's 3 (forget, input, output).
Fewer parameters → trains faster. On FD001: identical val NLL (0.721), 42 vs 49 epochs.

**GRU equations:**
```
r_t = sigmoid(W_r · [h_{t-1}, x_t] + b_r)       ← reset gate
z_t = sigmoid(W_z · [h_{t-1}, x_t] + b_z)       ← update gate
h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t] + b)       ← candidate state
h_t = (1-z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t           ← final state
```
The additive path (1-z_t)⊙h_{t-1} prevents vanishing gradients.

**softplus:** softplus(x) = log(1+eˣ). Always positive, smooth, ≈x for large x.

### gaussian_lstm.py

Same dual-head architecture using LSTM cells. Kept for ablation comparison.
Stage A trains both; Table 4 compares them. GRU wins on speed, ties on quality.

### baselines.py

**NaivePersistence:** x̂_{t+1} = x_t. MSE = 0.541. Absolute floor.

**RidgeBaseline:** Flattens (30,7) → 210 features → Ridge regression. MSE = 0.350.
Surprisingly strong — confirms the linear signal in the data is substantial.

**IsolationForestBaseline:** Isolation Forest on flattened windows. Standard baseline.

**OneClassSVMBaseline:** One-class SVM on flattened windows. Another standard baseline.

---

## 5. Training (src/training/)

### losses.py

**GaussianNLLLoss:** The heart of the probabilistic approach.
```
NLL = (1/d) Σ_j [ log(σ_j²) + (x_j-μ_j)²/σ_j² + log(2π) ]
```
Two competing terms:
- `log(σ)`: penalises large uncertainty → model wants to be confident
- `(x-μ)²/(2σ²)`: penalises errors scaled by confidence → if wrong, admit it

The tug-of-war produces calibrated σ: small when accurate, large when uncertain.

**MSELossWrapper:** Standard MSE with (mu, sigma, target) interface for compatibility.

### trainer.py

**Trainer class:** Universal training loop:
- Adam optimiser (lr=0.001, weight_decay=1e-5)
- ReduceLROnPlateau scheduler (factor=0.5, patience=7)
- Early stopping (patience=15 epochs)
- Gradient clipping (max_norm=1.0) — prevents LSTM/GRU gradient explosions
- Best-model checkpointing → `outputs/models/{model_name}_best.pt`

---

## 6. Anomaly Detection (src/anomaly/)

### scoring.py

**AnomalyScorer(score_type="nll"):**
- `fit_normalization(val_y, val_mu, val_sigma)` — computes μ,σ of NLL on normal data
- `score(y, mu, sigma, normalize=True)` — returns z-normalised NLL scores
- `compute_thresholds(y, mu, sigma, percentiles)` — p95/p97.5/p99

**Standalone functions:**
- `compute_nll_scores(targets, mu, sigma)` → `(scores, per_sensor_scores)`
- `compute_mse_scores(targets, mu)` → `(scores, per_sensor_scores)`

**Why NLL is better than MSE for scoring:**
Same error of 0.5 units:
- Noisy sensor (σ=2.0): NLL ∝ 0.5²/(2×4) = 0.031 — not alarming
- Precise sensor (σ=0.3): NLL ∝ 0.5²/(2×0.09) = 1.389 — very alarming
Same error, 45× different score. NLL correctly weights by sensor reliability.

### urd.py — ★ CORE CONTRIBUTION

**URDScorer(fde_window=5, run_delta=1e-4, run_bonus=2.0, run_threshold=2):**

`fit(targets, mu, sigma)` — from normal validation data, stores:
- σ_ref,j = median σ per sensor
- FDE_ref,j = mean squared step-to-step change per sensor
- normalisation statistics for D and S

`score(targets, mu, sigma, normalize=True)` → dict containing:
- `deviation`: D_t per timestep
- `uncertainty`: U_t per timestep
- `stationarity`: S_t per timestep
- `combined`: max(D_norm, S_norm)
- `signed_residuals`: r_{t,j} (with sign, for fingerprinting)
- `fde_scores`, `run_scores`: components of S

**Stationarity internals:**
```python
def _compute_fde_score(raw):
    #for each t: fde[j] = mean of (Δx)² over last w steps
    #score = max_j of max(0, -log(fde[j]/fde_ref[j] + ε))

def _compute_run_length(raw):
    #if |x[t,j] - x[t-1,j]| < δ: run[t,j] = run[t-1,j] + 1
    #return max over j

def _compute_stationarity(raw):
    #S[t] = fde_score[t] + γ × max(0, run_length[t] - threshold)
    #additive NOT max: run-length amplifies FDE, never fires alone
```

**Why additive (not max):** When fde_score ≈ 0 (normal data), S ≈ 0 regardless of run length.
Run-length cannot generate false positives independently — it only amplifies FDE.

**extract_urd_features(urd_result, event_center_idx, analysis_window=15):**
Returns 7 URD-specific features for one event: deviation_at_peak, uncertainty_at_peak,
stationarity_at_peak, uncertainty_slope, stationarity_max, du_ratio, signed_deviation_mean.

### smoothing.py

**exponential_moving_average(scores, alpha=0.2):**
`EMA_t = α × score_t + (1-α) × EMA_{t-1}`
Smooths noisy per-step scores to reveal temporal trends. Higher α = less smoothing.

---

## 7. Synthetic Data (src/synthetic/)

### anomaly_generator.py

**AnomalyGenerator** injects controlled anomalies into COPIES of real trajectories.

5 anomaly types:
- **spike:** +magnitude at one timestep on one sensor
- **drop:** -magnitude at one timestep
- **persistent_offset:** +magnitude sustained for duration steps
- **noise_burst:** random noise × variance_multiplier for duration steps
- **sensor_freeze:** sensor locked at current value for duration steps

**create_injected_trajectory(engine_data, anomaly_type, injection_life_frac, magnitude, duration):**
Returns `InjectedTrajectory` with `sensor_values` (modified), `labels` (0=normal, 1=anomaly), `events`.

**generate_test_suite(engine_data_list, magnitudes, injection_positions):**
Creates one trajectory per (engine × type × magnitude × position).

### drift_generator.py

**DriftGenerator** — same interface, 4 drift types, labels=2.

Types: gradual_shift (linear ramp), sigmoid_plateau (S-curve), accelerating (quadratic), multi_sensor.

---

## 8. Drift Classification (src/drift/)

### features.py

**extract_event_features(scores, residuals, sigmas, event_idx, analysis_window):**
9 standard features: max_score, mean_score, score_slope, score_curvature,
score_volatility, duration, sensor_concentration, num_sensors_flagged, max_single_sensor.

**extract_urd_features_for_trajectory(scores, residuals, sigmas, labels, urd_result, ...):**
Returns 16-feature matrix for all events in a trajectory (9 standard + 7 URD).

**FEATURE_NAMES:** canonical list of 12 original feature names.

### classifier.py

**DriftAnomalyClassifier(model_type, random_seed):**
Supports: logistic_regression, random_forest, xgboost.

`fit(X, y, feature_names)` — trains classifier, scales features internally.

`evaluate(X, y_true)` → returns:
- accuracy
- drift_as_anomaly_rate: fraction of drift misclassified as anomaly (critical metric)
- anomaly_as_drift_rate
- confusion_matrix
- classification_report

`get_feature_importance()` → dict feature→importance (from RF or LR coefficients).

---

## 9. Evaluation (src/evaluation/)

### metrics.py

**threshold_independent_metrics(y_true, scores):** ROC-AUC + PR-AUC. No threshold needed.

**point_level_metrics(y_true, scores, threshold):** Precision, Recall, F1 at one threshold.

**event_level_metrics(y_true, scores, threshold):** More realistic for time-series.
- event_recall: fraction of injected events actually detected
- mean_detection_delay: cycles from event onset to first detection
- event_precision: fraction of alarms that correspond to real events

### degradation.py

**full_degradation_report(engine_scores, engine_sigmas):**
- per_engine_score_correlation: Spearman ρ between score and life_fraction
- bucketed_score_analysis: early/middle/late life score comparison
- kruskal_wallis_test: statistical significance
- uncertainty_vs_degradation: does σ increase with engine age?

---

## 10. Experiment Scripts (experiments/)

### 01_train_baselines.py — Stage A ★ MUST RUN FIRST

6 phases:
1. Load FD001, compute life_fraction, select 7 sensors
2. Split 100 engines 70/15/15 by engine ID
3. Z-score normalise (fit on train only)
4. Create rolling windows (normal region only for train/val)
5. Train 4 neural models + 2 non-neural baselines
6. Evaluate, run degradation analysis, save everything

Models trained and saved to `outputs/models/`:
- `gaussian_gru_best.pt` ← ★ primary, loaded by all subsequent stages
- `gaussian_lstm_best.pt`
- `deterministic_gru_best.pt`
- `deterministic_lstm_best.pt`

Key results (from our runs):

| Model | Test MSE | Val NLL | Epochs | Time |
|-------|---------|---------|--------|------|
| Gaussian GRU ★ | 0.361 | 0.721 | 42 | 25.7s |
| Gaussian LSTM | 0.471 | 0.721 | 49 | 31.6s |
| Det. GRU | 0.267 | — | 31 | 15s |
| Ridge | 0.350 | — | — | <1s |
| Naive | 0.541 | — | — | — |

### 02_synthetic_evaluation.py — Stage C

1. Loads GRU + fits NLL scorer + URD scorer on validation data
2. Generates 1,080 synthetic anomaly trajectories
3. Scores with NLL-only and URD combined
4. Computes ROC-AUC + PR-AUC per anomaly type

Key result: sensor freeze ROC goes from 0.436 (NLL, sub-random!) to 0.713 (URD).

Saves: `stage_c_results.json`, `stage_c_head_to_head.csv`, `stage_c_threshold_metrics.csv`

### 02b_method_comparison.py — Stage C+

6 methods compared: NLL → D+Conformity → D+Variance → D+FDE → D+FDE+Run (URD) → IForest

Shows progression that motivated our design. Conformity (chi-sq on residuals) fails.
Only switching to raw sensor values (FDE) creates a working stationarity signal.

Saves: `method_comparison_roc.csv`, `method_comparison_pr.csv`,
`method_comparison_freeze_progression.csv`, `method_comparison_summary.csv`

### 03_drift_classification.py — Stage D

16 configurations: 3 classifiers × (9/12/16 features).
Key result: XGBoost + 16 URD features = 94.1% accuracy, 6.1% drift→anomaly rate.

Saves: `stage_d_classification.csv`

### 04_urd_fingerprinting.py — Stage E

5-class actionable taxonomy: 91.9% accuracy.
Spike vs drop with signed_deviation_mean: 95.0% (vs 58.3% without).

Saves: `stage_e_5class.csv`, `stage_e_9class.csv`, `stage_e_spike_drop.csv`,
`stage_e_ablation.csv`, `stage_e_feature_importance.csv`

### 05_generate_paper_outputs.py — ★ ALL PAPER OUTPUTS

Run after Stage A. Generates 7 figures + 4 tables to `outputs/for_paper/`.

Structure:
```
Phase 1 — no data needed:
  fig1_pipeline_overview.png        (static pipeline with math at every step)
  table4_model_comparison.csv       (static from known results)

Phase 2 — load data + GRU checkpoint

Phase 3 — fit URD and NLL scorers on validation data

Phase 4 — generate:
  fig7_prediction_bands.png         (μ ± 2σ on test engine)
  fig2_urd_channels.png             (D/U/S for spike/drift/freeze)
  fig3_sensor_freeze_blind_spot.png (NLL vs Stationarity on frozen sensor)
  fig4_roc_curves_by_type.png       (ROC per type, all 6 methods) + table1
  fig5_signature_heatmap.png        ((D,U,S) mean per event type)
  fig6_feature_importance.png       (RF importances Stage D)          + table2
  table3_fingerprint.csv            (5-class P/R/F1)
```

---

## 11. URD Results Summary

### Detection (Stage C):

| Metric | NLL Baseline | URD (D+FDE+Run) | Delta |
|--------|-------------|-----------------|-------|
| Overall ROC-AUC | 0.721 | **0.807** | +0.086 |
| Sensor Freeze ROC | 0.436 (sub-random!) | **0.713** | **+0.277** |
| Event Recall (p95) | 0.822 | **0.957** | +0.135 |

### Drift Classification (Stage D):

| Features | XGBoost Accuracy | Drift→Anomaly Rate |
|----------|-----------------|-------------------|
| 9-feat (no URD) | 88.8% | 11.5% |
| 12-feat (+prob) | 91.5% | 7.4% |
| **16-feat URD** | **94.1%** | **6.1%** |

### Fingerprinting (Stage E):

| Metric | Value |
|--------|-------|
| 5-class accuracy | 91.9% |
| Spike vs Drop (with signed_dev) | 95.0% |
| Spike vs Drop (without) | 58.3% |
| Sensor malfunction F1 | 0.914 |

---

## 12. The Complete Data Flow (Code Perspective)

```
train_FD001.txt
    ↓ load_train_data(data_dir, "FD001")
DataFrame (20,631 rows × 26 cols)
    ↓ compute_life_fraction(df)
DataFrame + life_fraction column
    ↓ select_sensors(df, sensors, keep_meta=True)
DataFrame (7 sensors + unit_nr, time_cycles, life_fraction)
    ↓ split_engines(df, 0.70, 0.15, 0.15, seed=42)
train_ids, val_ids, test_ids    (numpy arrays of engine IDs)
    ↓ apply_split(df, train_ids, val_ids, test_ids)
splits = {"train": df, "val": df, "test": df}
    ↓ SensorScaler(sensors).fit_transform(splits["train"])
    ↓ scaler.transform(splits["val"]) / scaler.transform(splits["test"])
Normalised splits
    ↓ create_windows(splits["train"], sensors, window_size=30, max_life_fraction=0.5)
X_train (N,30,7), y_train (N,7), meta_train (N,3)
    ↓ create_dataloader(X, y, batch_size=64, shuffle=True)
DataLoader objects
    ↓ GaussianGRU(input_size=7, hidden_size=64, ...) + Trainer(loss_type="nll")
    ↓ trainer.fit(train_loader, val_loader)
gaussian_gru_best.pt
    ↓
    ├── AnomalyScorer(score_type="nll").fit_normalization(val_y, val_mu, val_sigma)
    │     ↓ .score(test_y, test_mu, test_sigma) → NLL anomaly scores
    │
    ├── URDScorer(fde_window=5).fit(val_y, val_mu, val_sigma)
    │     ↓ .score(test_y, test_mu, test_sigma) → {D, U, S, combined}
    │
    └── extract_urd_features_for_trajectory(...)
          ↓ DriftAnomalyClassifier("random_forest").fit(X_events, y_events)
          ↓ clf.evaluate(X_test, y_test) → accuracy, drift_as_anomaly_rate
          ↓
      05_generate_paper_outputs.py → outputs/for_paper/ (7 figs + 4 tables)
```

---

## 13. How to Run Everything

```bash
#place C-MAPSS data files in data/raw/CMAPSSData/
pip install -r requirements.txt

#Stage A: train all models (~2-5 min on GPU)
python -m experiments.01_train_baselines

#Stage C: NLL vs URD head-to-head
python -m experiments.02_synthetic_evaluation

#Stage C+: all 6 methods compared
python -m experiments.02b_method_comparison

#Stage D: drift classification ablation
python -m experiments.03_drift_classification

#Stage E: 5-class fingerprinting
python -m experiments.04_urd_fingerprinting

#Generate ALL paper figures and tables
python -m experiments.05_generate_paper_outputs
#→ outputs/for_paper/ (fig1-7 + table1-4, publication-ready, 200 DPI)
```

All CSVs and figures are saved to `outputs/results/` and `outputs/figures/`.
Paper-ready outputs go to `outputs/for_paper/`.
