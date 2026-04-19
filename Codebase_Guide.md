# Complete Codebase Guide
## Every Python File Explained

---

## Project Structure

```
project/
├── config/config.yaml          ← All hyperparameters in one place
├── experiments/                 ← Runnable experiment scripts (numbered in order)
│   ├── 01_train_baselines.py   ← Stage A: Train models + degradation analysis
│   ├── 02_synthetic_evaluation.py  ← Stage C: NLL vs URD head-to-head
│   ├── 02b_method_comparison.py    ← Stage C+: Full 6-method comparison table
│   ├── 03_drift_classification.py  ← Stage D: Drift vs anomaly classification
│   └── 04_urd_fingerprinting.py    ← Stage E: 5-class + 9-class fingerprinting
├── src/                         ← Library code (imported by experiments)
│   ├── data/                    ← Data loading, preprocessing, splitting, windowing
│   ├── models/                  ← Neural network architectures
│   ├── training/                ← Training loop and loss functions
│   ├── anomaly/                 ← Scoring, URD decomposition, smoothing
│   ├── synthetic/               ← Anomaly and drift injection generators
│   ├── drift/                   ← Feature extraction and classifiers for Stage D/E
│   ├── evaluation/              ← Metrics (ROC, PR, event-level, degradation)
│   └── visualization/           ← Plotting functions
├── tests/test_pipeline.py       ← Smoke tests
└── outputs/                     ← Results, figures, CSVs (created at runtime)
```

---

## DATA PIPELINE (src/data/)

### src/data/loader.py
**Purpose:** Loads raw C-MAPSS text files into pandas DataFrames.

**Key function:** `load_train_data(data_dir, subset="FD001")`
- Reads `train_FD001.txt` (space-delimited, no header)
- Assigns column names: `unit_nr`, `time_cycles`, `setting_1`, `setting_2`, `setting_3`, `s_1`...`s_21`
- Returns a DataFrame with 20,631 rows × 26 columns (for FD001)

**Why it matters:** C-MAPSS files have no headers, so the column naming here is critical. Getting this wrong would misalign all sensor readings.

### src/data/preprocessing.py
**Purpose:** Computes life_fraction, selects sensors, and standardizes data.

**Key functions:**
- `compute_life_fraction(df)` — For each engine, adds `life_fraction = current_cycle / max_cycle`. Engine 1 with 192 cycles gets life_fraction = 0.0 at cycle 1 and 1.0 at cycle 192. This is the fundamental label we use for semi-supervised training.
- `select_sensors(df, sensor_list, keep_meta=True)` — Keeps only the 7 selected sensors plus metadata columns (unit_nr, time_cycles, life_fraction). The 7 sensors were chosen because they show meaningful degradation trends.
- `SensorScaler` — Wraps sklearn's StandardScaler. Fits on training data, transforms val/test. Each sensor is independently z-normalized to mean=0, std=1. This is crucial because raw sensor values span wildly different ranges (s_3 ≈ 1590°, s_20 ≈ 38).

### src/data/splits.py
**Purpose:** Splits 100 engines into train/val/test by ENGINE ID (not by row).

**Key function:** `split_engines(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)`
- Returns three lists of engine IDs: [70 train], [15 val], [15 test]
- `apply_split(df, train_ids, val_ids, test_ids)` — Filters the DataFrame

**Critical design decision:** Splitting by engine prevents temporal leakage. If we split by row, the model could see cycle 100 of engine 5 in training and cycle 50 in testing — it would already know that engine's behavior. By splitting by engine, the model never sees any data from test engines during training.

### src/data/windowing.py
**Purpose:** Creates rolling windows for LSTM input.

**Key function:** `create_windows(df, sensors, window_size=30, max_life_fraction=0.5)`
- For each engine, slides a window of 30 steps across the time series
- X[i] = sensor values at steps [t-29, t-28, ..., t], shape (30, 7)
- y[i] = sensor values at step t+1, shape (7,) — the prediction target
- `max_life_fraction=0.5` means we only create windows from the first half of each engine's life (the "normal" region)
- Returns: X (N, 30, 7), y (N, 7), metadata dict

---

## MODELS (src/models/)

### src/models/gaussian_lstm.py — PRIMARY MODEL
**Purpose:** LSTM that predicts μ and σ for each sensor.

**Architecture:**
```
Input (batch, 30, 7)
  → LSTM(7→64, 2 layers, dropout=0.2)
  → Last hidden state (batch, 64)
  → Linear(64→7)     → μ (predicted mean)
  → Linear(64→7)     → softplus → σ (predicted std, always positive)
```

**Key details:**
- `softplus(x) = log(1 + exp(x))` ensures σ > 0
- `sigma_min=0.01` prevents σ from collapsing to zero
- Forward pass returns `(mu, sigma)` — both shape (batch, 7)
- Trained with Gaussian NLL loss

### src/models/gaussian_gru.py
**Purpose:** Same architecture but with GRU instead of LSTM. GRU is faster (fewer parameters) and sometimes performs comparably. Used as a comparison model.

### src/models/baselines.py
**Purpose:** Non-neural baselines.
- `NaivePersistence` — Predicts x_{t+1} = x_t (just repeats the last value). MSE = 0.541.
- `RidgeRegression` — Flattens the window into a single vector and fits ridge regression. MSE = 0.350. This is surprisingly strong — it means the linear signal in the data is substantial.

### src/models/lstm_autoencoder.py
**Purpose:** Reconstruction-based baseline (encode-decode architecture). Not used in the final results but available for comparison.

---

## TRAINING (src/training/)

### src/training/losses.py
**Purpose:** Loss functions for training.

**Key functions:**
- `gaussian_nll_loss(mu, sigma, targets)` — The Gaussian NLL: `log(σ) + (x-μ)²/(2σ²)`. This is the key loss that teaches the model to predict both mean and uncertainty. It penalizes overconfidence (small σ with big error) more than honest uncertainty.
- `mse_loss(predictions, targets)` — Standard MSE for deterministic models.

### src/training/trainer.py
**Purpose:** Training loop with early stopping and learning rate scheduling.

**Key class:** `Trainer(model, optimizer, loss_fn, device, patience=15, lr_schedule_patience=8)`
- Trains for up to 100 epochs
- Monitors validation loss; stops if no improvement for 15 epochs
- Reduces learning rate by 0.5x if no improvement for 8 epochs
- Saves best model checkpoint

---

## ANOMALY DETECTION (src/anomaly/)

### src/anomaly/scoring.py
**Purpose:** Standard NLL-based anomaly scoring (the baseline).

**Key class:** `AnomalyScorer(score_type="nll")`
- `fit_normalization(y, mu, sigma)` — Computes mean/std of NLL scores on normal validation data
- `score(y, mu, sigma, normalize=True)` — Returns per-timestep anomaly scores (z-normalized NLL)
- `compute_thresholds(y, mu, sigma, percentiles)` — Computes score thresholds at p95, p97.5, p99

**Score types available:** "nll" (Gaussian NLL), "mse" (mean squared error), "mae" (mean absolute error)

### src/anomaly/urd.py — CORE CONTRIBUTION
**Purpose:** The URD (Uncertainty-Residual Decomposition) framework.

**Key class:** `URDScorer(fde_window=5, run_delta=1e-4, run_bonus=2.0)`

**Three channels:**
1. **Deviation (D):** `mean((x-μ)²/σ²)` per timestep — catches spikes, drops, offsets
2. **Uncertainty (U):** `mean(σ/σ_ref)` per timestep — catches drift (model becomes uncertain)
3. **Stationarity (S):** FDE + run-length bonus — catches sensor freeze

**The Stationarity channel (FDE+Run):**
- First-Difference Energy: `FDE(t,j) = mean of (Δx)² over window` — near-zero for frozen sensors
- Run-length bonus: counts consecutive near-identical values, adds bonus when run > 2
- Combined: `S = FDE_score + run_bonus * max(0, run_length - 2)` (additive, not max)

**Combined score:** `max(D_normalized, S_normalized)` — either high deviation OR high stationarity triggers

**Feature extraction:** `extract_urd_features(urd_result, event_idx)` — Extracts 7 URD-specific features for fingerprinting: deviation_at_peak, uncertainty_at_peak, stationarity_at_peak, uncertainty_slope, stationarity_max, du_ratio, signed_deviation_mean.

**Full 16-feature set** (URD_FEATURE_NAMES): 9 standard + 7 URD-specific.

### src/anomaly/smoothing.py
**Purpose:** Exponential Moving Average (EMA) smoothing for anomaly scores.
- Smooths noisy per-timestep scores into a cleaner signal
- `ema_smooth(scores, alpha=0.3)` — Higher α = more smoothing

---

## SYNTHETIC DATA (src/synthetic/)

### src/synthetic/anomaly_generator.py
**Purpose:** Injects 5 types of synthetic anomalies into real engine trajectories.

**Anomaly types:**
1. `spike` — Sudden upward jump on 1-2 sensors for 1-3 steps
2. `drop` — Sudden downward drop (same as spike but negative)
3. `persistent_offset` — Sustained shift in sensor value for 10-30 steps
4. `noise_burst` — Increased noise variance for 5-15 steps
5. `sensor_freeze` — Sensor value frozen (repeated) for 5-20 steps

**Key function:** `generate_test_suite(engine_data_list, magnitudes, injection_positions)`
- Takes real test engine data and injects anomalies at specified life_fraction positions
- Returns list of `InjectedTrajectory` objects, each with:
  - `sensor_values` — The modified sensor data
  - `labels` — Per-timestep: 0=normal, 1=anomaly
  - `events` — List of injected events with type, start, end, magnitude

### src/synthetic/drift_generator.py
**Purpose:** Injects 4 types of synthetic drift into real engine trajectories.

**Drift types:**
1. `gradual_shift` — Linear ramp up over duration
2. `sigmoid_plateau` — Sigmoid-shaped transition to new level
3. `accelerating` — Quadratic acceleration
4. `multi_sensor` — Drift affecting multiple sensors with different rates

**Key function:** `generate_test_suite(engine_data_list, rates, durations, drift_types)`
- Labels: 0=normal, 2=drift (distinct from anomaly label=1)

---

## DRIFT CLASSIFICATION (src/drift/)

### src/drift/features.py
**Purpose:** Extracts features from detected events for classification.

**Key functions:**
- `extract_event_features(scores, residuals, sigmas, event_idx, window)` — Computes 9 standard features: max_score, mean_score, score_slope, score_curvature, score_volatility, duration, sensor_concentration, num_sensors_flagged, max_single_sensor
- `extract_features_for_trajectory(...)` — Finds events in a trajectory and extracts 12-feature vectors (9 standard + 3 original probabilistic)
- `extract_urd_features_for_trajectory(...)` — Finds events and extracts full 16-feature vectors (9 standard + 7 URD-specific). This is the function used by Stages D and E.

### src/drift/classifier.py
**Purpose:** Lightweight classifiers for drift-vs-anomaly and fingerprinting.

**Key class:** `DriftAnomalyClassifier(model_type, random_seed=42)`
- Supports: "logistic_regression", "random_forest", "xgboost"
- `fit(X, y, feature_names)` — Trains the classifier
- `evaluate(X, y)` — Returns accuracy, confusion matrix, drift_as_anomaly_rate, anomaly_as_drift_rate
- `get_feature_importance()` — Returns dict of feature→importance

---

## EVALUATION (src/evaluation/)

### src/evaluation/metrics.py
**Purpose:** All evaluation metrics for the paper.

**Key functions:**
- `threshold_independent_metrics(labels, scores)` — Returns ROC-AUC and PR-AUC. These don't require a threshold and measure overall ranking quality.
- `point_level_metrics(labels, scores, threshold)` — Returns precision, recall, F1 at a given threshold. Each timestep is classified independently.
- `event_level_metrics(labels, scores, threshold)` — Returns event_recall (fraction of anomaly events that were detected) and mean_detection_delay (how many steps after event start before detection). This is more realistic than point-level because it evaluates whether each injected anomaly was caught, not just individual timesteps.

### src/evaluation/degradation.py
**Purpose:** Analyzes how anomaly scores correlate with engine degradation.

**Key function:** `degradation_report(engine_scores, engine_life_fracs)`
- Computes Spearman correlation between score and life_fraction per engine
- Reports early/middle/late mean scores
- Kruskal-Wallis test for statistical significance
- σ vs life_fraction correlation

---

## VISUALIZATION (src/visualization/)

### src/visualization/plots.py
**Purpose:** All paper figures.

**Key functions:**
- `plot_prediction_bands(targets, mu, sigma, engine_id)` — Shows model predictions with ±2σ uncertainty bands
- `plot_training_curves(train_losses, val_losses)` — Loss curves during training
- `plot_score_vs_life_fraction(scores, life_fracs)` — The degradation plot showing scores increase with engine age
- `plot_confusion_matrix_3way(cm, labels)` — Confusion matrix for drift/anomaly/normal
- `plot_feature_importance(importances, title)` — Bar chart of feature importances
- `plot_roc_curves(...)` — ROC curves for multiple methods

---

## EXPERIMENT SCRIPTS (experiments/)

### experiments/01_train_baselines.py — Stage A
**What it does:**
1. Loads FD001 data, computes life_fraction, selects 7 sensors
2. Splits 100 engines into 70/15/15 train/val/test
3. Standardizes using train statistics only
4. Creates rolling windows (30-step) from normal region (life_fraction ≤ 0.5)
5. Trains 4 models: Gaussian LSTM, Gaussian GRU, Deterministic LSTM, Deterministic GRU
6. Evaluates on test set: MSE, MAE, per-sensor MAE, mean σ
7. Computes degradation analysis: Spearman ρ, early/middle/late scores, Kruskal-Wallis test
8. Saves: model checkpoints (.pt), figures, results JSON

**Outputs:** `outputs/models/gaussian_lstm_best.pt` (used by all subsequent stages)

### experiments/02_synthetic_evaluation.py — Stage C
**What it does:**
1. Loads trained Gaussian LSTM and fits NLL scorer + URD scorer on validation data
2. Generates 1080 synthetic anomaly trajectories (5 types × multiple magnitudes × multiple injection positions)
3. Scores each trajectory with both NLL and URD, computes ROC-AUC and PR-AUC per type
4. Computes threshold-based metrics (precision, recall, F1, event recall) at p95/p97.5/p99
5. Prints head-to-head comparison table

**Key result:** Sensor freeze ROC goes from 0.436 (NLL) to ~0.71 (URD)

**CSVs saved:** `stage_c_head_to_head.csv`, `stage_c_threshold_metrics.csv`

### experiments/02b_method_comparison.py — Stage C+ (Paper Table)
**What it does:**
1. Fits 6 different scoring methods on validation data
2. Evaluates all 6 on the same 1080 synthetic trajectories
3. Produces 4 paper-ready tables: ROC per type, PR per type, event metrics, freeze progression
4. Shows the clear progression from residual-based (fail) to raw-value-based (succeed)

**The 6 methods:** NLL → D+Conformity → D+Variance → D+FDE → D+FDE+Run (URD) → IForest

**CSVs saved:** `method_comparison_roc.csv`, `method_comparison_pr.csv`, `method_comparison_freeze_progression.csv`, `method_comparison_event_metrics.csv`, `method_comparison_summary.csv`

### experiments/03_drift_classification.py — Stage D
**What it does:**
1. Generates synthetic anomalies AND synthetic drifts for val and test engines
2. Runs the Gaussian LSTM on each trajectory to get predictions
3. Extracts features for each detected event: 9 standard, 12 original, or 16 URD
4. Trains 3 classifiers (LR, RF, XGBoost) × 3 feature sets = 9 configurations
5. Reports accuracy, drift_as_anomaly_rate, anomaly_as_drift_rate for each

**Key result:** XGBoost + 16 URD features = 94.2% accuracy, 6.1% D→A rate

**CSVs saved:** `stage_d_classification.csv`

### experiments/04_urd_fingerprinting.py — Stage E
**What it does:**
1. Generates per-type synthetic data (5 anomaly types + 4 drift types)
2. Maps 9 types → 5 actionable categories (point_anomaly, persistent_shift, noise_anomaly, sensor_malfunction, drift)
3. Extracts 16-feature URD vectors for each event
4. Experiment 1: 5-class Random Forest classification → accuracy, per-class F1, confusion matrix, URD signature profiles
5. Experiment 2: 9-class per-type classification → shows within-category confusion
6. Experiment 3: Spike vs drop distinction with signed_deviation ablation
7. Experiment 4: Feature ablation (16 URD vs 9 standard)

**Key results:** 5-class = 92.1%, spike/drop = 95.0% (signed_deviation adds +36.7pp)

**CSVs saved:** `stage_e_5class.csv`, `stage_e_9class.csv`, `stage_e_spike_drop.csv`, `stage_e_ablation.csv`, `stage_e_feature_importance.csv`

---

## HOW TO RUN (in order)

```bash
python -m experiments.01_train_baselines        # Stage A (2 min on GPU)
python -m experiments.02_synthetic_evaluation    # Stage C (~30 sec)
python -m experiments.02b_method_comparison      # Stage C+ (~2 min)
python -m experiments.03_drift_classification    # Stage D (~1 min)
python -m experiments.04_urd_fingerprinting      # Stage E (~1 min)
```

All CSVs and figures are saved to `outputs/results/` and `outputs/figures/`.

---

## CONFIG (config/config.yaml)

Contains every hyperparameter:
- `dataset.selected_sensors`: The 7 sensors used
- `preprocessing.window_size`: 30 (LSTM input length)
- `preprocessing.normal_life_fraction_threshold`: 0.5 (train on first half only)
- `model.hidden_size`: 64, `num_layers`: 2, `dropout`: 0.2
- `training.max_epochs`: 100, `patience`: 15, `lr`: 0.001
- `synthetic_anomalies.magnitudes`: [3.0, 5.0] (injection strength in σ units)
- `drift_classifier.models`: ["logistic_regression", "random_forest", "xgboost"]

---

## COMPLETE CSV OUTPUT LIST

After running all experiments, `outputs/results/` contains:

| File | Stage | Contents |
|------|-------|----------|
| stage_a_results.json | A | Model metrics, degradation stats |
| stage_c_results.json | C | NLL vs URD overall + per-type |
| stage_c_head_to_head.csv | C | Per-type ROC/PR comparison |
| stage_c_threshold_metrics.csv | C | P/R/F1 at each threshold |
| method_comparison_roc.csv | C+ | ROC per type, all 6 methods |
| method_comparison_pr.csv | C+ | PR per type, all 6 methods |
| method_comparison_freeze_progression.csv | C+ | Freeze detection progression |
| method_comparison_event_metrics.csv | C+ | Event recall per method |
| method_comparison_summary.csv | C+ | Overall summary table |
| method_comparison_results.json | C+ | Full results (JSON) |
| stage_d_results.json | D | All 9 classifier configs |
| stage_d_classification.csv | D | Accuracy + error rates per config |
| stage_e_results.json | E | All fingerprinting results |
| stage_e_5class.csv | E | 5-class P/R/F1 + URD profiles |
| stage_e_9class.csv | E | 9-class P/R/F1 + URD profiles |
| stage_e_spike_drop.csv | E | Signed deviation ablation |
| stage_e_ablation.csv | E | 16 vs 9 feature comparison |
| stage_e_feature_importance.csv | E | Feature rankings |
