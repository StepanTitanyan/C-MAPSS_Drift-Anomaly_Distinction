# Drift-Aware Probabilistic Anomaly Detection in Multivariate Time Series

## A Complete Technical Guide to the URD Framework

*For the C-MAPSS FD001 Turbofan Engine Degradation Dataset*

---

## 1. What Are We Trying to Do?

Imagine you are monitoring a jet engine during a flight. The engine has dozens of sensors measuring things like temperature, pressure, and rotation speed. Your job is to answer three questions in real time:

1. **Is something wrong right now?** (Anomaly detection)
2. **Is the engine slowly wearing out?** (Drift detection)
3. **What kind of problem is it?** (Anomaly fingerprinting)

These three questions sound simple, but answering them simultaneously is surprisingly hard. The core difficulty is that engines naturally degrade over their lifetime — temperatures rise, pressures shift, vibrations increase. This gradual degradation is *expected* and not an emergency. But a sudden spike in temperature IS an emergency. And a sensor that freezes and stops reporting changes is a *different* kind of emergency. A good monitoring system must distinguish all of these.

Our framework, called **URD** (Uncertainty-Residual Decomposition), solves all three problems with a single unified approach. Here is what makes it novel: instead of just asking "is the sensor reading abnormal?", we decompose the anomaly signal into three independent channels that each capture a different *type* of abnormality. This decomposition turns out to be the key to everything.

---

## 2. The Dataset: C-MAPSS FD001

NASA's Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) is a standard benchmark for predictive maintenance research. The FD001 subset contains:

- **100 turbofan engines**, each running from brand-new until failure
- **21 sensor measurements** per time step (temperatures, pressures, speeds, etc.)
- Each engine has a different lifetime (128 to 362 operating cycles)
- No explicit labels — we know the engine was healthy at the start and failed at the end

We selected **7 sensors** that show meaningful degradation trends: s_3, s_4, s_7, s_11, s_12, s_17, s_20. The other 14 sensors are either nearly constant or redundant.

**The life_fraction concept:** For each engine, we define `life_fraction = current_cycle / total_cycles`. An engine at life_fraction = 0.0 is brand new; at 1.0, it is about to fail. We consider life_fraction ≤ 0.5 as the "normal" operating region and train our model only on this region. This is crucial — it means the model learns what "healthy" looks like, and anything beyond that is potentially anomalous.

**Example:** Engine #1 runs for 192 cycles. At cycle 96, life_fraction = 0.5. We train on cycles 1–96 (the healthy half). When we later score cycles 97–192, the model sees patterns it was never trained on, producing higher anomaly scores as the engine degrades.

---

## 3. The Pipeline: Bird's-Eye View

Our system is a three-stage pipeline:

```
Raw Sensor Data
      ↓
┌─────────────────────────────────────────┐
│  STAGE 1: Gaussian LSTM                 │
│  Learns to predict next sensor values   │
│  Outputs: predicted mean (μ) and        │
│           predicted uncertainty (σ)     │
└─────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────┐
│  STAGE 2: URD Decomposition             │
│  Decomposes prediction errors into      │
│  three channels: D, U, S               │
│  → Combined anomaly score               │
└─────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────┐
│  STAGE 3: Classification & Fingerprint  │
│  16-feature extraction per event        │
│  → Is it drift or anomaly?             │
│  → What type of anomaly?               │
└─────────────────────────────────────────┘
```

Let us walk through each stage in detail.

---

## 4. Stage 1: The Gaussian LSTM — Learning "Normal"

### 4.1 The Core Idea

An LSTM (Long Short-Term Memory) network is a type of recurrent neural network that excels at learning patterns in sequential data. We give it a window of the past 30 sensor readings and ask it to predict what the next reading will be.

But here is the twist: instead of predicting a single value, our model predicts a **probability distribution**. Specifically, for each sensor, it predicts:
- **μ** (mu): the expected value ("I think the temperature will be 1590°")
- **σ** (sigma): the uncertainty ("but I could be off by ±6°")

This is called a **Gaussian** (or Normal) prediction because we model each sensor's prediction as a bell curve centered at μ with width σ.

### 4.2 The Mathematical Formulation

Given a window of sensor readings **X** = [x_{t-29}, x_{t-28}, ..., x_{t}] (each x is a vector of 7 sensor values), the model outputs:

```
μ_{t+1} = f_μ(X)         — predicted mean (7 values, one per sensor)
σ_{t+1} = softplus(f_σ(X)) — predicted std dev (7 values, always positive)
```

The `softplus` function ensures σ is always positive: softplus(z) = log(1 + e^z). We also enforce a minimum σ of 0.01 to prevent numerical issues.

### 4.3 The Loss Function: Gaussian Negative Log-Likelihood

During training, we want the model to produce predictions where (a) μ is close to the true value, and (b) σ is appropriately sized — small when the prediction is easy, large when it is hard.

The Gaussian NLL loss for a single sensor j at time t is:

```
NLL_{t,j} = log(σ_{t,j}) + (x_{t,j} - μ_{t,j})² / (2 · σ_{t,j}²)
```

**Intuitive breakdown:**
- The first term, `log(σ)`, penalizes large uncertainty. The model cannot just say "I have no idea" (σ → ∞) to get zero error.
- The second term, `(x-μ)²/(2σ²)`, penalizes prediction errors, but scales them by σ². A big error with big uncertainty is less penalized than a big error with small uncertainty.

This loss function naturally teaches the model to be honest about what it knows and does not know. When patterns are predictable (healthy engine), σ is small. When patterns become unfamiliar (degrading engine), σ grows.

**Example:** If the true temperature is 1590° and the model predicts μ=1588°, σ=6°, the loss is:
- log(6) + (1590-1588)² / (2·36) = 1.79 + 0.056 = 1.85

If σ were instead 2° (overconfident), the loss would be:
- log(2) + 4/(2·4) = 0.69 + 0.5 = 1.19 (lower, because the prediction was actually close)

But if σ=2° and the error was 10°:
- log(2) + 100/8 = 0.69 + 12.5 = 13.19 (very high! Overconfidence with a big miss)

### 4.4 Training Strategy

We train ONLY on the normal region (life_fraction ≤ 0.5) of 70 engines. The remaining 15 engines are for validation, 15 for testing. This is a **semi-supervised** approach — the model never sees anomalous data during training.

The model converges to a validation NLL of ~0.721, with a mean σ of ~0.51 (in standardized units). The key validation: we checked that anomaly scores increase with life_fraction (Spearman ρ = 0.649, 100% of engines show positive correlation). This confirms the model correctly treats degradation as increasingly "surprising."

---

## 5. Stage 2: The URD Framework — Our Main Contribution

### 5.1 Why Standard Scoring Fails

The standard approach to anomaly detection with a predictive model is simple: if the prediction is very wrong, something is abnormal. The NLL score or the squared error serves as the anomaly score.

This works great for **high-deviation anomalies** — spikes, drops, offsets. If a temperature suddenly jumps by 50°, the model's prediction will be far off, NLL will be huge, and we catch it.

But it completely fails for **variance-suppressed anomalies** — specifically, sensor freeze. When a sensor freezes (gets stuck at one value), two things happen:

1. The model's prediction μ is based on the past 30 values, which included the frozen value. So μ is *close to* the frozen value.
2. The residual (x - μ) is therefore *small*, not large.

The NLL-based score is LOW during a sensor freeze. We tested this: sensor freeze detection has ROC-AUC of 0.436 with NLL — worse than random guessing (0.5 would be random). The standard approach is actively anti-correlated with this failure mode.

This is not a quirk of our implementation — it is a fundamental limitation of any residual-based scorer. If the model predicts close to the actual value, the residual is small, regardless of whether the sensor is actually working.

### 5.2 The URD Decomposition: Three Orthogonal Channels

Our solution is to decompose the anomaly signal into three independent channels, each designed to catch a different failure mode.

#### Channel 1: Deviation (D) — "How wrong is the prediction?"

This is the traditional anomaly signal. For each time step t, we compute:

```
D_t = (1/d) Σ_{j=1}^{d} [(x_{t,j} - μ_{t,j})² / σ_{t,j}²]
```

where d = 7 (number of sensors). Each term is a squared normalized residual — the prediction error divided by the model's stated uncertainty. Under normal conditions (the null hypothesis), each term follows a χ²(1) distribution, so D_t has expectation 1.

We then z-normalize using the mean and standard deviation of D computed on normal validation data:

```
D_normalized = (D_t - mean_D) / std_D
```

**What D catches:** Spikes (sudden jumps), drops (sudden falls), persistent offsets (shifted baselines), noise bursts (scattered large deviations). All of these produce large prediction errors.

**What D misses:** Sensor freeze (small residuals), gradual drift (residuals grow slowly, hard to threshold).

**Example:** Normal step: D ≈ 0. Spike of +5σ on one sensor: D ≈ (25 + 1+1+1+1+1+1)/7 ≈ 4.4. After normalization, this might be D ≈ 20 — clearly anomalous.

#### Channel 2: Uncertainty (U) — "How confident is the model?"

```
U_t = (1/d) Σ_{j=1}^{d} [σ_{t,j} / σ_ref,j]
```

where σ_ref,j is the median predicted σ for sensor j on normal validation data. This is a ratio: U = 1 means normal confidence, U > 1 means the model is less confident than usual.

**What U catches:** Drift and degradation. As the engine wears out, the model encounters patterns it has never seen, and its predicted σ grows. Our data shows σ correlates with life_fraction at r = 0.41 across all engines.

**What U does NOT catch:** Point anomalies (brief events that do not change σ much), sensor freeze (model confidence is unaffected by frozen input).

**Example:** In early life, U ≈ 1.0 (normal confidence). In late life with degradation, U ≈ 1.02–1.05 (the model is noticeably less sure).

#### Channel 3: Stationarity (S) — "Has sensor variability collapsed?"

This is our novel channel. Instead of looking at residuals (which depend on the model's prediction), we look directly at the **raw sensor values**. A frozen sensor has a unique property: its variance over a time window drops to zero. No other condition produces this — even a slowly trending sensor has nonzero step-to-step changes.

The Stationarity channel combines three complementary tests:

---

**Component A: First-Difference Energy (FDE)**

Instead of computing the variance of values (which can be low for trending sensors), we compute the energy of consecutive differences:

```
Δx_{t,j} = x_{t,j} - x_{t-1,j}      (step-to-step change)

FDE(t, j, w) = (1/w) Σ_{i=t-w+1}^{t} (Δx_{i,j})²     (mean squared change over window)
```

Under normal operation, FDE has a reference value FDE_ref,j estimated from training data. The score is:

```
S_fde(t) = max over sensors j of: max(0, -log(FDE(t,j,w) / FDE_ref,j + ε))
```

**Why FDE is better than variance:** Consider a sensor that slowly trends upward (e.g., temperature increasing by 0.5° per step). The variance of the values over 10 steps is small — they are clustered around the trend line. But the first differences are all ≈ 0.5, so FDE is NOT small. FDE correctly identifies this as "the sensor is changing" rather than "the sensor is static."

A frozen sensor, by contrast, has Δx = 0 for every step, so FDE = 0, and -log(0/FDE_ref) → very large.

**Example with numbers:**
- Normal sensor, 5 steps: values = [1.0, 1.3, 0.8, 1.1, 0.9]. Differences = [0.3, -0.5, 0.3, -0.2]. FDE = (0.09+0.25+0.09+0.04)/4 = 0.1175. With FDE_ref=0.10, ratio=1.175, score = max(0, -log(1.175)) = 0 (normal).
- Trending sensor: values = [1.0, 1.5, 2.0, 2.5, 3.0]. Differences = [0.5, 0.5, 0.5, 0.5]. FDE = 0.25. Ratio = 2.5, score = 0 (normal — even negative log).
- Frozen sensor: values = [1.0, 1.0, 1.0, 1.0, 1.0]. Differences = [0, 0, 0, 0]. FDE ≈ 0. Ratio ≈ 0, score = -log(ε) ≈ 23 (extremely high!).

---

**Component B: Multi-Scale Variance**

We compute the variance ratio at multiple window sizes {3, 5, 8} and take the maximum:

```
For each window size w ∈ {3, 5, 8}:
    V_obs(t, j, w) = Var(x_{t-w+1}, ..., x_{t})   for sensor j
    R(t, j, w) = V_obs / V_ref(j, w)
    S_w(t) = max over j of: max(0, -log(R + ε))

S_multi(t) = max over w of: S_w(t)
```

**Why multiple scales?** A freeze lasting only 3 steps is best caught by a window of 3. A freeze lasting 10 steps is caught by all windows, but the w=3 window catches it first. Using multiple scales gives us both fast detection (small windows) and strong statistical power (large windows).

**Critical insight:** The smallest window, w=3, can react after just 3 frozen steps. The old single-window approach with w=10 needed the freeze to fill the entire window before producing a strong signal, missing short freezes entirely.

---

**Component C: Consecutive-Identity Run Length**

This is the most direct test. In digital sensor systems, a frozen sensor produces values that are exactly identical — not just similar, but bit-for-bit the same. For continuous-valued data, the probability of two consecutive readings being exactly equal is essentially zero.

```
For each sensor j at each time step t:
    if |x_{t,j} - x_{t-1,j}| < δ:   (δ = 0.0001, a tiny tolerance)
        run(t, j) = run(t-1, j) + 1
    else:
        run(t, j) = 0

S_run(t) = max over j of: run(t, j) × α     (α = 3.0, scaling factor)
```

**Why this works so well:** Under normal conditions with continuous sensors, consecutive identical values almost never occur. If we set δ = 10^{-4}, the probability of any single pair being "identical" is some small value p (maybe 0.01). The probability of a run of k consecutive identical values is p^k. For k=3, that is 10^{-6}. For k=5, it is 10^{-10}. So ANY run of length ≥ 3 is almost certainly a freeze.

This test needs no window parameter, reacts within 2–3 steps, and has a near-zero false positive rate. It is the fastest and most specific of the three components.

**Example:** Normal sensor values at consecutive steps: 1.032, 1.078, 0.994, 1.015. No consecutive pair is within δ=0.0001, so run = 0 at every step.

Frozen sensor: 1.032, 1.032, 1.032, 1.032. After step 2, run=1. After step 3, run=2. After step 4, run=3. Score = 3 × 3.0 = 9.0 (highly anomalous even without normalization).

---

**Combining the Three Components:**

```
S(t) = max(S_fde(t), S_multi(t), S_run(t))
```

We take the maximum because each component catches different aspects of freeze. FDE catches the general stagnation pattern. Multi-scale variance catches it across different durations. Run-length catches exact repetition with near-zero false positives. Taking the max means ANY component firing is sufficient to trigger the alarm.

After computing S(t), we z-normalize it using the mean and standard deviation from normal validation data:

```
S_normalized = (S(t) - mean_S) / std_S
```

### 5.3 The Combined URD Score

The final anomaly score at each time step is:

```
URD_combined(t) = max(D_normalized(t), S_normalized(t))
```

This is the "bidirectional" detection:
- If D is high → the sensor reading deviates from predictions → traditional anomaly
- If S is high → the sensor variance has collapsed → sensor malfunction

Either condition flags the time step as anomalous. The `max` operator ensures that a sensor freeze (D ≈ 0, S >> 0) is caught just as well as a spike (D >> 0, S ≈ 0).

### 5.4 Why This Decomposition Matters

The D, U, S triplet gives each anomaly type a unique **signature**:

| Anomaly Type       | D (Deviation) | U (Uncertainty) | S (Stationarity) |
|-------------------|:------------:|:--------------:|:----------------:|
| Spike / Drop       |    HIGH      |     LOW        |      LOW         |
| Persistent Offset  |    HIGH      |     LOW        |      LOW         |
| Noise Burst        |    VERY HIGH |    MODERATE    |      LOW         |
| Drift/Degradation  |    RISING    |     HIGH       |      LOW         |
| **Sensor Freeze**  |  **LOW**     |   **LOW**      |    **HIGH**      |

Sensor freeze is the ONLY condition that produces high S with low D. This is why the Stationarity channel can catch it with high specificity — nothing else looks like this.

Our experimental results confirm these signatures:

```
drift              : D=  9.60   U=1.022   S= -0.18
noise_anomaly      : D= 40.01   U=1.001   S=  0.07
persistent_shift   : D= 23.38   U=1.007   S=  0.05
point_anomaly      : D= 19.45   U=1.002   S=  0.56
sensor_malfunction : D=  0.42   U=1.000   S=  2.41
```

Look at sensor_malfunction: D is near zero (the model is not surprised), but S is 2.41 (the sensor has stopped varying). This is the signature that no residual-based method can see.

---

## 6. Stage 3: Drift Classification

### 6.1 The Problem

Once we have detected an anomalous event, we need to classify it: is this a sudden fault (anomaly) that needs immediate attention, or a gradual trend (drift) that can be monitored?

This is important operationally. If a temperature spike is actually the early stage of a gradual degradation trend, the appropriate action is "schedule maintenance next week," not "emergency shutdown now."

### 6.2 Feature Extraction: The 16-Feature Set

For each detected event, we extract 16 features that capture the event's shape and URD signature:

**Standard features (9):**
1. **max_score** — peak anomaly score (how severe?)
2. **mean_score** — average score over the event (sustained or brief?)
3. **score_slope** — is the score increasing or decreasing? (evolving or static?)
4. **score_curvature** — is the slope changing? (accelerating degradation?)
5. **score_volatility** — standard deviation of scores (noisy or smooth?)
6. **duration** — how many time steps the event spans
7. **sensor_concentration** — is the anomaly spread across sensors or focused on one?
8. **num_sensors_flagged** — how many sensors exceed the threshold?
9. **max_single_sensor** — the highest per-sensor residual

**URD-specific features (7):**
10. **deviation_at_peak** — D channel value at the event's peak
11. **uncertainty_at_peak** — U channel value at the event's peak
12. **stationarity_at_peak** — S channel value at the event's peak
13. **uncertainty_slope** — is σ increasing over the event window? (sign of drift)
14. **stationarity_max** — maximum S in the analysis window
15. **du_ratio** — D/U ratio: was the deviation "expected" by the model?
16. **signed_deviation_mean** — mean signed residual (positive = upward spike, negative = drop)

### 6.3 The Classifier

We train lightweight classifiers (Random Forest, XGBoost) on these 16 features. The training data comes from synthetic anomalies and drifts injected into validation trajectories. The test data uses synthetic injections into test trajectories.

**Key result:** XGBoost with 16 URD features achieves **94.1% accuracy** at distinguishing drift from anomaly. The drift-as-anomaly error rate is only 6.1%, and anomaly-as-drift is 5.7%.

**The ablation study** proves the URD features matter: without them (9 features only), accuracy drops to 88.8%. The 7 URD features add +5.3 percentage points.

---

## 7. Stage 4: Anomaly Fingerprinting

### 7.1 The Five-Class Taxonomy

Beyond just "anomaly vs drift," we can identify the specific TYPE of anomaly from its URD signature. We define five actionable categories:

| Category              | Contains             | Operational Response                    |
|----------------------|---------------------|-----------------------------------------|
| **Point anomaly**     | spike + drop        | Investigate immediately; possible sensor glitch |
| **Persistent shift**  | persistent offset   | Check for physical change; recalibrate  |
| **Noise anomaly**     | noise burst          | Check sensor connection; possible interference |
| **Sensor malfunction**| sensor freeze        | Replace or recalibrate sensor           |
| **Drift**             | all gradual trends   | Schedule maintenance; monitor trend     |

Each category has a distinct operational response. This is the key — we did NOT design the taxonomy based on mathematical convenience. We designed it based on what the maintenance engineer would *do differently* for each case.

### 7.2 Results

A Random Forest classifier on 16 URD features achieves **91.9% accuracy** across all five categories:

| Category             | Precision | Recall | F1    |
|---------------------|-----------|--------|-------|
| drift                | 0.936    | 0.971  | 0.953 |
| noise_anomaly        | 0.920    | 0.767  | 0.836 |
| persistent_shift     | 0.859    | 0.917  | 0.887 |
| point_anomaly        | 0.885    | 0.900  | 0.893 |
| sensor_malfunction   | 0.946    | 0.883  | 0.914 |

Every category has F1 > 0.83. Sensor malfunction, which was completely invisible to NLL-based methods, is now identified with F1 = 0.914.

### 7.3 The Signed Deviation Feature and Spike vs Drop

Within the "point anomaly" category, spikes (upward) and drops (downward) require different investigations. But since most features use absolute residuals, the classifier cannot tell them apart.

The **signed_deviation_mean** feature solves this. It is simply the mean of the signed (not absolute) normalized residuals at the event peak:

```
signed_deviation_mean = (1/d) Σ_j [(x_{t,j} - μ_{t,j}) / σ_{t,j}]
```

Positive for spikes, negative for drops. With this feature, spike-vs-drop accuracy is **95.0%**. Without it: 58.3%. The improvement of +36.7 percentage points from a single feature demonstrates its discriminative power.

---

## 8. The Progression of Stationarity Methods

To understand why we ended up with the three-component approach, it helps to see the progression of attempts:

### Attempt 1: Conformity (chi-squared on residuals) — FAILED

The first idea was to detect freeze by checking if residuals are "suspiciously low." We computed the chi-squared statistic of normalized residuals over a window and tested if it was below the expected value.

**Why it failed:** A frozen sensor does NOT produce zero residuals. The model predicts μ based on recent inputs. If the sensor froze at value v, and the recent window shows v repeatedly, the model predicts something close to v. The residual |x - μ| is small but not zero, and within normal fluctuation. The chi-squared test on 5–10 samples with a modest variance reduction has almost no statistical power.

Result: sensor freeze ROC-AUC remained at ~0.44 (below random chance).

### Attempt 2: Variance ratio on raw values (single window) — PARTIAL SUCCESS

The insight: stop looking at residuals, look at the raw sensor signal. A frozen sensor has zero variance in its actual values, regardless of what the model predicts.

**Why it was only partial:** A single window of w=10 cannot catch a 5-step freeze effectively. At best, 5 of the 10 values are frozen and 5 are normal, reducing variance by about half. The signal is real but not overwhelming.

Result: sensor freeze ROC-AUC improved to ~0.55.

### Attempt 3: Three-component max (FDE + multi-scale + run-length) — v3

Adding first-difference energy (more specific to freeze), multiple window sizes (catches all durations), and run-length detection (near-zero false positive rate) created a stationarity channel with overwhelming detection power.

Result: sensor freeze ROC-AUC jumped to **0.706** — a +0.270 improvement over the NLL baseline. But a hidden problem emerged.

### Attempt 4: Confirmation-weighted FDE (v4) — BEST

Analyzing the v3 results revealed three hidden bottlenecks:

**Bottleneck 1: The `max` operator hurts precision.** FDE alone achieved PR-AUC = 0.169 for sensor freeze, but URD-v3 (which adds multi-scale and run-length via `max`) dropped it to 0.091. The `max` lets ANY single component trigger alone — and the multi-scale variance occasionally fires on normal data where variance naturally dips.

**Bottleneck 2: `max(D, S)` bleeds noise into non-freeze types.** Spike detection dropped from 0.911 (D-only) to 0.902 (URD-v3) because occasional stationarity spikes on normal data pushed up the combined score, creating false positives.

**Bottleneck 3: Score distribution plateau.** The `max` created many normal scores at exactly the same value, making the p95 and p97.5 thresholds collapse to identical values.

**The v4 fix uses two key changes:**

**Change 1: Confirmation-weighted combination** replaces `max(FDE, multi_scale, run)`:

```
S(t) = FDE_score(t) × confirmation(t)

confirmation(t) = 1 + α·min(run_length(t), K)/K + β·gate(var_score(t))
```

FDE is the PRIMARY signal. Run-length and multi-scale can only AMPLIFY it, never fire independently. The key property: when FDE ≈ 0 (normal data), even a long run-length produces S ≈ 0. This eliminates the false positives from independent component firing.

**Change 2: Margin-gated addition** replaces `max(D, S)`:

```
combined(t) = D(t) + λ·max(0, S(t) - margin)
```

where `margin` is the 97.5th percentile of S on normal data. When S is below the margin (which is true for nearly all normal data), the combined score equals D exactly — non-freeze types are completely unaffected. Only when S significantly exceeds the margin (true freeze) does the stationarity signal add to the combined score.

Result: v4 should maintain the high ROC-AUC of v3 while recovering the PR-AUC closer to FDE-alone levels, AND eliminating the degradation of non-freeze types.

---

## 9. Complete Results Summary

### Detection (Stage C):

| Metric                | NLL Baseline | URD (Our Method) | Improvement |
|----------------------|:------------:|:----------------:|:-----------:|
| Overall ROC-AUC       |    0.721     |     **0.803**    |   +0.082    |
| Sensor Freeze ROC-AUC |    0.436     |     **0.706**    |   +0.270    |
| Event Recall (p95)    |    0.822     |     **0.957**    |   +0.135    |

### Classification (Stage D):

| Feature Set         | XGBoost Accuracy | D→A Rate | A→D Rate |
|--------------------|:----------------:|:--------:|:--------:|
| 9 standard         |      0.888       |  11.5%   |  10.9%   |
| 12 original prob   |      0.915       |   7.4%   |   9.6%   |
| **16 URD (ours)**  |    **0.941**     | **6.1%** | **5.7%** |

### Fingerprinting (Stage E):

| Experiment                 | Accuracy  |
|---------------------------|:---------:|
| 5-class actionable         | **0.919** |
| 9-class per-type           |   0.627   |
| Spike vs drop (w/ signed)  | **0.950** |
| Spike vs drop (w/o signed) |   0.583   |

---

## 10. Key Design Decisions

1. **Train on normal only (semi-supervised):** We never show the model anomalies during training. This means it learns "what is healthy" rather than "what anomalies look like." Any new type of anomaly is automatically detectable as long as it produces unusual residuals or unusual sensor behavior.

2. **Split by engine, not by row:** We split data into train/val/test by engine ID, not by individual rows. This prevents temporal leakage — if we split by row, the model could see the middle of an engine's life in training and the beginning in testing, which would contaminate the evaluation.

3. **Raw sensor values for stationarity:** The Stationarity channel operates on raw sensor values, not residuals. This is the fundamental insight that fixed sensor freeze detection. Residuals depend on model predictions; raw values do not.

4. **Five-class taxonomy based on action:** The fingerprinting categories were designed around operational responses, not mathematical convenience. "Spike" and "drop" are merged into "point_anomaly" because the response is the same. All drift subtypes merge because the response is the same. This boosted accuracy from 62.7% (9-class) to 91.9% (5-class).

5. **Signed deviation for directionality:** Adding one feature (signed_deviation_mean) improved spike-vs-drop classification from 58.3% to 95.0%. This shows that sometimes a single well-chosen feature is worth more than a complex model.

---

## 11. How to Run the Experiments

```bash
# Stage A: Train all models (takes ~2 minutes on GPU)
python -m experiments.01_train_baselines

# Stage C: NLL vs URD head-to-head comparison
python -m experiments.02_synthetic_evaluation

# Stage C+: Comprehensive method comparison (7 methods side-by-side)
python -m experiments.02b_method_comparison

# Stage D: Drift-vs-anomaly classification with ablation
python -m experiments.03_drift_classification

# Stage E: Fingerprinting (5-class + 9-class + spike/drop)
python -m experiments.04_urd_fingerprinting
```

---

## 12. Notation Reference

| Symbol | Meaning |
|--------|---------|
| x_{t,j} | True sensor value: sensor j at time t |
| μ_{t,j} | Predicted mean: sensor j at time t |
| σ_{t,j} | Predicted standard deviation: sensor j at time t |
| D_t | Deviation score at time t |
| U_t | Uncertainty score at time t |
| S_t | Stationarity score at time t |
| d | Number of sensors (= 7) |
| w | Window size for rolling computations |
| σ_ref,j | Median predicted σ for sensor j (from normal validation data) |
| V_ref(j) | Reference variance of sensor j (from normal training data) |
| FDE_ref(j) | Reference first-difference energy for sensor j |
| ε | Small constant to prevent log(0), typically 10^{-10} |
| δ | Tolerance for consecutive-identity test, typically 10^{-4} |
