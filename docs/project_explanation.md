# Drift-Aware Probabilistic Anomaly Detection in Multivariate Time Series
## Complete Project Explanation — From Intuition to Implementation

---

## Table of Contents

1. [What Problem Are We Solving?](#1-what-problem-are-we-solving)
2. [The Dataset: NASA C-MAPSS](#2-the-dataset-nasa-c-mapss)
3. [Data Preprocessing Pipeline](#3-data-preprocessing-pipeline)
4. [The Probabilistic Model: Gaussian GRU](#4-the-probabilistic-model-gaussian-gru)
5. [Training with Gaussian NLL Loss](#5-training-with-gaussian-nll-loss)
6. [From Predictions to Anomaly Scores](#6-from-predictions-to-anomaly-scores)
7. [The Problem with Standard Scoring](#7-the-problem-with-standard-scoring)
8. [URD: Uncertainty-Residual Decomposition](#8-urd-uncertainty-residual-decomposition)
9. [Channel 1: Deviation — Too Far From Expected](#9-channel-1-deviation)
10. [Channel 2: Uncertainty — How Confident Is the Model?](#10-channel-2-uncertainty)
11. [Channel 3: Stationarity — Variance Collapse on Raw Signals](#11-channel-3-stationarity)
12. [How the Three Channels Combine](#12-how-the-three-channels-combine)
13. [Stage 2: Drift vs Anomaly Classification](#13-stage-2-drift-vs-anomaly-classification)
14. [Stage 3: Anomaly Fingerprinting](#14-stage-3-anomaly-fingerprinting)
15. [The Complete Pipeline](#15-the-complete-pipeline)
16. [Experimental Design and Evaluation](#16-experimental-design-and-evaluation)
17. [Summary of Contributions](#17-summary-of-contributions)

---

## 1. What Problem Are We Solving?

### The Setting

Imagine you are monitoring an aircraft engine. The engine has dozens of sensors that
measure things like temperature, pressure, rotation speed, and fuel flow. Every flight
cycle (takeoff, cruise, landing), these sensors produce a new set of readings. Over the
engine's lifetime — which might last hundreds of flight cycles — you collect a long
multivariate time series.

Your job is to detect when something goes wrong.

### Why Is This Hard?

The naive approach is: "Learn what normal looks like. Flag anything that's different."
The problem is that **not all changes are faults.**

Real engines change over time for many reasons:

- **Gradual degradation:** Components wear down slowly. Temperatures creep up by
  fractions of a degree per cycle. This is expected aging, not a fault.
- **Regime changes:** Operating conditions might shift (different altitude, different
  load). Sensor readings change, but the engine is fine.
- **Actual faults:** A bearing cracks. A valve sticks. A sensor malfunctions. These are
  the events you actually need to catch.

A standard anomaly detector treats ALL changes the same. It will flood the maintenance
team with false alarms every time the engine ages slightly, while potentially missing
subtle but dangerous faults.

### What We Want

We want a system that can do three things:

1. **Detect** that something unusual is happening (anomaly detection)
2. **Distinguish** whether the unusual behavior is a fault or just gradual
   drift/degradation (drift-awareness)
3. **Identify** what kind of anomaly it is (fingerprinting)

This is significantly harder than standard anomaly detection, and it requires a model
that doesn't just predict sensor values, but also understands its own uncertainty.

---

## 2. The Dataset: NASA C-MAPSS

### What It Is

C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) is a simulated turbofan
engine dataset from NASA. It contains sensor data from 100 engines, each running from
healthy until failure.

Each engine has:
- A unique ID (`unit_nr`)
- A sequence of cycles (`time_cycles`), numbered 1 to T (where T varies per engine)
- 3 operational settings
- 21 sensor measurements at each cycle

We use the **FD001** subset, which has a single operating condition and a single fault
mode. This makes it the cleanest starting point.

### What the Data Looks Like

Here is one row of data (one cycle of one engine):

```
unit_nr=1, time_cycles=1, setting_1=-0.0007, setting_2=-0.0004, setting_3=100.0,
s_1=518.67, s_2=641.82, s_3=1589.70, s_4=1400.60, s_5=14.62, ...
s_20=39.06, s_21=23.4190
```

Engine 1 lives for 192 cycles. Engine 5 lives for 269 cycles. The last recorded cycle
for each engine is the failure cycle.

### Key Property

Because each engine runs to failure, we know exactly where each engine is in its
lifetime. We define:

```
life_fraction = current_cycle / total_cycles_for_this_engine
```

So `life_fraction = 0.0` means the engine just started, and `life_fraction = 1.0`
means the engine is about to fail. This lets us study how sensor behavior changes
as engines degrade.

---

## 3. Data Preprocessing Pipeline

### Step 1: Sensor Selection

Not all 21 sensors are useful. Our analysis found:

**Zero-variance sensors** (always the same value): s_1, s_5, s_6, s_10, s_16, s_18, s_19.
These carry no information at all. Dropped.

**Weak-correlation sensors** (don't correlate with degradation): s_9, s_14.
Their correlation with `life_fraction` is below 0.45. Dropped.

**Redundancy check:** Among the remaining 12 sensors, many are highly correlated with
each other (correlation > 0.95). Keeping all of them would just add noise without new
information.

**Final selection: 7 sensors**
- Positive degradation trend (increase as engine degrades): s_3, s_4, s_11, s_17
- Negative degradation trend (decrease as engine degrades): s_7, s_12, s_20

These 7 sensors capture the full degradation signal with minimal redundancy.

### Step 2: Train/Validation/Test Split

We split by **engine**, not by individual rows. This is critical.

**Why?** If we split rows randomly, the same engine could have cycles 1-50 in training
and cycles 51-100 in validation. The model would "know" that engine's behavior pattern,
which leaks temporal information and makes validation scores unrealistically good.

Our split: 70 engines for training, 15 for validation, 15 for testing.

### Step 3: Normal-Only Training Region

This is a crucial design decision. We only train the model on the **first 50% of each
engine's life** (life_fraction ≤ 0.5). This means the model ONLY learns what healthy
engines look like. It never sees degraded or failing behavior during training.

**Why?** Because we're building an anomaly detector. The model needs to learn "normal"
so that it can flag "not normal." If we trained on the full lifecycle including failure,
the model would learn to predict failure patterns as normal, and it would never flag them.

This gives us 5,135 training windows from the healthy portion of 70 engines.

### Step 4: Standardization (Z-Score Normalization)

Each sensor operates at a different scale:
- s_3 has mean ≈ 1590 and std ≈ 6
- s_20 has mean ≈ 38.8 and std ≈ 0.18

If we feed raw values to the model, it would be dominated by s_3 (large numbers) and
ignore s_20 (tiny numbers). We fix this with z-score normalization:

```
x_scaled = (x - μ) / σ
```

where μ and σ are computed from training data ONLY. After scaling, every sensor has
mean ≈ 0 and standard deviation ≈ 1.

**Important:** We fit the scaler on training data and apply the same transformation
to validation and test data. We never compute statistics from test data — that would
be data leakage.

### Step 5: Rolling Window Creation

A sequence model needs a window of recent history to make predictions. We use a
sliding window of 30 cycles:

```
Input:  sensor values at cycles [t-29, t-28, ..., t]     → shape: (30, 7)
Target: sensor values at cycle [t+1]                      → shape: (7,)
```

For one engine with 192 cycles, this creates 192 - 30 = 162 windows.

The input is a matrix: 30 rows (time steps) × 7 columns (sensors). Each window
is one training example. The model looks at 30 cycles of history and predicts
what the next cycle's sensor readings will be.

---

## 4. The Probabilistic Model: Gaussian GRU

### What a Standard LSTM Does

A standard LSTM (Long Short-Term Memory) network processes a sequence of inputs and
produces a single output. For our task:

```
Input: 30 cycles of 7 sensor values → (30, 7) matrix
  ↓
LSTM layers process the sequence step by step
  ↓
Output: predicted next-step sensor values → (7,) vector
```

The LSTM has internal "memory cells" that learn to remember relevant past information
and forget irrelevant information. After processing all 30 input steps, its final
hidden state encodes a summary of the entire input sequence.

A standard LSTM trained with MSE loss outputs only **μ** (the predicted mean):

```
μ = Linear(h_final)      # h_final is the last hidden state
```

### What Our Gaussian GRU Does Differently

Our model has **two output heads** instead of one:

```
Input: (30, 7) matrix
  ↓
LSTM layers → final hidden state h
  ↓
Two parallel heads:
  Head 1: μ = Linear(h)                              # predicted mean
  Head 2: σ = softplus(Linear(h)) + 0.0001           # predicted std
```

**μ** is the same as a standard LSTM — the model's best guess for the next sensor values.

**σ** is new — it's the model's estimate of how uncertain it is about that guess. A
larger σ means "I'm less sure about this prediction."

The `softplus` function ensures σ is always positive (you can't have negative
uncertainty). The small constant 0.0001 prevents σ from ever reaching exactly zero
(which would cause mathematical problems during training).

```
softplus(x) = log(1 + exp(x))
```

This function is smooth, always positive, and approximately equal to x when x is large.

### What This Means Conceptually

Instead of predicting "sensor s_3 will read 1590 next cycle," the model predicts
"sensor s_3 will read 1590 ± 6 next cycle." The ±6 part is the uncertainty.

More precisely, the model says: "The next reading of sensor j follows a Gaussian
(normal) distribution:"

```
x_{t+1, j}  ~  Normal(μ_{t+1, j},  σ²_{t+1, j})
```

This is fundamentally more informative than a point prediction because it tells us
not just WHAT the model expects, but HOW CONFIDENT it is.

---

## 5. Training with Gaussian NLL Loss

### Why Not MSE?

A standard LSTM is trained with Mean Squared Error:

```
MSE = (1/d) Σ_j (x_j - μ_j)²
```

This only penalizes the model for being wrong. It doesn't use σ at all, so there's
no reason for the model to learn meaningful uncertainty.

### The Gaussian Negative Log-Likelihood (NLL) Loss

We train with NLL instead. The idea: given that our model predicts a Gaussian
distribution for each sensor, we ask "how likely is the true observation under
the predicted distribution?"

The probability density of observing x under Normal(μ, σ²) is:

```
p(x | μ, σ) = (1 / (σ√(2π))) × exp(-(x - μ)² / (2σ²))
```

Taking the negative log (because we want to minimize, not maximize):

```
NLL = -log(p(x | μ, σ))
    = log(σ) + (x - μ)² / (2σ²) + 0.5×log(2π)
```

The last term is a constant and doesn't affect optimization. For d sensors:

```
Loss = (1/d) Σ_j [ log(σ_j) + (x_j - μ_j)² / (2σ_j²) ]
```

### Why NLL Is Better: A Concrete Example

Suppose the true sensor reading is x = 5.0.

**Model A** predicts μ = 3.0, σ = 0.1 (very confident, but wrong):
```
NLL_A = log(0.1) + (5 - 3)² / (2 × 0.01) = -2.30 + 200 = 197.7
```

**Model B** predicts μ = 3.0, σ = 3.0 (honestly uncertain):
```
NLL_B = log(3.0) + (5 - 3)² / (2 × 9) = 1.10 + 0.22 = 1.32
```

Model A has the same mean prediction but is punished MUCH more harshly because it
was overconfident (small σ) while being wrong. Model B admitted its uncertainty and
is penalized far less.

This creates a powerful incentive: the model MUST learn accurate uncertainty.
If it predicts small σ, it better be right. If it might be wrong, it should predict
large σ.

### The Two-Term Tug-of-War

The NLL loss has two competing terms:

1. **log(σ)** — this REWARDS small σ (the model wants to be confident)
2. **(x - μ)² / (2σ²)** — this PUNISHES small σ when the prediction is wrong

The model finds the optimal balance: σ should be small when predictions are accurate,
and large when predictions are unreliable. This is exactly the calibrated uncertainty
we need.

---

## 6. From Predictions to Anomaly Scores

### The Basic Idea

Once the model is trained on normal data, we run it on new data (including potentially
abnormal data). At each time step:

1. The model sees the last 30 cycles
2. It predicts μ and σ for the next step
3. We observe the actual next step x
4. We compute how "surprising" x is given the prediction

If the engine is healthy, x should be close to μ (within the predicted uncertainty σ).
If something is wrong, x will deviate significantly from μ.

### The NLL Anomaly Score

The simplest anomaly score is the NLL itself:

```
Score_t = Σ_j [ log(σ_{t,j}) + (x_{t,j} - μ_{t,j})² / (2σ_{t,j}²) ]
```

Higher score = more anomalous (the observation was unlikely under the predicted
distribution).

### Why This Is Better Than Raw Error

Consider two scenarios where the prediction error is identical:

**Scenario 1:** Sensor s_3 (noisy sensor, σ = 2.0), error = 3.0
```
NLL contribution = log(2.0) + 9 / 8 = 0.69 + 1.125 = 1.82
```

**Scenario 2:** Sensor s_11 (quiet sensor, σ = 0.3), error = 3.0
```
NLL contribution = log(0.3) + 9 / 0.18 = -1.20 + 50.0 = 48.8
```

Same error, but the NLL score is 27× higher for the quiet sensor. This is correct:
a 3-unit deviation in a sensor that normally varies by ±0.3 is much more alarming
than the same deviation in a sensor that normally varies by ±2.0. The probabilistic
score automatically weights by the sensor's expected reliability.

### Score Normalization

To make scores interpretable across different models, we z-normalize them:

```
Normalized_Score = (Score - μ_normal) / σ_normal
```

where μ_normal and σ_normal are the mean and std of scores computed on normal
validation data. After this, a score of 0 means "typical normal behavior," and a
score of 2.0 means "2 standard deviations more anomalous than typical."

### Thresholds

We set detection thresholds from the normal validation data:
- **95th percentile:** flags the top 5% most unusual normal behavior
- **97.5th percentile:** flags the top 2.5%
- **99th percentile:** flags the top 1%

Anything scoring above the threshold on test data is flagged as potentially anomalous.

---

## 7. The Problem with Standard Scoring

### What Works

The NLL-based anomaly detector works excellently for anomalies that increase the
prediction error:

| Anomaly Type | What Happens | NLL Response | ROC-AUC |
|---|---|---|---|
| Spike | Sudden jump in one sensor | Large residual → High NLL | 0.91 |
| Drop | Sudden decrease | Large residual → High NLL | 0.90 |
| Persistent Offset | Sensor shifts to new level | Sustained residuals → High NLL | 0.92 |
| Noise Burst | Increased variance | Scattered large residuals | 0.78 |

### What Fails Completely

| Anomaly Type | What Happens | NLL Response | ROC-AUC |
|---|---|---|---|
| **Sensor Freeze** | Sensor stuck at one value | Residuals near zero → **Low NLL** | **0.44** |

A ROC-AUC of 0.44 is BELOW random chance (0.50). The detector actively mis-classifies
frozen sensors as extra-normal.

### Why Sensor Freeze Breaks NLL

When a sensor freezes, it repeats the same value every cycle. If the model's prediction
happens to be close to that value (which is likely, since the frozen value is typically
close to the recent average), then:

```
x_frozen ≈ μ_predicted
residual = x_frozen - μ ≈ 0
NLL contribution ≈ log(σ) + 0 / (2σ²) = log(σ)
```

The NLL is minimal! The detector thinks "great prediction, nothing wrong here."

But in reality, a sensor producing the EXACT same reading 20 times in a row is
extremely suspicious. Normal sensors have measurement noise — they fluctuate. A
perfectly constant reading is a sign of malfunction.

### The Deeper Issue

Standard anomaly detection asks only ONE question: "Is the observation far from
expected?" This is a one-directional test. It catches things that are too far,
but it cannot catch things that are too close.

**This is the gap that URD fills.**

---

## 8. URD: Uncertainty-Residual Decomposition

### The Core Idea

Instead of a single anomaly score, URD decomposes the signal into **three orthogonal
channels**, each measuring a different kind of statistical violation:

| Channel | Question | What It Detects |
|---|---|---|
| **Deviation (D)** | "Is this too far from expected?" | Spikes, drops, offsets |
| **Uncertainty (U)** | "Is the model unusually uncertain?" | Drift, degradation |
| **Stationarity (S)** | "Has variability collapsed?" | Sensor freeze, flatline |

Together, these three channels provide a complete picture. Standard detectors only
have Channel 1. Channel 2 is only possible with a probabilistic model. Channel 3
is the novel contribution — it requires the probabilistic model's σ to work.

### The Normalized Residual: The Foundation

Everything in URD starts from the **normalized residual**:

```
r_{t,j} = (x_{t,j} - μ_{t,j}) / σ_{t,j}
```

This is the raw prediction error divided by the predicted uncertainty. It answers:
"How many predicted standard deviations away is the observation?"

Under normal conditions (model well-trained, engine healthy), these normalized
residuals should follow a standard normal distribution:

```
r_{t,j}  ~  N(0, 1)
```

This means:
- About 68% of r values fall between -1 and +1
- About 95% fall between -2 and +2
- The mean is 0 (no systematic bias)
- The variance is 1 (properly calibrated uncertainty)

**Every kind of anomaly violates one of these properties.**

---

## 9. Channel 1: Deviation (D)

### What It Measures

The Deviation channel measures how far the observations are from the predictions,
normalized by the model's uncertainty:

```
D_t = (1/d) × Σ_{j=1}^{d}  r_{t,j}²
    = (1/d) × Σ_{j=1}^{d}  [(x_{t,j} - μ_{t,j})² / σ_{t,j}²]
```

where d = 7 (number of sensors).

### Statistical Properties Under Normal Conditions

Each r_{t,j}² follows a **chi-squared distribution** with 1 degree of freedom.
A chi-squared(1) random variable has:
- Mean = 1
- Variance = 2

Since D_t is the average of d such variables:
- E[D_t] = 1 (expected value under normal)
- Var[D_t] = 2/d

For d = 7: E[D_t] = 1, std(D_t) ≈ √(2/7) ≈ 0.53.

After z-normalization, D has mean 0 and std 1 on normal data.

### Worked Example

**Normal cycle:**
```
Sensor   x_actual    μ_pred    σ_pred    r = (x-μ)/σ    r²
s_3      1591.5      1590.5    6.0       0.167           0.028
s_4      1410.0      1408.8    9.0       0.133           0.018
s_11     47.6        47.5      0.27      0.370           0.137
s_17     393.0       393.2     1.5       -0.133          0.018
s_7      553.0       553.4     0.9       -0.444          0.198
s_12     521.0       521.4     0.7       -0.571          0.327
s_20     38.8        38.8      0.18      0.000           0.000

D = (0.028 + 0.018 + 0.137 + 0.018 + 0.198 + 0.327 + 0.000) / 7
  = 0.726 / 7
  = 0.104
```

This is well below E[D] = 1. Perfectly normal.

**Spike cycle (s_3 jumps by 30 units):**
```
Sensor   x_actual    μ_pred    σ_pred    r = (x-μ)/σ    r²
s_3      1620.0      1590.5    6.0       4.917           24.17   ← HUGE
s_4      1410.0      1408.8    9.0       0.133           0.018
s_11     47.6        47.5      0.27      0.370           0.137
... (others same as before)

D = (24.17 + 0.018 + 0.137 + 0.018 + 0.198 + 0.327 + 0.000) / 7
  = 24.87 / 7
  = 3.55
```

D jumps from 0.10 to 3.55 — far above the normal expected value of 1.0.
After z-normalization, this would be around 4-5 standard deviations above normal.
The Deviation channel fires strongly.

---

## 10. Channel 2: Uncertainty (U)

### What It Measures

The Uncertainty channel tracks the model's own confidence level over time:

```
U_t = (1/d) × Σ_{j=1}^{d}  [σ_{t,j} / σ_ref,j]
```

where σ_ref,j is the **median predicted σ** for sensor j across all normal validation
data. This is the model's "baseline" uncertainty for each sensor.

### Interpretation

- U_t = 1.0: The model is exactly as confident as usual
- U_t = 1.5: The model is 50% more uncertain than usual
- U_t = 0.8: The model is slightly more confident than usual

### Why This Channel Exists

When the Gaussian GRU sees input patterns that are different from what it learned
during training (which was healthy-only data), its internal hidden states reflect
this unfamiliarity. The σ head, trained to output calibrated uncertainty, responds
by producing larger σ values.

This happens BEFORE the engine's sensor values deviate dramatically. The model
"senses" something is off in the temporal patterns even when individual readings
are still within normal ranges.

### Worked Example: Gradual Degradation

Consider an engine going through gradual degradation:

```
Cycle    life_frac    σ_s3    σ_s4    ...    Mean σ    σ_ref    U
50       0.26         5.8     8.5     ...    0.49      0.51     0.96
100      0.52         6.0     8.9     ...    0.51      0.51     1.00
150      0.78         6.5     9.8     ...    0.56      0.51     1.10
180      0.94         7.2     11.0    ...    0.63      0.51     1.24
190      0.99         8.0     12.5    ...    0.71      0.51     1.39
```

The model becomes progressively more uncertain as the engine degrades. U rises
from 0.96 (slightly below baseline) to 1.39 (39% more uncertain than normal).

This gradual rise in U is the signature of drift — the model recognizes
unfamiliar patterns developing.

### What Makes This Channel Unique

**A deterministic model cannot produce this signal.** If you train a standard LSTM
with MSE loss, it outputs σ = 1.0 always (a dummy constant). U would be 1.0
forever, regardless of whether the engine is healthy or failing. You'd lose all
drift-detection capability.

---

## 11. Channel 3: Stationarity (S)

### The Core Insight

Standard residual-based detectors look for values that are **too far** from the model
prediction. Sensor freeze is different. Once a sensor becomes stuck, the model often
predicts the stuck value well, so the residual can become *small* instead of large.

That means freeze is better detected in the **raw dynamics** than in residual magnitude.
The practical signature is not “big error”, but “the signal stopped moving.”

### The Tuned Stationarity Score

For each sensor we compute first-difference energy over a short window of length `w=5`:

```
FDE_t(j) = (1/w) Σ_{i=t-w+1}^{t} (x_{i,j} - x_{i-1,j})^2
```

This is then compared to a healthy reference level `FDE_ref,j` estimated from validation
trajectories. If the energy collapses far below its healthy level, the stationarity score
rises.

The deployed baseline also adds a run-length bonus:

```
S_t = FDE_score(t) + 3 * max(0, run_t - 1)
```

where `run_t` is the longest current sequence of nearly unchanged values across sensors.
The tuned parameters mean the bonus starts after **two nearly identical consecutive steps**
and then grows quickly for sustained flatlines.

### Why This Works Better Than Residual-Only Freeze Detection

A frozen sensor often has:
- low residual magnitude, because the model predicts the repeated value
- normal-looking uncertainty, because the input sequence itself looks stable
- but **very low first-difference energy** and **long run lengths**

This is exactly why the stationarity channel is computed on **raw sensor values**, not on
residuals alone.

### Latest Empirical Evidence

In the current baseline comparison:

- NLL freeze ROC-AUC = **0.4398**
- URD baseline freeze ROC-AUC = **0.8230**
- NLL freeze PR-AUC = **0.0347**
- URD baseline freeze PR-AUC = **0.4467**

So the stationarity channel is not a small tweak. It is the main reason the framework can
reliably detect frozen or flatlined sensors. The repo now also includes a practical TranAD-style
transformer baseline trained on the same FD001 split and healthy-only windows so this claim can be
checked against a strong external TSAD family inside the same experiment pipeline.

## 12. How the Three Channels Combine

### The Combined Score

For anomaly detection, the current baseline uses a **weighted fusion** of the normalised
deviation and stationarity channels:

```
Combined_t = 0.35 * D_t + 0.65 * S_t
```

This weighting came from the upgrade search on synthetic validation data. It preserves the
broad anomaly sensitivity of deviation while giving extra emphasis to the stationarity
channel, which is the key for sensor freeze.

### The Uncertainty Channel's Role

U is still **not** part of the deployed detection score. It is retained as a characterization
signal used downstream for drift-vs-anomaly classification and fingerprinting:

- High D + low U → sharp surprise anomaly
- Moderate D + high U → drift / regime shift
- Low D + high S → sensor malfunction / freeze

### The Anomaly Signature Table

Each anomaly type produces a distinct pattern across the three channels:

```
Type                  D (Deviation)    U (Uncertainty)    S (Stationarity)
────────────────────────────────────────────────────────────────────────────
Spike / Drop          VERY HIGH        LOW (~1.0)         LOW (~0)
Persistent Offset     HIGH             LOW-MODERATE       LOW (~0)
Noise Burst           HIGH             MODERATE           LOW (~0)
Sensor Freeze         LOW-MODERATE     LOW (~1.0)         VERY HIGH
Gradual Drift         MODERATE         HIGH (>1.2)        LOW (~0)
Accelerating Drift    RISING FAST      HIGH (>1.3)        LOW (~0)
```

This table is the foundation of both drift/anomaly classification (Stage 2) and
anomaly fingerprinting (Stage 3).

---

## 13. Stage 2: Drift vs Anomaly Classification

### The Problem

The first stage (anomaly detection with URD combined score) flags unusual events.
But not all flagged events are the same. Some are real faults that need immediate
attention. Others are gradual drift that should be monitored but doesn't require
emergency action.

A standard anomaly detector treats them identically. Our second stage distinguishes them.

### Feature Extraction

For each flagged event, we extract **16 features** from the score trajectory and
URD channels. These features capture the temporal and spatial structure of the event.

**Standard features (1-9):** computed from the anomaly score trajectory

| # | Feature | What It Captures |
|---|---|---|
| 1 | max_score | Peak severity of the event |
| 2 | mean_score | Average severity over the analysis window |
| 3 | score_slope | Is the score rising (drift) or spiking (anomaly)? |
| 4 | score_curvature | Sharp peak (anomaly) vs gradual curve (drift)? |
| 5 | score_volatility | Jagged score (anomaly) vs smooth (drift)? |
| 6 | duration | How long the elevated score persists |
| 7 | sensor_concentration | One sensor (anomaly) or many (drift)? |
| 8 | num_sensors_flagged | Count of simultaneously abnormal sensors |
| 9 | max_single_sensor | Largest single-sensor deviation |

**URD features (10-16):** only available from the probabilistic model

| # | Feature | What It Captures |
|---|---|---|
| 10 | deviation_at_peak | D channel value at the event center |
| 11 | uncertainty_at_peak | U channel value — was the model surprised? |
| 12 | stationarity_at_peak | S channel value — frozen sensor? |
| 13 | uncertainty_slope | Is U rising (drift) or flat (anomaly)? |
| 14 | stationarity_max | Peak stationarity in the analysis window |
| 15 | D/U ratio | High ratio = surprise (anomaly), low = expected (drift) |

### The D/U Ratio: Why It Works

This feature deserves special attention. It's the ratio of Deviation to Uncertainty:

```
D/U ratio = Deviation_at_peak / Uncertainty_at_peak
```

For a **sudden anomaly** (spike, drop):
- D is high (large unexpected deviation)
- U is low (model had no warning, was confident before the event)
- D/U ratio is HIGH (e.g., 20/1.0 = 20)

For **drift**:
- D may also be high (deviation is real)
- But U is ALSO high (model recognizes unfamiliar patterns)
- D/U ratio is MODERATE (e.g., 5/1.3 = 3.8)

This single feature captures whether the model was "surprised" or "prepared" for the
deviation. Surprise = anomaly. Preparation = drift.

### The Classifier

We train a lightweight classifier (Logistic Regression, Random Forest, or XGBoost) on
these 16 features. The training data comes from synthetic anomalies and drifts
injected into validation engine trajectories.

Input: 16-dimensional feature vector
Output: "anomaly" or "drift"

**Why a lightweight classifier?** If even a simple model can distinguish the two
classes, it proves the features are doing the real work — which means the URD
decomposition is the actual contribution, not the classifier complexity.

### Results Summary

| Configuration | Accuracy | Drift Misclassified as Anomaly |
|---|---|---|
| 9 features (no probabilistic info) | 90.9% | 8.6% |
| 12 features (original probabilistic) | 92.1% | 5.4% |
| **16 features (URD, RF best)** | **95.5%** | **2.9%** |

The URD features reduce the drift-as-anomaly rate sharply relative to weaker feature sets. In the latest run, logistic regression reaches 91.3% accuracy with URD features, Random Forest reaches the best overall accuracy at 95.5% with only 2.9% of drift events misclassified as anomalies, and XGBoost remains very close at 95.1%.

---

## 14. Stage 3: Anomaly Fingerprinting

### The Idea

The previous stage distinguishes drift from anomaly. But we can go further: the
(D, U, S) signature is distinct enough to identify the **specific type** of anomaly or drift pattern.

This is like a doctor who doesn't just say "you're sick" but says "you have the flu"
vs "you have a broken bone" vs "you have food poisoning" — based on the pattern of
symptoms.

### The Anomaly Types as "Diseases" with Different "Symptom Profiles"

```
"Disease" (anomaly type)     "Symptoms" (D, U, S)
───────────────────────────────────────────────────
Spike                        D=HIGH,  U=LOW,   C=LOW
Drop                         D=HIGH,  U=LOW,   C=LOW
Persistent offset            D=HIGH,  U=LOW,   C=LOW
Noise burst                  D=HIGH,  U=MED,   C=LOW
Sensor freeze                D=LOW,   U=LOW,   C=HIGH
Gradual drift                D=MED,   U=HIGH,  C=LOW
Sigmoid drift                D=MED,   U=HIGH,  C=LOW
Accelerating drift           D=HIGH,  U=HIGH,  C=LOW
Multi-sensor drift           D=MED,   U=HIGH,  C=LOW
```

### The Classifier

We train a multi-class Random Forest on the same 16 URD features, but now with
9 classes instead of 2. The training data labels each event with its specific type
(spike, drop, persistent_offset, etc.).

### Why This Is Novel

No existing time-series anomaly detection method provides automatic anomaly taxonomy
from a single model's output. Existing methods:
- Detect anomalies: yes/no (binary)
- Some distinguish point anomalies from collective anomalies

Our method automatically categorizes INTO specific types using the residual structure.
This is possible because the three URD channels capture fundamentally different
physical phenomena, and different anomaly types disturb different combinations of
these phenomena.

---

## 15. The Complete Pipeline

### End-to-End Flow

```
Raw Sensor Data
    │
    ▼
[Preprocessing]
    │  • Select 7 informative sensors
    │  • Normalize with z-score (fit on training only)
    │  • Create sliding windows of 30 cycles
    │
    ▼
[Gaussian GRU]  ← trained on normal data only (first 50% of engine life)
    │
    │  Outputs: μ (predicted mean) and σ (predicted uncertainty)
    │           for each sensor at each time step
    │
    ▼
[URD Decomposition]
    │
    ├── Channel D: Deviation = mean of squared normalized residuals
    │   "How wrong is the prediction?"
    │
    ├── Channel U: Uncertainty = σ_current / σ_reference
    │   "How confident is the model?"
    │
    ├── Channel S: Stationarity = tuned FDE + run-length
    │   "Has variability collapsed?"
    │
    ▼
[Combined Detection]
    │  Combined Score = 0.35 D + 0.65 S
    │  Flag if Combined > threshold
    │
    │  At this point we know: "something unusual is happening"
    │
    ▼
[Feature Extraction]  ← for each flagged event
    │  Extract 16 features from (D, U, S) trajectories
    │  and score patterns around the event
    │
    ▼
[Stage 2: Drift vs Anomaly Classifier]
    │  Random Forest on 16 features
    │  Output: "anomaly" or "drift"
    │
    │  If anomaly:
    │      ▼
    │  [Stage 3: Fingerprint Classifier]
    │      Random Forest on 16 features
    │      Output: specific type (spike, freeze, offset, etc.)
    │
    ▼
Final Output:
    • "Normal" — no action needed
    • "Drift" — monitor, schedule maintenance
    • "Anomaly: spike in sensor s_3" — investigate immediately
    • "Anomaly: sensor s_7 frozen" — replace sensor
```

### What Each Stage Requires

| Stage | What It Needs | What It Produces |
|---|---|---|
| Gaussian GRU | Normal training data | μ and σ predictions |
| URD Decomposition | μ, σ, and actual observations | D, U, S scores |
| Detection | 0.35 D + 0.65 S, threshold | Binary flag: normal/abnormal |
| Drift Classifier | 16 URD features per event | anomaly vs drift |
| Fingerprinting | 16 URD features per event | specific anomaly type |

---

## 16. Experimental Design and Evaluation

### Why Synthetic Anomalies?

The C-MAPSS dataset contains real degradation but no labeled point anomalies. We
can't evaluate an anomaly detector without knowing where the anomalies are. So we
create them synthetically:

**Anomaly types we inject:**
1. **Spike:** Add a large value to one sensor for 1 cycle
2. **Drop:** Subtract a large value for 1 cycle
3. **Persistent offset:** Shift a sensor by a fixed amount for 5-15 cycles
4. **Noise burst:** Increase sensor variance for 5-15 cycles
5. **Sensor freeze:** Lock a sensor at a constant value for 5-20 cycles

**Drift types we inject:**
1. **Gradual shift:** Linear ramp over 30-100 cycles
2. **Sigmoid plateau:** Smooth S-curve transition to a new level
3. **Accelerating drift:** Quadratic ramp (slow start, fast later)
4. **Multi-sensor drift:** 2-3 sensors shift together

Each injection is applied to a COPY of a test engine trajectory. The original data
is never modified. We record exactly where each injection starts, how long it lasts,
which sensors are affected, and what type it is.

### Evaluation Protocol

**Stage C (detection):** For each injected trajectory, we compute anomaly scores
and evaluate whether the injected points are flagged. We measure:
- ROC-AUC and PR-AUC (threshold-independent)
- Precision, recall, F1 at fixed thresholds
- Event recall (was the injected event detected at all?)
- Detection delay (how many cycles after onset until detection?)

We run this TWICE: once with NLL-only scoring, once with URD combined scoring.
The comparison shows the improvement from bidirectional detection.

**Stage D (drift classification):** For events flagged by the detector, we extract
16 features and classify them as anomaly or drift. We test three ablation
configurations:
- 9 features (no probabilistic information)
- 12 features (original probabilistic features)
- 16 features (URD features)

Each configuration is tested with 3 classifiers (Logistic Regression, Random Forest,
XGBoost), producing a 3×3 ablation table.

**Stage E (fingerprinting):** We train a multi-class classifier on all anomaly and
drift types simultaneously, using the 16 URD features. We evaluate with a multi-class
confusion matrix and per-type precision/recall.

### Key Results

**Detection (Stage C):**
- NLL-only: Overall ROC-AUC = 0.72, sensor freeze ROC-AUC = 0.44
- URD combined: sensor freeze detection should improve dramatically

**Drift Classification (Stage D):**
- Without URD features: 88% accuracy, 11% drift-as-anomaly rate
- With URD features: 90% accuracy, 6% drift-as-anomaly rate

**Degradation Sanity Check:**
- Anomaly scores correlate with engine life: Spearman ρ = 0.67
- 100% of test engines show positive score-life correlation
- Kruskal-Wallis test: p = 9.27 × 10^{-247} (overwhelmingly significant)
- Model uncertainty also increases with degradation: ρ = 0.47

---

## 17. Summary of Contributions

### Contribution 1: Bidirectional Anomaly Detection
Standard anomaly detectors only look for observations that deviate FROM predictions.
URD also detects observations that conform TOO WELL to predictions. The Conformity
channel uses chi-squared statistics on normalized residuals to catch sensor freeze
and flatline anomalies that are invisible to standard methods.

### Contribution 2: Drift-Aware Anomaly Classification
The three URD channels produce features that distinguish sudden faults from gradual
drift. The D/U ratio — deviation relative to model uncertainty — captures whether
the model was "surprised" (anomaly) or "prepared" (drift). This reduces
drift-as-anomaly misclassification by 44%.

### Contribution 3: Anomaly Fingerprinting
The (D, U, S) signature profile is distinct for each anomaly type, enabling automatic
identification of WHAT went wrong — not just that something did. This is a capability
no existing time-series anomaly detection method provides.

### Why the Probabilistic Model Is Essential
All three contributions REQUIRE the probabilistic output (σ):
- Channel 1 (D) uses σ to properly weight residuals across sensors
- Channel 2 (U) IS the σ trajectory — impossible without it
- Channel 3 (C) requires σ to define what "normal residual magnitude" should be

A deterministic model (MSE-trained, σ = constant) can only use Channel 1 in a
degraded form. It cannot produce Channels 2 or 3. The entire URD framework is
fundamentally enabled by the probabilistic approach.

---

*End of document.*


## 18. Detailed update on the current baseline and TranAD comparison

The earlier sections explain the original logic of the project. This section adds the newer mathematical refinements that now define the deployed baseline.

### 18.1 What exactly changed?

Originally, the project’s detector was closest to a `max(D,S)` score with an older deviation channel. The current repository now uses:

\[
A_t = 0.35\widetilde D_t + 0.65\widetilde S_t
\]

where the new pieces are:
- calibrated sigma before residual scoring,
- Mahalanobis energy on calibrated normalised residuals for \(D\),
- the same general stationarity philosophy for \(S\), but with tuned run/FDE settings,
- uncertainty \(U\) retained for interpretation and downstream classifiers.

### 18.2 Detailed mathematics of the deployed baseline

The Gaussian GRU predicts a mean and standard deviation for every next-step sensor value:

\[
\mu_t, \sigma_t \in \mathbb{R}^7
\]

Healthy validation windows are used to fit a per-sensor calibration temperature. First compute raw normalised residuals:

\[
 r^{raw}_{t,j} = rac{x_{t,j}-\mu_{t,j}}{\sigma_{t,j}}
\]

Then fit

\[
	au_j = \sqrt{rac{1}{N}\sum_t (r^{raw}_{t,j})^2}
\]

and define

\[
\sigma^{eff}_{t,j} = 	au_j\sigma_{t,j}
\]

This gives the calibrated residual vector

\[
 r_t = \left(rac{x_{t,1}-\mu_{t,1}}{\sigma^{eff}_{t,1}},\ldots,rac{x_{t,7}-\mu_{t,7}}{\sigma^{eff}_{t,7}}ight)^	op
\]

The new deviation channel is

\[
D_t = r_t^	op\Sigma_r^{-1}r_t
\]

where \(\Sigma_r\) is the covariance of healthy validation residuals. This is then standardised to \(\widetilde D_t\).

The uncertainty channel uses the calibrated sigma values:

\[
U_t = rac{1}{7}\sum_{j=1}^{7}rac{\sigma^{eff}_{t,j}}{\sigma_j^{ref}}, \qquad \sigma_j^{ref}=\operatorname{median}(\sigma^{eff}_{t,j}	ext{ on val})
\]

The stationarity channel is still computed from raw targets rather than residuals. With window size 5:

\[
\Delta x_{t,j}=x_{t,j}-x_{t-1,j}, \qquad \operatorname{FDE}_{t,j}=rac{1}{5}\sum_{k=t-4}^{t}(\Delta x_{k,j})^2
\]

\[
S_t^{fde}=\max_j\max\left(0,-\lograc{\operatorname{FDE}_{t,j}}{\operatorname{FDE}_j^{ref}+arepsilon}ight)
\]

The run bonus is

\[
3\max(0,run_t-1)
\]

so

\[
S_t = S_t^{fde} + 3\max(0,run_t-1)
\]

and then normalised to \(\widetilde S_t\).

The final anomaly score is

\[
A_t = 0.35\widetilde D_t + 0.65\widetilde S_t
\]

### 18.3 Why these refinements helped

- **Sigma calibration** makes residual scaling trustworthy.
- **Mahalanobis D** captures unusual multivariate combinations of residuals.
- **Weighted fusion** prevents stationarity from being ignored when deviation is only moderate.
- **Raw-signal S** still preserves the project’s main freeze-detection advantage.

### 18.4 Matched TranAD baseline

The project now includes a practical TranAD-style comparator under the same data protocol. It uses a two-pass transformer next-step predictor. If the first-pass prediction is \(\hat y_t^{(1)}\), then the model constructs a causal self-conditioning focus term from disagreement with the latest observed step and uses it in the second pass to obtain \(\hat y_t^{(2)}\). The final anomaly score is the validation-normalised next-step MSE:

\[
score_t^{TranAD} = rac{rac{1}{d}\sum_j (y_{t,j}-\hat y_{t,j}^{(2)})^2 - \mu_{val}}{\sigma_{val}}
\]

This makes the comparison fair: same engines, same sensors, same windows, same healthy-only training region, different anomaly-detection philosophy.

### 18.5 Latest empirical snapshot

**Stage C: URD baseline vs TranAD**

| Metric | URD baseline | TranAD |
|---|---:|---:|
| Overall ROC-AUC | **0.8636** | 0.7379 |
| Overall PR-AUC | **0.4250** | 0.2475 |
| Freeze ROC-AUC | **0.8230** | 0.4621 |
| Freeze PR-AUC | **0.4467** | 0.0362 |
| p99 precision | **0.356** | 0.129 |
| p99 false alarms / 1000 windows | **33.8** | 110.3 |

**Stage D: drift vs anomaly**

- Logistic regression URD-16: **0.913**
- Random Forest URD-16: **0.955**
- XGBoost URD-16: **0.951**

**Stage E: fingerprinting**

- 5-class actionable: **0.900**
- 9-class per-type: **0.631**
- Spike vs Drop with signed deviation: **0.950**
- Without signed deviation: **0.567**
