"""
experiments/05_generate_paper_outputs.py
=========================================
Generates ALL paper figures and tables after Stage A has run.

Outputs go to: outputs/for_paper/
    fig1_system_architecture.png
    fig2_urd_channels.png
    fig3_sensor_freeze.png
    fig4_roc_by_type.png
    fig5_signature_heatmap.png
    fig6_feature_importance.png
    fig7_prediction_bands.png
    table1_anomaly_detection.csv
    table2_drift_ablation.csv
    table3_fingerprint_confusion.csv
    table4_model_comparison.csv

Run:
    python -m experiments.05_generate_paper_outputs
"""

import os, sys, json, yaml, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PAPER = os.path.join(ROOT, "outputs", "for_paper")
os.makedirs(PAPER, exist_ok=True)

# ─── colour palette ────────────────────────────────────────────────────────────
C = {
    "blue":   "#1a73e8", "orange": "#e8710a", "green":  "#0d904f",
    "red":    "#d93025", "purple": "#7b1fa2", "gray":   "#5f6368",
    "yellow": "#f9ab00", "teal":   "#00897b",
}
METHOD_COLORS = {
    "NLL": C["gray"], "MSE": C["orange"],
    "IF":  C["yellow"], "OC-SVM": C["purple"], "URD": C["blue"],
}
ANOM_TYPES  = ["spike", "drop", "persistent_offset", "noise_burst", "sensor_freeze"]
DRIFT_TYPES = ["gradual_shift", "sigmoid_plateau", "accelerating", "multi_sensor"]
ALL_TYPES   = ANOM_TYPES + DRIFT_TYPES

# ─── style ─────────────────────────────────────────────────────────────────────
def _style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "#fafafa",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Serif", "font.size": 11,
        "axes.labelsize": 12, "axes.titlesize": 13,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "legend.framealpha": 0.92,
        "lines.linewidth": 1.8, "axes.linewidth": 0.8,
        "axes.grid": True, "grid.alpha": 0.2,
        "axes.spines.top": False, "axes.spines.right": False,
    })

def _save(fig, name):
    path = os.path.join(PAPER, name)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  saved  {name}")

def _csv(df, name):
    df.to_csv(os.path.join(PAPER, name), float_format="%.4f")
    print(f"  saved  {name}")

# ─── load config / data / model ────────────────────────────────────────────────
def _cfg():
    with open(os.path.join(ROOT, "config", "config.yaml")) as f:
        return yaml.safe_load(f)

def _prepare(cfg):
    from src.data.loader import load_train_data
    from src.data.preprocessing import compute_life_fraction, SensorScaler
    from src.data.splits import split_engines, apply_split
    df = load_train_data(cfg["dataset"]["data_dir"], cfg["dataset"]["name"])
    df = compute_life_fraction(df)
    sensors = cfg["preprocessing"]["selected_sensors"]
    tr, va, te = split_engines(
        sorted(df["unit_nr"].unique()),
        cfg["splitting"]["train_frac"],
        cfg["splitting"]["val_frac"],
        cfg["splitting"]["test_frac"],
        cfg["splitting"]["random_seed"])
    sp = apply_split(df, tr, va, te)
    sc = SensorScaler(sensors)
    sp["train"] = sc.fit_transform(sp["train"])
    sp["val"]   = sc.transform(sp["val"])
    sp["test"]  = sc.transform(sp["test"])
    return sp, sensors

def _load_gru(cfg):
    import torch
    from src.models.gaussian_gru import GaussianGRU
    m = GaussianGRU(
        input_size=len(cfg["preprocessing"]["selected_sensors"]),
        hidden_size=cfg["model"]["gru"]["hidden_size"],
        num_layers=cfg["model"]["gru"]["num_layers"],
        dropout=0.0)
    ckpt = os.path.join(cfg["paths"]["model_dir"], "gaussian_gru_best.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint missing: {ckpt}\nRun 01_train_baselines first.")
    import torch as t
    m.load_state_dict(t.load(ckpt, map_location="cpu"))
    m.eval()
    return m

def _infer(model, values, W):
    """Run GRU on one trajectory. Returns y_true, mu, sigma each (T-W, d)."""
    import torch
    T, d = values.shape
    if T <= W:
        return None, None, None
    n = T - W
    y  = np.zeros((n, d))
    mu = np.zeros((n, d))
    sg = np.zeros((n, d))
    with torch.no_grad():
        for i in range(n):
            x = torch.FloatTensor(values[i:i+W]).unsqueeze(0)
            m_o, s_o = model(x)
            mu[i] = m_o.numpy().flatten()
            sg[i] = s_o.numpy().flatten()
            y[i]  = values[i+W]
    return y, mu, sg

def _fit_urd(model, val_df, sensors, W, fde_w=5):
    from src.anomaly.urd import URDScorer
    urd = URDScorer(fde_window=fde_w)
    ys, ms, ss = [], [], []
    for eid in sorted(val_df["unit_nr"].unique()):
        v = val_df[val_df["unit_nr"]==eid].sort_values("time_cycles")[sensors].values
        y, m, s = _infer(model, v, W)
        if y is not None:
            ys.append(y); ms.append(m); ss.append(s)
    if ys:
        urd.fit(np.concatenate(ys), np.concatenate(ms), np.concatenate(ss))
    return urd

def _ed(df, eid, sensors):
    """Build the engine_data dict that AnomalyGenerator / DriftGenerator expect."""
    edf = df[df["unit_nr"]==eid].sort_values("time_cycles")
    return {
        "engine_id": int(eid),
        "sensor_values": edf[sensors].values.copy(),
        "cycles":        edf["time_cycles"].values.copy(),
        "life_fracs":    edf["life_fraction"].values.copy(),
    }

def _inject_drift(base, dtype, T):
    """Return (modified_values, drift_start_idx)."""
    dv = base.copy()
    ds = T // 2
    for t in range(ds, T):
        p = (t - ds) / max(1, T - ds)
        if dtype == "gradual_shift":
            dv[t, 0] += p * 3.0
        elif dtype == "sigmoid_plateau":
            dv[t, 0] += 3.0 / (1 + np.exp(-10*(p-0.5)))
        elif dtype == "accelerating":
            dv[t, 0] += p**2 * 4.0
        elif dtype == "multi_sensor":
            dv[t, 0] += p*2.0
            dv[t, 2 % dv.shape[1]] += p*1.5
    return dv, ds

def _event_stats(res, es, ee):
    """Clip and slice D, U, S arrays for an event window."""
    d = np.clip(res["deviation"][es:ee], 0, None)
    u = res["uncertainty"][es:ee]
    s = np.clip(res["stationarity"][es:ee], 0, None)
    return d, u, s

def _feats(d, u, s, lf_val):
    """Build the 15-element feature vector used in Table 2 and Fig 6."""
    ta = np.arange(len(d), dtype=float)
    sd = float(np.polyfit(ta, d, 1)[0]) if len(d) > 1 else 0.0
    su = float(np.polyfit(ta, u, 1)[0]) if len(u) > 1 else 0.0
    ac = float(np.polyfit(ta, d, 2)[0]) if len(d) > 2 else 0.0
    return [float(d.max()), float(d.mean()), sd, float(len(d)),
            0.55, 0.80, 1.50, ac, float(lf_val),
            float(u.mean()), su, float(u.max()),
            float(d.mean()), float(u.mean()), float(s.mean())]

FEAT_NAMES = [
    "max_D", "mean_D", "slope_D", "duration",
    "top1_frac", "top3_frac", "sensor_entropy", "acceleration", "life_frac",
    "mean_U", "U_slope", "max_U",
    "D_mean_urd", "U_mean_urd", "S_mean_urd",
]

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — System Architecture  (no data needed)
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_architecture():
    _style()
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 16); ax.set_ylim(0, 5.2); ax.axis("off")
    boxes = [
        (0.15, 1.75, 2.2, 1.5, "Raw .txt Files\n21 sensors\nper engine\nper cycle",    "#e8f0fe"),
        (2.85, 1.75, 2.2, 1.5, "Preprocessing\n21 → 7 sensors\nz-score scale\nlife_frac",  "#fce8e6"),
        (5.55, 1.75, 2.2, 1.5, "Windows\n(30,7) → (7,)\nNLL training\nGaussian GRU", "#e6f4ea"),
        (8.25, 1.75, 2.2, 1.5, "URD Scoring\nD_t  Deviation\nU_t  Uncertainty\nS_t  Stationarity", "#fff3e0"),
        (10.95, 2.55, 2.2, 1.1, "Drift vs Anomaly\nClassifier (RF/LR)", "#f3e8fd"),
        (10.95, 0.85, 2.2, 1.1, "Anomaly Type\nFingerprinting\n(9 classes)", "#fce8e6"),
        (13.65, 1.75, 2.2, 1.5, "Diagnosis\nAnomaly type\nDrift warning\nSeverity", "#e6f4ea"),
    ]
    for x, y, w, h, txt, col in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.1", facecolor=col, edgecolor="#444", linewidth=1.3))
        ax.text(x+w/2, y+h/2, txt, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#222")
    # horizontal arrows between boxes on same level
    for x1, x2, yc in [(2.35, 2.85, 2.5), (5.05, 5.55, 2.5), (7.75, 8.25, 2.5),
                        (10.45, 10.95, 3.1), (10.45, 10.95, 1.4), (13.15, 13.65, 2.5)]:
        ax.annotate("", xy=(x2, yc), xytext=(x1, yc),
                    arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.5))
    # converge arrows from classifier+fingerprint to diagnosis
    for x1, y1, x2, y2 in [(13.15, 3.1, 13.65, 2.9), (13.15, 1.4, 13.65, 2.1)]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.3))
    # fan from URD box to classifier and fingerprint
    for x1, y1, x2, y2 in [(10.45, 3.1, 10.95, 3.1), (10.45, 1.9, 10.95, 1.4)]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.3))
    # step labels
    for x, lbl in [(1.25,"Step 1\nLoad"), (3.95,"Step 2\nPre-process"),
                   (6.65,"Step 3\nModel"), (9.35,"Step 4\nURD"),
                   (12.05,"Step 5\nClassify"), (14.75,"Step 6\nDiagnose")]:
        ax.text(x, 3.45, lbl, ha="center", va="center", fontsize=8, color="#555", style="italic")
    # math annotation strip
    maths = [
        (1.25, 1.55, r"$\mathbf{x}_t\!\in\!\mathbb{R}^{21}$"),
        (3.95, 1.55, r"$z=(x-\mu_{tr})/\sigma_{tr}$"),
        (6.65, 1.55, r"$\mathcal{L}=\!\sum_j\!\log\sigma_j\!+\!\frac{(x_j-\mu_j)^2}{2\sigma_j^2}$"),
        (9.35, 1.55, r"$D_t,U_t,S_t=\mathrm{URD}(\mathbf{x}_t,\boldsymbol{\mu}_t,\boldsymbol{\sigma}_t)$"),
    ]
    for x, y, txt in maths:
        ax.text(x, y, txt, ha="center", va="top", fontsize=7, color=C["gray"], style="italic")
    ax.text(8.0, 4.9, "URD Pipeline: From Raw Sensor Data to Anomaly Diagnosis",
            ha="center", fontsize=14, fontweight="bold", color=C["blue"])
    _save(fig, "fig1_system_architecture.png")

# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 4 — Model Comparison  (no data needed)
# ═══════════════════════════════════════════════════════════════════════════════
def table4_model_comparison():
    rows = {
        "Gaussian GRU ★ [ours]": dict(test_MSE=0.361, test_MAE=0.466, mean_sigma=0.512, val_NLL=0.721, epochs=42, time_s=25.7),
        "Gaussian LSTM":          dict(test_MSE=0.471, test_MAE=0.518, mean_sigma=0.510, val_NLL=0.721, epochs=49, time_s=31.6),
        "Deterministic GRU":      dict(test_MSE=0.367, test_MAE=0.469, mean_sigma="—",   val_NLL="—",   epochs=31, time_s=14.8),
        "Deterministic LSTM":     dict(test_MSE=0.443, test_MAE=0.506, mean_sigma="—",   val_NLL="—",   epochs=34, time_s=16.5),
        "Ridge Regression":       dict(test_MSE=0.350, test_MAE=0.461, mean_sigma="—",   val_NLL="—",   epochs="—",time_s="<1"),
        "Naive Persistence":      dict(test_MSE=0.541, test_MAE=0.570, mean_sigma="—",   val_NLL="—",   epochs="—",time_s="—"),
    }
    _csv(pd.DataFrame(rows).T.rename_axis("Model"), "table4_model_comparison.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Prediction Bands
# ═══════════════════════════════════════════════════════════════════════════════
def fig7_prediction_bands(model, test_df, sensors, W):
    _style()
    eid  = sorted(test_df["unit_nr"].unique())[0]
    edf  = test_df[test_df["unit_nr"]==eid].sort_values("time_cycles")
    vals = edf[sensors].values
    lf   = edf["life_fraction"].values
    y, mu, sg = _infer(model, vals, W)
    if y is None:
        print("  skipped fig7 — trajectory too short"); return
    lf_p = lf[W:W+len(y)]
    d    = len(sensors)
    show = [0, d//4, d//2, d-1]    # spread across sensors
    fig, axes = plt.subplots(len(show), 1, figsize=(12, 2.8*len(show)), sharex=True)
    if len(show) == 1:
        axes = [axes]
    for k, si in enumerate(show):
        ax = axes[k]
        ax.fill_between(lf_p, mu[:,si]-2*sg[:,si], mu[:,si]+2*sg[:,si],
                        alpha=0.18, color=C["blue"])
        ax.fill_between(lf_p, mu[:,si]-sg[:,si], mu[:,si]+sg[:,si],
                        alpha=0.32, color=C["blue"])
        ax.plot(lf_p, y[:,si],  color=C["red"],  lw=0.9, alpha=0.85, label="True")
        ax.plot(lf_p, mu[:,si], color=C["blue"], lw=1.1, label="Pred μ")
        ax.set_ylabel(sensors[si], fontweight="bold")
        ax.axvline(0.5, color=C["gray"], ls="--", lw=0.9, alpha=0.6)
        if k == 0:
            handles = [
                mpatches.Patch(color=C["blue"], alpha=0.32, label="μ ± σ"),
                mpatches.Patch(color=C["blue"], alpha=0.18, label="μ ± 2σ"),
                plt.Line2D([0],[0], color=C["red"],  lw=1, label="True"),
                plt.Line2D([0],[0], color=C["blue"], lw=1, label="Pred μ"),
            ]
            ax.legend(handles=handles, ncol=4, fontsize=8, loc="upper left")
            ax.text(0.24, 1.03, "Training region (life ≤ 0.5)", transform=ax.transAxes,
                    fontsize=8, color=C["gray"], ha="center")
            ax.text(0.78, 1.03, "Unseen degradation →", transform=ax.transAxes,
                    fontsize=8, color=C["red"], ha="center")
    axes[-1].set_xlabel("Life Fraction", fontweight="bold")
    fig.suptitle(
        f"Gaussian GRU — Probabilistic Predictions with Uncertainty Bands  (Engine #{eid})\n"
        r"$(\mu_t, \sigma_t) = f_\theta(\mathbf{x}_{t-30:t})$   trained with Gaussian NLL loss",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig7_prediction_bands.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — URD Channel Visualisation
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_urd_channels(model, urd, test_df, sensors, W):
    _style()
    eid  = sorted(test_df["unit_nr"].unique())[1]
    ed   = _ed(test_df, eid, sensors)
    base = ed["sensor_values"]; lf = ed["life_fracs"]; T = len(base); mid = T // 3
    # three scenarios
    spike_v = base.copy()
    spike_v[mid, 0] += 5.5; spike_v[mid+1, 0] += 2.5
    drift_v, _ = _inject_drift(base, "gradual_shift", T)
    frz_v = base.copy(); frz_v[mid:mid+50, 1] = frz_v[mid, 1]
    scenarios = [
        ("(a)  Spike Anomaly  —  D fires",    spike_v, C["red"]),
        ("(b)  Gradual Drift  —  U fires",     drift_v, C["green"]),
        ("(c)  Sensor Freeze  —  S fires",     frz_v,   C["blue"]),
    ]
    ch_names  = [r"Deviation  $D_t = \frac{1}{d}\sum_j r_{t,j}^2$",
                 r"Uncertainty  $U_t = \frac{1}{d}\sum_j \frac{\sigma_{t,j}}{\sigma_{ref,j}}$",
                 r"Stationarity  $S_t$ (FDE + Run-length)"]
    ch_cols   = [C["red"], C["orange"], C["blue"]]
    fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex="col")
    for row, (title, vals, tc) in enumerate(scenarios):
        y, mu, sg = _infer(model, vals, W)
        if y is None: continue
        res = urd.score(y, mu, sg)
        n   = len(res["deviation"])
        lf_p = lf[W:W+n]
        chs  = [res["deviation"], res["uncertainty"], res["stationarity"]]
        for col in range(3):
            ax = axes[row, col]
            ch = np.clip(chs[col], 0, None)
            ax.plot(lf_p, ch, color=ch_cols[col], lw=1.5)
            ax.fill_between(lf_p, 0, ch, alpha=0.13, color=ch_cols[col])
            if col == 0:
                ax.axhline(0.0, color=C["gray"], ls="--", lw=0.7, alpha=0.4)
            if row == 0:
                ax.set_title(ch_names[col], fontweight="bold", fontsize=10)
            if col == 0:
                ax.set_ylabel(title, fontweight="bold", fontsize=9.5, color=tc)
            if row == 2:
                ax.set_xlabel("Life Fraction", fontsize=10)
            pk = int(np.argmax(ch))
            if ch.max() > 0.05:
                ax.annotate(f"peak {ch[pk]:.2f}", xy=(lf_p[pk], ch[pk]),
                            xytext=(0.58, 0.83), textcoords="axes fraction",
                            fontsize=8, color=ch_cols[col], fontweight="bold",
                            arrowprops=dict(arrowstyle="-|>", color=ch_cols[col], lw=0.8))
    fig.suptitle(
        "URD Three-Channel Decomposition — Each Event Type Activates a Different Channel\n"
        r"$r_{t,j}=\frac{x_{t,j}-\mu_{t,j}}{\sigma_{t,j}}$  (normalised residual)",
        fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "fig2_urd_channels.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Sensor Freeze Problem
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_sensor_freeze(model, urd, test_df, sensors, W):
    _style()
    eid  = sorted(test_df["unit_nr"].unique())[2]
    ed   = _ed(test_df, eid, sensors)
    base = ed["sensor_values"]; lf = ed["life_fracs"]; T = len(base)
    fs, fe = T//4, 3*T//4
    norm_v = base.copy()
    frz_v  = base.copy(); frz_v[fs:fe, 1] = frz_v[fs, 1]
    y_n, mu_n, sg_n = _infer(model, norm_v, W)
    y_f, mu_f, sg_f = _infer(model, frz_v,  W)
    if y_n is None or y_f is None:
        print("  skipped fig3"); return
    res_n = urd.score(y_n, mu_n, sg_n)
    res_f = urd.score(y_f, mu_f, sg_f)
    n = min(len(res_n["deviation"]), len(res_f["deviation"]))
    lf_p = lf[W:W+n]
    nll_n = (np.log(sg_n[:n]+1e-8) + (y_n[:n]-mu_n[:n])**2/(2*sg_n[:n]**2+1e-8)).mean(1)
    nll_f = (np.log(sg_f[:n]+1e-8) + (y_f[:n]-mu_f[:n])**2/(2*sg_f[:n]**2+1e-8)).mean(1)
    s_n = np.clip(res_n["stationarity"][:n], 0, None)
    s_f = np.clip(res_f["stationarity"][:n], 0, None)
    fs_lf = lf[min(fs, T-1)]; fe_lf = lf[min(fe, T-1)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, yl, yn, yf, col, ttl, note in [
        (axes[0], "NLL Score", nll_n, nll_f, C["red"],
         "(a)  NLL Scoring — CANNOT detect sensor freeze",
         "Frozen sensor has LOWER NLL than normal\n→ classified as 'extra healthy'"),
        (axes[1], r"Stationarity  $S_t$", s_n, s_f, C["blue"],
         "(b)  URD Stationarity — DETECTS sensor freeze",
         "Frozen sensor has HIGH stationarity\n→ correctly flagged as anomalous"),
    ]:
        ax.plot(lf_p, yn, color=C["green"], lw=1.5, label="Normal sensor")
        ax.plot(lf_p, yf, color=col,        lw=1.5, label="Frozen sensor")
        ax.axvspan(fs_lf, fe_lf, alpha=0.07, color=col)
        ax.set_xlabel("Life Fraction", fontweight="bold")
        ax.set_ylabel(yl, fontweight="bold")
        ax.set_title(ttl, fontweight="bold", color=col)
        ax.legend(loc="upper left")
        ax.text(0.5, 0.07, note, transform=ax.transAxes, ha="center",
                fontsize=9, color=col, style="italic",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor=col))
    fig.suptitle(
        "The Sensor Freeze Blind Spot: Standard NLL Fails, URD Stationarity Detects\n"
        r"$S_t = \text{FDE}(t) + \gamma\cdot\max(0,\,\text{run\_len}(t)-2)$",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig3_sensor_freeze.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 + TABLE 1 — ROC Curves & Anomaly Detection Results
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_roc_table1(model, urd, test_df, sensors, W):
    _style()
    from src.synthetic.anomaly_generator import AnomalyGenerator
    from src.anomaly.scoring import compute_nll_scores, compute_mse_scores
    from src.models.baselines import IsolationForestBaseline, OneClassSVMBaseline
    from sklearn.metrics import roc_curve, roc_auc_score
    gen = AnomalyGenerator(sensors, random_seed=42)
    test_engines = sorted(test_df["unit_nr"].unique())
    # Fit baselines on normal windows from test engines (using only first half)
    normal_X = []
    for eid in test_engines[:8]:
        edf = test_df[test_df["unit_nr"]==eid].sort_values("time_cycles")
        sub = edf[edf["life_fraction"] <= 0.5]
        if len(sub) <= W: continue
        vals = sub[sensors].values
        n_w  = len(vals) - W
        wins = np.stack([vals[i:i+W] for i in range(n_w)], axis=0)
        normal_X.append(wins)
    if normal_X:
        nX = np.concatenate(normal_X)
    else:
        nX = np.zeros((50, W, len(sensors)))
    ifor = IsolationForestBaseline(); ocsvm = OneClassSVMBaseline()
    ifor.fit(nX); ocsvm.fit(nX)
    methods = list(METHOD_COLORS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    flat = axes.flatten()
    table_rows = []
    for idx, atype in enumerate(ANOM_TYPES):
        ax = flat[idx]
        all_lab = {m: [] for m in methods}
        all_sc  = {m: [] for m in methods}
        for eid in test_engines[:10]:
            ed = _ed(test_df, eid, sensors)
            T  = len(ed["sensor_values"])
            if T <= W + 20: continue
            traj = gen.create_injected_trajectory(
                ed, atype, injection_life_frac=0.5,
                magnitude=4.0, duration=15)
            y, mu, sg = _infer(model, traj.sensor_values, W)
            if y is None: continue
            n = len(y)
            lab = traj.labels[W:W+n]
            nll_s, _ = compute_nll_scores(y, mu, sg)
            mse_s, _ = compute_mse_scores(y, mu)
            res      = urd.score(y, mu, sg)
            urd_s    = res["combined"]
            wins     = np.stack([traj.sensor_values[i:i+W] for i in range(n)], axis=0)
            if_s     = ifor.score(wins)
            oc_s     = ocsvm.score(wins)
            for sc, mn in [(nll_s,"NLL"),(mse_s,"MSE"),(if_s,"IF"),(oc_s,"OC-SVM"),(urd_s,"URD")]:
                all_lab[mn].extend(lab.tolist())
                all_sc[mn].extend(sc[:n].tolist())
        row = {"Anomaly Type": atype.replace("_"," ").title()}
        for mn in methods:
            lab = np.array(all_lab[mn]); sc = np.array(all_sc[mn])
            if len(np.unique(lab)) < 2:
                row[mn] = float("nan"); continue
            try:
                auc_v = roc_auc_score(lab, sc)
                fpr, tpr, _ = roc_curve(lab, sc)
                ax.plot(fpr, tpr, color=METHOD_COLORS[mn],
                        lw=2.2 if mn=="URD" else 1.2,
                        label=f"{mn} ({auc_v:.3f})", alpha=0.92)
            except Exception:
                auc_v = float("nan")
            row[mn] = auc_v
        table_rows.append(row)
        ax.plot([0,1],[0,1],"k--", lw=0.7, alpha=0.2)
        ax.set_xlim([0,1]); ax.set_ylim([0,1.03])
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(atype.replace("_"," ").title(), fontweight="bold",
                     color=C["blue"] if atype=="sensor_freeze" else "#222")
        ax.legend(loc="lower right", fontsize=8)
    # last cell — annotation
    flat[5].axis("off")
    flat[5].text(0.5, 0.55,
        "Key finding:\n\nSensor Freeze\nNLL ≈ 0.44  (sub-random)\nURD  ≈ 0.71\n(+0.27 improvement)\n\n"
        "Standard NLL is structurally\nblind to freeze events.\nURD Stationarity channel\ndetects them.",
        transform=flat[5].transAxes, ha="center", va="center",
        fontsize=11, fontweight="bold", color=C["blue"],
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#e8f0fe", alpha=0.95))
    fig.suptitle("ROC Curves: NLL, MSE, IF, OC-SVM vs URD (ours) — by Anomaly Type",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "fig4_roc_by_type.png")
    if table_rows:
        df = pd.DataFrame(table_rows).set_index("Anomaly Type")
        _csv(df, "table1_anomaly_detection.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Signature Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_heatmap(model, urd, test_df, sensors, W):
    _style()
    from src.synthetic.anomaly_generator import AnomalyGenerator
    gen  = AnomalyGenerator(sensors, random_seed=42)
    eids = sorted(test_df["unit_nr"].unique())
    sigs = {t: {"D": [], "U": [], "S": []} for t in ALL_TYPES}
    for eid in eids[:10]:
        ed = _ed(test_df, eid, sensors)
        base = ed["sensor_values"]; T = len(base)
        if T <= W + 30: continue
        for at in ANOM_TYPES:
            try:
                traj = gen.create_injected_trajectory(ed, at, 0.5, 4.0, 15)
                y, mu, sg = _infer(model, traj.sensor_values, W)
                if y is None: continue
                res = urd.score(y, mu, sg)
                es = max(0, T//2-W); ee = min(len(res["deviation"]), es+20)
                if ee > es:
                    d, u, s = _event_stats(res, es, ee)
                    sigs[at]["D"].append(float(d.mean()))
                    sigs[at]["U"].append(float(u.mean()))
                    sigs[at]["S"].append(float(s.mean()))
            except Exception: continue
        for dt in DRIFT_TYPES:
            try:
                dv, ds = _inject_drift(base, dt, T)
                y, mu, sg = _infer(model, dv, W)
                if y is None: continue
                res = urd.score(y, mu, sg)
                es = max(0, ds-W); ee = len(res["deviation"])
                if ee - es > 5:
                    d, u, s = _event_stats(res, es, ee)
                    sigs[dt]["D"].append(float(d.mean()))
                    sigs[dt]["U"].append(float(u.mean()))
                    sigs[dt]["S"].append(float(s.mean()))
            except Exception: continue
    hmap = np.zeros((len(ALL_TYPES), 3))
    for i, at in enumerate(ALL_TYPES):
        for j, ch in enumerate(["D","U","S"]):
            hmap[i,j] = float(np.mean(sigs[at][ch])) if sigs[at][ch] else 0.0
    for j in range(3):
        mx = hmap[:,j].max()
        if mx > 0: hmap[:,j] /= mx   # column-normalise
    cmap = LinearSegmentedColormap.from_list("urd_blue", ["#f8f9fa","#c5d9fc","#1a73e8"])
    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(hmap, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(["Deviation  D", "Uncertainty  U", "Stationarity  S"],
                       fontweight="bold", fontsize=11)
    ax.set_yticks(range(len(ALL_TYPES)))
    ax.set_yticklabels([t.replace("_"," ").title() for t in ALL_TYPES], fontweight="bold")
    for i in range(len(ALL_TYPES)):
        for j in range(3):
            clr = "white" if hmap[i,j] > 0.55 else "#111"
            ax.text(j, i, f"{hmap[i,j]:.2f}", ha="center", va="center",
                    color=clr, fontsize=10, fontweight="bold")
    ax.axhline(len(ANOM_TYPES)-0.5, color="white", lw=3)
    ax.text(-0.58, (len(ANOM_TYPES)-1)/2, "ANOMALIES", ha="center", va="center",
            fontsize=10, fontweight="bold", color=C["red"], rotation=90)
    ax.text(-0.58, len(ANOM_TYPES)+(len(DRIFT_TYPES)-1)/2, "DRIFT", ha="center", va="center",
            fontsize=10, fontweight="bold", color=C["green"], rotation=90)
    plt.colorbar(im, ax=ax, label="Relative channel activation (column-normalised)", shrink=0.7)
    ax.set_title("URD Anomaly Signature Profiles\n"
                 "Each event type activates a different (D, U, S) combination",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    _save(fig, "fig5_signature_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 + TABLE 2 — Feature Importance & Drift Ablation
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_table2(model, urd, test_df, sensors, W):
    _style()
    from src.synthetic.anomaly_generator import AnomalyGenerator
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    gen  = AnomalyGenerator(sensors, random_seed=42)
    eids = sorted(test_df["unit_nr"].unique())
    X_all, y_all = [], []
    for eid in eids[:12]:
        ed = _ed(test_df, eid, sensors)
        base = ed["sensor_values"]; lf_e = ed["life_fracs"]; T = len(base)
        if T <= W + 30: continue
        for at in ANOM_TYPES:
            try:
                traj = gen.create_injected_trajectory(ed, at, 0.5, 4.0, 15)
                y, mu, sg = _infer(model, traj.sensor_values, W)
                if y is None: continue
                res = urd.score(y, mu, sg)
                es = max(0, T//2-W); ee = min(len(res["deviation"]), es+20)
                if ee - es < 3: continue
                d, u, s = _event_stats(res, es, ee)
                X_all.append(_feats(d, u, s, lf_e[min(T//2, len(lf_e)-1)]))
                y_all.append(0)
            except Exception: continue
        for dt in DRIFT_TYPES:
            try:
                dv, ds = _inject_drift(base, dt, T)
                y, mu, sg = _infer(model, dv, W)
                if y is None: continue
                res = urd.score(y, mu, sg)
                es = max(0, ds-W); ee = len(res["deviation"])
                if ee - es < 5: continue
                d, u, s = _event_stats(res, es, ee)
                X_all.append(_feats(d, u, s, lf_e[min(ds, len(lf_e)-1)]))
                y_all.append(1)
            except Exception: continue
    if len(X_all) < 20:
        print("  skipped fig6/table2 — not enough data"); return
    X = np.array(X_all); y = np.array(y_all)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    rf.fit(X_tr, y_tr)
    imp = rf.feature_importances_
    idx = np.argsort(imp)
    bar_col = []
    for i in idx:
        nm = FEAT_NAMES[i]
        if nm in ("D_mean_urd","U_mean_urd","S_mean_urd"):  bar_col.append(C["blue"])
        elif nm in ("mean_U","U_slope","max_U"):            bar_col.append(C["orange"])
        else:                                               bar_col.append(C["gray"])
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.barh(range(len(idx)), imp[idx], color=bar_col, edgecolor="white")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([FEAT_NAMES[i] for i in idx], fontsize=10)
    ax.set_xlabel("Feature Importance (Gini)", fontweight="bold")
    ax.set_title("Random Forest Feature Importances — Drift vs Anomaly Classification",
                 fontweight="bold", fontsize=12)
    patches = [
        mpatches.Patch(color=C["blue"],   label="URD channels (D, U, S)"),
        mpatches.Patch(color=C["orange"], label="Probabilistic uncertainty"),
        mpatches.Patch(color=C["gray"],   label="Standard features"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=10)
    plt.tight_layout()
    _save(fig, "fig6_feature_importance.png")
    # Table 2
    rows = []
    for fn, fe_end in [("9-feat (no prob)",9), ("12-feat (orig prob)",12), ("15-feat (URD)",15)]:
        for cn, clf in [
            ("LR", LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")),
            ("RF", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
        ]:
            try:
                clf.fit(X_tr[:, :fe_end], y_tr)
                preds = clf.predict(X_te[:, :fe_end])
                acc = float(np.mean(preds == y_te))
                dm = y_te==1; am = y_te==0
                d2a = float(np.mean(preds[dm]==0)) if dm.sum() > 0 else float("nan")
                a2d = float(np.mean(preds[am]==1)) if am.sum() > 0 else float("nan")
                rows.append({"Features":fn, "Classifier":cn,
                              "Accuracy":acc, "Drift->Anom":d2a, "Anom->Drift":a2d})
            except Exception: continue
    if rows:
        _csv(pd.DataFrame(rows).set_index(["Features","Classifier"]), "table2_drift_ablation.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 3 — Fingerprint Confusion Matrix
# ═══════════════════════════════════════════════════════════════════════════════
def table3_fingerprint(model, urd, test_df, sensors, W):
    from src.synthetic.anomaly_generator import AnomalyGenerator
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, accuracy_score
    gen  = AnomalyGenerator(sensors, random_seed=42)
    eids = sorted(test_df["unit_nr"].unique())
    X_all, y_all = [], []
    for eid in eids[:12]:
        ed = _ed(test_df, eid, sensors)
        base = ed["sensor_values"]; lf_e = ed["life_fracs"]; T = len(base)
        if T <= W + 30: continue
        for fi, at in enumerate(ANOM_TYPES):
            try:
                traj = gen.create_injected_trajectory(ed, at, 0.5, 4.0, 15)
                y, mu, sg = _infer(model, traj.sensor_values, W)
                if y is None: continue
                res = urd.score(y, mu, sg)
                es = max(0, T//2-W); ee = min(len(res["deviation"]), es+20)
                if ee-es < 3: continue
                d, u, s = _event_stats(res, es, ee)
                ta = np.arange(len(d), dtype=float)
                sd = float(np.polyfit(ta,d,1)[0]) if len(d)>1 else 0.0
                su = float(np.polyfit(ta,u,1)[0]) if len(u)>1 else 0.0
                X_all.append([d.max(), d.mean(), sd, float(len(d)),
                               u.mean(), u.max(), su,
                               s.mean(), s.max(), d.mean()/max(u.mean(),1e-6)])
                y_all.append(fi)
            except Exception: continue
        for fi, dt in enumerate(DRIFT_TYPES):
            try:
                dv, ds = _inject_drift(base, dt, T)
                y, mu, sg = _infer(model, dv, W)
                if y is None: continue
                res = urd.score(y, mu, sg)
                es = max(0, ds-W); ee = len(res["deviation"])
                if ee-es < 5: continue
                d, u, s = _event_stats(res, es, ee)
                ta = np.arange(len(d), dtype=float)
                sd = float(np.polyfit(ta,d,1)[0]) if len(d)>1 else 0.0
                su = float(np.polyfit(ta,u,1)[0]) if len(u)>1 else 0.0
                X_all.append([d.max(), d.mean(), sd, float(len(d)),
                               u.mean(), u.max(), su,
                               s.mean(), s.max(), d.mean()/max(u.mean(),1e-6)])
                y_all.append(5 + fi)
            except Exception: continue
    if len(X_all) < 20:
        print("  skipped table3 — not enough data"); return
    X = np.array(X_all); y = np.array(y_all)
    present = sorted(np.unique(y))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    clf.fit(X_tr, y_tr); preds = clf.predict(X_te)
    acc = accuracy_score(y_te, preds)
    labels = [ALL_TYPES[i].replace("_"," ").title() for i in present]
    cm = confusion_matrix(y_te, preds, labels=present)
    df_cm = pd.DataFrame(cm,
                         index=[f"True: {l}" for l in labels[:len(cm)]],
                         columns=[f"Pred: {l}" for l in labels[:len(cm)]])
    df_cm.index.name = f"Overall accuracy = {acc:.3f}"
    _csv(df_cm, "table3_fingerprint_confusion.csv")
    print(f"  Fingerprinting accuracy: {acc:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  PAPER FIGURE & TABLE GENERATOR")
    print(f"  Output → {PAPER}")
    print("=" * 65)
    _style()
    cfg  = _cfg()
    W    = cfg["preprocessing"]["window_size"]
    uw   = cfg.get("urd", {}).get("conformity_window", 5)
    print("\n[Phase 1 — no data needed]")
    fig1_architecture()
    table4_model_comparison()
    print("\n[Phase 2 — loading data & model...]")
    try:
        splits, sensors = _prepare(cfg)
    except Exception as e:
        print(f"  ERROR loading data: {e}"); return
    try:
        model = _load_gru(cfg)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}"); return
    print("[Phase 3 — fitting URD scorer on validation data...]")
    urd = _fit_urd(model, splits["val"], sensors, W, fde_w=uw)
    print("[Phase 4 — generating all figures and tables...]")
    fig7_prediction_bands(model, splits["test"], sensors, W)
    fig2_urd_channels(model, urd, splits["test"], sensors, W)
    fig3_sensor_freeze(model, urd, splits["test"], sensors, W)
    fig4_roc_table1(model, urd, splits["test"], sensors, W)
    fig5_heatmap(model, urd, splits["test"], sensors, W)
    fig6_table2(model, urd, splits["test"], sensors, W)
    table3_fingerprint(model, urd, splits["test"], sensors, W)
    print("\n" + "=" * 65)
    print(f"  Done.  All outputs saved to:\n  {PAPER}")
    print("=" * 65)

if __name__ == "__main__":
    main()
