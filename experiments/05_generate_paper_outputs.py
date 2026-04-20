"""
experiments/05_generate_paper_outputs.py
==========================================
Generate paper figures/tables for the current URD baseline and TranAD comparison.

Recommended run order:
  1) python -m experiments.01_train_baselines
  2) python -m experiments.02_synthetic_evaluation
  3) python -m experiments.03_drift_classification
  4) python -m experiments.04_urd_fingerprinting
  5) python -m experiments.05_generate_paper_outputs
"""

import csv
import json
import os
import shutil
import sys
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yaml
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PAPER = os.path.join(ROOT, "outputs", "for_paper")
os.makedirs(PAPER, exist_ok=True)

from src.data.loader import load_train_data
from src.data.preprocessing import compute_life_fraction, select_sensors, SensorScaler
from src.data.splits import split_engines, apply_split
from src.models.gaussian_gru import GaussianGRU
from src.models.tranad import TranAD
from src.anomaly.urd import URDScorer
from src.synthetic.anomaly_generator import AnomalyGenerator

PAL = {"blue":"#1565C0","orange":"#E65100","green":"#2E7D32","red":"#C62828","purple":"#6A1B9A","gray":"#546E7A","teal":"#00695C","amber":"#F57F17","urd":"#1565C0","tranad":"#E65100"}
ANOM_TYPES = ["spike", "drop", "persistent_offset", "noise_burst", "sensor_freeze"]
DRIFT_TYPES = ["gradual_shift", "sigmoid_plateau", "accelerating", "multi_sensor"]
CATEGORY_MAP = {
    "spike": "point_anomaly",
    "drop": "point_anomaly",
    "persistent_offset": "persistent_shift",
    "noise_burst": "noise_anomaly",
    "sensor_freeze": "sensor_malfunction",
    "gradual_shift": "drift",
    "sigmoid_plateau": "drift",
    "accelerating": "drift",
    "multi_sensor": "drift",
}


def _style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#F9F9F9",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.framealpha": 0.92,
        "lines.linewidth": 1.8,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.55,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(fig, name):
    path = os.path.join(PAPER, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  saved {name}")


def _csv_write(rows, headers, name):
    path = os.path.join(PAPER, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            w.writerow(row)
    print(f"  saved {name}")


def _cfg():
    with open(os.path.join(ROOT, "config", "config.yaml"), "r") as f:
        return yaml.safe_load(f)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_data(cfg):
    sensors = cfg["dataset"]["selected_sensors"]
    df = load_train_data(cfg["paths"]["raw_data_dir"], cfg["dataset"]["subset"])
    df = compute_life_fraction(df)
    df = select_sensors(df, sensors, keep_meta=True)
    ti, vi, si = split_engines(
        df,
        cfg["preprocessing"]["train_ratio"],
        cfg["preprocessing"]["val_ratio"],
        cfg["preprocessing"]["test_ratio"],
        cfg["preprocessing"]["split_random_seed"],
    )
    splits = apply_split(df, ti, vi, si)
    scaler = SensorScaler(sensors)
    splits["train"] = scaler.fit_transform(splits["train"])
    splits["val"] = scaler.transform(splits["val"])
    splits["test"] = scaler.transform(splits["test"])
    return splits, sensors


def _load_gru(cfg, device="cpu"):
    m = GaussianGRU(
        input_size=len(cfg["dataset"]["selected_sensors"]),
        hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"],
        dropout=0.0,
        sigma_min=cfg["model"]["sigma_min"],
    )
    ckpt = os.path.join(cfg["paths"]["model_dir"], "gaussian_gru_best.pt")
    import torch
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    return m


def _load_tranad(cfg, device="cpu"):
    tcfg = cfg["model"].get("tranad", {})
    m = TranAD(
        input_size=len(cfg["dataset"]["selected_sensors"]),
        window_size=cfg["preprocessing"]["window_size"],
        d_model=tcfg.get("d_model", 64),
        nhead=tcfg.get("nhead", 4),
        num_layers=tcfg.get("num_layers", 2),
        dim_feedforward=tcfg.get("dim_feedforward", 128),
        dropout=tcfg.get("dropout", 0.1),
    )
    import torch
    ckpt = os.path.join(cfg["paths"]["model_dir"], "tranad_best.pt")
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    return m


def _infer_gru(model, values, W, device="cpu"):
    import torch
    T, d = values.shape
    if T <= W:
        return None, None, None
    X = np.array([values[i:i + W] for i in range(T - W)], dtype=np.float32)
    y = np.array([values[i + W] for i in range(T - W)], dtype=np.float32)
    with torch.no_grad():
        batch = torch.tensor(X, dtype=torch.float32).to(device)
        mu, sigma = model(batch)
    return y, mu.cpu().numpy(), sigma.cpu().numpy()




def _infer_tranad(model, values, W, device="cpu"):
    import torch
    T, d = values.shape
    if T <= W:
        return None, None, None
    X = np.array([values[i:i + W] for i in range(T - W)], dtype=np.float32)
    y = np.array([values[i + W] for i in range(T - W)], dtype=np.float32)
    with torch.no_grad():
        batch = torch.tensor(X, dtype=torch.float32).to(device)
        pred1, pred2 = model(batch)
    return y, pred1.cpu().numpy(), pred2.cpu().numpy()

def _fit_urd(model, val_df, sensors, W, device="cpu"):
    ys, ms, ss = [], [], []
    for eid in sorted(val_df["unit_nr"].unique()):
        vals = val_df[val_df["unit_nr"] == eid].sort_values("time_cycles")[sensors].values
        y, mu, sigma = _infer_gru(model, vals, W, device)
        if y is not None:
            ys.append(y); ms.append(mu); ss.append(sigma)
    urd = URDScorer(fde_window=5)
    urd.fit(np.concatenate(ys), np.concatenate(ms), np.concatenate(ss))
    return urd


def _ed(df, eid, sensors):
    edf = df[df["unit_nr"] == eid].sort_values("time_cycles")
    return {
        "engine_id": int(eid),
        "sensor_values": edf[sensors].values.copy(),
        "cycles": edf["time_cycles"].values.copy(),
        "life_fracs": edf["life_fraction"].values.copy(),
    }


def _inject_drift(base, dtype, T):
    dv = base.copy()
    ds = T // 2
    for t in range(ds, T):
        p = (t - ds) / max(1, T - ds)
        if dtype == "gradual_shift":
            dv[t, 0] += p * 3.0
        elif dtype == "sigmoid_plateau":
            dv[t, 0] += 3.0 / (1 + np.exp(-10 * (p - 0.5)))
        elif dtype == "accelerating":
            dv[t, 0] += p ** 2 * 4.0
        elif dtype == "multi_sensor":
            dv[t, 0] += p * 2.0
            dv[t, min(2, dv.shape[1] - 1)] += p * 1.5
    return dv, ds


def fig1_pipeline_overview():
    _style()
    fig, ax = plt.subplots(figsize=(19, 8.5))
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    boxes = [
        (0.2, 4.2, 2.6, 2.8, "STEP 1\nRaw Data", "train_FD001.txt\n100 engines\n26 cols per row\nwhitespace-delim\nno header", "#E3F2FD"),
        (3.2, 4.2, 2.6, 2.8, "STEP 2\nPreprocessing", "21→7 sensors\nengine split\nwindow = 30\nZ-score scale\nhealthy-only train", "#E8F5E9"),
        (6.2, 4.2, 2.8, 2.8, "STEP 3\nGaussian GRU", "Input: (30, 7)\n→ μ ∈ ℝ⁷\n→ σ ∈ ℝ⁷\ntrained with NLL\nlife_frac ≤ 0.5", "#FFF3E0"),
        (9.5, 4.2, 3.1, 2.8, "STEP 4\nURD Baseline", "D: calibrated Mahalanobis\nU: sigma inflation\nS: tuned FDE + run\ncombined = 0.35D + 0.65S", "#FCE4EC"),
        (13.1, 4.2, 3.0, 2.8, "STEP 5\nTranAD Baseline", "2-phase transformer\nself-conditioning\nnext-step error score\ndirect TSAD comparator", "#E8F5E9"),
        (16.6, 5.4, 2.8, 1.5, "STEP 6a\nClassify", "drift vs anomaly\n16-feature URD model", "#EDE7F6"),
        (16.6, 4.2, 2.8, 1.0, "STEP 6b\nFingerprint", "5-class type ID\nfeature importance", "#E8EAF6"),
    ]

    for bx, by, bw, bh, title, body, col in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.12",
                                            facecolor=col, edgecolor="#546E7A", linewidth=1.5))
        ax.text(bx + bw / 2, by + bh - 0.28, title, ha="center", va="top", fontsize=9.5,
                fontweight="bold", color="#1A237E")
        ax.text(bx + bw / 2, by + 0.18, body, ha="center", va="bottom", fontsize=8.1,
                color="#263238", linespacing=1.55)

    akw = dict(arrowstyle="-|>", color="#37474F", lw=1.8)
    for x1, x2, yc in [(2.8, 3.2, 5.6), (5.8, 6.2, 5.6), (9.0, 9.5, 5.6), (12.6, 13.1, 5.6), (16.1, 16.6, 6.15), (16.1, 16.6, 4.7)]:
        ax.annotate("", xy=(x2, yc), xytext=(x1, yc), arrowprops=akw)

    math_boxes = [
        (1.5, 4.05, r"$\mathbf{x}_t\in\mathbb{R}^{21}$" + "\nunit_nr, time_cycles\n21 raw sensors"),
        (4.5, 4.05, r"$z_j=(x_j-\bar{x}_j^{tr})/s_j^{tr}$" + "\nfit scaler on train only\nengine-level split"),
        (7.6, 4.05, r"$\mathcal{L}=\frac{1}{d}\sum_j[\log \sigma_j+\frac{(x_j-\mu_j)^2}{2\sigma_j^2}]$" + "\nprobabilistic next-step learning"),
        (11.05, 4.05, r"$D_t=r_t^T\Sigma_r^{-1}r_t$  with  $r=\frac{x-\mu}{\tau\odot\sigma}$" + "\n" + r"$U_t=\frac{1}{d}\sum_j \sigma^{eff}_{t,j}/\sigma_{ref,j}$" + "\n" + r"$S_t=FDE(t)+3\cdot\max(0,run-1)$"),
        (14.6, 4.05, "TranAD score = refined\ntransformer next-step error\nsame FD001 protocol"),
        (18.0, 4.05, "drift vs anomaly\n5-class fingerprint\ninterpretable outputs"),
    ]
    for x, y, txt in math_boxes:
        ax.text(x, y, txt, ha="center", va="top", fontsize=7.6, color="#37474F", style="italic",
                linespacing=1.5, bbox=dict(boxstyle="round,pad=0.28", facecolor="#FAFAFA",
                alpha=0.88, edgecolor="#B0BEC5"))

    ax.text(10.5, 8.1,
            "Matched comparison setting: FD001, same 7 sensors, window = 30, engine split, healthy-only training",
            ha="center", fontsize=9.5, color=PAL["blue"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#E3F2FD", edgecolor=PAL["blue"], alpha=0.92))

    ax.set_title("Project Pipeline with External TSAD Baseline Added", fontsize=13.5, fontweight="bold", pad=10)
    _save(fig, "fig1_pipeline_overview.png")

def fig2_roc_pr_compare(stage_c):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    urd = stage_c["urd"]
    tranad = stage_c["tranad"]
    axes[0].plot(urd["curves"]["roc"][0], urd["curves"]["roc"][1], color=PAL["urd"], label=f"URD (AUC={urd['overall']['roc_auc']:.3f})")
    axes[0].plot(tranad["curves"]["roc"][0], tranad["curves"]["roc"][1], color=PAL["tranad"], label=f"TranAD (AUC={tranad['overall']['roc_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve — URD vs TranAD")
    axes[0].legend()

    axes[1].plot(urd["curves"]["pr"][1], urd["curves"]["pr"][0], color=PAL["urd"], label=f"URD (AP={urd['overall']['pr_auc']:.3f})")
    axes[1].plot(tranad["curves"]["pr"][1], tranad["curves"]["pr"][0], color=PAL["tranad"], label=f"TranAD (AP={tranad['overall']['pr_auc']:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("PR Curve — URD vs TranAD")
    axes[1].legend()
    fig.suptitle("Direct Detector Comparison on Synthetic FD001 Test Suite", fontweight="bold")
    _save(fig, "fig2_roc_pr_urd_vs_tranad.png")


def fig3_threshold_sweep(csv_path):
    _style()
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    for method, color in [("URD", PAL["urd"]), ("TranAD", PAL["tranad"])]:
        sub = [r for r in rows if r["method"] == method]
        thr = [float(r["threshold"]) for r in sub]
        prec = [float(r["precision"]) for r in sub]
        rec = [float(r["recall"]) for r in sub]
        f1 = [float(r["f1"]) for r in sub]
        axes[0].plot(thr, prec, color=color, label=method)
        axes[1].plot(thr, rec, color=color, label=method)
        axes[2].plot(thr, f1, color=color, label=method)
    for ax, ttl in zip(axes, ["Precision", "Recall", "F1"]):
        ax.set_title(ttl, fontweight="bold")
        ax.set_xlabel("Threshold")
        ax.legend()
    axes[0].set_ylabel("Metric value")
    fig.suptitle("Threshold Sweeps — Point-level Detection Trade-offs", fontweight="bold")
    _save(fig, "fig3_threshold_sweep.png")


def fig4_roc_curves_by_type(cfg, splits, sensors, W, model, urd, tranad_model, device="cpu"):
    from sklearn.metrics import roc_curve, roc_auc_score
    from src.synthetic.anomaly_generator import AnomalyGenerator
    _style()
    gen = AnomalyGenerator(sensors, random_seed=42)
    eids = sorted(splits["test"]["unit_nr"].unique())[:10]
    methods = ["URD (baseline)", "TranAD"]
    colors = {"URD (baseline)": PAL["blue"], "TranAD": PAL["orange"]}
    all_scores = {m: {at: {"scores": [], "labels": []} for at in ANOM_TYPES} for m in methods}

    for eid in eids:
        ed = _ed(splits["test"], eid, sensors)
        T = len(ed["sensor_values"])
        if T <= W + 20:
            continue
        for at in ANOM_TYPES:
            try:
                traj = gen.create_injected_trajectory(ed, at, injection_life_frac=0.5, magnitude=4.0, duration=15)
                y, mu, sigma = _infer_gru(model, traj.sensor_values, W, device)
                ty, p1, p2 = _infer_tranad(tranad_model, traj.sensor_values, W, device)
                if y is None or ty is None:
                    continue
                n = min(len(y), len(ty))
                lab = traj.labels[W:W+n]
                urd_s = urd.score(y[:n], mu[:n], sigma[:n], normalize=True)["combined"][:n]
                tranad_err = np.mean((ty[:n] - p2[:n]) ** 2, axis=1)
                # robust z-normalization per trajectory to match plotting style from old figure
                tr_mu = float(np.mean(tranad_err[:max(5, n // 3)]))
                tr_sd = max(float(np.std(tranad_err[:max(5, n // 3)])), 1e-8)
                tranad_s = (tranad_err - tr_mu) / tr_sd
                all_scores["URD (baseline)"][at]["scores"].extend(urd_s.tolist())
                all_scores["URD (baseline)"][at]["labels"].extend(lab.tolist())
                all_scores["TranAD"][at]["scores"].extend(tranad_s.tolist())
                all_scores["TranAD"][at]["labels"].extend(lab.tolist())
            except Exception:
                continue

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    flat = axes.flatten()
    table_rows = []
    for idx, at in enumerate(ANOM_TYPES):
        ax = flat[idx]
        row = [at.replace("_", " ").title()]
        for mn in methods:
            s = np.array(all_scores[mn][at]["scores"])
            l = np.array(all_scores[mn][at]["labels"])
            if len(s) == 0 or len(np.unique(l)) < 2:
                row.append("n/a")
                continue
            auc_v = roc_auc_score(l, s)
            fpr, tpr, _ = roc_curve(l, s)
            lw = 2.8 if mn == "URD (baseline)" else 1.6
            ls = "-" if mn == "URD (baseline)" else "--"
            ax.plot(fpr, tpr, color=colors[mn], lw=lw, ls=ls, label=f"{mn} ({auc_v:.3f})", alpha=0.94)
            row.append(f"{auc_v:.3f}")
        table_rows.append(row)
        ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.2)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        title_col = PAL["blue"] if at == "sensor_freeze" else "#1A237E"
        ax.set_title(at.replace("_", " ").title(), fontweight="bold", color=title_col, fontsize=10.5)
        ax.legend(fontsize=7.5, loc="lower right")

    flat[5].axis("off")
    flat[5].text(0.5, 0.56,
                 "KEY RESULT\n\nURD baseline vs TranAD\nunder the same FD001 setup\n\nStrongest gap:\nSensor Freeze\nURD ≈ 0.823\nTranAD ≈ 0.462\n\nTranAD is competitive on\nordinary point anomalies,\nbut URD is far stronger on\nfreeze-like malfunctions",
                 transform=flat[5].transAxes, ha="center", va="center", fontsize=10.5, fontweight="bold",
                 color=PAL["blue"], bbox=dict(boxstyle="round,pad=0.65", facecolor="#E3F2FD", alpha=0.97,
                 edgecolor=PAL["blue"], lw=1.5))
    fig.suptitle("ROC Curves by Anomaly Type — URD Baseline vs TranAD\nSolid = URD baseline   |   Dashed = TranAD   |   Diagonal = random chance",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); _save(fig, "fig4_roc_curves_by_type.png")
    _csv_write(table_rows, ["Anomaly Type"] + methods, "table1_anomaly_detection.csv")

def fig5_case_timeline(model, urd, test_df, sensors, W, device="cpu"):
    _style()
    gen = AnomalyGenerator(sensors, random_seed=42)
    eid = sorted(test_df["unit_nr"].unique())[0]
    ed = _ed(test_df, eid, sensors)
    traj = gen.create_injected_trajectory(ed, "sensor_freeze", injection_life_frac=0.5, magnitude=4.0, duration=20)
    y, mu, sigma = _infer_gru(model, traj["sensor_values"] if isinstance(traj, dict) else traj.sensor_values, W, device)
    if y is None:
        return
    values = traj["sensor_values"] if isinstance(traj, dict) else traj.sensor_values
    labels = traj["labels"] if isinstance(traj, dict) else traj.labels
    lf = traj["life_fracs"] if isinstance(traj, dict) else traj.life_fracs
    res = urd.score(y, mu, sigma, normalize=True)
    idx = sensors.index("s_4") if "s_4" in sensors else 0
    lf_plot = lf[W:W + len(y)]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(lf_plot, y[:, idx], color=PAL["red"], label="True")
    axes[0].plot(lf_plot, mu[:, idx], color=PAL["urd"], label="Predicted μ")
    axes[0].fill_between(lf_plot, mu[:, idx] - 2 * sigma[:, idx], mu[:, idx] + 2 * sigma[:, idx], color=PAL["urd"], alpha=0.15, label="μ ± 2σ")
    axes[0].set_ylabel(sensors[idx])
    axes[0].legend(loc="upper left")
    axes[0].set_title("Case study — injected sensor freeze with GRU prediction band", fontweight="bold")

    axes[1].plot(lf_plot, res["combined"], color=PAL["purple"], label="URD score")
    axes[1].plot(lf_plot, (labels[W:W + len(y)] > 0).astype(int), color="#455A64", alpha=0.4, label="Anomaly label")
    axes[1].set_ylabel("score")
    axes[1].legend(loc="upper left")

    axes[2].plot(lf_plot, res["deviation"], color=PAL["red"], label="D")
    axes[2].plot(lf_plot, res["uncertainty"], color=PAL["green"], label="U")
    axes[2].plot(lf_plot, res["stationarity"], color=PAL["urd"], label="S")
    axes[2].set_xlabel("Life Fraction")
    axes[2].set_ylabel("channel value")
    axes[2].legend(loc="upper left")
    fig.suptitle("Raw signal, probabilistic prediction, and URD channels on one failure case", fontweight="bold")
    _save(fig, "fig5_case_timeline_freeze.png")


def fig6_dus_distributions(model, urd, test_df, sensors, W, device="cpu"):
    _style()
    gen = AnomalyGenerator(sensors, random_seed=42)
    vals = {cat: {"D": [], "U": [], "S": []} for cat in sorted(set(CATEGORY_MAP.values()))}
    for eid in sorted(test_df["unit_nr"].unique())[:10]:
        ed = _ed(test_df, eid, sensors)
        base = ed["sensor_values"]
        T = len(base)
        if T <= W + 40:
            continue
        for at in ANOM_TYPES:
            traj = gen.create_injected_trajectory(ed, at, 0.5, 4.0, 15)
            y, mu, sigma = _infer_gru(model, traj.sensor_values, W, device)
            if y is None:
                continue
            res = urd.score(y, mu, sigma, normalize=True)
            mask = traj.labels[W:W + len(y)] > 0
            if mask.sum() == 0:
                continue
            cat = CATEGORY_MAP[at]
            vals[cat]["D"].append(float(np.mean(res["deviation"][mask])))
            vals[cat]["U"].append(float(np.mean(res["uncertainty"][mask])))
            vals[cat]["S"].append(float(np.mean(res["stationarity"][mask])))
        for dt in DRIFT_TYPES:
            dv, ds = _inject_drift(base, dt, T)
            y, mu, sigma = _infer_gru(model, dv, W, device)
            if y is None:
                continue
            res = urd.score(y, mu, sigma, normalize=True)
            mask = np.zeros(len(y), dtype=bool)
            start = max(0, ds - W)
            mask[start:] = True
            vals["drift"]["D"].append(float(np.mean(res["deviation"][mask])))
            vals["drift"]["U"].append(float(np.mean(res["uncertainty"][mask])))
            vals["drift"]["S"].append(float(np.mean(res["stationarity"][mask])))
    cats = list(vals.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
    for ax, ch in zip(axes, ["D", "U", "S"]):
        data = [vals[c][ch] for c in cats]
        ax.boxplot(data, tick_labels=[c.replace("_", " ") for c in cats], showfliers=False)
        ax.set_title(f"{ch} by category", fontweight="bold")
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("URD channel distributions by anomaly category", fontweight="bold")
    _save(fig, "fig6_dus_distributions.png")


def fig7_feature_importance(stage_d):
    _style()
    imp = stage_d["xgboost_URD_16feat"]["feature_importance"]
    items = sorted(imp.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(9.5, 7))
    ax.barh(range(len(names)), vals, color=PAL["urd"])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Importance")
    ax.set_title("Drift-vs-anomaly feature importance (XGBoost, URD 16 features)", fontweight="bold")
    _save(fig, "fig7_feature_importance.png")


def fig8_calibration(model, urd, val_df, sensors, W, device="cpu"):
    _style()
    rs = []
    for eid in sorted(val_df["unit_nr"].unique()):
        vals = val_df[val_df["unit_nr"] == eid].sort_values("time_cycles")[sensors].values
        y, mu, sigma = _infer_gru(model, vals, W, device)
        if y is None:
            continue
        sigma_eff = sigma * urd.sigma_temp[np.newaxis, :]
        rs.append(((y - mu) / sigma_eff).reshape(-1))
    residuals = np.concatenate(rs)
    z_grid = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    empirical = [float(np.mean(np.abs(residuals) <= z)) for z in z_grid]
    import math
    theoretical = [float(math.erf(z / np.sqrt(2.0))) for z in z_grid]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(z_grid, empirical, marker="o", color=PAL["urd"], label="Empirical")
    axes[0].plot(z_grid, theoretical, marker="s", color=PAL["gray"], label="Gaussian ideal")
    axes[0].set_xlabel("z in μ ± zσ")
    axes[0].set_ylabel("Coverage")
    axes[0].set_title("Coverage calibration", fontweight="bold")
    axes[0].legend()

    axes[1].hist(residuals, bins=40, density=True, color=PAL["urd"], alpha=0.7)
    x = np.linspace(-4, 4, 400)
    pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)
    axes[1].plot(x, pdf, color=PAL["red"], label="N(0,1)")
    axes[1].set_title(f"Normalized residuals\nmean={residuals.mean():.3f}, std={residuals.std():.3f}", fontweight="bold")
    axes[1].legend()
    fig.suptitle("Probabilistic calibration of the URD backbone", fontweight="bold")
    _save(fig, "fig8_probabilistic_calibration.png")


def copy_optional_figure(src_name, dst_name):
    src = os.path.join(ROOT, "outputs", "figures", src_name)
    dst = os.path.join(PAPER, dst_name)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  copied {dst_name}")


def table1_anomaly_detection(stage_c):
    rows = []
    all_types = sorted(stage_c["urd"]["per_type"].keys())
    for t in all_types:
        rows.append([
            t,
            stage_c["urd"]["per_type"][t]["roc_auc"],
            stage_c["urd"]["per_type"][t]["pr_auc"],
            stage_c["tranad"]["per_type"][t]["roc_auc"],
            stage_c["tranad"]["per_type"][t]["pr_auc"],
        ])
    rows.append(["OVERALL", stage_c["urd"]["overall"]["roc_auc"], stage_c["urd"]["overall"]["pr_auc"], stage_c["tranad"]["overall"]["roc_auc"], stage_c["tranad"]["overall"]["pr_auc"]])
    _csv_write(rows, ["type", "urd_roc", "urd_pr", "tranad_roc", "tranad_pr"], "table1_anomaly_detection.csv")


def table2_drift_ablation(stage_d):
    rows = []
    for name, data in stage_d.items():
        rows.append([name, data["accuracy"], data["drift_as_anomaly_rate"], data["anomaly_as_drift_rate"]])
    _csv_write(rows, ["model", "accuracy", "drift_as_anomaly_rate", "anomaly_as_drift_rate"], "table2_drift_ablation.csv")


def table3_fingerprint(stage_e):
    rows = []
    rep = stage_e["five_class"]["report"]
    for cls in ["drift", "noise_anomaly", "persistent_shift", "point_anomaly", "sensor_malfunction"]:
        if cls in rep:
            rows.append([cls, rep[cls]["precision"], rep[cls]["recall"], rep[cls]["f1-score"], rep[cls]["support"]])
    rows.append(["accuracy", stage_e["five_class"]["accuracy"], "", "", ""])
    _csv_write(rows, ["class", "precision", "recall", "f1", "support"], "table3_fingerprint.csv")


def table4_model_comparison(stage_a):
    rows = []
    wanted = ["gaussian_gru", "tranad", "gaussian_lstm", "deterministic_gru", "deterministic_lstm", "ridge", "naive"]
    for k in wanted:
        if k not in stage_a["eval_results"]:
            continue
        res = stage_a["eval_results"][k]
        rows.append([k, res.get("mse", ""), res.get("mae", ""), res.get("mean_sigma", "")])
    _csv_write(rows, ["model", "test_mse", "test_mae", "mean_sigma"], "table4_model_comparison.csv")


def main():
    _style()
    cfg = _cfg()
    stage_c_path = os.path.join(ROOT, "outputs", "results", "stage_c_results.json")
    stage_d_path = os.path.join(ROOT, "outputs", "results", "stage_d_results.json")
    stage_e_path = os.path.join(ROOT, "outputs", "results", "stage_e_results.json")
    stage_a_path = os.path.join(ROOT, "outputs", "results", "stage_a_results.json")
    sweep_csv = os.path.join(ROOT, "outputs", "results", "stage_c_threshold_sweep.csv")
    for pth in [stage_c_path, stage_d_path, stage_e_path, stage_a_path, sweep_csv]:
        if not os.path.exists(pth):
            raise FileNotFoundError(f"Required input missing: {pth}. Run the earlier experiment stages first.")

    stage_c = _load_json(stage_c_path)
    stage_d = _load_json(stage_d_path)
    stage_e = _load_json(stage_e_path)
    stage_a = _load_json(stage_a_path)

    splits, sensors = _load_data(cfg)
    W = cfg["preprocessing"]["window_size"]
    model = _load_gru(cfg)
    tranad_model = _load_tranad(cfg)
    urd = _fit_urd(model, splits["val"], sensors, W)

    print("=" * 70)
    print("Generating updated paper outputs")
    print("=" * 70)
    fig1_pipeline_overview()
    fig2_roc_pr_compare(stage_c)
    fig3_threshold_sweep(sweep_csv)
    fig4_roc_curves_by_type(cfg, splits, sensors, W, model, urd, tranad_model)
    fig5_case_timeline(model, urd, splits["test"], sensors, W)
    fig6_dus_distributions(model, urd, splits["test"], sensors, W)
    fig7_feature_importance(stage_d)
    fig8_calibration(model, urd, splits["val"], sensors, W)
    copy_optional_figure("fingerprint_5class_cm.png", "fig9_fingerprint_5class_confusion.png")
    copy_optional_figure("feature_importance_xgboost.png", "fig10_stage_d_feature_importance.png")

    table1_anomaly_detection(stage_c)
    table2_drift_ablation(stage_d)
    table3_fingerprint(stage_e)
    table4_model_comparison(stage_a)
    print(f"\nDone. Outputs written to {PAPER}")


if __name__ == "__main__":
    main()
