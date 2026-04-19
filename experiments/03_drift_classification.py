"""
experiments/03_drift_classification.py
=========================================
Stage D: Drift-vs-Anomaly Classification with URD (16 features).

Primary model: Gaussian GRU (gaussian_gru_best.pt).

Ablation: 16 URD vs 12 original vs 9 standard features.

Usage:
    python -m experiments.03_drift_classification
"""

import os, sys, json, yaml, numpy as np, torch, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_train_data
from src.data.preprocessing import compute_life_fraction, select_sensors, SensorScaler
from src.data.splits import split_engines, apply_split
from src.data.windowing import create_windows
from src.models.gaussian_gru import GaussianGRU
from src.anomaly.scoring import AnomalyScorer
from src.anomaly.urd import URDScorer, URD_FEATURE_NAMES
from src.synthetic.anomaly_generator import AnomalyGenerator
from src.synthetic.drift_generator import DriftGenerator
from src.drift.features import (extract_features_for_trajectory, extract_urd_features_for_trajectory, FEATURE_NAMES)
from src.drift.classifier import DriftAnomalyClassifier
from src.visualization.plots import plot_confusion_matrix_3way, plot_feature_importance


def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_model(model, values, window_size, device):
    T, d = values.shape
    if T <= window_size:
        return None, None, None
    X, y = [], []
    for i in range(T - window_size):
        X.append(values[i:i+window_size])
        y.append(values[i+window_size])
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        batch = torch.tensor(X, dtype=torch.float32).to(device)
        mu, sigma = model(batch)
    return y, mu.cpu().numpy(), sigma.cpu().numpy()


def score_and_extract(model, trajectories, window_size, device, scorer, urd_scorer, analysis_window, use_urd=True):
    all_features, all_labels = [], []
    model.eval()
    for traj in trajectories:
        y, mu, sigma = run_model(model, traj.sensor_values, window_size, device)
        if y is None:
            continue
        labels = traj.labels[window_size:]
        ml = min(len(labels), len(y))
        labels, y_a, mu_a, sigma_a = labels[:ml], y[:ml], mu[:ml], sigma[:ml]
        if not np.any(labels > 0):
            continue
        scores, _ = scorer.score(y_a, mu_a, sigma_a, normalize=True)
        residuals = np.abs(y_a - mu_a)
        urd_result = None
        if use_urd and urd_scorer is not None:
            urd_result = urd_scorer.score(y_a, mu_a, sigma_a, normalize=True)
        if use_urd and urd_result is not None:
            feats, el = extract_urd_features_for_trajectory(scores, residuals, sigma_a, labels, urd_result, threshold=2.0, analysis_window=analysis_window)
        else:
            feats, el = extract_features_for_trajectory(scores, residuals, sigma_a, labels, threshold=2.0, analysis_window=analysis_window)
        if len(feats) > 0:
            all_features.append(feats)
            all_labels.append(el)
    if all_features:
        return np.concatenate(all_features), np.concatenate(all_labels)
    n = len(URD_FEATURE_NAMES) if use_urd else 12
    return np.empty((0, n)), np.empty((0,), dtype=np.int32)


def main():
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sensors = cfg["dataset"]["selected_sensors"]
    ws = cfg["preprocessing"]["window_size"]
    aw = cfg["drift_classifier"]["analysis_window"]

    print("=" * 70)
    print("Stage D: Drift-vs-Anomaly (URD — 16 features)  |  Model: Gaussian GRU")
    print("=" * 70)

    df = load_train_data(cfg["paths"]["raw_data_dir"], cfg["dataset"]["subset"])
    df = compute_life_fraction(df)
    df = select_sensors(df, sensors, keep_meta=True)
    train_ids, val_ids, test_ids = split_engines(df, cfg["preprocessing"]["train_ratio"], cfg["preprocessing"]["val_ratio"], cfg["preprocessing"]["test_ratio"], cfg["preprocessing"]["split_random_seed"])
    splits = apply_split(df, train_ids, val_ids, test_ids)
    scaler = SensorScaler(sensors)
    splits["train"] = scaler.fit_transform(splits["train"])
    splits["val"] = scaler.transform(splits["val"])
    splits["test"] = scaler.transform(splits["test"])

    model = GaussianGRU(input_size=len(sensors), hidden_size=cfg["model"]["hidden_size"], num_layers=cfg["model"]["num_layers"], dropout=0.0, sigma_min=cfg["model"]["sigma_min"])
    model.load_state_dict(torch.load(os.path.join(cfg["paths"]["model_dir"], "gaussian_gru_best.pt"), map_location=device))
    model.to(device)

    X_val, y_val, _ = create_windows(splits["val"], sensors, window_size=ws, max_life_fraction=cfg["preprocessing"]["normal_life_fraction_threshold"])
    with torch.no_grad():
        batch = torch.tensor(X_val, dtype=torch.float32).to(device)
        vm, vs = model(batch)
        vm, vs = vm.cpu().numpy(), vs.cpu().numpy()

    scorer = AnomalyScorer(score_type="nll")
    scorer.fit_normalization(y_val, vm, vs)
    urd_scorer = URDScorer(fde_window=5)
    urd_scorer.fit(y_val, vm, vs)

    def prep(s):
        return [{"engine_id": int(eid), "sensor_values": s[s["unit_nr"] == eid].sort_values("time_cycles")[sensors].values.copy(), "cycles": s[s["unit_nr"] == eid].sort_values("time_cycles")["time_cycles"].values.copy(), "life_fracs": s[s["unit_nr"] == eid].sort_values("time_cycles")["life_fraction"].values.copy()} for eid in s["unit_nr"].unique() if len(s[s["unit_nr"] == eid]) > ws]

    ve, te = prep(splits["val"]), prep(splits["test"])
    ag = AnomalyGenerator(sensors, random_seed=cfg["synthetic_anomalies"]["random_seed"])
    dg = DriftGenerator(sensors, random_seed=cfg["synthetic_drift"]["random_seed"])

    va = ag.generate_test_suite(ve, magnitudes=[3.0, 5.0])
    vd = dg.generate_test_suite(ve, rates=[0.03, 0.05])
    ta = ag.generate_test_suite(te, magnitudes=[3.0, 5.0])
    td = dg.generate_test_suite(te, rates=[0.03, 0.05])
    print(f"Val: {len(va)} anom + {len(vd)} drift | Test: {len(ta)} anom + {len(td)} drift")

    print(f"\nExtracting features (URD={len(URD_FEATURE_NAMES)} feat)...")
    Xtr_u, ytr = score_and_extract(model, va+vd, ws, device, scorer, urd_scorer, aw, use_urd=True)
    Xte_u, yte = score_and_extract(model, ta+td, ws, device, scorer, urd_scorer, aw, use_urd=True)
    print(f"  URD: Train={len(Xtr_u)} Test={len(Xte_u)} dim={Xtr_u.shape[1]}")

    Xtr_12, _ = score_and_extract(model, va+vd, ws, device, scorer, None, aw, use_urd=False)
    Xte_12, _ = score_and_extract(model, ta+td, ws, device, scorer, None, aw, use_urd=False)
    print(f"  Original: Train={len(Xtr_12)} Test={len(Xte_12)} dim={Xtr_12.shape[1]}")

    Xtr_9, Xte_9 = Xtr_u[:, :9], Xte_u[:, :9]

    if len(Xtr_u) == 0 or len(Xte_u) == 0:
        print("ERROR: No events."); return

    results = {}
    configs = [
        (f"URD_{len(URD_FEATURE_NAMES)}feat", Xtr_u, Xte_u, URD_FEATURE_NAMES),
        ("Original_12feat", Xtr_12, Xte_12, FEATURE_NAMES),
        ("NoProbabilistic_9feat", Xtr_9, Xte_9, URD_FEATURE_NAMES[:9])]

    print("\n" + "=" * 70)
    for mt in cfg["drift_classifier"]["models"]:
        for cn, xtr, xte, fn in configs:
            name = f"{mt}_{cn}"
            clf = DriftAnomalyClassifier(mt, random_seed=cfg["drift_classifier"]["random_seed"])
            clf.fit(xtr, ytr, feature_names=fn)
            ev = clf.evaluate(xte, yte)
            results[name] = ev
            imp = clf.get_feature_importance()
            if imp:
                results[name]["feature_importance"] = imp
            print(f"{name:<55} Acc={ev['accuracy']:.3f}  D→A={ev['drift_as_anomaly_rate']:.3f}  A→D={ev['anomaly_as_drift_rate']:.3f}")

    fig_dir, res_dir = cfg["paths"]["figure_dir"], cfg["paths"]["results_dir"]
    os.makedirs(fig_dir, exist_ok=True); os.makedirs(res_dir, exist_ok=True)
    for n, r in results.items():
        if "confusion_matrix" in r:
            plot_confusion_matrix_3way(r["confusion_matrix"], ["Anomaly", "Drift"], title=n, save_path=os.path.join(fig_dir, f"confusion_{n}.png"))
        if "feature_importance" in r:
            plot_feature_importance(r["feature_importance"], title=f"Importance — {n}", save_path=os.path.join(fig_dir, f"importance_{n}.png"))
    sd = {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray)} for k, v in results.items()}
    with open(os.path.join(res_dir, "stage_d_results.json"), "w") as f:
        json.dump(sd, f, indent=2, default=str)

    with open(os.path.join(res_dir, "stage_d_classification.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["classifier", "feature_set", "n_features", "accuracy", "drift_as_anomaly_rate", "anomaly_as_drift_rate"])
        for name, r in sorted(results.items()):
            parts = name.split("_", 1)
            clf_name = parts[0] if len(parts) > 0 else name
            feat_name = parts[1] if len(parts) > 1 else ""
            nf = 16 if "16" in feat_name else (12 if "12" in feat_name else 9)
            w.writerow([clf_name, feat_name, nf, f"{r['accuracy']:.4f}", f"{r['drift_as_anomaly_rate']:.4f}", f"{r['anomaly_as_drift_rate']:.4f}"])

    print(f"\n  Stage D Complete! CSVs saved to {res_dir}/")


if __name__ == "__main__":
    main()
