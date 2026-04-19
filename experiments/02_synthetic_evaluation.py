"""
experiments/02_synthetic_evaluation.py — v3
============================================
Stage C: NLL-only vs URD (with Stationarity) comparison.

Fixes from v2:
1. Uses VALIDATION-CALIBRATED thresholds for both methods (not test-derived)
2. Channel 3 is now Stationarity (variance ratio on raw values) not Conformity
3. Consistent threshold application

Usage:
    python -m experiments.02_synthetic_evaluation
"""

import os, sys, json, yaml, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.loader import load_train_data
from src.data.preprocessing import compute_life_fraction, select_sensors, SensorScaler
from src.data.splits import split_engines, apply_split
from src.data.windowing import create_windows
from src.models.gaussian_lstm import GaussianLSTM
from src.anomaly.scoring import AnomalyScorer
from src.anomaly.urd import URDScorer
from src.synthetic.anomaly_generator import AnomalyGenerator
from src.evaluation.metrics import (threshold_independent_metrics, point_level_metrics, event_level_metrics,)


def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "config", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def score_trajectory(model, values, window_size, device):
    T, d = values.shape
    if T <= window_size:
        return None, None, None
    X_list, y_list = [], []
    for i in range(T - window_size):
        X_list.append(values[i:i + window_size])
        y_list.append(values[i + window_size])
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        batch = torch.tensor(X, dtype=torch.float32).to(device)
        mu, sigma = model(batch)
    return y, mu.cpu().numpy(), sigma.cpu().numpy()


def evaluate_method(model, trajectories, window_size, device, score_fn, thresholds):
    """Score all trajectories and compute metrics using pre-calibrated thresholds."""
    all_scores, all_labels = [], []
    per_type = {}

    for traj in trajectories:
        y, mu, sigma = score_trajectory(model, traj.sensor_values, window_size, device)
        if y is None:
            continue
        labels = traj.labels[window_size:]
        min_len = min(len(labels), len(y))
        labels, y_a, mu_a, sigma_a = labels[:min_len], y[:min_len], mu[:min_len], sigma[:min_len]

        # Also pass raw sensor values for stationarity computation
        raw_values = traj.sensor_values[window_size:window_size + min_len]
        scores = score_fn(y_a, mu_a, sigma_a, raw_values)
        binary = (labels > 0).astype(int)

        all_scores.extend(scores.tolist())
        all_labels.extend(binary.tolist())

        if traj.events:
            atype = traj.events[0].anomaly_type
            if atype not in per_type:
                per_type[atype] = {"scores": [], "labels": []}
            per_type[atype]["scores"].extend(scores.tolist())
            per_type[atype]["labels"].extend(binary.tolist())

    all_scores, all_labels = np.array(all_scores), np.array(all_labels)

    #Overall AUC
    overall = threshold_independent_metrics(all_labels, all_scores)

    #Per-type AUC
    type_results = {}
    for atype, data in per_type.items():
        s, l = np.array(data["scores"]), np.array(data["labels"])
        if len(np.unique(l)) >= 2:
            type_results[atype] = threshold_independent_metrics(l, s)
        else:
            type_results[atype] = {"roc_auc": float("nan"), "pr_auc": float("nan")}

    #Threshold-based metrics using VALIDATION-calibrated thresholds
    thr_results = {}
    for pct, thr in thresholds.items():
        pm = point_level_metrics(all_labels, all_scores, thr)
        em = event_level_metrics(all_labels, all_scores, thr)
        thr_results[pct] = {"point": pm, "event": em}

    return {"overall": overall, "per_type": type_results, "threshold_results": thr_results}


def main():
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sensors = cfg["dataset"]["selected_sensors"]
    window_size = cfg["preprocessing"]["window_size"]

    print("=" * 70)
    print("Stage C: Anomaly Detection — NLL vs URD (Stationarity) Comparison")
    print("=" * 70)

    #Load data and model
    print("\n[1] Loading data and model...")
    df = load_train_data(cfg["paths"]["raw_data_dir"], cfg["dataset"]["subset"])
    df = compute_life_fraction(df)
    df = select_sensors(df, sensors, keep_meta=True)
    train_ids, val_ids, test_ids = split_engines(
        df, cfg["preprocessing"]["train_ratio"], cfg["preprocessing"]["val_ratio"],
        cfg["preprocessing"]["test_ratio"], cfg["preprocessing"]["split_random_seed"])
    splits = apply_split(df, train_ids, val_ids, test_ids)
    scaler = SensorScaler(sensors)
    splits["train"] = scaler.fit_transform(splits["train"])
    splits["val"] = scaler.transform(splits["val"])
    splits["test"] = scaler.transform(splits["test"])

    model = GaussianLSTM(
        input_size=len(sensors), hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"], dropout=cfg["model"]["dropout"],
        sigma_min=cfg["model"]["sigma_min"])
    model_path = os.path.join(cfg["paths"]["model_dir"], "gaussian_lstm_best.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device); model.eval()

    #Fit scorers on VALIDATION data
    print("[2] Fitting scorers on validation data...")
    X_val, y_val, _ = create_windows(
        splits["val"], sensors, window_size=window_size,
        max_life_fraction=cfg["preprocessing"]["normal_life_fraction_threshold"])
    with torch.no_grad():
        batch = torch.tensor(X_val, dtype=torch.float32).to(device)
        vm, vs = model(batch)
        vm, vs = vm.cpu().numpy(), vs.cpu().numpy()

    nll_scorer = AnomalyScorer(score_type="nll")
    nll_scorer.fit_normalization(y_val, vm, vs)

    urd_scorer = URDScorer(fde_window=5)
    urd_scorer.fit(y_val, vm, vs)

    #Compute VALIDATION-calibrated thresholds for NLL
    nll_thresholds = nll_scorer.compute_thresholds(
        y_val, vm, vs, percentiles=[95.0, 97.5, 99.0])
    print(f"   NLL thresholds: {nll_thresholds}")

    #Compute VALIDATION-calibrated thresholds for URD combined
    urd_thresholds_all = urd_scorer.compute_thresholds(y_val, vm, vs)
    urd_combined_thresholds = urd_thresholds_all["combined"]
    print(f"   URD combined thresholds: {urd_combined_thresholds}")

    #Define scoring functions
    def nll_score_fn(y, mu, sigma, raw_values):
        scores, _ = nll_scorer.score(y, mu, sigma, normalize=True)
        return scores

    def urd_score_fn(y, mu, sigma, raw_values):
        result = urd_scorer.score(y, mu, sigma, normalize=True)
        return result["combined"]

    #Generate synthetic anomalies
    print("[3] Generating synthetic anomalies...")
    engine_data_list = []
    for eid in splits["test"]["unit_nr"].unique():
        edf = splits["test"][splits["test"]["unit_nr"] == eid].sort_values("time_cycles")
        if len(edf) <= window_size:
            continue
        engine_data_list.append({
            "engine_id": int(eid),
            "sensor_values": edf[sensors].values.copy(),
            "cycles": edf["time_cycles"].values.copy(),
            "life_fracs": edf["life_fraction"].values.copy()})
    print(f"   {len(engine_data_list)} test engines")

    generator = AnomalyGenerator(sensors, random_seed=cfg["synthetic_anomalies"]["random_seed"])
    injected = generator.generate_test_suite(
        engine_data_list=engine_data_list,
        magnitudes=cfg["synthetic_anomalies"]["magnitudes"],
        injection_positions=cfg["synthetic_anomalies"]["injection_positions"])
    print(f"   {len(injected)} injected trajectories")

    #=== EVALUATION 1: NLL-only ===
    print("\n" + "=" * 70)
    print("EVALUATION: NLL-Only Scoring (Baseline)")
    print("=" * 70)
    nll_results = evaluate_method(
        model, injected, window_size, device, nll_score_fn, nll_thresholds)

    print(f"\n   Overall: ROC-AUC={nll_results['overall']['roc_auc']:.4f}  "
          f"PR-AUC={nll_results['overall']['pr_auc']:.4f}")
    print(f"\n   Per Anomaly Type:")
    for atype, m in sorted(nll_results["per_type"].items()):
        print(f"   {atype:>20s}: ROC={m['roc_auc']:.4f}  PR={m['pr_auc']:.4f}")
    for pct, res in sorted(nll_results["threshold_results"].items()):
        pm, em = res["point"], res["event"]
        print(f"\n   p{pct}: P={pm['precision']:.3f} R={pm['recall']:.3f} F1={pm['f1']:.3f} "
              f"| EventR={em['event_recall']:.3f} Delay={em['mean_detection_delay']:.1f}")

    #=== EVALUATION 2: URD Combined (with Stationarity) ===
    print("\n" + "=" * 70)
    print("EVALUATION: URD Combined Scoring (Deviation + Stationarity)")
    print("=" * 70)
    urd_results = evaluate_method(
        model, injected, window_size, device, urd_score_fn, urd_combined_thresholds)

    print(f"\n   Overall: ROC-AUC={urd_results['overall']['roc_auc']:.4f}  "
          f"PR-AUC={urd_results['overall']['pr_auc']:.4f}")
    print(f"\n   Per Anomaly Type:")
    for atype, m in sorted(urd_results["per_type"].items()):
        print(f"   {atype:>20s}: ROC={m['roc_auc']:.4f}  PR={m['pr_auc']:.4f}")
    for pct, res in sorted(urd_results["threshold_results"].items()):
        pm, em = res["point"], res["event"]
        print(f"\n   p{pct}: P={pm['precision']:.3f} R={pm['recall']:.3f} F1={pm['f1']:.3f} "
              f"| EventR={em['event_recall']:.3f} Delay={em['mean_detection_delay']:.1f}")

    #=== HEAD-TO-HEAD ===
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD: NLL-only vs URD (Stationarity)")
    print("=" * 70)
    print(f"\n{'Anomaly Type':<22} {'NLL ROC':>9} {'URD ROC':>9} {'Change':>9} "
          f"{'NLL PR':>9} {'URD PR':>9} {'Change':>9}")
    print("-" * 80)
    all_types = sorted(set(list(nll_results["per_type"].keys()) +
                           list(urd_results["per_type"].keys())))
    for atype in all_types:
        nr = nll_results["per_type"].get(atype, {}).get("roc_auc", 0)
        ur = urd_results["per_type"].get(atype, {}).get("roc_auc", 0)
        np_ = nll_results["per_type"].get(atype, {}).get("pr_auc", 0)
        up = urd_results["per_type"].get(atype, {}).get("pr_auc", 0)
        marker = " <<<" if abs(ur - nr) > 0.05 else ""
        print(f"{atype:<22} {nr:>9.4f} {ur:>9.4f} {ur-nr:>+9.4f} "
              f"{np_:>9.4f} {up:>9.4f} {up-np_:>+9.4f}{marker}")

    no, uo = nll_results["overall"], urd_results["overall"]
    print(f"\n{'OVERALL':<22} {no['roc_auc']:>9.4f} {uo['roc_auc']:>9.4f} "
          f"{uo['roc_auc']-no['roc_auc']:>+9.4f} "
          f"{no['pr_auc']:>9.4f} {uo['pr_auc']:>9.4f} "
          f"{uo['pr_auc']-no['pr_auc']:>+9.4f}")

    #Save
    res_dir = cfg["paths"]["results_dir"]
    os.makedirs(res_dir, exist_ok=True)
    save_data = {"nll_only": {"overall": nll_results["overall"],
                              "per_type": nll_results["per_type"]},
                 "urd_stationarity": {"overall": urd_results["overall"],
                                       "per_type": urd_results["per_type"]}}
    with open(os.path.join(res_dir, "stage_c_results.json"), "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    #CSV: head-to-head comparison
    import csv
    with open(os.path.join(res_dir, "stage_c_head_to_head.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["anomaly_type", "nll_roc", "urd_roc", "delta_roc", "nll_pr", "urd_pr", "delta_pr"])
        for atype in sorted(set(list(nll_results["per_type"].keys()) + list(urd_results["per_type"].keys()))):
            nr = nll_results["per_type"].get(atype, {}).get("roc_auc", 0)
            ur = urd_results["per_type"].get(atype, {}).get("roc_auc", 0)
            np_ = nll_results["per_type"].get(atype, {}).get("pr_auc", 0)
            up = urd_results["per_type"].get(atype, {}).get("pr_auc", 0)
            w.writerow([atype, f"{nr:.4f}", f"{ur:.4f}", f"{ur-nr:+.4f}", f"{np_:.4f}", f"{up:.4f}", f"{up-np_:+.4f}"])
        w.writerow(["OVERALL", f"{no['roc_auc']:.4f}", f"{uo['roc_auc']:.4f}",
                     f"{uo['roc_auc']-no['roc_auc']:+.4f}", f"{no['pr_auc']:.4f}",
                     f"{uo['pr_auc']:.4f}", f"{uo['pr_auc']-no['pr_auc']:+.4f}"])

    #CSV: threshold-based metrics
    with open(os.path.join(res_dir, "stage_c_threshold_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "threshold_pct", "precision", "recall", "f1", "event_recall", "mean_delay"])
        for method, res in [("NLL", nll_results), ("URD", urd_results)]:
            for pct, r in sorted(res.get("threshold_results", {}).items()):
                pm, em = r["point"], r["event"]
                w.writerow([method, pct, f"{pm['precision']:.4f}", f"{pm['recall']:.4f}",
                            f"{pm['f1']:.4f}", f"{em['event_recall']:.4f}", f"{em['mean_detection_delay']:.1f}"])

    print(f"\n✓ Stage C Complete! CSVs saved to {res_dir}/")


if __name__ == "__main__":
    main()
