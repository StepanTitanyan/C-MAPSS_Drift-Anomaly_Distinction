"""
experiments/02_synthetic_evaluation.py
========================================
Stage C: direct comparison between the current URD baseline and TranAD.

Both methods use:
- same FD001 engine split
- same 7 selected sensors
- same window length
- same healthy-only training assumption
- same synthetic anomaly generation and evaluation protocol
"""

import os
import sys
import json
import yaml
import numpy as np
import torch
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_train_data
from src.data.preprocessing import compute_life_fraction, select_sensors, SensorScaler
from src.data.splits import split_engines, apply_split
from src.data.windowing import create_windows
from src.models.gaussian_gru import GaussianGRU
from src.models.tranad import TranAD, TranADScorer
from src.anomaly.urd import URDScorer
from src.synthetic.anomaly_generator import AnomalyGenerator
from src.evaluation.metrics import (
    threshold_independent_metrics,
    point_level_metrics,
    event_level_metrics,
    compute_curves,
    false_alarms_per_1000,
    threshold_sweep_metrics,
)


def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)



def gaussian_predictions(model, values, window_size, device):
    T, d = values.shape
    if T <= window_size:
        return None, None, None
    X = np.array([values[i:i + window_size] for i in range(T - window_size)], dtype=np.float32)
    y = np.array([values[i + window_size] for i in range(T - window_size)], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        batch = torch.tensor(X, dtype=torch.float32).to(device)
        mu, sigma = model(batch)
    return y, mu.cpu().numpy(), sigma.cpu().numpy()



def tranad_predictions(model, values, window_size, device):
    T, d = values.shape
    if T <= window_size:
        return None, None
    X = np.array([values[i:i + window_size] for i in range(T - window_size)], dtype=np.float32)
    y = np.array([values[i + window_size] for i in range(T - window_size)], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        batch = torch.tensor(X, dtype=torch.float32).to(device)
        pred = model.predict_next(batch)
    return y, pred.cpu().numpy()



def aggregate_threshold_metrics(all_labels, all_scores, labels_by_traj, scores_by_traj, thresholds):
    out = {}
    n_traj = max(len(labels_by_traj), 1)
    for pct, thr in thresholds.items():
        pm = point_level_metrics(all_labels, all_scores, thr)
        em = event_level_metrics(all_labels, all_scores, thr)
        fp_total = 0
        for lab, sc in zip(labels_by_traj, scores_by_traj):
            fp_total += int(np.sum((sc >= thr) & (lab == 0)))
        out[pct] = {
            "point": pm,
            "event": em,
            "false_alarms_per_1000": false_alarms_per_1000(all_labels, all_scores, thr),
            "false_alarms_per_engine": float(fp_total / n_traj),
        }
    return out



def evaluate_method(trajectories, score_traj_fn, thresholds):
    all_scores = []
    all_labels = []
    labels_by_traj = []
    scores_by_traj = []
    per_type = {}

    for traj in trajectories:
        scores = score_traj_fn(traj.sensor_values)
        if scores is None:
            continue
        labels = traj.labels[len(traj.sensor_values) - len(scores):]
        binary = (labels > 0).astype(int)

        all_scores.extend(scores.tolist())
        all_labels.extend(binary.tolist())
        labels_by_traj.append(binary)
        scores_by_traj.append(scores)

        if traj.events:
            atype = traj.events[0].anomaly_type
            if atype not in per_type:
                per_type[atype] = {"scores": [], "labels": []}
            per_type[atype]["scores"].extend(scores.tolist())
            per_type[atype]["labels"].extend(binary.tolist())

    all_scores = np.array(all_scores, dtype=np.float64)
    all_labels = np.array(all_labels, dtype=np.int64)
    overall = threshold_independent_metrics(all_labels, all_scores)
    curves = compute_curves(all_labels, all_scores)

    type_results = {}
    for atype, data in per_type.items():
        s = np.array(data["scores"], dtype=np.float64)
        l = np.array(data["labels"], dtype=np.int64)
        if len(np.unique(l)) >= 2:
            type_results[atype] = threshold_independent_metrics(l, s)
        else:
            type_results[atype] = {"roc_auc": float("nan"), "pr_auc": float("nan")}

    thr_results = aggregate_threshold_metrics(all_labels, all_scores, labels_by_traj, scores_by_traj, thresholds)
    sweep_grid = np.quantile(all_scores, np.linspace(0.80, 0.995, 40)) if len(all_scores) > 0 else np.array([])
    sweep = threshold_sweep_metrics(all_labels, all_scores, np.unique(sweep_grid)) if len(sweep_grid) > 0 else []

    return {
        "overall": overall,
        "per_type": type_results,
        "threshold_results": thr_results,
        "curves": curves,
        "sweep": sweep,
    }



def print_method_results(title, results):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"\n   Overall: ROC-AUC={results['overall']['roc_auc']:.4f}  PR-AUC={results['overall']['pr_auc']:.4f}")
    print("\n   Per Anomaly Type:")
    for atype, m in sorted(results["per_type"].items()):
        print(f"   {atype:>20s}: ROC={m['roc_auc']:.4f}  PR={m['pr_auc']:.4f}")
    for pct, res in sorted(results["threshold_results"].items()):
        pm, em = res["point"], res["event"]
        print(
            f"\n   p{pct}: P={pm['precision']:.3f} R={pm['recall']:.3f} F1={pm['f1']:.3f} | "
            f"EventP={em['event_precision']:.3f} EventR={em['event_recall']:.3f} EventF1={em['event_f1']:.3f} | "
            f"Delay={em['mean_detection_delay']:.1f} | FP/1k={res['false_alarms_per_1000']:.1f} | FP/eng={res['false_alarms_per_engine']:.2f}"
        )



def main():
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sensors = cfg["dataset"]["selected_sensors"]
    window_size = cfg["preprocessing"]["window_size"]

    print("=" * 70)
    print("Stage C: URD Baseline vs TranAD | Same FD001 / sensors / window / healthy-only setup")
    print("=" * 70)

    print("\n[1] Loading data and models...")
    df = load_train_data(cfg["paths"]["raw_data_dir"], cfg["dataset"]["subset"])
    df = compute_life_fraction(df)
    df = select_sensors(df, sensors, keep_meta=True)
    train_ids, val_ids, test_ids = split_engines(
        df,
        cfg["preprocessing"]["train_ratio"],
        cfg["preprocessing"]["val_ratio"],
        cfg["preprocessing"]["test_ratio"],
        cfg["preprocessing"]["split_random_seed"],
    )
    splits = apply_split(df, train_ids, val_ids, test_ids)
    scaler = SensorScaler(sensors)
    splits["train"] = scaler.fit_transform(splits["train"])
    splits["val"] = scaler.transform(splits["val"])
    splits["test"] = scaler.transform(splits["test"])

    gru = GaussianGRU(
        input_size=len(sensors),
        hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"],
        dropout=0.0,
        sigma_min=cfg["model"]["sigma_min"],
    )
    gru.load_state_dict(torch.load(os.path.join(cfg["paths"]["model_dir"], "gaussian_gru_best.pt"), map_location=device))
    gru.to(device)
    gru.eval()

    tcfg = cfg["model"].get("tranad", {})
    tranad = TranAD(
        input_size=len(sensors),
        window_size=window_size,
        d_model=tcfg.get("d_model", 64),
        nhead=tcfg.get("nhead", 4),
        num_layers=tcfg.get("num_layers", 2),
        dim_feedforward=tcfg.get("dim_feedforward", 128),
        dropout=tcfg.get("dropout", 0.1),
    )
    tranad.load_state_dict(torch.load(os.path.join(cfg["paths"]["model_dir"], "tranad_best.pt"), map_location=device))
    tranad.to(device)
    tranad.eval()

    print("[2] Fitting scorers on healthy validation windows...")
    X_val, y_val, _ = create_windows(
        splits["val"], sensors, window_size=window_size,
        max_life_fraction=cfg["preprocessing"]["normal_life_fraction_threshold"]
    )
    with torch.no_grad():
        batch = torch.tensor(X_val, dtype=torch.float32).to(device)
        vm, vs = gru(batch)
        vm = vm.cpu().numpy(); vs = vs.cpu().numpy()
        vp = tranad.predict_next(batch).cpu().numpy()

    urd_scorer = URDScorer(fde_window=5)
    urd_scorer.fit(y_val, vm, vs)
    urd_thresholds = urd_scorer.compute_thresholds(y_val, vm, vs)["combined"]

    tranad_scorer = TranADScorer()
    tranad_scorer.fit(y_val, vp)
    tranad_thresholds = tranad_scorer.compute_thresholds(y_val, vp, percentiles=[95.0, 97.5, 99.0])
    print(f"   URD thresholds:    {urd_thresholds}")
    print(f"   TranAD thresholds: {tranad_thresholds}")

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
            "life_fracs": edf["life_fraction"].values.copy(),
        })
    generator = AnomalyGenerator(sensors, random_seed=cfg["synthetic_anomalies"]["random_seed"])
    injected = generator.generate_test_suite(
        engine_data_list=engine_data_list,
        magnitudes=cfg["synthetic_anomalies"]["magnitudes"],
        injection_positions=cfg["synthetic_anomalies"]["injection_positions"],
    )
    print(f"   {len(injected)} injected trajectories")

    def urd_score_traj(values):
        y, mu, sigma = gaussian_predictions(gru, values, window_size, device)
        if y is None:
            return None
        return urd_scorer.score(y, mu, sigma, normalize=True)["combined"]

    def tranad_score_traj(values):
        y, pred = tranad_predictions(tranad, values, window_size, device)
        if y is None:
            return None
        return tranad_scorer.score(y, pred, normalize=True)

    urd_results = evaluate_method(injected, urd_score_traj, urd_thresholds)
    tranad_results = evaluate_method(injected, tranad_score_traj, tranad_thresholds)

    print_method_results("EVALUATION: URD Baseline", urd_results)
    print_method_results("EVALUATION: TranAD", tranad_results)

    print("\n" + "=" * 82)
    print("HEAD-TO-HEAD: URD Baseline vs TranAD")
    print("=" * 82)
    print(f"\n{'Anomaly Type':<22} {'URD ROC':>9} {'TranAD ROC':>11} {'Change':>9} {'URD PR':>9} {'TranAD PR':>11} {'Change':>9}")
    print("-" * 82)
    all_types = sorted(set(list(urd_results["per_type"].keys()) + list(tranad_results["per_type"].keys())))
    for atype in all_types:
        ur = urd_results["per_type"].get(atype, {}).get("roc_auc", 0)
        tr = tranad_results["per_type"].get(atype, {}).get("roc_auc", 0)
        up = urd_results["per_type"].get(atype, {}).get("pr_auc", 0)
        tp = tranad_results["per_type"].get(atype, {}).get("pr_auc", 0)
        marker = " <<<" if abs(ur - tr) > 0.05 else ""
        print(f"{atype:<22} {ur:>9.4f} {tr:>11.4f} {tr-ur:>+9.4f} {up:>9.4f} {tp:>11.4f} {tp-up:>+9.4f}{marker}")
    uo, to = urd_results["overall"], tranad_results["overall"]
    print(f"\n{'OVERALL':<22} {uo['roc_auc']:>9.4f} {to['roc_auc']:>11.4f} {to['roc_auc']-uo['roc_auc']:>+9.4f} {uo['pr_auc']:>9.4f} {to['pr_auc']:>11.4f} {to['pr_auc']-uo['pr_auc']:>+9.4f}")

    res_dir = cfg["paths"]["results_dir"]
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, "stage_c_results.json"), "w") as f:
        json.dump({"urd": urd_results, "tranad": tranad_results}, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    # threshold sweep CSV
    sweep_path = os.path.join(res_dir, "stage_c_threshold_sweep.csv")
    with open(sweep_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "threshold", "precision", "recall", "f1", "event_precision", "event_recall", "event_f1", "delay", "false_alarms_per_1000"])
        for method_name, data in [("URD", urd_results), ("TranAD", tranad_results)]:
            for row in data["sweep"]:
                w.writerow([
                    method_name, row["threshold"], row["precision"], row["recall"], row["f1"],
                    row["event_precision"], row["event_recall"], row["event_f1"], row["delay"], row["false_alarms_per_1000"],
                ])

    print(f"\nResults saved to {res_dir}/")
    print("\nStage C Complete!")


if __name__ == "__main__":
    main()
