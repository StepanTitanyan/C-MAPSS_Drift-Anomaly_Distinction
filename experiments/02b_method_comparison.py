"""
experiments/02b_method_comparison.py
======================================
Paper-ready scoring comparison including the current URD baseline and TranAD.
"""

import os
import sys
import json
import yaml
import warnings
import numpy as np
import torch
import csv
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_train_data
from src.data.preprocessing import compute_life_fraction, select_sensors, SensorScaler
from src.data.splits import split_engines, apply_split
from src.data.windowing import create_windows
from src.models.gaussian_gru import GaussianGRU
from src.models.tranad import TranAD, TranADScorer
from src.anomaly.scoring import AnomalyScorer
from src.anomaly.urd import URDScorer
from src.synthetic.anomaly_generator import AnomalyGenerator
from src.evaluation.metrics import threshold_independent_metrics, event_level_metrics


def load_config():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.yaml")
    with open(p) as f:
        return yaml.safe_load(f)


class ConformityScorer:
    def __init__(self, window=10):
        self.window = window
        self.d_mean = self.d_std = self.c_mean = self.c_std = None
    def fit(self, y, mu, sigma):
        d = np.mean(((y - mu) / sigma) ** 2, axis=1)
        self.d_mean, self.d_std = d.mean(), max(d.std(), 1e-8)
        c = self._conf(y, mu, sigma)
        v = c[~np.isnan(c)]
        self.c_mean, self.c_std = v.mean(), max(v.std(), 1e-8)
    def _conf(self, y, mu, sigma):
        T, d = y.shape
        w = self.window
        z_sq = ((y - mu) / sigma) ** 2
        scores = np.full(T, np.nan)
        for t in range(w - 1, T):
            q = np.sum(z_sq[t - w + 1:t + 1], axis=0)
            pv = stats.chi2.cdf(q, df=w)
            scores[t] = max(0.0, -np.log(np.min(pv) + 1e-15))
        return scores
    def score(self, y, mu, sigma, raw=None):
        d = (np.mean(((y - mu) / sigma) ** 2, axis=1) - self.d_mean) / self.d_std
        c = self._conf(y, mu, sigma)
        c = np.nan_to_num((c - self.c_mean) / self.c_std, nan=0.0)
        return np.maximum(d, c)


class VarianceScorer:
    def __init__(self, window=10, eps=1e-10):
        self.window = window
        self.eps = eps
        self.var_ref = None
        self.d_mean = self.d_std = self.s_mean = self.s_std = None
    def fit(self, y, mu, sigma):
        self.var_ref = np.maximum(np.var(y, axis=0), self.eps)
        ds = np.mean(((y - mu) / sigma) ** 2, axis=1)
        self.d_mean, self.d_std = ds.mean(), max(ds.std(), 1e-8)
        ss = self._stat(y)
        v = ss[~np.isnan(ss)]
        self.s_mean, self.s_std = v.mean(), max(v.std(), 1e-8)
    def _stat(self, raw):
        T, d = raw.shape
        w = self.window
        scores = np.full(T, np.nan)
        for t in range(w - 1, T):
            var = np.var(raw[t - w + 1:t + 1], axis=0)
            ratios = var / self.var_ref
            scores[t] = np.max(np.maximum(-np.log(ratios + self.eps), 0.0))
        return scores
    def score(self, y, mu, sigma, raw=None):
        raw = y if raw is None else raw
        d = (np.mean(((y - mu) / sigma) ** 2, axis=1) - self.d_mean) / self.d_std
        s = self._stat(raw)
        s = np.nan_to_num((s - self.s_mean) / self.s_std, nan=0.0)
        return np.maximum(d, s)


class FDEScorer:
    def __init__(self, window=5, eps=1e-10):
        self.window = window
        self.eps = eps
        self.fde_ref = None
        self.d_mean = self.d_std = self.s_mean = self.s_std = None
    def fit(self, y, mu, sigma):
        diffs = np.diff(y, axis=0)
        self.fde_ref = np.maximum(np.mean(diffs ** 2, axis=0), self.eps)
        ds = np.mean(((y - mu) / sigma) ** 2, axis=1)
        self.d_mean, self.d_std = ds.mean(), max(ds.std(), 1e-8)
        ss = self._stat(y)
        v = ss[~np.isnan(ss)]
        self.s_mean, self.s_std = v.mean(), max(v.std(), 1e-8)
    def _stat(self, raw):
        T, d = raw.shape
        w = self.window
        diffs = np.zeros_like(raw)
        diffs[1:] = raw[1:] - raw[:-1]
        sq = diffs ** 2
        scores = np.full(T, np.nan)
        for t in range(w, T):
            f = np.mean(sq[t - w + 1:t + 1], axis=0)
            scores[t] = np.max(np.maximum(-np.log(f / self.fde_ref + self.eps), 0.0))
        return scores
    def score(self, y, mu, sigma, raw=None):
        raw = y if raw is None else raw
        d = (np.mean(((y - mu) / sigma) ** 2, axis=1) - self.d_mean) / self.d_std
        s = self._stat(raw)
        s = np.nan_to_num((s - self.s_mean) / self.s_std, nan=0.0)
        return np.maximum(d, s)


class IsolationForestScorer:
    def __init__(self, contamination=0.05):
        from sklearn.ensemble import IsolationForest
        self.iforest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    def fit(self, y, mu, sigma):
        self.iforest.fit(np.abs(y - mu))
    def score(self, y, mu, sigma, raw=None):
        return -self.iforest.score_samples(np.abs(y - mu))



def gaussian_predictions(model, values, ws, device):
    T, d = values.shape
    if T <= ws:
        return None, None, None
    X = np.array([values[i:i + ws] for i in range(T - ws)], dtype=np.float32)
    y = np.array([values[i + ws] for i in range(T - ws)], dtype=np.float32)
    with torch.no_grad():
        mu, sigma = model(torch.tensor(X).to(device))
    return y, mu.cpu().numpy(), sigma.cpu().numpy()


def tranad_predictions(model, values, ws, device):
    T, d = values.shape
    if T <= ws:
        return None, None
    X = np.array([values[i:i + ws] for i in range(T - ws)], dtype=np.float32)
    y = np.array([values[i + ws] for i in range(T - ws)], dtype=np.float32)
    with torch.no_grad():
        pred = model.predict_next(torch.tensor(X).to(device))
    return y, pred.cpu().numpy()


def make_gaussian_score_fn(model, ws, device, fn):
    def scorer(values):
        out = gaussian_predictions(model, values, ws, device)
        if out[0] is None:
            return None
        y, mu, sigma = out
        raw = values[ws:ws + len(y)]
        return fn(y, mu, sigma, raw)
    return scorer


def make_tranad_score_fn(model, ws, device, scorer):
    def score(values):
        out = tranad_predictions(model, values, ws, device)
        if out[0] is None:
            return None
        y, pred = out
        return scorer.score(y, pred, normalize=True)
    return score



def evaluate_method(injected, score_traj_fn):
    per_type = {}
    all_s, all_l = [], []
    for traj in injected:
        scores = score_traj_fn(traj.sensor_values)
        if scores is None:
            continue
        labels = traj.labels[len(traj.sensor_values) - len(scores):]
        binary = (labels > 0).astype(int)
        all_s.extend(scores.tolist())
        all_l.extend(binary.tolist())
        if traj.events:
            at = traj.events[0].anomaly_type
            per_type.setdefault(at, {"s": [], "l": []})
            per_type[at]["s"].extend(scores.tolist())
            per_type[at]["l"].extend(binary.tolist())
    all_s, all_l = np.array(all_s), np.array(all_l)
    overall = threshold_independent_metrics(all_l, all_s)
    thr = np.percentile(all_s[all_l == 0], 95) if np.any(all_l == 0) else np.percentile(all_s, 95)
    ev = event_level_metrics(all_l, all_s, thr)
    type_results = {}
    for at, data in per_type.items():
        s, l = np.array(data["s"]), np.array(data["l"])
        type_results[at] = threshold_independent_metrics(l, s) if len(np.unique(l)) >= 2 else {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {"overall": overall, "per_type": type_results, "event": ev}



def main():
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sensors = cfg["dataset"]["selected_sensors"]
    ws = cfg["preprocessing"]["window_size"]

    print("=" * 116)
    print("PAPER TABLE: Scoring Method Comparison  |  Model: Gaussian GRU + TranAD baseline")
    print("=" * 116)

    df = load_train_data(cfg["paths"]["raw_data_dir"], cfg["dataset"]["subset"])
    df = compute_life_fraction(df)
    df = select_sensors(df, sensors, keep_meta=True)
    ti, vi, si = split_engines(df, cfg["preprocessing"]["train_ratio"], cfg["preprocessing"]["val_ratio"], cfg["preprocessing"]["test_ratio"], cfg["preprocessing"]["split_random_seed"])
    splits = apply_split(df, ti, vi, si)
    sc = SensorScaler(sensors)
    splits["train"] = sc.fit_transform(splits["train"])
    splits["val"] = sc.transform(splits["val"])
    splits["test"] = sc.transform(splits["test"])

    model = GaussianGRU(input_size=len(sensors), hidden_size=cfg["model"]["hidden_size"], num_layers=cfg["model"]["num_layers"], dropout=0.0, sigma_min=cfg["model"]["sigma_min"])
    model.load_state_dict(torch.load(os.path.join(cfg["paths"]["model_dir"], "gaussian_gru_best.pt"), map_location=device))
    model.to(device)
    model.eval()

    tcfg = cfg["model"].get("tranad", {})
    tranad = TranAD(input_size=len(sensors), window_size=ws, d_model=tcfg.get("d_model", 64), nhead=tcfg.get("nhead", 4), num_layers=tcfg.get("num_layers", 2), dim_feedforward=tcfg.get("dim_feedforward", 128), dropout=tcfg.get("dropout", 0.1))
    tranad.load_state_dict(torch.load(os.path.join(cfg["paths"]["model_dir"], "tranad_best.pt"), map_location=device))
    tranad.to(device)
    tranad.eval()

    print("\n[1] Fitting scorers on healthy validation windows...")
    Xv, yv, _ = create_windows(splits["val"], sensors, window_size=ws, max_life_fraction=cfg["preprocessing"]["normal_life_fraction_threshold"])
    with torch.no_grad():
        b = torch.tensor(Xv, dtype=torch.float32).to(device)
        vm, vs = model(b)
        vm, vs = vm.cpu().numpy(), vs.cpu().numpy()
        vp = tranad.predict_next(b).cpu().numpy()

    nll_sc = AnomalyScorer(score_type="nll")
    nll_sc.fit_normalization(yv, vm, vs)
    conf_sc = ConformityScorer(window=10)
    conf_sc.fit(yv, vm, vs)
    var_sc = VarianceScorer(window=10)
    var_sc.fit(yv, vm, vs)
    fde_sc = FDEScorer(window=5)
    fde_sc.fit(yv, vm, vs)
    urd_base = URDScorer(fde_window=5)
    urd_base.fit(yv, vm, vs)
    tranad_sc = TranADScorer()
    tranad_sc.fit(yv, vp)
    if_sc = IsolationForestScorer(contamination=0.05)
    if_sc.fit(yv, vm, vs)

    methods = [
        ("NLL", make_gaussian_score_fn(model, ws, device, lambda y, mu, sigma, raw: nll_sc.score(y, mu, sigma, normalize=True)[0])),
        ("D+Conformity", make_gaussian_score_fn(model, ws, device, lambda y, mu, sigma, raw: conf_sc.score(y, mu, sigma, raw))),
        ("D+Variance", make_gaussian_score_fn(model, ws, device, lambda y, mu, sigma, raw: var_sc.score(y, mu, sigma, raw))),
        ("D+FDE", make_gaussian_score_fn(model, ws, device, lambda y, mu, sigma, raw: fde_sc.score(y, mu, sigma, raw))),
        ("URD (baseline)", make_gaussian_score_fn(model, ws, device, lambda y, mu, sigma, raw: urd_base.score(y, mu, sigma, normalize=True)["combined"])),
        ("TranAD", make_tranad_score_fn(tranad, ws, device, tranad_sc)),
        ("IForest", make_gaussian_score_fn(model, ws, device, lambda y, mu, sigma, raw: if_sc.score(y, mu, sigma, raw))),
    ]

    print("\n[2] Generating synthetic anomalies...")
    engine_list = []
    for eid in splits["test"]["unit_nr"].unique():
        edf = splits["test"][splits["test"]["unit_nr"] == eid].sort_values("time_cycles")
        if len(edf) <= ws:
            continue
        engine_list.append({
            "engine_id": int(eid),
            "sensor_values": edf[sensors].values.copy(),
            "cycles": edf["time_cycles"].values.copy(),
            "life_fracs": edf["life_fraction"].values.copy(),
        })
    ag = AnomalyGenerator(sensors, random_seed=cfg["synthetic_anomalies"]["random_seed"])
    injected = ag.generate_test_suite(engine_list, magnitudes=cfg["synthetic_anomalies"]["magnitudes"], injection_positions=cfg["synthetic_anomalies"]["injection_positions"])
    print(f"   {len(injected)} trajectories")

    print("\n[3] Evaluating all methods...")
    results = {}
    for name, fn in methods:
        print(f"   {name}...", end="", flush=True)
        results[name] = evaluate_method(injected, fn)
        print(f" ROC={results[name]['overall']['roc_auc']:.4f}")

    all_types = sorted(set().union(*(r["per_type"].keys() for r in results.values())))
    mnames = [m[0] for m in methods]

    cw = 16
    print("\n" + "=" * 160)
    print("TABLE 1: ROC-AUC")
    print("=" * 160)
    print(f"{'Type':<24}" + "".join(f"{m:>{cw}}" for m in mnames))
    print("-" * (24 + cw * len(mnames)))
    roc_rows = []
    pr_rows = []
    for at in all_types:
        vals_roc = []
        vals_pr = []
        for m in mnames:
            rr = results[m]["per_type"].get(at, {})
            vals_roc.append(rr.get("roc_auc", float("nan")))
            vals_pr.append(rr.get("pr_auc", float("nan")))
        print(f"{at:<24}" + "".join(f"{v:>{cw}.4f}" if not np.isnan(v) else f"{'n/a':>{cw}}" for v in vals_roc))
        roc_rows.append([at] + vals_roc)
        pr_rows.append([at] + vals_pr)
    ovals = [results[m]["overall"]["roc_auc"] for m in mnames]
    print("-" * (24 + cw * len(mnames)))
    print(f"{'OVERALL':<24}" + "".join(f"{v:>{cw}.4f}" for v in ovals))
    roc_rows.append(["OVERALL"] + ovals)

    print("\n" + "=" * 160)
    print("TABLE 4: Sensor Freeze Progression")
    print("=" * 160)
    freeze0 = results["NLL"]["per_type"].get("sensor_freeze", {}).get("roc_auc", 0.0)
    freeze_rows = []
    desc = {
        "NLL": "[residuals (prob)]",
        "D+Conformity": "[residuals (chi²)]",
        "D+Variance": "[RAW (variance)]",
        "D+FDE": "[RAW (Δx²)]",
        "URD (baseline)": "[calibrated Mahalanobis D + tuned S + weighted fusion]",
        "TranAD": "[two-phase transformer next-step error]",
        "IForest": "[residuals (tree)]",
    }
    for m in mnames:
        fr = results[m]["per_type"].get("sensor_freeze", {}).get("roc_auc", float("nan"))
        fp = results[m]["per_type"].get("sensor_freeze", {}).get("pr_auc", float("nan"))
        print(f"  {m:<22} ROC={fr:.4f} PR={fp:.4f} ΔROC={fr-freeze0:+.4f}  {desc.get(m,'')}")
        freeze_rows.append([m, fr, fp, fr - freeze0, desc.get(m, "")])

    res_dir = cfg["paths"]["results_dir"]
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, "method_comparison_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    with open(os.path.join(res_dir, "method_comparison_roc.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type"] + mnames)
        for row in roc_rows:
            w.writerow(row)

    with open(os.path.join(res_dir, "method_comparison_pr.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type"] + mnames)
        for row in pr_rows:
            w.writerow(row)

    with open(os.path.join(res_dir, "method_comparison_freeze_progression.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "freeze_roc", "freeze_pr", "delta_roc_vs_nll", "description"])
        for row in freeze_rows:
            w.writerow(row)

    print("\n  CSVs saved to outputs/results/")
    print("  Method Comparison Complete!")


if __name__ == "__main__":
    main()
