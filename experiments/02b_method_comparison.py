"""
experiments/02b_method_comparison.py
======================================
Stage C+: Comprehensive scoring method comparison for the paper.

Primary model: Gaussian GRU (gaussian_gru_best.pt).

6 methods showing the progression:
  1. NLL              — Standard baseline (residual-based)
  2. D+Conformity     — Chi-squared on residuals (shows why residuals fail)
  3. D+Variance       — Variance ratio on RAW values (first improvement)
  4. D+FDE            — First-difference energy (better statistic)
  5. D+FDE+Run (URD)  — FDE + additive run bonus (our best method)
  6. IForest          — Isolation Forest baseline

Usage:
    python -m experiments.02b_method_comparison
"""

import os, sys, json, yaml, warnings, numpy as np, torch, csv
from scipy import stats
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_train_data
from src.data.preprocessing import compute_life_fraction, select_sensors, SensorScaler
from src.data.splits import split_engines, apply_split
from src.data.windowing import create_windows
from src.models.gaussian_gru import GaussianGRU
from src.anomaly.scoring import AnomalyScorer
from src.anomaly.urd import URDScorer
from src.synthetic.anomaly_generator import AnomalyGenerator
from src.evaluation.metrics import threshold_independent_metrics, event_level_metrics


def load_config():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.yaml")
    with open(p) as f:
        return yaml.safe_load(f)


class DeviationScorer:
    def __init__(self):
        self.mean = self.std = None
    def fit(self, y, mu, sigma):
        d = np.mean(((y-mu)/sigma)**2, axis=1)
        self.mean, self.std = d.mean(), max(d.std(), 1e-8)
    def score(self, y, mu, sigma, raw=None):
        return (np.mean(((y-mu)/sigma)**2, axis=1) - self.mean) / self.std


class ConformityScorer:
    def __init__(self, window=10):
        self.window = window
        self.d_mean = self.d_std = self.c_mean = self.c_std = None
    def fit(self, y, mu, sigma):
        d = np.mean(((y-mu)/sigma)**2, axis=1)
        self.d_mean, self.d_std = d.mean(), max(d.std(), 1e-8)
        c = self._conf(y, mu, sigma)
        v = c[~np.isnan(c)]
        self.c_mean, self.c_std = v.mean(), max(v.std(), 1e-8)
    def _conf(self, y, mu, sigma):
        T, d = y.shape
        w = self.window
        z_sq = ((y-mu)/sigma)**2
        scores = np.full(T, np.nan)
        for t in range(w-1, T):
            q = np.sum(z_sq[t-w+1:t+1], axis=0)
            pv = stats.chi2.cdf(q, df=w)
            scores[t] = max(0, -np.log(np.min(pv) + 1e-15))
        return scores
    def score(self, y, mu, sigma, raw=None):
        d = (np.mean(((y-mu)/sigma)**2, axis=1) - self.d_mean) / self.d_std
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
        ds = np.mean(((y-mu)/sigma)**2, axis=1)
        self.d_mean, self.d_std = ds.mean(), max(ds.std(), 1e-8)
        ss = self._stat(y)
        v = ss[~np.isnan(ss)]
        self.s_mean, self.s_std = v.mean(), max(v.std(), 1e-8)
    def _stat(self, raw):
        T, d = raw.shape
        w = self.window
        scores = np.full(T, np.nan)
        for t in range(w-1, T):
            v = np.var(raw[t-w+1:t+1], axis=0)
            r = v / self.var_ref
            scores[t] = np.max(np.maximum(-np.log(r + self.eps), 0.0))
        return scores
    def score(self, y, mu, sigma, raw=None):
        if raw is None:
            raw = y
        d = (np.mean(((y-mu)/sigma)**2, axis=1) - self.d_mean) / self.d_std
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
        self.fde_ref = np.maximum(np.mean(diffs**2, axis=0), self.eps)
        ds = np.mean(((y-mu)/sigma)**2, axis=1)
        self.d_mean, self.d_std = ds.mean(), max(ds.std(), 1e-8)
        ss = self._stat(y)
        v = ss[~np.isnan(ss)]
        self.s_mean, self.s_std = v.mean(), max(v.std(), 1e-8)
    def _stat(self, raw):
        T, d = raw.shape
        w = self.window
        diffs = np.zeros_like(raw)
        diffs[1:] = raw[1:] - raw[:-1]
        sq = diffs**2
        scores = np.full(T, np.nan)
        for t in range(w, T):
            f = np.mean(sq[t-w+1:t+1], axis=0)
            scores[t] = np.max(np.maximum(-np.log(f / self.fde_ref + self.eps), 0.0))
        return scores
    def score(self, y, mu, sigma, raw=None):
        if raw is None:
            raw = y
        d = (np.mean(((y-mu)/sigma)**2, axis=1) - self.d_mean) / self.d_std
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


def score_trajectory(model, values, ws, device):
    T, d = values.shape
    if T <= ws:
        return None, None, None
    X, y = [], []
    for i in range(T - ws):
        X.append(values[i:i+ws])
        y.append(values[i+ws])
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        mu, sigma = model(torch.tensor(X).to(device))
    return y, mu.cpu().numpy(), sigma.cpu().numpy()


def evaluate_method(model, injected, ws, device, fn):
    per_type = {}
    all_s, all_l = [], []
    for traj in injected:
        y, mu, sigma = score_trajectory(model, traj.sensor_values, ws, device)
        if y is None:
            continue
        labels = traj.labels[ws:]
        ml = min(len(labels), len(y))
        labels, y_a, mu_a, sigma_a = labels[:ml], y[:ml], mu[:ml], sigma[:ml]
        raw = traj.sensor_values[ws:ws+ml]
        scores = fn(y_a, mu_a, sigma_a, raw)
        binary = (labels > 0).astype(int)
        all_s.extend(scores.tolist())
        all_l.extend(binary.tolist())
        if traj.events:
            at = traj.events[0].anomaly_type
            if at not in per_type:
                per_type[at] = {"s": [], "l": []}
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

    print("=" * 100)
    print("PAPER TABLE: Scoring Method Comparison  |  Model: Gaussian GRU")
    print("=" * 100)

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
    model.to(device); model.eval()

    print("\n[1] Fitting scorers...")
    Xv, yv, _ = create_windows(splits["val"], sensors, window_size=ws, max_life_fraction=cfg["preprocessing"]["normal_life_fraction_threshold"])
    with torch.no_grad():
        b = torch.tensor(Xv, dtype=torch.float32).to(device)
        vm, vs = model(b)
        vm, vs = vm.cpu().numpy(), vs.cpu().numpy()

    nll_sc = AnomalyScorer(score_type="nll"); nll_sc.fit_normalization(yv, vm, vs)
    conf_sc = ConformityScorer(window=10); conf_sc.fit(yv, vm, vs)
    var_sc = VarianceScorer(window=10); var_sc.fit(yv, vm, vs)
    fde_sc = FDEScorer(window=5); fde_sc.fit(yv, vm, vs)
    urd_sc = URDScorer(fde_window=5); urd_sc.fit(yv, vm, vs)
    if_sc = IsolationForestScorer(contamination=0.05); if_sc.fit(yv, vm, vs)

    methods = [
        ("NLL", lambda y, mu, s, r: nll_sc.score(y, mu, s, normalize=True)[0]),
        ("D+Conformity", lambda y, mu, s, r: conf_sc.score(y, mu, s, r)),
        ("D+Variance", lambda y, mu, s, r: var_sc.score(y, mu, s, r)),
        ("D+FDE", lambda y, mu, s, r: fde_sc.score(y, mu, s, r)),
        ("D+FDE+Run (URD)", lambda y, mu, s, r: urd_sc.score(y, mu, s, normalize=True)["combined"]),
        ("IForest", lambda y, mu, s, r: if_sc.score(y, mu, s, r))]

    print("\n[2] Generating synthetic anomalies...")
    el = []
    for eid in splits["test"]["unit_nr"].unique():
        edf = splits["test"][splits["test"]["unit_nr"] == eid].sort_values("time_cycles")
        if len(edf) <= ws:
            continue
        el.append({"engine_id": int(eid), "sensor_values": edf[sensors].values.copy(), "cycles": edf["time_cycles"].values.copy(), "life_fracs": edf["life_fraction"].values.copy()})
    ag = AnomalyGenerator(sensors, random_seed=cfg["synthetic_anomalies"]["random_seed"])
    injected = ag.generate_test_suite(el, magnitudes=cfg["synthetic_anomalies"]["magnitudes"], injection_positions=cfg["synthetic_anomalies"]["injection_positions"])
    print(f"   {len(injected)} trajectories")

    print("\n[3] Evaluating all 6 methods...")
    results = {}
    for name, fn in methods:
        print(f"   {name}...", end="", flush=True)
        results[name] = evaluate_method(model, injected, ws, device, fn)
        print(f" ROC={results[name]['overall']['roc_auc']:.4f}")

    all_types = sorted(set().union(*(r["per_type"].keys() for r in results.values())))
    mnames = [m[0] for m in methods]

    cw = 16
    print("\n" + "=" * 100)
    print("TABLE 1: ROC-AUC")
    print("=" * 100)
    hdr = f"{'Type':<20}" + "".join(f"{m:>{cw}}" for m in mnames)
    print(hdr); print("-" * len(hdr))
    for at in all_types:
        row = f"{at:<20}"
        for mn in mnames:
            row += f"{results[mn]['per_type'].get(at, {}).get('roc_auc', 0):>{cw}.4f}"
        print(row)
    print("-" * len(hdr))
    row = f"{'OVERALL':<20}"
    for mn in mnames:
        row += f"{results[mn]['overall']['roc_auc']:>{cw}.4f}"
    print(row)

    print("\n" + "=" * 100)
    print("TABLE 4: Sensor Freeze Progression")
    print("=" * 100)
    nll_sf = results["NLL"]["per_type"].get("sensor_freeze", {}).get("roc_auc", 0)
    sigs = {"NLL": "residuals", "D+Conformity": "residuals (chi²)", "D+Variance": "RAW (variance)", "D+FDE": "RAW (Δx²)", "D+FDE+Run (URD)": "RAW (Δx²+run)", "IForest": "residuals (tree)"}
    for mn in mnames:
        roc = results[mn]["per_type"].get("sensor_freeze", {}).get("roc_auc", 0)
        pr = results[mn]["per_type"].get("sensor_freeze", {}).get("pr_auc", 0)
        print(f"  {mn:<25} ROC={roc:.4f} PR={pr:.4f} ΔROC={roc-nll_sf:+.4f}  [{sigs.get(mn, '')}]")

    #Save CSVs
    res_dir = cfg["paths"]["results_dir"]
    os.makedirs(res_dir, exist_ok=True)

    with open(os.path.join(res_dir, "method_comparison_roc.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["anomaly_type"] + mnames)
        for at in all_types:
            w.writerow([at] + [f"{results[mn]['per_type'].get(at, {}).get('roc_auc', 0):.4f}" for mn in mnames])
        w.writerow(["OVERALL"] + [f"{results[mn]['overall']['roc_auc']:.4f}" for mn in mnames])

    with open(os.path.join(res_dir, "method_comparison_pr.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["anomaly_type"] + mnames)
        for at in all_types:
            w.writerow([at] + [f"{results[mn]['per_type'].get(at, {}).get('pr_auc', 0):.4f}" for mn in mnames])
        w.writerow(["OVERALL"] + [f"{results[mn]['overall']['pr_auc']:.4f}" for mn in mnames])

    with open(os.path.join(res_dir, "method_comparison_freeze_progression.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "signal_space", "freeze_roc", "freeze_pr", "delta_roc_vs_nll"])
        for mn in mnames:
            roc = results[mn]["per_type"].get("sensor_freeze", {}).get("roc_auc", 0)
            pr = results[mn]["per_type"].get("sensor_freeze", {}).get("pr_auc", 0)
            w.writerow([mn, sigs.get(mn, ""), f"{roc:.4f}", f"{pr:.4f}", f"{roc-nll_sf:+.4f}"])

    with open(os.path.join(res_dir, "method_comparison_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "overall_roc", "overall_pr", "freeze_roc", "freeze_pr", "event_recall"])
        for mn in mnames:
            w.writerow([mn, f"{results[mn]['overall']['roc_auc']:.4f}", f"{results[mn]['overall']['pr_auc']:.4f}", f"{results[mn]['per_type'].get('sensor_freeze', {}).get('roc_auc', 0):.4f}", f"{results[mn]['per_type'].get('sensor_freeze', {}).get('pr_auc', 0):.4f}", f"{results[mn]['event']['event_recall']:.4f}"])

    with open(os.path.join(res_dir, "method_comparison_results.json"), "w") as f:
        json.dump({mn: {"overall": results[mn]["overall"], "per_type": results[mn]["per_type"], "event": results[mn]["event"]} for mn in mnames}, f, indent=2, default=str)

    print(f"\n  CSVs saved to {res_dir}/")
    print(f"  Method Comparison Complete!")


if __name__ == "__main__":
    main()
