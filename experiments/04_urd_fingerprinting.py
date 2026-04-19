"""
experiments/04_urd_fingerprinting.py
========================================
Stage E: URD Anomaly Fingerprinting.

Primary model: Gaussian GRU (gaussian_gru_best.pt).

4 Experiments:
  1. 5-class actionable taxonomy (main paper result)
  2. 9-class per-type breakdown (supplementary)
  3. Spike vs drop distinction (signed_deviation ablation)
  4. Feature ablation: 16 URD vs 9 standard

Usage:
    python -m experiments.04_urd_fingerprinting
"""

import os, sys, json, yaml, numpy as np, torch, csv
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_train_data
from src.data.preprocessing import compute_life_fraction, select_sensors, SensorScaler
from src.data.splits import split_engines, apply_split
from src.data.windowing import create_windows
from src.models.gaussian_gru import GaussianGRU
from src.anomaly.scoring import AnomalyScorer
from src.anomaly.urd import URDScorer, extract_urd_features, URD_FEATURE_NAMES
from src.synthetic.anomaly_generator import AnomalyGenerator
from src.synthetic.drift_generator import DriftGenerator
from src.drift.features import extract_event_features

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


#Type → actionable category mapping
TYPE_TO_CATEGORY = {
    "spike": "point_anomaly", "drop": "point_anomaly",
    "persistent_offset": "persistent_shift",
    "noise_burst": "noise_anomaly",
    "sensor_freeze": "sensor_malfunction",
    "gradual_shift": "drift", "sigmoid_plateau": "drift",
    "accelerating": "drift", "multi_sensor": "drift"}


def run_model(model, values, ws, device):
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
        b = torch.tensor(X, dtype=torch.float32).to(device)
        mu, sigma = model(b)
    return y, mu.cpu().numpy(), sigma.cpu().numpy()


def extract_fp(model, traj, ws, device, nll_sc, urd_sc, aw=15):
    y, mu, sigma = run_model(model, traj.sensor_values, ws, device)
    if y is None:
        return None
    labels = traj.labels[ws:]
    ml = min(len(labels), len(y))
    labels, y_a, mu_a, sigma_a = labels[:ml], y[:ml], mu[:ml], sigma[:ml]
    if not np.any(labels > 0):
        return None
    nll_scores, _ = nll_sc.score(y_a, mu_a, sigma_a, normalize=True)
    urd_result = urd_sc.score(y_a, mu_a, sigma_a, normalize=True)
    residuals = np.abs(y_a - mu_a)
    events = []
    in_ev, start = False, 0
    for i in range(len(labels)):
        if labels[i] > 0 and not in_ev:
            in_ev, start = True, i
        elif labels[i] == 0 and in_ev:
            events.append((start, i))
            in_ev = False
    if in_ev:
        events.append((start, len(labels)))
    if not events:
        return None
    std_names = URD_FEATURE_NAMES[:9]
    rows = []
    for s, e in events:
        esc = nll_scores[s:e]
        ci = s + np.argmax(np.abs(esc))
        sf = extract_event_features(nll_scores, residuals, sigma_a, ci, aw)
        uf = extract_urd_features(urd_result, ci, aw)
        row = [sf[n] for n in std_names]
        row.extend([uf["deviation_at_peak"], uf["uncertainty_at_peak"], uf["stationarity_at_peak"], uf["uncertainty_slope"], uf["stationarity_max"], uf["du_ratio"], uf["signed_deviation_mean"]])
        rows.append(row)
    return rows


def plot_cm(cm, names, title, save_path):
    fig, ax = plt.subplots(figsize=(max(8, len(names)*1.1), max(6, len(names)*0.9)))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    n = len(names)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=12); ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=13)
    for i in range(n):
        for j in range(n):
            c = "white" if cm[i, j] > cm.max()/2 else "black"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=c, fontsize=10)
    plt.tight_layout(); plt.savefig(save_path, bbox_inches="tight", dpi=150); plt.close()


def plot_signatures(profiles, names, save_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    channels = ["Deviation (D)", "Uncertainty (U)", "Stationarity (S)"]
    x = np.arange(len(channels))
    w = min(0.14, 0.8 / len(names))
    colors = ["#E53935", "#FF7043", "#FFA726", "#66BB6A", "#42A5F5", "#7E57C2", "#26A69A", "#EF5350", "#78909C"]
    for i, (nm, p) in enumerate(zip(names, profiles)):
        ax.bar(x + (i - len(names)/2 + 0.5)*w, p, w, label=nm, color=colors[i % len(colors)])
    ax.set_xticks(x); ax.set_xticklabels(channels, fontsize=13)
    ax.set_ylabel("Mean Value", fontsize=12)
    ax.set_title("URD Signature Profiles", fontsize=14)
    ax.legend(fontsize=8, ncol=2); ax.axhline(y=0, color="gray", lw=0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, bbox_inches="tight", dpi=150); plt.close()


def main():
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sensors = cfg["dataset"]["selected_sensors"]
    ws = cfg["preprocessing"]["window_size"]

    print("=" * 70)
    print("Stage E: URD Anomaly Fingerprinting  |  Model: Gaussian GRU")
    print("=" * 70)

    df = load_train_data(cfg["paths"]["raw_data_dir"], cfg["dataset"]["subset"])
    df = compute_life_fraction(df); df = select_sensors(df, sensors, keep_meta=True)
    ti, vi, si = split_engines(df, cfg["preprocessing"]["train_ratio"], cfg["preprocessing"]["val_ratio"], cfg["preprocessing"]["test_ratio"], cfg["preprocessing"]["split_random_seed"])
    splits = apply_split(df, ti, vi, si)
    sc = SensorScaler(sensors)
    splits["train"] = sc.fit_transform(splits["train"])
    splits["val"] = sc.transform(splits["val"])
    splits["test"] = sc.transform(splits["test"])

    model = GaussianGRU(input_size=len(sensors), hidden_size=cfg["model"]["hidden_size"], num_layers=cfg["model"]["num_layers"], dropout=0.0, sigma_min=cfg["model"]["sigma_min"])
    model.load_state_dict(torch.load(os.path.join(cfg["paths"]["model_dir"], "gaussian_gru_best.pt"), map_location=device))
    model.to(device)

    Xv, yv, _ = create_windows(splits["val"], sensors, window_size=ws, max_life_fraction=cfg["preprocessing"]["normal_life_fraction_threshold"])
    with torch.no_grad():
        b = torch.tensor(Xv, dtype=torch.float32).to(device)
        vm, vs = model(b); vm, vs = vm.cpu().numpy(), vs.cpu().numpy()
    nll_sc = AnomalyScorer(score_type="nll"); nll_sc.fit_normalization(yv, vm, vs)
    urd_sc = URDScorer(fde_window=5); urd_sc.fit(yv, vm, vs)

    print("\n[1] Generating typed trajectories...")
    def prep(s):
        return [{"engine_id": int(eid), "sensor_values": s[s["unit_nr"] == eid].sort_values("time_cycles")[sensors].values.copy(), "cycles": s[s["unit_nr"] == eid].sort_values("time_cycles")["time_cycles"].values.copy(), "life_fracs": s[s["unit_nr"] == eid].sort_values("time_cycles")["life_fraction"].values.copy()} for eid in s["unit_nr"].unique() if len(s[s["unit_nr"] == eid]) > ws]

    ve, te = prep(splits["val"]), prep(splits["test"])
    ag = AnomalyGenerator(sensors, random_seed=123)
    dg = DriftGenerator(sensors, random_seed=456)
    atypes = ["spike", "drop", "persistent_offset", "noise_burst", "sensor_freeze"]
    dtypes = ["gradual_shift", "sigmoid_plateau", "accelerating", "multi_sensor"]

    def gen_typed(engines):
        typed = {}
        for at in atypes:
            typed[at] = ag.generate_test_suite(engines, magnitudes=[3.0, 5.0], injection_positions={"mid": 0.5}, anomaly_types=[at])
        for dt in dtypes:
            typed[dt] = dg.generate_test_suite(engines, rates=[0.03, 0.05], durations=[30, 60], injection_positions={"mid": 0.5}, drift_types=[dt])
        return typed

    val_typed, test_typed = gen_typed(ve), gen_typed(te)

    print("\n[2] Extracting features...")
    def extract_all(typed):
        feats, fine_labels, cat_labels = [], [], []
        for tname, trajs in typed.items():
            cat = TYPE_TO_CATEGORY[tname]
            for traj in trajs:
                rows = extract_fp(model, traj, ws, device, nll_sc, urd_sc)
                if rows:
                    for r in rows:
                        feats.append(r); fine_labels.append(tname); cat_labels.append(cat)
        return (np.array(feats) if feats else np.empty((0, len(URD_FEATURE_NAMES))), fine_labels, cat_labels)

    X_tr, fine_tr, cat_tr = extract_all(val_typed)
    X_te, fine_te, cat_te = extract_all(test_typed)
    print(f"   Train: {len(X_tr)} events | Test: {len(X_te)} events")
    print(f"   Fine-type dist (test): {dict(Counter(fine_te))}")
    print(f"   Category dist (test): {dict(Counter(cat_te))}")

    if len(X_tr) == 0 or len(X_te) == 0:
        print("ERROR: no events"); return

    fsc = StandardScaler()
    X_tr_s, X_te_s = fsc.fit_transform(X_tr), fsc.transform(X_te)
    d_col, u_col, s_col = 9, 10, 11

    print("\n" + "=" * 70)
    print("EXPERIMENT 1: 5-Class Actionable Category Classification")
    print("=" * 70)
    le_cat = LabelEncoder()
    y_tr_cat, y_te_cat = le_cat.fit_transform(cat_tr), le_cat.transform(cat_te)
    cat_names = le_cat.classes_.tolist()
    clf_5 = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
    clf_5.fit(X_tr_s, y_tr_cat)
    pred_5 = clf_5.predict(X_te_s)
    acc_5 = float(np.mean(pred_5 == y_te_cat))
    report_5 = classification_report(y_te_cat, pred_5, target_names=cat_names, output_dict=True)
    cm_5 = confusion_matrix(y_te_cat, pred_5)
    print(f"\n   5-Class Accuracy: {acc_5:.3f}")
    print(f"\n   {'Category':<22} {'Prec':>8} {'Recall':>8} {'F1':>8} {'N':>6}")
    print("   " + "-" * 52)
    for cn in cat_names:
        r = report_5[cn]
        print(f"   {cn:<22} {r['precision']:>8.3f} {r['recall']:>8.3f} {r['f1-score']:>8.3f} {r['support']:>6.0f}")

    profiles_5 = []
    print(f"\n   URD Signature Profiles (5-class):")
    for cn in cat_names:
        mask = np.array(cat_te) == cn
        if mask.sum() > 0:
            p = [X_te[mask, d_col].mean(), X_te[mask, u_col].mean(), X_te[mask, s_col].mean()]
            profiles_5.append(p)
            print(f"   {cn:<22}: D={p[0]:>8.2f}  U={p[1]:>8.3f}  S={p[2]:>8.2f}")
        else:
            profiles_5.append([0, 0, 0])

    imp_5 = dict(zip(URD_FEATURE_NAMES, clf_5.feature_importances_))
    top_5 = sorted(imp_5.items(), key=lambda x: -x[1])[:6]
    print(f"\n   Top features: {[(n, round(v, 4)) for n, v in top_5]}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: 9-Class Per-Type Classification")
    print("=" * 70)
    le_fine = LabelEncoder()
    y_tr_fine, y_te_fine = le_fine.fit_transform(fine_tr), le_fine.transform(fine_te)
    fine_names = le_fine.classes_.tolist()
    clf_9 = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
    clf_9.fit(X_tr_s, y_tr_fine)
    pred_9 = clf_9.predict(X_te_s)
    acc_9 = float(np.mean(pred_9 == y_te_fine))
    report_9 = classification_report(y_te_fine, pred_9, target_names=fine_names, output_dict=True)
    cm_9 = confusion_matrix(y_te_fine, pred_9)
    print(f"\n   9-Class Accuracy: {acc_9:.3f}")
    profiles_9 = []
    for fn in fine_names:
        mask = np.array(fine_te) == fn
        p = [X_te[mask, d_col].mean(), X_te[mask, u_col].mean(), X_te[mask, s_col].mean()] if mask.sum() > 0 else [0, 0, 0]
        profiles_9.append(p)

    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Spike vs Drop Distinction")
    print("=" * 70)
    sd_mask_tr = np.array([(f in ["spike", "drop"]) for f in fine_tr])
    sd_mask_te = np.array([(f in ["spike", "drop"]) for f in fine_te])
    acc_sd, acc_nosign = 0.0, 0.0
    if sd_mask_tr.sum() > 0 and sd_mask_te.sum() > 0:
        X_sd_tr, X_sd_te = X_tr_s[sd_mask_tr], X_te_s[sd_mask_te]
        y_sd_tr, y_sd_te = np.array(fine_tr)[sd_mask_tr], np.array(fine_te)[sd_mask_te]
        le_sd = LabelEncoder()
        y_sd_tr_e, y_sd_te_e = le_sd.fit_transform(y_sd_tr), le_sd.transform(y_sd_te)
        clf_sd = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42)
        clf_sd.fit(X_sd_tr, y_sd_tr_e)
        acc_sd = float(np.mean(clf_sd.predict(X_sd_te) == y_sd_te_e))
        signed_idx = URD_FEATURE_NAMES.index("signed_deviation_mean")
        clf_nosign = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42)
        clf_nosign.fit(np.delete(X_sd_tr, signed_idx, axis=1), y_sd_tr_e)
        acc_nosign = float(np.mean(clf_nosign.predict(np.delete(X_sd_te, signed_idx, axis=1)) == y_sd_te_e))
        print(f"\n   With signed_deviation:    {acc_sd:.3f}")
        print(f"   Without signed_deviation: {acc_nosign:.3f}")
        print(f"   Improvement:              {acc_sd - acc_nosign:+.3f}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Feature Ablation (5-Class)")
    print("=" * 70)
    fsc_9 = StandardScaler()
    X_tr_9 = fsc_9.fit_transform(X_tr[:, :9]); X_te_9 = fsc_9.transform(X_te[:, :9])
    clf_abl = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
    clf_abl.fit(X_tr_9, y_tr_cat)
    acc_abl = float(np.mean(clf_abl.predict(X_te_9) == y_te_cat))
    print(f"\n   16 URD features (5-class): {acc_5:.3f}")
    print(f"    9 standard features:      {acc_abl:.3f}")
    print(f"   Improvement from URD:      {acc_5 - acc_abl:+.3f}")

    #Save
    fig_dir = cfg["paths"]["figure_dir"]
    res_dir = cfg["paths"]["results_dir"]
    os.makedirs(fig_dir, exist_ok=True); os.makedirs(res_dir, exist_ok=True)

    plot_cm(cm_5, cat_names, f"5-Class Fingerprinting (Acc: {acc_5:.1%})", os.path.join(fig_dir, "fingerprint_5class_cm.png"))
    plot_cm(cm_9, fine_names, f"9-Class Per-Type (Acc: {acc_9:.1%})", os.path.join(fig_dir, "fingerprint_9class_cm.png"))
    plot_signatures(profiles_5, cat_names, os.path.join(fig_dir, "urd_profiles_5class.png"))
    plot_signatures(profiles_9, fine_names, os.path.join(fig_dir, "urd_profiles_9class.png"))

    save_data = {
        "five_class": {"accuracy": acc_5, "report": report_5, "profiles": {n: {"D": p[0], "U": p[1], "S": p[2]} for n, p in zip(cat_names, profiles_5)}},
        "nine_class": {"accuracy": acc_9, "report": report_9, "profiles": {n: {"D": p[0], "U": p[1], "S": p[2]} for n, p in zip(fine_names, profiles_9)}},
        "spike_vs_drop": {"accuracy_with_signed": acc_sd, "accuracy_without_signed": acc_nosign},
        "ablation": {"16feat": acc_5, "9feat": acc_abl}}
    with open(os.path.join(res_dir, "stage_e_results.json"), "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    with open(os.path.join(res_dir, "stage_e_5class.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "precision", "recall", "f1", "support", "D_mean", "U_mean", "S_mean"])
        for i, cn in enumerate(cat_names):
            r = report_5[cn]; p = profiles_5[i]
            w.writerow([cn, f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1-score']:.4f}", int(r['support']), f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}"])
        w.writerow(["ACCURACY", f"{acc_5:.4f}", "", "", "", "", "", ""])

    with open(os.path.join(res_dir, "stage_e_9class.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "precision", "recall", "f1", "support", "D_mean", "U_mean", "S_mean"])
        for i, fn in enumerate(fine_names):
            r = report_9[fn]; p = profiles_9[i]
            w.writerow([fn, f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1-score']:.4f}", int(r['support']), f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}"])
        w.writerow(["ACCURACY", f"{acc_9:.4f}", "", "", "", "", "", ""])

    with open(os.path.join(res_dir, "stage_e_spike_drop.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "accuracy"])
        w.writerow(["with_signed_deviation", f"{acc_sd:.4f}"])
        w.writerow(["without_signed_deviation", f"{acc_nosign:.4f}"])
        w.writerow(["improvement", f"{acc_sd - acc_nosign:+.4f}"])

    with open(os.path.join(res_dir, "stage_e_ablation.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature_set", "n_features", "accuracy_5class"])
        w.writerow(["URD_full", 16, f"{acc_5:.4f}"])
        w.writerow(["standard_only", 9, f"{acc_abl:.4f}"])
        w.writerow(["improvement", "", f"{acc_5-acc_abl:+.4f}"])

    with open(os.path.join(res_dir, "stage_e_feature_importance.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "importance"])
        for fn, fv in sorted(imp_5.items(), key=lambda x: -x[1]):
            w.writerow([fn, f"{fv:.4f}"])

    print(f"\n  Stage E Complete! CSVs saved to {res_dir}/")


if __name__ == "__main__":
    main()
