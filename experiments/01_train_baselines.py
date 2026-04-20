"""
experiments/01_train_baselines.py
=================================
Stage A: Train all baseline and probabilistic models.

This script runs the complete pipeline:
1. Load FD001 data
2. Preprocess (sensor selection, life_fraction, scaling)
3. Split engines into train/val/test
4. Create rolling windows (normal-only for training)
5. Train: Gaussian LSTM, Gaussian GRU, Deterministic LSTM, Deterministic GRU
6. Evaluate prediction quality
7. Compute initial anomaly scores on test data
8. Run degradation analysis (score vs life_fraction)
9. Save all models, results, and figures

Usage:
    cd project/
    python -m experiments.01_train_baselines

    Or with custom config:
    python -m experiments.01_train_baselines --config config/config.yaml
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
import torch
import time

#Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_train_data
from src.data.preprocessing import (compute_life_fraction, select_sensors, SensorScaler, filter_normal_region)
from src.data.splits import split_engines, apply_split, get_split_summary
from src.data.windowing import (create_windows, create_dataloader, create_full_sequence_windows)
from src.models.gaussian_lstm import GaussianLSTM, DeterministicLSTM
from src.models.gaussian_gru import GaussianGRU, DeterministicGRU
from src.models.tranad import TranAD, TranADTrainer
from src.models.baselines import NaivePersistence, RidgeBaseline
from src.training.trainer import Trainer
from src.anomaly.scoring import AnomalyScorer
from src.anomaly.smoothing import smooth_engine_scores
from src.evaluation.degradation import full_degradation_report
from src.visualization.plots import (plot_training_curves, plot_prediction_bands, plot_score_vs_life_fraction_aggregated)


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_data(cfg: dict):
    """
    Full data preparation pipeline.
    Returns: splits dict, scaler, windowed data, metadata.
    """
    print("=" * 70)
    print("PHASE 1: Data Preparation")
    print("=" * 70)

    #1. Load raw data
    data_dir = cfg["paths"]["raw_data_dir"]
    subset = cfg["dataset"]["subset"]
    print(f"\n[1/6] Loading {subset} training data from {data_dir}...")
    df = load_train_data(data_dir, subset)
    print(f"       Loaded {len(df)} rows, {df['unit_nr'].nunique()} engines")

    #2. Compute life fraction
    print("[2/6] Computing life_fraction...")
    df = compute_life_fraction(df)

    #3. Select sensors
    sensors = cfg["dataset"]["selected_sensors"]
    print(f"[3/6] Selecting {len(sensors)} sensors: {sensors}")
    df = select_sensors(df, sensors, keep_meta=True)

    #4. Split engines
    print("[4/6] Splitting engines into train/val/test...")
    train_ids, val_ids, test_ids = split_engines(
        df,
        train_ratio=cfg["preprocessing"]["train_ratio"],
        val_ratio=cfg["preprocessing"]["val_ratio"],
        test_ratio=cfg["preprocessing"]["test_ratio"],
        random_seed=cfg["preprocessing"]["split_random_seed"])
    splits = apply_split(df, train_ids, val_ids, test_ids)
    summary = get_split_summary(splits)
    print(summary.to_string(index=False))

    #5. Scale sensors (fit on training data only!)
    print("[5/6] Fitting scaler on training data...")
    scaler = SensorScaler(sensors)
    splits["train"] = scaler.fit_transform(splits["train"])
    splits["val"] = scaler.transform(splits["val"])
    splits["test"] = scaler.transform(splits["test"])
    print(f"       Train sensor means: {scaler.means_.values.round(2)}")
    print(f"       Train sensor stds:  {scaler.stds_.values.round(2)}")

    #6.Create windowed sequences
    print("[6/6] Creating rolling windows...")
    window_size = cfg["preprocessing"]["window_size"]
    normal_threshold = cfg["preprocessing"]["normal_life_fraction_threshold"]

    #Normal-only windows for training (first 50% of each engine's life)
    X_train, y_train, meta_train = create_windows(
        splits["train"], sensors, window_size=window_size,
        max_life_fraction=normal_threshold)
    #Normal-only windows for validation
    X_val, y_val, meta_val = create_windows(
        splits["val"], sensors, window_size=window_size,
        max_life_fraction=normal_threshold)
    #ALL windows for test (full trajectories — we need late life for evaluation)
    X_test, y_test, meta_test = create_windows(
        splits["test"], sensors, window_size=window_size)

    print(f"       Train (normal): {X_train.shape[0]} windows")
    print(f"       Val (normal):   {X_val.shape[0]} windows")
    print(f"       Test (full):    {X_test.shape[0]} windows")

    #Create DataLoaders
    batch_size = cfg["training"]["batch_size"]
    train_loader = create_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = create_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)

    #Also create per-engine test windows for trajectory-level analysis
    test_engine_windows = create_full_sequence_windows(splits["test"], sensors, window_size=window_size)

    return {
        "splits": splits,
        "scaler": scaler,
        "sensors": sensors,
        "loaders": {"train": train_loader, "val": val_loader, "test": test_loader},
        "arrays": {
            "X_train": X_train, "y_train": y_train, "meta_train": meta_train,
            "X_val": X_val, "y_val": y_val, "meta_val": meta_val,
            "X_test": X_test, "y_test": y_test, "meta_test": meta_test,
        },
        "test_engine_windows": test_engine_windows}


def train_neural_model(model_name: str, model: torch.nn.Module, loss_type: str, loaders: dict, cfg: dict, device: str) -> dict:
    """Train one neural model and return results."""
    print(f"\n--- Training {model_name} (loss={loss_type}) ---")

    trainer = Trainer(
        model=model,
        loss_type=loss_type,
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        max_epochs=cfg["training"]["max_epochs"],
        patience=cfg["training"]["early_stopping_patience"],
        lr_patience=cfg["training"]["lr_scheduler"]["patience"],
        lr_factor=cfg["training"]["lr_scheduler"]["factor"],
        min_lr=cfg["training"]["lr_scheduler"]["min_lr"],
        gradient_clip_norm=cfg["training"]["gradient_clip_norm"],
        device=device,
        checkpoint_dir=cfg["paths"]["model_dir"],
        model_name=model_name)

    history = trainer.fit(loaders["train"], loaders["val"], verbose=True)

    return {"model": model, "trainer": trainer, "history": history}




def train_tranad_model(model_name: str, model: torch.nn.Module, loaders: dict, cfg: dict, device: str) -> dict:
    """Train the practical TranAD baseline on healthy windows."""
    print(f"\n--- Training {model_name} (loss=tranad) ---")

    tcfg = cfg["model"].get("tranad", {})
    trainer = TranADTrainer(
        model=model,
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        max_epochs=cfg["training"]["max_epochs"],
        patience=cfg["training"]["early_stopping_patience"],
        lr_patience=cfg["training"]["lr_scheduler"]["patience"],
        lr_factor=cfg["training"]["lr_scheduler"]["factor"],
        min_lr=cfg["training"]["lr_scheduler"]["min_lr"],
        gradient_clip_norm=cfg["training"]["gradient_clip_norm"],
        phase2_weight=tcfg.get("phase2_weight", 1.5),
        device=device,
        checkpoint_dir=cfg["paths"]["model_dir"],
        model_name=model_name)

    history = trainer.fit(loaders["train"], loaders["val"], verbose=True)
    return {"model": model, "trainer": trainer, "history": history}


def evaluate_tranad_predictions(model: torch.nn.Module, X: np.ndarray, y: np.ndarray, device: str, batch_size: int = 256) -> dict:
    """Run TranAD inference and compute next-step prediction metrics."""
    model.eval()
    model.to(device)

    all_pred = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            pred = model.predict_next(batch)
            all_pred.append(pred.cpu().numpy())

    pred = np.concatenate(all_pred, axis=0)
    errors = y - pred
    mse = float(np.mean(errors ** 2))
    mae = float(np.mean(np.abs(errors)))
    per_sensor_mae = np.mean(np.abs(errors), axis=0)
    return {
        "mu": pred,
        "sigma": np.ones_like(pred),
        "mse": mse,
        "mae": mae,
        "per_sensor_mae": per_sensor_mae.tolist(),
        "mean_sigma": 1.0,
    }

def evaluate_model_predictions(model: torch.nn.Module, X: np.ndarray, y: np.ndarray, device: str, batch_size: int = 256) -> dict:
    """Run model inference and compute prediction metrics."""
    model.eval()
    model.to(device)

    all_mu = []
    all_sigma = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            mu, sigma = model(batch)
            all_mu.append(mu.cpu().numpy())
            all_sigma.append(sigma.cpu().numpy())

    mu = np.concatenate(all_mu, axis=0)
    sigma = np.concatenate(all_sigma, axis=0)

    #Compute MSE and MAE
    errors = y - mu
    mse = float(np.mean(errors ** 2))
    mae = float(np.mean(np.abs(errors)))

    #Per-sensor MAE
    per_sensor_mae = np.mean(np.abs(errors), axis=0)

    #Mean predicted sigma (for probabilistic models)
    mean_sigma = float(np.mean(sigma))

    return {
        "mu": mu, "sigma": sigma,
        "mse": mse, "mae": mae,
        "per_sensor_mae": per_sensor_mae.tolist(),
        "mean_sigma": mean_sigma}


def run_degradation_analysis(model: torch.nn.Module, test_engine_windows: dict, scorer: AnomalyScorer, device: str, cfg: dict):
    """Score full test trajectories and analyze score vs degradation."""
    print("\n--- Degradation Analysis ---")

    engine_scores = {}

    model.eval()
    model.to(device)

    for engine_id, data in test_engine_windows.items():
        X = data["X"]
        y = data["y"]

        if len(X) == 0:
            continue

        #Run inference
        with torch.no_grad():
            batch = torch.tensor(X, dtype=torch.float32).to(device)
            mu, sigma = model(batch)
            mu = mu.cpu().numpy()
            sigma = sigma.cpu().numpy()

        #Compute scores
        scores, per_sensor = scorer.score(y, mu, sigma, normalize=True)

        engine_scores[engine_id] = {
            "scores": scores,
            "cycles": data["cycles"],
            "life_fracs": data["life_fracs"],
            "mu": mu,
            "sigma": sigma,
            "y": y}

    #Smooth scores
    smoothing_cfg = cfg["anomaly_scoring"]["smoothing"]
    if smoothing_cfg["enabled"]:
        engine_scores = smooth_engine_scores(engine_scores, method="ema", alpha=smoothing_cfg["alpha"])

    #Run degradation report
    #Build sigma data for uncertainty analysis
    engine_sigmas = {}
    for eid, data in engine_scores.items():
        if "sigma" in data:
            engine_sigmas[eid] = {"sigmas": data["sigma"], "life_fracs": data["life_fracs"]}

    report = full_degradation_report(engine_scores, engine_sigmas)
    return engine_scores, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    #Load config
    cfg = load_config(args.config)

    #Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    #Set random seeds
    seed = cfg["training"]["random_seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    #Create output directories
    for d in ["model_dir", "figure_dir", "results_dir", "log_dir"]:
        os.makedirs(cfg["paths"][d], exist_ok=True)

    # =========================================================================
    #Prepare Data
    # =========================================================================
    data = prepare_data(cfg)
    sensors = data["sensors"]
    n_sensors = len(sensors)

    # =========================================================================
    #Train Models
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Model Training")
    print("=" * 70)

    model_cfg = cfg["model"]
    results = {}

    #--- Gaussian LSTM (PRIMARY MODEL) ---
    g_lstm = GaussianLSTM(
        input_size=n_sensors,
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        sigma_min=model_cfg["sigma_min"])
    results["gaussian_lstm"] = train_neural_model("gaussian_lstm", g_lstm, "nll", data["loaders"], cfg, device)

    #--- Gaussian GRU ---
    g_gru = GaussianGRU(
        input_size=n_sensors,
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
        sigma_min=model_cfg["sigma_min"])
    results["gaussian_gru"] = train_neural_model("gaussian_gru", g_gru, "nll", data["loaders"], cfg, device)

    #--- Deterministic LSTM (MSE baseline) ---
    d_lstm = DeterministicLSTM(
        input_size=n_sensors,
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"])
    results["deterministic_lstm"] = train_neural_model("deterministic_lstm", d_lstm, "mse", data["loaders"], cfg, device)

    #--- Deterministic GRU (MSE baseline) ---
    d_gru = DeterministicGRU(
        input_size=n_sensors,
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"])
    results["deterministic_gru"] = train_neural_model("deterministic_gru", d_gru, "mse", data["loaders"], cfg, device)

    #--- Practical TranAD baseline ---
    tranad_cfg = model_cfg.get("tranad", {})
    tranad = TranAD(
        input_size=n_sensors,
        window_size=cfg["preprocessing"]["window_size"],
        d_model=tranad_cfg.get("d_model", 64),
        nhead=tranad_cfg.get("nhead", 4),
        num_layers=tranad_cfg.get("num_layers", 2),
        dim_feedforward=tranad_cfg.get("dim_feedforward", 128),
        dropout=tranad_cfg.get("dropout", 0.1),
    )
    results["tranad"] = train_tranad_model("tranad", tranad, data["loaders"], cfg, device)

    #--- Non-neural baselines ---
    print("\n--- Non-neural Baselines ---")

    #Naive persistence
    naive = NaivePersistence()
    naive_mu, naive_sigma = naive.predict(data["arrays"]["X_test"])
    naive_mse = float(np.mean((data["arrays"]["y_test"] - naive_mu) ** 2))
    naive_mae = float(np.mean(np.abs(data["arrays"]["y_test"] - naive_mu)))
    print(f"Naive Persistence — Test MSE: {naive_mse:.6f}, MAE: {naive_mae:.6f}")

    #Ridge regression
    ridge = RidgeBaseline(alpha=1.0)
    ridge.fit(data["arrays"]["X_train"], data["arrays"]["y_train"])
    ridge_mu, ridge_sigma = ridge.predict(data["arrays"]["X_test"])
    ridge_mse = float(np.mean((data["arrays"]["y_test"] - ridge_mu) ** 2))
    ridge_mae = float(np.mean(np.abs(data["arrays"]["y_test"] - ridge_mu)))
    print(f"Ridge Regression — Test MSE: {ridge_mse:.6f}, MAE: {ridge_mae:.6f}")

    # =========================================================================
    #Evaluate All Models
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Evaluation")
    print("=" * 70)

    eval_results = {}
    for name, res in results.items():
        print(f"\nEvaluating {name}...")
        if name == "tranad":
            eval_res = evaluate_tranad_predictions(res["model"], data["arrays"]["X_test"], data["arrays"]["y_test"], device)
        else:
            eval_res = evaluate_model_predictions(res["model"], data["arrays"]["X_test"], data["arrays"]["y_test"], device)
        eval_results[name] = eval_res
        print(f"  MSE: {eval_res['mse']:.6f}")
        print(f"  MAE: {eval_res['mae']:.6f}")
        print(f"  Mean σ: {eval_res['mean_sigma']:.6f}")
        print(f"  Per-sensor MAE: {[f'{v:.4f}' for v in eval_res['per_sensor_mae']]}")

    eval_results["naive"] = {"mse": naive_mse, "mae": naive_mae}
    eval_results["ridge"] = {"mse": ridge_mse, "mae": ridge_mae}

    # =========================================================================
    #Anomaly Scoring & Degradation Analysis (Primary Model)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: Anomaly Scoring & Degradation Analysis")
    print("=" * 70)

    primary_name = cfg["model"]["primary"]
    primary_model = results[primary_name]["model"]
    primary_eval = eval_results[primary_name]

    #Fit anomaly scorer on validation normal data
    print(f"\nFitting anomaly scorer ({cfg['anomaly_scoring']['score_type']}) "
          f"on validation data...")
    scorer = AnomalyScorer(score_type=cfg["anomaly_scoring"]["score_type"])

    if primary_name == "tranad":
        val_eval = evaluate_tranad_predictions(primary_model, data["arrays"]["X_val"], data["arrays"]["y_val"], device)
    else:
        val_eval = evaluate_model_predictions(primary_model, data["arrays"]["X_val"], data["arrays"]["y_val"], device)
    scorer.fit_normalization(data["arrays"]["y_val"], val_eval["mu"], val_eval["sigma"])

    #Compute thresholds
    thresholds = scorer.compute_thresholds(data["arrays"]["y_val"], val_eval["mu"], val_eval["sigma"], percentiles=cfg["anomaly_scoring"]["thresholds"]["percentiles"],)
    print(f"Thresholds: {thresholds}")

    #Run degradation analysis
    engine_scores, deg_report = run_degradation_analysis(primary_model, data["test_engine_windows"], scorer, device, cfg)

    print(f"\nDegradation Report:")
    corr = deg_report["score_correlation"]
    print(f"  Mean Spearman ρ (score vs life): {corr['mean_spearman']:.4f}")
    print(f"  % engines with positive corr:    {corr['pct_positive']*100:.1f}%")

    buckets = deg_report["bucketed_analysis"]["bucket_stats"]
    for bname, bstats in buckets.items():
        print(f"  {bname:>8s}: mean={bstats['mean']:.4f}, median={bstats['median']:.4f}")

    kw = deg_report["bucketed_analysis"]["kruskal_wallis"]
    print(f"  Kruskal-Wallis: H={kw['statistic']:.2f}, p={kw['p_value']:.2e}")

    if "uncertainty_analysis" in deg_report:
        ua = deg_report["uncertainty_analysis"]
        print(f"  σ vs life corr: {ua['mean_sigma_life_corr']:.4f} "
              f"({ua['pct_positive']*100:.1f}% positive)")

    # =========================================================================
    #Save Results & Figures
    # =========================================================================
    print("\n" + "=" * 70)
    print("Saving Results & Figures")
    print("=" * 70)

    fig_dir = cfg["paths"]["figure_dir"]
    res_dir = cfg["paths"]["results_dir"]

    #Training curves
    for name, res in results.items():
        plot_training_curves(
            res["history"],
            title=f"Training Curves — {name}",
            save_path=os.path.join(fig_dir, f"training_curves_{name}.png"))

    #Prediction bands for primary model (first 3 test engines)
    for idx, (eid, edata) in enumerate(engine_scores.items()):
        if idx >= 3:
            break
        plot_prediction_bands(
            true_values=edata["y"],
            mu=edata["mu"],
            sigma=edata["sigma"],
            cycles=edata["cycles"],
            sensor_names=sensors,
            engine_id=eid,
            n_sensors=4,
            save_path=os.path.join(fig_dir, f"prediction_bands_engine_{eid}.png"))

    #Aggregated score vs life fraction
    plot_score_vs_life_fraction_aggregated(
        engine_scores,
        title="Mean Anomaly Score vs Life Fraction (All Test Engines)",
        save_path=os.path.join(fig_dir, "score_vs_life_fraction.png"))

    #Save numerical results
    save_results = {
        "eval_results": {
            k: {kk: vv for kk, vv in v.items()
                 if not isinstance(vv, np.ndarray)}
            for k, v in eval_results.items()
        },
        "thresholds": thresholds,
        "degradation_report": {
            "score_correlation": {
                k: v for k, v in deg_report["score_correlation"].items()
                if k != "per_engine"
            },
            "bucketed_analysis": deg_report["bucketed_analysis"]}}

    with open(os.path.join(res_dir, "stage_a_results.json"), "w") as f:
        json.dump(save_results, f, indent=2, default=str)

    print(f"\nFigures saved to {fig_dir}/")
    print(f"Results saved to {res_dir}/")
    print("\n✓ Stage A Complete!")


if __name__ == "__main__":
    main()
