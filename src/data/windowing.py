"""
src/data/windowing.py
=====================
Create rolling-window sequences from engine trajectories.

For each engine, we slide a window of size W across the sensor time series
and pair it with the next-step target:

    Input:  x[t-W+1], x[t-W+2], ..., x[t]     shape: (W, d)
    Target: x[t+1]                               shape: (d,)

where d = number of sensors.

Windows are created per-engine (never crossing engine boundaries).
Optionally, windows can be filtered to only include the "normal" region
of engine life for training purposes.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

#Lazy torch import — numpy-only functions work without torch installed
try:
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def create_windows(df: pd.DataFrame, sensor_cols: List[str], window_size: int = 30, forecast_horizon: int = 1, max_life_fraction: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build rolling-window sequences from a dataframe of engine trajectories.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed and scaled dataframe. Must contain 'unit_nr', 'time_cycles',
        and all columns in sensor_cols.
    sensor_cols : list of str
        Which sensor columns to include in the windows.
    window_size : int
        Number of past time steps in each input window.
    forecast_horizon : int
        How many steps ahead to predict. Default 1 = next-step prediction.
    max_life_fraction : float or None
        If set, only create windows whose TARGET falls within this life fraction.
        Used to create "normal-only" training sets. This means both the input
        window AND the target must come from the early portion of engine life.

    Returns
    -------
    X : np.ndarray of shape (N, window_size, num_sensors)
        Input windows.
    y : np.ndarray of shape (N, num_sensors)
        Target vectors (next-step sensor values).
    meta : np.ndarray of shape (N, 3)
        Metadata per window: [engine_id, target_cycle, target_life_fraction].
        This is essential for later analysis (e.g., plotting scores vs life stage).
    """
    X_list = []
    y_list = []
    meta_list = []

    for engine_id in df["unit_nr"].unique():
        #Get this engine's data, sorted by cycle
        engine_df = df[df["unit_nr"] == engine_id].sort_values("time_cycles")
        sensor_values = engine_df[sensor_cols].values  # shape: (T, d)

        #Get life_fraction for filtering
        if "life_fraction" in engine_df.columns:
            life_fracs = engine_df["life_fraction"].values
        else:
            life_fracs = np.ones(len(engine_df))  # dummy if not available

        cycles = engine_df["time_cycles"].values
        T = len(sensor_values)

        #Slide the window
        for i in range(T - window_size - forecast_horizon + 1):
            target_idx = i + window_size + forecast_horizon - 1
            target_lf = life_fracs[target_idx]

            # Filter: if max_life_fraction is set, skip windows beyond it
            if max_life_fraction is not None and target_lf > max_life_fraction:
                continue

            window = sensor_values[i: i + window_size]     # (W, d)
            target = sensor_values[target_idx]              # (d,)

            X_list.append(window)
            y_list.append(target)
            meta_list.append([engine_id, cycles[target_idx], target_lf])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta = np.array(meta_list, dtype=np.float32)

    return X, y, meta


def create_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle: bool = True, num_workers: int = 0,) -> "DataLoader":
    """
    Wrap numpy arrays into a PyTorch DataLoader.

    Parameters
    ----------
    X : np.ndarray of shape (N, W, d)
    y : np.ndarray of shape (N, d)
    batch_size : int
    shuffle : bool
    num_workers : int

    Returns
    -------
    torch.utils.data.DataLoader
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for create_dataloader()")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available())
    return loader


def create_full_sequence_windows(df: pd.DataFrame, sensor_cols: List[str], window_size: int = 30) -> dict:
    """
    Create windows for each engine separately, preserving engine identity.

    This is used during evaluation when we need to reconstruct per-engine
    anomaly score trajectories (not shuffled across engines).

    Parameters
    ----------
    df : pd.DataFrame
    sensor_cols : list of str
    window_size : int

    Returns
    -------
    dict mapping engine_id -> {"X": array, "y": array, "cycles": array, "life_fracs": array}
    """
    result = {}

    for engine_id in df["unit_nr"].unique():
        engine_df = df[df["unit_nr"] == engine_id].sort_values("time_cycles")
        sensor_values = engine_df[sensor_cols].values
        cycles = engine_df["time_cycles"].values
        life_fracs = engine_df["life_fraction"].values if "life_fraction" in engine_df.columns \
            else np.ones(len(engine_df))

        T = len(sensor_values)
        if T <= window_size:
            continue  # Engine too short for even one window

        X_list = []
        y_list = []
        cycle_list = []
        lf_list = []

        for i in range(T - window_size):
            target_idx = i + window_size
            X_list.append(sensor_values[i: i + window_size])
            y_list.append(sensor_values[target_idx])
            cycle_list.append(cycles[target_idx])
            lf_list.append(life_fracs[target_idx])

        result[int(engine_id)] = {
            "X": np.array(X_list, dtype=np.float32),
            "y": np.array(y_list, dtype=np.float32),
            "cycles": np.array(cycle_list),
            "life_fracs": np.array(lf_list)}

    return result
