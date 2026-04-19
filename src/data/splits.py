"""
src/data/splits.py
==================
Split C-MAPSS data by engine unit into train/validation/test sets.

CRITICAL DESIGN DECISION:
We split by engine ID, not by individual rows. If we split rows randomly,
windows from the same engine could appear in both train and validation,
leaking temporal information and inflating performance metrics.

The split produces:
- Train engines: used to train the normal-behavior model
- Validation engines: used for early stopping and threshold calibration
- Test engines: held out for final evaluation (never touched during development)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict


def split_engines(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split engine IDs into train/validation/test groups.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'unit_nr' column.
    train_ratio, val_ratio, test_ratio : float
        Proportions for each split. Must sum to ~1.0.
    random_seed : int
        For reproducibility.

    Returns
    -------
    Tuple of (train_ids, val_ids, test_ids), each a numpy array of engine IDs.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    engine_ids = df["unit_nr"].unique()

    #First split: train vs (val + test)
    temp_ratio = val_ratio + test_ratio
    train_ids, temp_ids = train_test_split(engine_ids, test_size=temp_ratio, random_state=random_seed, shuffle=True)

    #Second split: val vs test (from the temp portion)
    val_proportion_of_temp = val_ratio / temp_ratio
    val_ids, test_ids = train_test_split(temp_ids, test_size=(1 - val_proportion_of_temp), random_state=random_seed, shuffle=True)

    return np.sort(train_ids), np.sort(val_ids), np.sort(test_ids)


def apply_split(df: pd.DataFrame, train_ids: np.ndarray, val_ids: np.ndarray, test_ids: np.ndarray) -> Dict[str, pd.DataFrame]:
    """
    Split a dataframe into train/val/test based on engine IDs.

    Parameters
    ----------
    df : pd.DataFrame
        Full preprocessed dataframe.
    train_ids, val_ids, test_ids : np.ndarray
        Engine IDs for each split.

    Returns
    -------
    Dict with keys "train", "val", "test", each a pd.DataFrame.
    """
    return {
        "train": df[df["unit_nr"].isin(train_ids)].copy().reset_index(drop=True),
        "val": df[df["unit_nr"].isin(val_ids)].copy().reset_index(drop=True),
        "test": df[df["unit_nr"].isin(test_ids)].copy().reset_index(drop=True)}


def create_evaluation_groups( test_df: pd.DataFrame, early_threshold: float = 0.3) -> Dict[str, pd.DataFrame]:
    """
    Split test data into evaluation groups for different analysis purposes.

    Groups:
    - "clean_early": Early-life windows (life_fraction < threshold).
      Used to measure false positive rate on clearly normal data.
    - "full_trajectories": Complete test engine trajectories.
      Used to check whether anomaly scores rise with degradation.

    Note: Groups for synthetic anomaly/drift injection are created later
    by the synthetic generators — they operate on copies of test data.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test split dataframe with 'life_fraction' column.
    early_threshold : float
        life_fraction cutoff for "clearly normal" data.

    Returns
    -------
    Dict with group names as keys and DataFrames as values.
    """
    groups = {
        "clean_early": test_df[test_df["life_fraction"] < early_threshold].copy(),
        "full_trajectories": test_df.copy()}

    return groups


def get_split_summary( splits: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Print a summary of the train/val/test split.

    Returns a DataFrame with engine counts, row counts, and cycle statistics.
    """
    rows = []
    for name, df in splits.items():
        n_engines = df["unit_nr"].nunique()
        n_rows = len(df)
        cycles = df.groupby("unit_nr")["time_cycles"].max()
        rows.append({
            "split": name,
            "n_engines": n_engines,
            "n_rows": n_rows,
            "min_life": cycles.min(),
            "max_life": cycles.max(),
            "mean_life": round(cycles.mean(), 1),
        })

    summary = pd.DataFrame(rows)
    return summary
