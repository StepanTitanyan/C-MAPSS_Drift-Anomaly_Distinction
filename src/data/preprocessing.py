"""
src/data/preprocessing.py
=========================
Preprocessing pipeline for C-MAPSS sensor data.

This module encapsulates the sensor selection and normalization decisions
made during the exploratory analysis phase:

1. Remove zero/near-zero variance sensors (s_1, s_5, s_6, s_10, s_16, s_18, s_19)
2. Remove weak-correlation sensors (s_9, s_14)
3. Select final subset based on redundancy analysis (7 sensors)
4. Compute life_fraction = time_cycles / max_cycles_per_engine
5. Standardize (z-score) using training statistics only
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional


def compute_life_fraction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a life_fraction column: normalized position in engine lifecycle.

    life_fraction = time_cycles / max(time_cycles) for each engine.
    This goes from ~0 at the start of life to 1.0 at the last recorded cycle.

    For the training data in FD001, the last cycle IS the failure cycle,
    so life_fraction=1.0 means the engine has failed.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'unit_nr' and 'time_cycles' columns.

    Returns
    -------
    pd.DataFrame with 'life_fraction' column added.
    """
    df = df.copy()
    max_cycles = df.groupby("unit_nr")["time_cycles"].transform("max")
    df["life_fraction"] = df["time_cycles"] / max_cycles
    return df


def select_sensors(df: pd.DataFrame, sensor_list: List[str],
                   keep_meta: bool = True) -> pd.DataFrame:
    """
    Keep only the specified sensors (plus metadata columns if requested).

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe with all columns.
    sensor_list : list of str
        Which sensor columns to keep (e.g., ["s_3", "s_4", ...]).
    keep_meta : bool
        If True, also keep 'unit_nr', 'time_cycles', 'life_fraction'.

    Returns
    -------
    pd.DataFrame with only the requested columns.
    """
    meta_cols = []
    if keep_meta:
        for col in ["unit_nr", "time_cycles", "life_fraction"]:
            if col in df.columns:
                meta_cols.append(col)

    cols = meta_cols + [s for s in sensor_list if s in df.columns]
    return df[cols].copy()


class SensorScaler:
    """
    Standardizes sensor columns using z-score normalization.

    Critically, the scaler is fit ONLY on training data to prevent data leakage.
    The same mean/std are then applied to validation and test data.

    Attributes
    ----------
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler object.
    sensor_cols : list of str
        Which columns to scale.
    is_fitted : bool
        Whether fit() has been called.
    """

    def __init__(self, sensor_cols: List[str]):
        self.sensor_cols = sensor_cols
        self.scaler = StandardScaler()
        self.is_fitted = False
        # Store means and stds for easy access (useful for anomaly score interpretation)
        self.means_ = None
        self.stds_ = None

    def fit(self, df: pd.DataFrame) -> "SensorScaler":
        """
        Compute mean and std from training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training data (should only contain normal-regime data).

        Returns
        -------
        self
        """
        self.scaler.fit(df[self.sensor_cols])
        self.means_ = pd.Series(self.scaler.mean_, index=self.sensor_cols)
        self.stds_ = pd.Series(self.scaler.scale_, index=self.sensor_cols)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted scaling to a dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Data to transform (train, val, or test).

        Returns
        -------
        pd.DataFrame with sensor columns replaced by scaled values.
        """
        if not self.is_fitted:
            raise RuntimeError("SensorScaler must be fit before transform.")

        df = df.copy()
        df[self.sensor_cols] = self.scaler.transform(df[self.sensor_cols])
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on df and then transform it."""
        self.fit(df)
        return self.transform(df)

    def inverse_transform_array(self, arr: np.ndarray) -> np.ndarray:
        """
        Convert scaled values back to original sensor units.

        Parameters
        ----------
        arr : np.ndarray of shape (..., num_sensors)

        Returns
        -------
        np.ndarray in original scale.
        """
        return self.scaler.inverse_transform(arr.reshape(-1, len(self.sensor_cols))) \
                   .reshape(arr.shape)


def filter_normal_region(df: pd.DataFrame,
                         threshold: float = 0.5) -> pd.DataFrame:
    """
    Keep only rows from the early portion of each engine's life.

    This is used to create the "normal training set" — the model should
    only learn from healthy engine behavior, not from degraded/failing states.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'life_fraction' column.
    threshold : float
        Maximum life_fraction to include. Default 0.5 means first 50% of life.

    Returns
    -------
    pd.DataFrame containing only early-life rows.
    """
    if "life_fraction" not in df.columns:
        raise ValueError("DataFrame must have 'life_fraction' column. "
                         "Call compute_life_fraction() first.")
    return df[df["life_fraction"] <= threshold].copy()


def run_preprocessing_pipeline(df: pd.DataFrame, sensor_list: List[str], normal_threshold: Optional[float] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Run the full preprocessing pipeline on a raw dataframe.

    Steps:
    1. Compute life_fraction
    2. Select specified sensors
    3. Optionally filter to normal region

    Note: Scaling is NOT done here because it requires train/val/test split
    information (must fit on train only). Scaling happens after splitting.

    Parameters
    ----------
    df : pd.DataFrame
        Raw C-MAPSS dataframe.
    sensor_list : list of str
        Sensors to keep.
    normal_threshold : float or None
        If provided, also return a filtered normal-only dataframe.

    Returns
    -------
    Tuple of (full_df, normal_df_or_None)
    """
    #Step 1: life fraction
    df = compute_life_fraction(df)

    #Step 2: sensor selection
    df = select_sensors(df, sensor_list, keep_meta=True)

    #Step 3: optional normal filtering
    normal_df = None
    if normal_threshold is not None:
        normal_df = filter_normal_region(df, threshold=normal_threshold)

    return df, normal_df
