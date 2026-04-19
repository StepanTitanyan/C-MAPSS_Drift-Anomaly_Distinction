"""
src/data/loader.py
==================
Loads raw NASA C-MAPSS dataset files into pandas DataFrames.

The C-MAPSS dataset comes as whitespace-separated .txt files with no headers.
This module handles reading them and assigning the standard column names.

Each subset (FD001–FD004) has three files:
  - train_{subset}.txt  : full run-to-failure trajectories
  - test_{subset}.txt   : partial trajectories (cut before failure)
  - RUL_{subset}.txt    : remaining useful life for each test engine
"""

import os
import pandas as pd
from typing import Tuple, Dict


#Standard C-MAPSS column definitions
INDEX_COLS = ["unit_nr", "time_cycles"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS


def load_single_file(filepath: str, columns: list) -> pd.DataFrame:
    """
    Read one C-MAPSS .txt file into a DataFrame.

    The raw files are whitespace-delimited with no header row.
    Each row represents one cycle of one engine unit.

    Parameters
    ----------
    filepath : str
        Path to the .txt file.
    columns : list
        Column names to assign.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=columns)
    return df


def load_subset(data_dir: str, subset: str = "FD001") -> Dict[str, pd.DataFrame]:
    """
    Load all three files (train, test, RUL) for one C-MAPSS subset.

    Parameters
    ----------
    data_dir : str
        Directory containing the raw .txt files.
    subset : str
        One of "FD001", "FD002", "FD003", "FD004".

    Returns
    -------
    dict with keys "train", "test", "rul", each mapping to a DataFrame.

    Example
    -------
    >>> data = load_subset("data/raw/CMAPSSData", "FD001")
    >>> data["train"].shape
    (20631, 26)
    """
    result = {}

    #Train file: full run-to-failure trajectories
    train_path = os.path.join(data_dir, f"train_{subset}.txt")
    result["train"] = load_single_file(train_path, ALL_COLS)

    #Test file: partial trajectories
    test_path = os.path.join(data_dir, f"test_{subset}.txt")
    result["test"] = load_single_file(test_path, ALL_COLS)

    #RUL file: single column with remaining useful life per test engine
    rul_path = os.path.join(data_dir, f"RUL_{subset}.txt")
    result["rul"] = load_single_file(rul_path, ["RUL"])

    return result


def load_train_data(data_dir: str, subset: str = "FD001") -> pd.DataFrame:
    """
    Convenience function: load only the training data for a subset.

    In our project we primarily work with train_FD001.txt because it contains
    full run-to-failure trajectories (100 engines), which we split ourselves
    into train/val/test by engine ID.

    Parameters
    ----------
    data_dir : str
        Directory containing raw .txt files.
    subset : str
        Subset name.

    Returns
    -------
    pd.DataFrame with columns: unit_nr, time_cycles, setting_1-3, s_1 through s_21
    """
    filepath = os.path.join(data_dir, f"train_{subset}.txt")
    df = load_single_file(filepath, ALL_COLS)

    #Sort by engine and cycle for consistency
    df = df.sort_values(["unit_nr", "time_cycles"]).reset_index(drop=True)

    return df
