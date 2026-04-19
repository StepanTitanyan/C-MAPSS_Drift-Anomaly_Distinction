"""
src/synthetic/anomaly_generator.py
==================================
Generate synthetic anomalies by injecting controlled disturbances into
copies of real engine trajectories.

Anomaly Types:
  A. Spike — sudden single-step jump in 1+ sensors
  B. Drop — sudden single-step decrease
  C. Persistent offset — sensor shifts and stays shifted for several cycles
  D. Noise burst — temporarily increased variance
  E. Sensor freeze — sensor gets stuck at a constant value
  F. Multi-sensor coordinated — simultaneous anomaly in 2-3 sensors

Each injection records:
  - anomaly_type, magnitude, duration, affected sensors
  - start_cycle, engine_id, life_fraction at injection
  - per-step labels: 0=normal, 1=anomaly

This metadata is essential for breakdown analysis (which anomaly types
are detected, at what severity, at what life stage).
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class AnomalyEvent:
    """Metadata for one injected anomaly."""
    anomaly_type: str
    magnitude: float
    duration: int
    sensors_affected: List[int]       # Indices into sensor_cols
    start_idx: int                    # Index in the trajectory array
    engine_id: int
    life_fraction_at_start: float


@dataclass
class InjectedTrajectory:
    """A modified engine trajectory with anomaly labels."""
    engine_id: int
    sensor_values: np.ndarray         # (T, d) — modified sensor data
    labels: np.ndarray                # (T,) — 0=normal, 1=anomaly
    events: List[AnomalyEvent]        # All injected events
    cycles: np.ndarray                # (T,) — cycle numbers
    life_fracs: np.ndarray            # (T,) — life fractions


class AnomalyGenerator:
    """
    Injects synthetic anomalies into engine trajectories.

    Parameters
    ----------
    sensor_cols : list of str
        Names of sensor columns (for metadata).
    random_seed : int
        For reproducibility.
    """

    def __init__(self, sensor_cols: List[str], random_seed: int = 123):
        self.sensor_cols = sensor_cols
        self.num_sensors = len(sensor_cols)
        self.rng = np.random.RandomState(random_seed)

    def inject_spike(
        self,
        values: np.ndarray,
        labels: np.ndarray,
        start_idx: int,
        sensor_indices: List[int],
        magnitude: float,
        direction: int = 1,
    ) -> AnomalyEvent:
        """
        Inject a single-step spike anomaly.

        Parameters
        ----------
        values : (T, d) — sensor array (modified in-place)
        labels : (T,) — label array (modified in-place)
        start_idx : int — where to inject
        sensor_indices : which sensors to affect
        magnitude : shift size (in the data's scale, typically std units)
        direction : +1 for spike up, -1 for spike down
        """
        for s in sensor_indices:
            values[start_idx, s] += direction * magnitude
        labels[start_idx] = 1

    def inject_persistent_offset(
        self,
        values: np.ndarray,
        labels: np.ndarray,
        start_idx: int,
        sensor_indices: List[int],
        magnitude: float,
        duration: int,
        direction: int = 1,
    ):
        """Inject a persistent level shift for 'duration' steps."""
        end_idx = min(start_idx + duration, len(values))
        for t in range(start_idx, end_idx):
            for s in sensor_indices:
                values[t, s] += direction * magnitude
            labels[t] = 1

    def inject_noise_burst(
        self,
        values: np.ndarray,
        labels: np.ndarray,
        start_idx: int,
        sensor_indices: List[int],
        variance_multiplier: float,
        duration: int,
    ):
        """Inject a period of increased noise variance."""
        end_idx = min(start_idx + duration, len(values))
        for t in range(start_idx, end_idx):
            for s in sensor_indices:
                noise = self.rng.normal(0, variance_multiplier)
                values[t, s] += noise
            labels[t] = 1

    def inject_sensor_freeze(
        self,
        values: np.ndarray,
        labels: np.ndarray,
        start_idx: int,
        sensor_indices: List[int],
        duration: int,
    ):
        """Freeze sensors at their current value for 'duration' steps."""
        end_idx = min(start_idx + duration, len(values))
        frozen_values = values[start_idx, sensor_indices].copy()
        for t in range(start_idx, end_idx):
            values[t, sensor_indices] = frozen_values
            labels[t] = 1

    def inject_anomaly(
        self,
        values: np.ndarray,
        labels: np.ndarray,
        anomaly_type: str,
        start_idx: int,
        sensor_indices: List[int],
        magnitude: float = 3.0,
        duration: int = 1,
        direction: int = 1,
        variance_multiplier: float = 3.0,
    ):
        """
        Dispatch to the appropriate injection method.

        Parameters
        ----------
        anomaly_type : str
            One of "spike", "drop", "persistent_offset", "noise_burst",
            "sensor_freeze", "multi_sensor"
        """
        if anomaly_type == "spike":
            self.inject_spike(values, labels, start_idx, sensor_indices,
                              magnitude, direction=1)
        elif anomaly_type == "drop":
            self.inject_spike(values, labels, start_idx, sensor_indices,
                              magnitude, direction=-1)
        elif anomaly_type == "persistent_offset":
            self.inject_persistent_offset(values, labels, start_idx,
                                          sensor_indices, magnitude,
                                          duration, direction)
        elif anomaly_type == "noise_burst":
            self.inject_noise_burst(values, labels, start_idx,
                                    sensor_indices, variance_multiplier,
                                    duration)
        elif anomaly_type == "sensor_freeze":
            self.inject_sensor_freeze(values, labels, start_idx,
                                      sensor_indices, duration)
        else:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")

    def create_injected_trajectory(
        self,
        engine_data: dict,
        anomaly_type: str,
        injection_life_frac: float,
        magnitude: float = 3.0,
        duration: int = 1,
        num_sensors_affected: int = 1,
        direction: int = 1,
        variance_multiplier: float = 3.0,
    ) -> InjectedTrajectory:
        """
        Create a copy of an engine trajectory with one anomaly injected.

        Parameters
        ----------
        engine_data : dict
            Must have keys "sensor_values" (T, d), "cycles" (T,),
            "life_fracs" (T,), "engine_id" (int).
        anomaly_type : str
        injection_life_frac : float
            Approximate life_fraction at which to inject.
        magnitude : float
        duration : int
        num_sensors_affected : int
        direction : int
        variance_multiplier : float

        Returns
        -------
        InjectedTrajectory
        """
        # Copy data
        values = engine_data["sensor_values"].copy()
        cycles = engine_data["cycles"].copy()
        life_fracs = engine_data["life_fracs"].copy()
        engine_id = engine_data["engine_id"]
        T = len(values)

        labels = np.zeros(T, dtype=np.int32)

        # Find injection start index closest to target life_fraction
        start_idx = np.argmin(np.abs(life_fracs - injection_life_frac))

        # Ensure there's room for the duration
        start_idx = min(start_idx, T - duration - 1)
        start_idx = max(start_idx, 0)

        # Select which sensors to affect
        sensor_indices = self.rng.choice(
            self.num_sensors, size=min(num_sensors_affected, self.num_sensors),
            replace=False
        ).tolist()

        # Inject
        self.inject_anomaly(
            values, labels, anomaly_type, start_idx, sensor_indices,
            magnitude, duration, direction, variance_multiplier
        )

        # Record event metadata
        event = AnomalyEvent(
            anomaly_type=anomaly_type,
            magnitude=magnitude,
            duration=duration,
            sensors_affected=sensor_indices,
            start_idx=start_idx,
            engine_id=engine_id,
            life_fraction_at_start=float(life_fracs[start_idx]),
        )

        return InjectedTrajectory(
            engine_id=engine_id,
            sensor_values=values,
            labels=labels,
            events=[event],
            cycles=cycles,
            life_fracs=life_fracs,
        )

    def generate_test_suite(
        self,
        engine_data_list: List[dict],
        magnitudes: List[float] = [2.0, 3.0, 5.0],
        injection_positions: Dict[str, float] = None,
        anomaly_types: List[str] = None,
        durations: Dict[str, List[int]] = None,
    ) -> List[InjectedTrajectory]:
        """
        Generate a full suite of injected trajectories for evaluation.

        Creates one trajectory per (engine × anomaly_type × magnitude × position).

        Parameters
        ----------
        engine_data_list : list of dict
            Each dict has "sensor_values", "cycles", "life_fracs", "engine_id".
        magnitudes : list of float
        injection_positions : dict mapping name → life_fraction
        anomaly_types : list of str

        Returns
        -------
        List of InjectedTrajectory objects.
        """
        if injection_positions is None:
            injection_positions = {"early": 0.2, "mid": 0.5, "late": 0.8}
        if anomaly_types is None:
            anomaly_types = ["spike", "drop", "persistent_offset",
                             "noise_burst", "sensor_freeze"]
        if durations is None:
            durations = {
                "spike": [1], "drop": [1],
                "persistent_offset": [5, 10],
                "noise_burst": [5, 10],
                "sensor_freeze": [5, 10],
            }

        results = []

        for engine_data in engine_data_list:
            for atype in anomaly_types:
                for mag in magnitudes:
                    for pos_name, pos_lf in injection_positions.items():
                        for dur in durations.get(atype, [1]):
                            traj = self.create_injected_trajectory(
                                engine_data=engine_data,
                                anomaly_type=atype,
                                injection_life_frac=pos_lf,
                                magnitude=mag,
                                duration=dur,
                                num_sensors_affected=1 if atype != "multi_sensor" else 2,
                            )
                            results.append(traj)

        return results
