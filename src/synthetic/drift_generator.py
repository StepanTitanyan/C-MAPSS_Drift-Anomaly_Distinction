"""
src/synthetic/drift_generator.py
================================
Generate synthetic drift patterns by injecting gradual, structured changes
into copies of real engine trajectories.

Drift differs from anomaly in its temporal and spatial properties:
- Anomalies are abrupt, short, possibly sparse across sensors.
- Drift is gradual, sustained, and often affects sensors in a correlated way.

Drift Types:
  A. Gradual linear shift — slow linear ramp over many cycles
  B. Sigmoid plateau — smooth transition to a new level (S-curve)
  C. Regime trend change — the underlying slope changes
  D. Multi-sensor coordinated drift — 2-3 sensors shift together
  E. Accelerating drift — quadratic ramp (slow start, faster later)

Labels: 0=normal, 1=anomaly, 2=drift
The distinct label is critical: we never mix drift into the anomaly class.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class DriftEvent:
    """Metadata for one injected drift episode."""
    drift_type: str
    rate: float                       # Drift speed parameter
    duration: int                     # Number of affected cycles
    sensors_affected: List[int]       # Indices into sensor_cols
    start_idx: int                    # Index in the trajectory array
    engine_id: int
    life_fraction_at_start: float
    final_magnitude: float            # Total shift at end of drift


@dataclass
class DriftedTrajectory:
    """A modified engine trajectory with drift labels."""
    engine_id: int
    sensor_values: np.ndarray         # (T, d) — modified sensor data
    labels: np.ndarray                # (T,) — 0=normal, 2=drift
    events: List[DriftEvent]          # All injected drift events
    cycles: np.ndarray                # (T,) — cycle numbers
    life_fracs: np.ndarray            # (T,) — life fractions


class DriftGenerator:
    """
    Injects synthetic drift patterns into engine trajectories.

    Parameters
    ----------
    sensor_cols : list of str
        Names of sensor columns (for metadata).
    random_seed : int
        For reproducibility.
    """

    def __init__(self, sensor_cols: List[str], random_seed: int = 456):
        self.sensor_cols = sensor_cols
        self.num_sensors = len(sensor_cols)
        self.rng = np.random.RandomState(random_seed)

    def _linear_ramp(self, duration: int, rate: float) -> np.ndarray:
        """
        Create a linear ramp: starts at 0, increases by 'rate' per step.

        Returns shape (duration,) with values [0, rate, 2*rate, ..., (dur-1)*rate].
        """
        return np.arange(duration, dtype=np.float64) * rate

    def _sigmoid_ramp(self, duration: int, final_magnitude: float,
                      transition_width: int = 20) -> np.ndarray:
        """
        Create a sigmoid (S-curve) transition to a new level.

        Starts near 0, smoothly transitions to 'final_magnitude'.
        The transition_width controls how sharp the S-curve is.
        """
        t = np.arange(duration, dtype=np.float64)
        midpoint = duration / 2.0
        # Sigmoid centered at midpoint
        k = 6.0 / transition_width  # Controls steepness
        sigmoid = 1.0 / (1.0 + np.exp(-k * (t - midpoint)))
        return sigmoid * final_magnitude

    def _quadratic_ramp(self, duration: int, rate: float) -> np.ndarray:
        """
        Create an accelerating (quadratic) ramp.

        Starts very slowly, accelerates over time.
        Final magnitude = rate * duration².
        """
        t = np.arange(duration, dtype=np.float64)
        return rate * (t ** 2) / duration  # Normalize so max ≈ rate * duration

    def inject_gradual_shift(
        self,
        values: np.ndarray,
        labels: np.ndarray,
        start_idx: int,
        sensor_indices: List[int],
        rate: float,
        duration: int,
        direction: int = 1,
    ) -> float:
        """
        Inject a gradual linear mean shift.

        Parameters
        ----------
        values : (T, d) — modified in-place
        labels : (T,) — modified in-place (set to 2 for drift)
        start_idx : int
        sensor_indices : list of int
        rate : float — shift per step
        duration : int — number of steps
        direction : +1 or -1

        Returns
        -------
        final_magnitude : float — total shift at end of drift
        """
        ramp = self._linear_ramp(duration, rate) * direction
        end_idx = min(start_idx + duration, len(values))
        actual_duration = end_idx - start_idx

        for t_offset in range(actual_duration):
            for s in sensor_indices:
                values[start_idx + t_offset, s] += ramp[t_offset]
            labels[start_idx + t_offset] = 2  # drift label

        return float(ramp[actual_duration - 1]) if actual_duration > 0 else 0.0

    def inject_sigmoid_plateau(
        self,
        values: np.ndarray,
        labels: np.ndarray,
        start_idx: int,
        sensor_indices: List[int],
        final_magnitude: float,
        duration: int,
        transition_width: int = 20,
        direction: int = 1,
    ) -> float:
        """Inject a smooth sigmoid transition to a new level."""
        ramp = self._sigmoid_ramp(duration, final_magnitude, transition_width) * direction
        end_idx = min(start_idx + duration, len(values))
        actual_duration = end_idx - start_idx

        for t_offset in range(actual_duration):
            for s in sensor_indices:
                values[start_idx + t_offset, s] += ramp[t_offset]
            labels[start_idx + t_offset] = 2

        return float(ramp[actual_duration - 1]) if actual_duration > 0 else 0.0

    def inject_accelerating_drift(
        self,
        values: np.ndarray,
        labels: np.ndarray,
        start_idx: int,
        sensor_indices: List[int],
        rate: float,
        duration: int,
        direction: int = 1,
    ) -> float:
        """Inject a quadratic (accelerating) drift."""
        ramp = self._quadratic_ramp(duration, rate) * direction
        end_idx = min(start_idx + duration, len(values))
        actual_duration = end_idx - start_idx

        for t_offset in range(actual_duration):
            for s in sensor_indices:
                values[start_idx + t_offset, s] += ramp[t_offset]
            labels[start_idx + t_offset] = 2

        return float(ramp[actual_duration - 1]) if actual_duration > 0 else 0.0

    def create_drifted_trajectory(
        self,
        engine_data: dict,
        drift_type: str,
        injection_life_frac: float,
        rate: float = 0.03,
        duration: int = 60,
        num_sensors_affected: int = 1,
        direction: int = 1,
        transition_width: int = 20,
    ) -> DriftedTrajectory:
        """
        Create a copy of an engine trajectory with one drift episode injected.

        Parameters
        ----------
        engine_data : dict
            Keys: "sensor_values" (T, d), "cycles" (T,), "life_fracs" (T,),
            "engine_id" (int).
        drift_type : str
            "gradual_shift", "sigmoid_plateau", "accelerating",
            "regime_change", "multi_sensor"
        injection_life_frac : float
        rate : float — drift speed
        duration : int — drift length in cycles
        num_sensors_affected : int
        direction : int — +1 or -1

        Returns
        -------
        DriftedTrajectory
        """
        values = engine_data["sensor_values"].copy()
        cycles = engine_data["cycles"].copy()
        life_fracs = engine_data["life_fracs"].copy()
        engine_id = engine_data["engine_id"]
        T = len(values)

        labels = np.zeros(T, dtype=np.int32)

        # Find start index
        start_idx = np.argmin(np.abs(life_fracs - injection_life_frac))
        start_idx = min(start_idx, T - duration - 1)
        start_idx = max(start_idx, 0)

        # Select sensors
        n_sensors = num_sensors_affected
        if drift_type == "multi_sensor":
            n_sensors = min(3, self.num_sensors)

        sensor_indices = self.rng.choice(
            self.num_sensors, size=min(n_sensors, self.num_sensors), replace=False
        ).tolist()

        # Inject drift
        final_mag = 0.0
        if drift_type in ("gradual_shift", "regime_change"):
            final_mag = self.inject_gradual_shift(
                values, labels, start_idx, sensor_indices, rate, duration, direction
            )
        elif drift_type == "sigmoid_plateau":
            final_magnitude = rate * duration  # Total desired shift
            final_mag = self.inject_sigmoid_plateau(
                values, labels, start_idx, sensor_indices,
                final_magnitude, duration, transition_width, direction
            )
        elif drift_type in ("accelerating",):
            final_mag = self.inject_accelerating_drift(
                values, labels, start_idx, sensor_indices, rate, duration, direction
            )
        elif drift_type == "multi_sensor":
            final_mag = self.inject_gradual_shift(
                values, labels, start_idx, sensor_indices, rate, duration, direction
            )
        else:
            raise ValueError(f"Unknown drift type: {drift_type}")

        event = DriftEvent(
            drift_type=drift_type,
            rate=rate,
            duration=duration,
            sensors_affected=sensor_indices,
            start_idx=start_idx,
            engine_id=engine_id,
            life_fraction_at_start=float(life_fracs[start_idx]),
            final_magnitude=abs(final_mag),
        )

        return DriftedTrajectory(
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
        rates: List[float] = [0.01, 0.03, 0.05],
        durations: List[int] = [30, 60, 100],
        injection_positions: Dict[str, float] = None,
        drift_types: List[str] = None,
    ) -> List[DriftedTrajectory]:
        """
        Generate a full suite of drifted trajectories.

        Parameters
        ----------
        engine_data_list : list of dict
        rates : drift rates to test
        durations : drift durations to test
        injection_positions : where to inject
        drift_types : which drift types

        Returns
        -------
        List of DriftedTrajectory objects.
        """
        if injection_positions is None:
            injection_positions = {"early": 0.2, "mid": 0.5}
        if drift_types is None:
            drift_types = ["gradual_shift", "sigmoid_plateau",
                           "accelerating", "multi_sensor"]

        results = []
        for engine_data in engine_data_list:
            for dtype in drift_types:
                for rate in rates:
                    for dur in durations:
                        for pos_name, pos_lf in injection_positions.items():
                            traj = self.create_drifted_trajectory(
                                engine_data=engine_data,
                                drift_type=dtype,
                                injection_life_frac=pos_lf,
                                rate=rate,
                                duration=dur,
                                num_sensors_affected=1 if dtype != "multi_sensor" else 3,
                            )
                            results.append(traj)

        return results
