"""Industrial Machine Sensor Data Simulator.

Generates realistic time-series sensor data for a rotating machine operating
under Hidden Markov Model dynamics (Idle, Nominal Load, Peak Load).

Produces:
1. data/machine_sensor_normal.csv: Clean normal operating data for HMM training.
2. data/machine_sensor_test_stream.csv: Continuous test stream with injected
   industrial anomalies (spikes, mean drift, variance chatter, forbidden transitions, flatline).
"""

import os
from typing import Tuple
import numpy as np
import pandas as pd

# Physical parameters of simulated machine
NORMAL_STATES = {
    0: {"name": "Idle", "mean": 1.5, "std": 0.35},
    1: {"name": "Nominal", "mean": 4.0, "std": 0.45},
    2: {"name": "Peak", "mean": 7.0, "std": 0.60},
}

# Normal state transition matrix (Idle <-> Nominal <-> Peak)
# Direct Idle <-> Peak transitions are structurally prohibited
NORMAL_TRANSITION_MATRIX = np.array(
    [
        [0.92, 0.08, 0.00],  # From Idle: 92% stay, 8% to Nominal, 0% to Peak
        [0.05, 0.90, 0.05],  # From Nominal: 5% to Idle, 90% stay, 5% to Peak
        [0.00, 0.10, 0.90],  # From Peak: 0% to Idle, 10% to Nominal, 90% stay
    ]
)

NORMAL_INITIAL_PROBS = np.array([0.5, 0.4, 0.1])


def generate_normal_sequence(
    length: int, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a normal Markov state sequence and corresponding Gaussian

    emissions.
    """
    rng = np.random.RandomState(random_state)
    states = np.zeros(length, dtype=int)
    readings = np.zeros(length, dtype=float)

    # Sample initial state
    states[0] = rng.choice(3, p=NORMAL_INITIAL_PROBS)
    readings[0] = rng.normal(
        NORMAL_STATES[states[0]]["mean"], NORMAL_STATES[states[0]]["std"]
    )

    # Markov chain progression
    for t in range(1, length):
        current_state = states[t - 1]
        next_state = rng.choice(3, p=NORMAL_TRANSITION_MATRIX[current_state])
        states[t] = next_state

        # Gaussian emission with small harmonic background vibration
        harmonic = 0.12 * np.sin(2 * np.pi * t / 25.0)
        noise = rng.normal(0, NORMAL_STATES[next_state]["std"])
        readings[t] = NORMAL_STATES[next_state]["mean"] + harmonic + noise

    return readings, states


def generate_sensor_datasets(
    output_dir: str = None, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Creates the training normal dataset and the anomalous test stream."""
    output_dir = output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(random_state)

    # 1. Training Normal Dataset (3,000 steps)
    train_length = 3000
    train_readings, train_states = generate_normal_sequence(
        train_length, random_state=random_state
    )
    df_train = pd.DataFrame(
        {
            "timestamp": np.arange(train_length),
            "sensor_reading": np.round(train_readings, 4),
            "operational_state": train_states,
            "state_label": [NORMAL_STATES[s]["name"] for s in train_states],
        }
    )
    train_path = os.path.join(output_dir, "machine_sensor_normal.csv")
    df_train.to_csv(train_path, index=False)
    print(f"Generated normal training data: {train_path} ({len(df_train)} records)")

    # 2. Test Stream with Real-World Injected Anomalies (2,200 steps)
    test_length = 2200
    test_readings, test_states = generate_normal_sequence(
        test_length, random_state=random_state + 100
    )
    anomaly_mask = np.zeros(test_length, dtype=int)
    anomaly_types = ["normal"] * test_length

    # --- Anomaly 1: Sudden Mechanical Impact Spikes (t: 300 - 325) ---
    spike_indices = np.arange(300, 326)
    anomaly_mask[spike_indices] = 1
    for i in spike_indices:
        anomaly_types[i] = "mechanical_spike"
        test_readings[i] = rng.uniform(11.5, 14.5)

    # --- Anomaly 2: Thermal & Bearing Degradation Mean Shift (t: 650 - 730) ---
    shift_indices = np.arange(650, 731)
    anomaly_mask[shift_indices] = 1
    for i in shift_indices:
        anomaly_types[i] = "mean_shift_drift"
        # Reading shifted to an uncharacteristic 6.25 level (unstable abnormal intermediate regime)
        test_readings[i] = rng.normal(6.25, 0.25)

    # --- Anomaly 3: High-Frequency Resonance & Chatter / Variance Explosion (t: 1050 - 1130) ---
    chatter_indices = np.arange(1050, 1131)
    anomaly_mask[chatter_indices] = 1
    for i in chatter_indices:
        anomaly_types[i] = "variance_chatter"
        test_readings[i] = rng.normal(4.0, 2.6)

    # --- Anomaly 4: Forbidden Transition Violation (t: 1450 - 1530) ---
    # Rapidly alternating between Idle (1.5) and Peak (7.0) every 2 steps
    # The individual values look 'valid', but the sequence transition violates normal machine dynamics!
    transition_indices = np.arange(1450, 1531)
    anomaly_mask[transition_indices] = 1
    for idx, i in enumerate(transition_indices):
        anomaly_types[i] = "transition_violation"
        target_state = 0 if (idx // 2) % 2 == 0 else 2
        test_readings[i] = rng.normal(
            NORMAL_STATES[target_state]["mean"], NORMAL_STATES[target_state]["std"]
        )

    # --- Anomaly 5: Sensor Disconnection / Flatline Dropout (t: 1820 - 1880) ---
    freeze_indices = np.arange(1820, 1881)
    anomaly_mask[freeze_indices] = 1
    for i in freeze_indices:
        anomaly_types[i] = "sensor_freeze_dropout"
        test_readings[i] = 0.0  # Open-circuit / transducer disconnect flatline

    df_test = pd.DataFrame(
        {
            "timestamp": np.arange(test_length),
            "sensor_reading": np.round(test_readings, 4),
            "is_anomaly": anomaly_mask,
            "anomaly_type": anomaly_types,
            "operational_state": test_states,
        }
    )
    test_path = os.path.join(output_dir, "machine_sensor_test_stream.csv")
    df_test.to_csv(test_path, index=False)
    print(f"Generated test stream with anomalies: {test_path} ({len(df_test)} records)")

    return df_train, df_test


if __name__ == "__main__":
    generate_sensor_datasets()
