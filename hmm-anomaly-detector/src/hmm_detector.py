"""Hidden Markov Model Anomaly Detector for Machine Sensor Time-Series.

Implements:
1. Gaussian HMM training with hmmlearn.
2. Anomaly threshold calibration derived from training log-likelihood distribution.
3. Vectorized sliding window processing for batch continuous stream analysis.
4. Point-by-point real-time streaming engine with rolling window buffer.
5. Viterbi hidden state decoding for machine operational regime tracking.
"""

from collections import deque
import json
import os
import sys
from typing import Dict, Generator, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class HMMAnomalyDetector:
    """Production-grade Anomaly Detector using Hidden Markov Models (GaussianHMM)."""

    def __init__(
        self,
        n_components: int = 3,
        window_size: int = 20,
        covariance_type: str = "diag",
        n_iter: int = 200,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.window_size = window_size
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        self.model: Optional[GaussianHMM] = None
        self.threshold: Optional[float] = None
        self.train_likelihood_stats: Dict[str, float] = {}
        self.state_order_: Optional[np.ndarray] = None  # Mapping to sort states by mean

        # Streaming buffer for real-time sample-by-sample ingestion
        self._stream_buffer: deque = deque(maxlen=self.window_size)
        self._stream_step: int = 0

    def fit(
        self,
        X_train: np.ndarray,
        n_restarts: int = 5,
        threshold_method: str = "percentile",
        percentile: float = 1.0,
        sigma: float = 3.0,
    ) -> "HMMAnomalyDetector":
        """Trains GaussianHMM on normal sequential sensor data and calibrates

        the anomaly detection threshold based on the sliding window training
        likelihood distribution.
        """
        X = np.asarray(X_train, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Multi-restart EM fitting to ensure convergence to global maximum likelihood
        best_model = None
        best_ll = -np.inf

        for seed in range(n_restarts):
            m = GaussianHMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state + seed,
            )
            try:
                m.fit(X)
                ll = m.score(X)
                if ll > best_ll:
                    best_ll = ll
                    best_model = m
            except Exception:
                continue

        if best_model is None:
            raise RuntimeError("GaussianHMM fitting failed across all restarts.")

        self.model = best_model

        # Sort hidden states in ascending order of emission means for interpretability
        # State 0: Low/Idle, State 1: Nominal, State 2: Peak
        self.state_order_ = np.argsort(self.model.means_.flatten())

        # Calibrate anomaly threshold on training sliding windows
        self.calibrate_threshold(
            X, method=threshold_method, percentile=percentile, sigma=sigma
        )

        return self

    def calibrate_threshold(
        self,
        X_train: np.ndarray,
        method: str = "percentile",
        percentile: float = 1.0,
        sigma: float = 3.0,
    ) -> float:
        """Slides a window of size `window_size` across normal training data,

        computes normalized log-likelihoods, and derives the anomaly threshold.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before calibrating threshold.")

        X = np.asarray(X_train, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = len(X)
        if n_samples < self.window_size:
            raise ValueError(
                f"Training sequence length ({n_samples}) must be >= window_size ({self.window_size})"
            )

        # Extract sliding windows from training data
        window_lls = []
        for i in range(n_samples - self.window_size + 1):
            window = X[i : i + self.window_size]
            norm_ll = self.model.score(window) / self.window_size
            window_lls.append(norm_ll)

        window_lls = np.array(window_lls)

        mean_ll = float(np.mean(window_lls))
        std_ll = float(np.std(window_lls))
        min_ll = float(np.min(window_lls))
        pct_ll = float(np.percentile(window_lls, percentile))
        sigma_ll = float(mean_ll - sigma * std_ll)

        self.train_likelihood_stats = {
            "mean_log_likelihood": round(mean_ll, 4),
            "std_log_likelihood": round(std_ll, 4),
            "min_log_likelihood": round(min_ll, 4),
            "percentile_1_log_likelihood": round(pct_ll, 4),
            "three_sigma_log_likelihood": round(sigma_ll, 4),
            "num_calibration_windows": len(window_lls),
        }

        if method == "percentile":
            self.threshold = pct_ll
        elif method == "sigma":
            self.threshold = sigma_ll
        else:
            raise ValueError(f"Unknown threshold method: {method}")

        print(
            f"Anomaly threshold calibrated ({method}={percentile if method=='percentile' else sigma}): "
            f"Threshold = {self.threshold:.4f} (Mean: {mean_ll:.4f}, Std: {std_ll:.4f})"
        )
        return self.threshold

    def score_window(self, window: np.ndarray) -> float:
        """Computes normalized log-likelihood (per sample) for a single window."""
        if self.model is None:
            raise ValueError("Model not fitted.")
        w = np.asarray(window, dtype=float)
        if w.ndim == 1:
            w = w.reshape(-1, 1)
        return self.model.score(w) / len(w)

    def process_test_stream_sliding_window(
        self, X_stream: np.ndarray
    ) -> pd.DataFrame:
        """Processes continuous time-series stream using a sliding window.

        For each timestep t >= window_size - 1, computes the normalized
        log-likelihood of the preceding window [t - window_size + 1 : t + 1] and
        flags anomalies.
        """
        if self.model is None or self.threshold is None:
            raise ValueError("Detector must be fitted and calibrated before processing.")

        X = np.asarray(X_stream, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        N = len(X)
        norm_log_likelihoods = np.full(N, np.nan, dtype=float)
        is_anomaly = np.zeros(N, dtype=int)
        viterbi_states = self.model.predict(X)

        # Sliding window evaluation
        for t in range(self.window_size - 1, N):
            window = X[t - self.window_size + 1 : t + 1]
            try:
                ll = self.score_window(window)
                norm_log_likelihoods[t] = ll
                if ll < self.threshold:
                    is_anomaly[t] = 1
            except Exception:
                norm_log_likelihoods[t] = -np.inf
                is_anomaly[t] = 1

        df_results = pd.DataFrame(
            {
                "timestamp": np.arange(N),
                "sensor_reading": X.flatten(),
                "normalized_log_likelihood": norm_log_likelihoods,
                "is_anomaly_detected": is_anomaly,
                "anomaly_threshold": np.full(N, self.threshold),
                "inferred_state": viterbi_states,
            }
        )
        return df_results

    def reset_stream(self):
        """Resets real-time streaming buffer state."""
        self._stream_buffer.clear()
        self._stream_step = 0

    def process_stream_sample(
        self, reading: float
    ) -> Tuple[bool, Optional[float], Optional[int]]:
        """Real-time streaming ingestion interface.

        Takes a single live sensor reading, pushes it into the circular buffer,
        and returns:
            (is_anomaly, normalized_log_likelihood, inferred_state)
        """
        if self.model is None or self.threshold is None:
            raise ValueError("Detector must be fitted and calibrated.")

        self._stream_buffer.append(float(reading))
        self._stream_step += 1

        if len(self._stream_buffer) < self.window_size:
            # Buffer warming up; insufficient samples for full window log-likelihood
            return False, None, None

        window_arr = np.array(self._stream_buffer).reshape(-1, 1)
        ll = self.score_window(window_arr)
        is_anom = ll < self.threshold
        current_state = int(self.model.predict(window_arr)[-1])

        return bool(is_anom), float(ll), current_state

    def save(self, model_dir: str = "models"):
        """Serializes detector weights and configuration."""
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "trained_hmm.joblib")
        config_path = os.path.join(model_dir, "detector_config.json")

        joblib.dump(self.model, model_path)

        # Extract sorted state emission params
        means = self.model.means_.flatten().tolist()
        covars = self.model.covars_.flatten().tolist()
        transmat = self.model.transmat_.tolist()
        startprob = self.model.startprob_.tolist()

        config = {
            "n_components": self.n_components,
            "window_size": self.window_size,
            "covariance_type": self.covariance_type,
            "anomaly_threshold": float(self.threshold) if self.threshold else None,
            "train_likelihood_stats": self.train_likelihood_stats,
            "state_emissions": {
                f"state_{i}": {"mean": round(m, 4), "variance": round(c, 4)}
                for i, (m, c) in enumerate(zip(means, covars))
            },
            "transition_matrix": transmat,
            "initial_probabilities": startprob,
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"HMM model and configuration saved to {model_dir}/")

    @classmethod
    def load(cls, model_dir: str = "models") -> "HMMAnomalyDetector":
        """Loads serialized detector from disk."""
        model_path = os.path.join(model_dir, "trained_hmm.joblib")
        config_path = os.path.join(model_dir, "detector_config.json")

        if not os.path.exists(model_path) or not os.path.exists(config_path):
            raise FileNotFoundError(f"Model artifacts not found in {model_dir}")

        with open(config_path, "r") as f:
            config = json.load(f)

        detector = cls(
            n_components=config["n_components"],
            window_size=config["window_size"],
            covariance_type=config["covariance_type"],
        )
        detector.model = joblib.load(model_path)
        detector.threshold = config.get("anomaly_threshold")
        detector.train_likelihood_stats = config.get("train_likelihood_stats", {})
        return detector
