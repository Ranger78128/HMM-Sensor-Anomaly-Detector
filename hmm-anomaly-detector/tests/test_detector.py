"""Unit tests for Hidden Markov Model Anomaly Detector."""

import os
import sys
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.hmm_detector import HMMAnomalyDetector
from src.model_selection import calculate_hmm_free_parameters


def test_calculate_free_parameters():
    """Verifies parameter count formula for GaussianHMM:

    p = (K - 1) + K*(K - 1) + 2*K*d
    For K=3, d=1: (2) + 6 + 6 = 14
    """
    assert calculate_hmm_free_parameters(1, 1) == 2
    assert calculate_hmm_free_parameters(2, 1) == 7
    assert calculate_hmm_free_parameters(3, 1) == 14
    assert calculate_hmm_free_parameters(4, 1) == 23


def test_fit_and_threshold_calibration():
    """Tests fitting GaussianHMM on synthetic multi-state sequence."""
    rng = np.random.RandomState(42)
    s1 = rng.normal(1.5, 0.2, 200)
    s2 = rng.normal(4.0, 0.3, 300)
    s3 = rng.normal(7.0, 0.4, 200)
    data = np.concatenate([s1, s2, s3, s1, s2]).reshape(-1, 1)

    detector = HMMAnomalyDetector(n_components=3, window_size=15, random_state=42)
    detector.fit(data, n_restarts=2, threshold_method="percentile", percentile=1.0)

    assert detector.model is not None
    assert detector.threshold is not None
    assert detector.threshold < 0.0
    assert "mean_log_likelihood" in detector.train_likelihood_stats
    assert len(detector.model.means_) == 3


def test_sliding_window_stream_processing():
    """Tests batch sliding window processing output structure and types."""
    detector = HMMAnomalyDetector.load(os.path.join(PROJECT_ROOT, "models"))

    test_stream = np.array([4.0] * 50 + [12.0] * 10 + [4.0] * 40).reshape(-1, 1)
    results = detector.process_test_stream_sliding_window(test_stream)

    assert isinstance(results, pd.DataFrame)
    assert len(results) == len(test_stream)
    assert "normalized_log_likelihood" in results.columns
    assert "is_anomaly_detected" in results.columns
    assert "inferred_state" in results.columns

    # First W-1 samples should have NaN log-likelihood (warmup)
    assert np.isnan(results["normalized_log_likelihood"].iloc[0])

    # The spike segment around index 55 should be flagged as anomaly
    spike_slice = results.iloc[52:62]
    assert spike_slice["is_anomaly_detected"].sum() > 0


def test_real_time_sample_streaming():
    """Tests live sample-by-sample ingestion interface."""
    detector = HMMAnomalyDetector.load(os.path.join(PROJECT_ROOT, "models"))
    detector.reset_stream()

    W = detector.window_size

    # Ingest W-1 samples (warm-up phase)
    for i in range(W - 1):
        is_anom, ll, state = detector.process_stream_sample(4.0)
        assert is_anom is False
        assert ll is None
        assert state is None

    # Sample W completes first window (normal reading 4.0)
    is_anom, ll, state = detector.process_stream_sample(4.0)
    assert isinstance(is_anom, bool)
    assert isinstance(ll, float)
    assert isinstance(state, int)
    assert is_anom is False

    # Ingest massive spike (should trigger live alert)
    for _ in range(5):
        is_anom, ll, state = detector.process_stream_sample(14.0)

    assert is_anom is True
    assert ll < detector.threshold


def test_model_persistence():
    """Tests saving and reloading detector, ensuring identical inference

    scores.
    """
    original = HMMAnomalyDetector.load(os.path.join(PROJECT_ROOT, "models"))
    test_window = np.full((original.window_size, 1), 4.0)

    score_orig = original.score_window(test_window)

    temp_dir = os.path.join(PROJECT_ROOT, "models", "temp_test_dir")
    original.save(temp_dir)

    reloaded = HMMAnomalyDetector.load(temp_dir)
    score_reloaded = reloaded.score_window(test_window)

    assert np.isclose(score_orig, score_reloaded, atol=1e-5)
    assert reloaded.threshold == original.threshold

    # Clean up temp test artifacts
    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
    os.rmdir(temp_dir)
