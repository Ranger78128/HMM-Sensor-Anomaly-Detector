"""End-to-End Orchestrator for Hidden Markov Model Anomaly Detection.

Executes:
1. Loading normal operating machine sensor data.
2. Model selection and hidden state validation (AIC, BIC, cross-validation).
3. Training Gaussian HMM with EM parameter estimation.
4. Anomaly threshold calibration from training likelihood distribution.
5. Model persistence (models/trained_hmm.joblib, detector_config.json).
6. Parameter inspection & state transition visualization.
7. Sliding-window processing of continuous test stream with real-world anomaly injections.
8. Quantitative sample-level & event-level evaluation.
9. Generating comprehensive 4-panel anomaly overlay visualization.
10. Simulating live point-by-point real-time streaming ingestion.

Run with:
    python main.py
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluate import run_evaluation
from src.hmm_detector import HMMAnomalyDetector
from src.model_selection import run_model_selection
from src.visualize import plot_anomaly_detection_dashboard, plot_hmm_parameters


def run_pipeline():
    print("=================================================================")
    print("   HIDDEN MARKOV MODEL ANOMALY DETECTOR - FULL PIPELINE   ")
    print("=================================================================")

    data_dir = os.path.join(PROJECT_ROOT, "data")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    reports_dir = os.path.join(PROJECT_ROOT, "reports")

    normal_csv = os.path.join(data_dir, "machine_sensor_normal.csv")
    test_csv = os.path.join(data_dir, "machine_sensor_test_stream.csv")

    # Step 1: Verify / Generate Datasets
    print("\n--- STEP 1: Verifying Industrial Sensor Datasets ---")
    if not os.path.exists(normal_csv) or not os.path.exists(test_csv):
        from data.data_generator import generate_sensor_datasets

        generate_sensor_datasets(output_dir=data_dir)

    df_train = pd.read_csv(normal_csv)
    df_test = pd.read_csv(test_csv)
    print(f"Normal training samples: {len(df_train)}")
    print(f"Test stream samples:     {len(df_test)}")
    print(f"Ground truth anomalies in test stream: {df_test['is_anomaly'].sum()} steps ({df_test['is_anomaly'].mean()*100:.1f}%)")

    # Step 2: Model Selection Rationale
    print("\n--- STEP 2: Running Model Selection & Hidden State Analysis ---")
    optimal_k, selection_df = run_model_selection(
        normal_csv, reports_dir=reports_dir
    )
    print(f"Empirically determined optimal hidden states: k={optimal_k}")

    # Step 3: Train Chosen HMM on Normal Data
    print(f"\n--- STEP 3: Training GaussianHMM (k={optimal_k}) on Normal Sensor Data ---")
    X_train = df_train[["sensor_reading"]].values

    # Using sliding window W=20
    window_size = 20
    detector = HMMAnomalyDetector(
        n_components=optimal_k,
        window_size=window_size,
        covariance_type="diag",
        n_iter=250,
        random_state=42,
    )

    detector.fit(
        X_train,
        n_restarts=5,
        threshold_method="percentile",
        percentile=1.0,  # 1st percentile of normal sliding-window likelihood
    )

    # Step 4: Save Model Artifacts
    print("\n--- STEP 4: Serializing Production Detector Artifacts ---")
    detector.save(model_dir=models_dir)

    # Step 5: Visualize Learned HMM Parameters
    print("\n--- STEP 5: Visualizing Learned State Emissions & Transitions ---")
    plot_hmm_parameters(
        detector,
        X_train,
        output_path=os.path.join(reports_dir, "state_emissions_and_transitions.png"),
    )

    # Step 6: Process Continuous Test Stream via Sliding Window
    print("\n--- STEP 6: Processing Continuous Test Stream via Sliding Window ---")
    X_test = df_test[["sensor_reading"]].values
    start_time = time.time()
    results_df = detector.process_test_stream_sliding_window(X_test)
    elapsed = time.time() - start_time
    print(f"Batch sliding-window processing completed in {elapsed:.3f} seconds ({len(X_test)/elapsed:.1f} samples/sec)")

    detected_count = results_df["is_anomaly_detected"].sum()
    print(f"Detected anomalies in test stream: {detected_count} timesteps")

    # Step 7: Quantitative Evaluation
    print("\n--- STEP 7: Evaluating Detection Performance ---")
    eval_results = run_evaluation(results_df, df_test, output_dir=reports_dir)

    # Step 8: Multi-Panel Visualization
    print("\n--- STEP 8: Generating Anomaly Overlay Dashboard ---")
    overlay_path = os.path.join(reports_dir, "anomaly_detection_overlay.png")
    plot_anomaly_detection_dashboard(
        results_df,
        ground_truth_df=df_test,
        threshold=detector.threshold,
        output_path=overlay_path,
    )

    # Step 9: Real-Time Streaming Demonstration
    print("\n--- STEP 9: Demonstrating Live Point-by-Point Real-Time Streaming Ingestion ---")
    detector.reset_stream()
    # Stream a 100-sample window traversing into an anomaly event
    stream_slice = df_test.iloc[280:340]
    print(f"Streaming live telemetry samples 280 to 340 (Mechanical Shock Anomaly at 300-325)...")

    alarm_raised_at = None
    for _, row in stream_slice.iterrows():
        t_step = int(row["timestamp"])
        val = float(row["sensor_reading"])
        gt = int(row["is_anomaly"])
        anom_type = str(row["anomaly_type"])

        is_anom, norm_ll, state = detector.process_stream_sample(val)

        if is_anom and alarm_raised_at is None:
            alarm_raised_at = t_step
            print(f"  🚨 [LIVE ALERT at t={t_step}] Anomaly Detected! Reading={val:.2f}, Window LL={norm_ll:.3f} (Threshold: {detector.threshold:.3f}), State={state}")
        elif is_anom and t_step % 5 == 0:
            print(f"  🚨 [ALERT CONTINUING at t={t_step}] Reading={val:.2f}, Window LL={norm_ll:.3f}, State={state}")
        elif not is_anom and t_step % 15 == 0 and norm_ll is not None:
            print(f"  🟢 [NORMAL at t={t_step}] Reading={val:.2f}, Window LL={norm_ll:.3f}, State={state}")

    print(f"\nLive streaming test: First alarm triggered at t={alarm_raised_at} (Actual anomaly injected at t=300).")
    print("\n=================================================================")
    print("   PIPELINE EXECUTION COMPLETED SUCCESSFULLY!   ")
    print("=================================================================\n")


if __name__ == "__main__":
    run_pipeline()
