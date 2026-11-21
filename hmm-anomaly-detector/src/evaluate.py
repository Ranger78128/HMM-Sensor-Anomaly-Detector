"""Evaluation module for HMM Anomaly Detector.

Computes sample-level and event-level anomaly detection metrics:
- Precision, Recall, F1-Score, False Alarm Rate (FAR)
- Event-level detection rate (coverage across distinct anomaly types)
- Detection latency / delay for each anomaly episode
"""

import json
import os
import sys
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def evaluate_sample_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Any]:
    """Computes point-by-point classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    far = fp / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "accuracy": round(float(acc), 4),
        "false_alarm_rate": round(float(far), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def evaluate_event_metrics(
    df_stream: pd.DataFrame,
    detected_mask: np.ndarray,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluates detection performance on contiguous anomaly episodes.

    Computes:
    - Event detection success (was the anomaly flagged during its duration?)
    - Detection latency (timesteps from event start to first alarm)
    - Coverage percentage within the event
    """
    df = df_stream.copy()
    df["detected"] = detected_mask

    # Find contiguous anomaly episodes
    gt_anom = df["is_anomaly"].values
    diff = np.diff(np.pad(gt_anom.astype(int), (1, 1)))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1  # Inclusive end index

    episodes: List[Dict[str, Any]] = []

    for idx, (s, e) in enumerate(zip(starts, ends)):
        event_slice = df.iloc[s : e + 1]
        event_type = event_slice["anomaly_type"].iloc[0]
        event_len = e - s + 1

        detected_in_event = event_slice["detected"].values
        has_detection = bool(np.any(detected_in_event))

        if has_detection:
            first_alarm_offset = int(np.where(detected_in_event == 1)[0][0])
            detection_latency = first_alarm_offset
        else:
            detection_latency = None

        coverage = float(np.mean(detected_in_event))

        episodes.append(
            {
                "event_id": idx + 1,
                "anomaly_type": event_type,
                "start_idx": int(s),
                "end_idx": int(e),
                "duration_steps": int(event_len),
                "detected": has_detection,
                "detection_latency_steps": detection_latency,
                "alarm_coverage_pct": round(coverage * 100.0, 2),
            }
        )

    detected_count = sum(ep["detected"] for ep in episodes)
    total_events = len(episodes)
    event_recall = (
        round((detected_count / total_events) * 100.0, 2)
        if total_events > 0
        else 0.0
    )

    valid_latencies = [
        ep["detection_latency_steps"]
        for ep in episodes
        if ep["detection_latency_steps"] is not None
    ]
    mean_latency = (
        round(float(np.mean(valid_latencies)), 2) if valid_latencies else None
    )

    event_summary = {
        "total_anomaly_events": total_events,
        "detected_anomaly_events": detected_count,
        "event_detection_rate_pct": event_recall,
        "mean_detection_latency_steps": mean_latency,
    }

    return episodes, event_summary


def run_evaluation(
    df_results: pd.DataFrame,
    df_ground_truth: pd.DataFrame,
    output_dir: str = "reports",
) -> Dict[str, Any]:
    """Runs full sample-level and event-level evaluation and exports reports."""
    # Exclude initial warm-up period where sliding window is warming up
    warmup_period = np.isnan(df_results["normalized_log_likelihood"].values)
    valid_indices = ~warmup_period

    y_true = df_ground_truth["is_anomaly"].values[valid_indices]
    y_pred = df_results["is_anomaly_detected"].values[valid_indices]

    sample_metrics = evaluate_sample_metrics(y_true, y_pred)
    episodes, event_summary = evaluate_event_metrics(
        df_ground_truth.iloc[valid_indices], y_pred
    )

    full_evaluation = {
        "sample_level_metrics": sample_metrics,
        "event_level_summary": event_summary,
        "anomaly_episodes": episodes,
    }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(json_path, "w") as f:
        json.dump(full_evaluation, f, indent=2)

    df_episodes = pd.DataFrame(episodes)
    csv_path = os.path.join(output_dir, "event_detection_summary.csv")
    df_episodes.to_csv(csv_path, index=False)

    print("\n=== ANOMALY DETECTION EVALUATION SUMMARY ===")
    print(f"Sample-level Precision: {sample_metrics['precision']:.4f}")
    print(f"Sample-level Recall:    {sample_metrics['recall']:.4f}")
    print(f"Sample-level F1-Score:  {sample_metrics['f1_score']:.4f}")
    print(f"False Alarm Rate (FAR): {sample_metrics['false_alarm_rate']:.4f}")
    print(f"Event Detection Rate:   {event_summary['event_detection_rate_pct']:.1f}% ({event_summary['detected_anomaly_events']}/{event_summary['total_anomaly_events']} events)")
    print(f"Mean Detection Latency: {event_summary['mean_detection_latency_steps']} steps")
    print(f"\nDetailed event breakdown saved to: {csv_path}")

    return full_evaluation
