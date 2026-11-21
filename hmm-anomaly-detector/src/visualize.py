"""Visualization module for HMM Anomaly Detector.

Creates publication-grade visual overlays:
1. Multi-panel real-time streaming anomaly detection overlay.
2. Sliding-window log-likelihood trace against calibrated threshold.
3. Viterbi decoded operational regimes.
4. Learned Gaussian emission densities and state transition matrix heatmap.
"""

import os
import sys
from typing import Optional
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def plot_anomaly_detection_dashboard(
    df_results: pd.DataFrame,
    ground_truth_df: Optional[pd.DataFrame] = None,
    threshold: float = -1.5,
    output_path: str = "reports/anomaly_detection_overlay.png",
):
    """Generates a comprehensive 4-panel anomaly detection dashboard:

    Panel 1: Raw Sensor Readings with Ground Truth & Detected Anomaly Overlay
    Panel 2: Sliding-Window Normalized Log-Likelihood vs Calibrated Threshold
    Panel 3: Viterbi Inferred Machine Operational States (0: Idle, 1: Nominal, 2: Peak)
    Panel 4: Likelihood Distribution Separation (Normal vs Anomalous Windows)
    """
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(15, 12), dpi=300)
    gs = fig.add_gridspec(4, 2, height_ratios=[2.2, 1.8, 1.3, 1.5], width_ratios=[3, 1])

    ax_raw = fig.add_subplot(gs[0, :])
    ax_ll = fig.add_subplot(gs[1, :], sharex=ax_raw)
    ax_state = fig.add_subplot(gs[2, :], sharex=ax_raw)
    ax_kde = fig.add_subplot(gs[3, :])

    t = df_results["timestamp"]
    readings = df_results["sensor_reading"]
    norm_ll = df_results["normalized_log_likelihood"]
    is_detected = df_results["is_anomaly_detected"] == 1
    states = df_results["inferred_state"]

    # -------------------------------------------------------------
    # PANEL 1: RAW SENSOR READINGS & ANOMALY OVERLAYS
    # -------------------------------------------------------------
    ax_raw.plot(
        t,
        readings,
        color="#2c3e50",
        linewidth=1.2,
        alpha=0.85,
        label="Raw Sensor Stream (Vibration / Temp)",
    )

    # Highlight ground truth anomaly spans if available
    if ground_truth_df is not None and "is_anomaly" in ground_truth_df.columns:
        gt_anom = ground_truth_df["is_anomaly"] == 1
        # Find contiguous ground truth regions
        gt_diff = np.diff(np.pad(gt_anom.astype(int), (1, 1)))
        starts = np.where(gt_diff == 1)[0]
        ends = np.where(gt_diff == -1)[0]
        for s, e in zip(starts, ends):
            ax_raw.axvspan(s, e, color="#f39c12", alpha=0.22, label="_nolegend_")

    # Overlay HMM detected anomalies in bright crimson
    detected_indices = t[is_detected]
    detected_readings = readings[is_detected]
    ax_raw.scatter(
        detected_indices,
        detected_readings,
        color="#e74c3c",
        s=16,
        alpha=0.9,
        zorder=5,
        label="HMM Anomaly Flagged (Window LL < Threshold)",
    )

    # Annotation of specific industrial anomaly events
    annotations = [
        (312, 13.0, "Mechanical Shock / Spikes", (330, 14.5)),
        (690, 5.7, "Bearing Thermal Drift", (690, 8.5)),
        (1090, 6.2, "Chatter / Variance Explosion", (1090, 10.0)),
        (1490, 4.0, "Forbidden State Flutter", (1490, 8.5)),
        (1850, 4.0, "Sensor Freeze / Flatline", (1850, 7.5)),
    ]
    for x_pos, y_pos, label_text, text_pos in annotations:
        if x_pos < len(t):
            ax_raw.annotate(
                label_text,
                xy=(x_pos, y_pos),
                xytext=text_pos,
                arrowprops=dict(facecolor="#c0392b", shrink=0.08, width=1.5, headwidth=6),
                bbox=dict(boxstyle="round,pad=0.3", fc="#fbeee6", ec="#c0392b", lw=1),
                fontsize=8.5,
                fontweight="bold",
            )

    # Legend for Panel 1
    patch_gt = mpatches.Patch(color="#f39c12", alpha=0.35, label="Ground Truth Anomaly Region")
    handles, labels = ax_raw.get_legend_handles_labels()
    handles.append(patch_gt)
    labels.append("Ground Truth Anomaly Region")
    ax_raw.legend(handles=handles, labels=labels, loc="upper right", fontsize=9.5, framealpha=0.9)
    ax_raw.set_ylabel("Sensor Reading", fontsize=11, fontweight="bold")
    ax_raw.set_title(
        "HMM Real-Time Sensor Anomaly Detection Dashboard",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )

    # -------------------------------------------------------------
    # PANEL 2: SLIDING-WINDOW LOG-LIKELIHOOD VS THRESHOLD
    # -------------------------------------------------------------
    ax_ll.plot(
        t,
        norm_ll,
        color="#2980b9",
        linewidth=1.2,
        label="Sliding-Window Normalized Log-Likelihood (per sample)",
    )
    # Highlight points below threshold
    ll_below = norm_ll.copy()
    ll_below[norm_ll >= threshold] = np.nan
    ax_ll.plot(
        t,
        ll_below,
        color="#c0392b",
        linewidth=1.8,
        label="Anomalous Log-Likelihood (< Threshold)",
    )

    ax_ll.axhline(
        y=threshold,
        color="#c0392b",
        linestyle="--",
        linewidth=1.8,
        label=f"Calibrated Anomaly Threshold (tau = {threshold:.2f})",
    )
    ax_ll.fill_between(
        t,
        threshold,
        norm_ll,
        where=(norm_ll < threshold),
        color="#e74c3c",
        alpha=0.25,
        label="_nolegend_",
    )
    ax_ll.set_ylabel("Log-Likelihood / sample", fontsize=11, fontweight="bold")
    ax_ll.set_ylim([max(np.nanmin(norm_ll) - 0.5, -8.0), np.nanmax(norm_ll) + 0.5])
    ax_ll.legend(loc="lower left", fontsize=9.5, framealpha=0.9)

    # -------------------------------------------------------------
    # PANEL 3: VITERBI OPERATIONAL REGIMES
    # -------------------------------------------------------------
    state_colors = {0: "#27ae60", 1: "#2980b9", 2: "#8e44ad"}
    state_labels = {0: "0: Idle", 1: "1: Nominal", 2: "2: Peak"}

    ax_state.step(
        t,
        states,
        where="mid",
        color="#34495e",
        linewidth=1.0,
        alpha=0.6,
        label="State Transitions",
    )
    for s_val, col in state_colors.items():
        s_mask = states == s_val
        ax_state.scatter(
            t[s_mask],
            states[s_mask],
            color=col,
            s=4,
            alpha=0.8,
            label=state_labels[s_val],
        )

    ax_state.set_yticks([0, 1, 2])
    ax_state.set_yticklabels(["Idle", "Nominal", "Peak"], fontsize=9.5, fontweight="bold")
    ax_state.set_ylabel("Inferred Regime", fontsize=11, fontweight="bold")
    ax_state.legend(loc="upper right", fontsize=9, ncol=3, framealpha=0.9)

    # -------------------------------------------------------------
    # PANEL 4: LIKELIHOOD DENSITY DISTRIBUTION SEPARATION
    # -------------------------------------------------------------
    valid_mask = ~np.isnan(norm_ll)
    if ground_truth_df is not None:
        gt_anom = ground_truth_df["is_anomaly"].values == 1
        ll_normal = norm_ll[valid_mask & (~gt_anom)]
        ll_anom = norm_ll[valid_mask & gt_anom]

        sns.kdeplot(
            ll_normal,
            ax=ax_kde,
            color="#27ae60",
            fill=True,
            alpha=0.35,
            linewidth=2,
            label="Normal Operating Windows Density",
        )
        sns.kdeplot(
            ll_anom,
            ax=ax_kde,
            color="#e74c3c",
            fill=True,
            alpha=0.35,
            linewidth=2,
            label="Anomalous Windows Density",
        )
        ax_kde.axvline(
            x=threshold,
            color="#c0392b",
            linestyle="--",
            linewidth=2,
            label=f"Decision Threshold ({threshold:.2f})",
        )
        ax_kde.set_xlabel("Normalized Log-Likelihood", fontsize=11, fontweight="bold")
        ax_kde.set_ylabel("Empirical Probability Density", fontsize=11, fontweight="bold")
        ax_kde.set_title("Distribution Separability: Normal vs Anomalous Windows", fontsize=12, fontweight="bold")
        ax_kde.legend(loc="upper left", fontsize=10)

    ax_state.set_xlabel("Timestamp / Sequence Step", fontsize=11, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Anomaly detection dashboard saved to: {output_path}")


def plot_hmm_parameters(
    detector,
    train_data: np.ndarray,
    output_path: str = "reports/state_emissions_and_transitions.png",
):
    """Plots the learned Gaussian emissions over the training data histogram and

    the transition probability matrix heatmap.
    """
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Left: Gaussian Emission Distributions
    sns.histplot(
        train_data.flatten(),
        ax=ax1,
        bins=50,
        stat="density",
        color="#bdc3c7",
        alpha=0.5,
        label="Training Sensor Data Histogram",
    )

    x_range = np.linspace(train_data.min() - 1, train_data.max() + 1, 500)
    means = detector.model.means_.flatten()
    covars = detector.model.covars_.flatten()
    sorted_order = np.argsort(means)

    colors = ["#27ae60", "#2980b9", "#8e44ad"]
    names = ["State 0 (Idle)", "State 1 (Nominal)", "State 2 (Peak)"]

    for i, state_idx in enumerate(sorted_order):
        mu = means[state_idx]
        var = covars[state_idx]
        sigma = np.sqrt(var)
        pdf = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
        ax1.plot(
            x_range,
            pdf,
            color=colors[i % len(colors)],
            linewidth=2.5,
            label=f"{names[i]}: mu={mu:.2f}, sigma={sigma:.2f}",
        )

    ax1.set_xlabel("Sensor Reading Amplitude", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax1.set_title("Learned Gaussian Emission Distributions", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9.5)

    # Right: Transition Probability Matrix Heatmap
    transmat = detector.model.transmat_[sorted_order, :][:, sorted_order]
    sns.heatmap(
        transmat,
        ax=ax2,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        cbar=True,
        xticklabels=["Idle", "Nominal", "Peak"],
        yticklabels=["Idle", "Nominal", "Peak"],
    )
    ax2.set_xlabel("Target State", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Source State", fontsize=11, fontweight="bold")
    ax2.set_title("Learned State Transition Probability Matrix", fontsize=12, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"HMM parameters visualization saved to: {output_path}")
