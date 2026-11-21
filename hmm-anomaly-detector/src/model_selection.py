"""HMM Model Selection & Hidden State Optimization Module.

Evaluates Hidden Markov Models across candidate hidden state counts K in [1, 6]
using:
1. Training Log-Likelihood (ln L)
2. Akaike Information Criterion (AIC): -2*ln(L) + 2*p
3. Bayesian Information Criterion (BIC): -2*ln(L) + p*ln(N)
4. Held-out Validation Log-Likelihood (Out-of-sample Generalization)

Generates empirical comparison tables and publication-quality diagnostic plots.
"""

import os
import sys
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def calculate_hmm_free_parameters(n_components: int, n_features: int = 1) -> int:
    """Calculates the number of free parameters for a GaussianHMM with diagonal

    covariance:

    - Initial state probabilities: (K - 1)
    - Transition probability matrix: K * (K - 1)
    - Emission means: K * n_features
    - Emission variances: K * n_features
    Total = (K - 1) + K*(K - 1) + 2*K*n_features = K^2 + 2*K*n_features - 1
    """
    K = n_components
    p_init = K - 1
    p_trans = K * (K - 1)
    p_emiss = 2 * K * n_features
    return p_init + p_trans + p_emiss


def evaluate_hidden_state_candidates(
    X_train: np.ndarray,
    X_val: np.ndarray,
    candidate_states: List[int] = [1, 2, 3, 4, 5, 6],
    n_iter: int = 150,
    n_restarts: int = 5,
) -> pd.DataFrame:
    """Fits GaussianHMM for each candidate state count with multiple random restarts
    and computes AIC, BIC, and held-out log-likelihood.
    """
    results: List[Dict] = []
    N = len(X_train)

    for k in candidate_states:
        best_model = None
        best_train_ll = -np.inf

        for seed in range(n_restarts):
            try:
                model = GaussianHMM(
                    n_components=k,
                    covariance_type="diag",
                    n_iter=n_iter,
                    random_state=seed,
                )
                model.fit(X_train)
                train_ll = model.score(X_train)
                if train_ll > best_train_ll:
                    best_train_ll = train_ll
                    best_model = model
            except Exception:
                continue

        if best_model is not None:
            val_ll = best_model.score(X_val)
            p = calculate_hmm_free_parameters(k, n_features=X_train.shape[1])
            aic = -2.0 * best_train_ll + 2.0 * p
            bic = -2.0 * best_train_ll + p * np.log(N)

            results.append(
                {
                    "n_states": k,
                    "free_parameters": p,
                    "train_log_likelihood": round(best_train_ll, 2),
                    "val_log_likelihood": round(val_ll, 2),
                    "norm_train_ll": round(best_train_ll / len(X_train), 4),
                    "norm_val_ll": round(val_ll / len(X_val), 4),
                    "AIC": round(aic, 2),
                    "BIC": round(bic, 2),
                    "converged": best_model.monitor_.converged,
                }
            )

    return pd.DataFrame(results)


def plot_model_selection_curves(
    results_df: pd.DataFrame, output_path: str = "reports/model_selection_aic_bic.png"
):
    """Plots BIC, AIC, and Held-out Log-Likelihood as a function of hidden

    states.
    """
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    states = results_df["n_states"]

    # Left plot: BIC & AIC (Information Criteria - lower is better)
    ax1.plot(
        states,
        results_df["BIC"],
        marker="o",
        linewidth=2.5,
        color="#c0392b",
        label="BIC (Schwarz Criterion)",
    )
    ax1.plot(
        states,
        results_df["AIC"],
        marker="s",
        linewidth=2.0,
        color="#2980b9",
        linestyle="--",
        label="AIC (Akaike Criterion)",
    )

    # Highlight minimum BIC
    min_bic_idx = results_df["BIC"].idxmin()
    best_k_bic = results_df.loc[min_bic_idx, "n_states"]
    min_bic_val = results_df.loc[min_bic_idx, "BIC"]
    ax1.scatter(
        [best_k_bic],
        [min_bic_val],
        color="gold",
        s=150,
        zorder=5,
        edgecolors="black",
        linewidth=1.5,
        label=f"Optimal BIC (k={best_k_bic})",
    )

    ax1.set_xlabel("Number of Hidden States (K)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Information Criterion Value", fontsize=11, fontweight="bold")
    ax1.set_title("Model Selection: AIC & BIC Curves (Lower is Better)", fontsize=12, fontweight="bold")
    ax1.set_xticks(states)
    ax1.legend(fontsize=10)

    # Right plot: Validation Normalized Log-Likelihood (Elbow Curve - higher is better)
    ax2.plot(
        states,
        results_df["norm_val_ll"],
        marker="^",
        linewidth=2.5,
        color="#27ae60",
        label="Held-out Val Log-Likelihood / sample",
    )
    ax2.plot(
        states,
        results_df["norm_train_ll"],
        marker="v",
        linewidth=1.8,
        color="#7f8c8d",
        linestyle=":",
        label="Train Log-Likelihood / sample",
    )

    ax2.axvline(
        x=best_k_bic,
        color="crimson",
        linestyle="--",
        alpha=0.7,
        label=f"Selected Model (k={best_k_bic})",
    )
    ax2.set_xlabel("Number of Hidden States (K)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Normalized Log-Likelihood (per sample)", fontsize=11, fontweight="bold")
    ax2.set_title("Generalization: Held-out Log-Likelihood", fontsize=12, fontweight="bold")
    ax2.set_xticks(states)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Model selection visualization saved to: {output_path}")


def run_model_selection(
    normal_data_path: str, reports_dir: str = "reports"
) -> Tuple[int, pd.DataFrame]:
    """Runs empirical state exploration and returns optimal state count and

    summary table.
    """
    df = pd.read_csv(normal_data_path)
    X = df[["sensor_reading"]].values

    # 75/25 train/validation split
    X_train, X_val = train_test_split(X, test_size=0.25, shuffle=False)

    print("Evaluating GaussianHMM candidate state counts k in [1, 6]...")
    results_df = evaluate_hidden_state_candidates(X_train, X_val)

    table_path = os.path.join(reports_dir, "model_selection_table.csv")
    os.makedirs(reports_dir, exist_ok=True)
    results_df.to_csv(table_path, index=False)

    plot_path = os.path.join(reports_dir, "model_selection_aic_bic.png")
    plot_model_selection_curves(results_df, plot_path)

    # Optimal K from minimum BIC
    best_row = results_df.loc[results_df["BIC"].idxmin()]
    optimal_k = int(best_row["n_states"])

    print("\n=== MODEL SELECTION RESULTS ===")
    print(results_df.to_string(index=False))
    print(f"\nEmpirical Optimal States: k={optimal_k} (BIC: {best_row['BIC']})")

    return optimal_k, results_df


if __name__ == "__main__":
    normal_csv = os.path.join(PROJECT_ROOT, "data", "machine_sensor_normal.csv")
    run_model_selection(normal_csv)
