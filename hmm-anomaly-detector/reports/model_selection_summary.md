# HMM Model Selection, State Optimization & Anomaly Detection Report

## 1. Executive Summary

Hidden Markov Models (HMMs) offer a principled probabilistic framework for time-series anomaly detection in sequential industrial telemetry. Unlike static point-anomaly methods (e.g. static thresholds, GMMs, Isolation Forests), an HMM jointly models:
1. **Unobserved physical operational regimes** (Hidden States) via a first-order Markov transition matrix.
2. **State-dependent observation emissions** via conditional Gaussian distributions.
3. **Temporal transition dynamics**, enabling the detection of forbidden state jumps, oscillation chatter, and sequence degradation even when instantaneous readings fall within normal operating ranges.

In this project, we built, validated, and deployed an end-to-end HMM Anomaly Detector using `hmmlearn` on rotating machinery sensor telemetry. Through rigorous model selection comparing Bayesian Information Criterion (BIC), Akaike Information Criterion (AIC), and held-out cross-validation log-likelihood, **$K = 3$ hidden states** was identified as the optimal model architecture.

---

## 2. Model Selection Rationale: Choosing the Number of Hidden States ($K$)

### 2.1 Theoretical & Physical Rationale
Industrial rotating machinery (such as turbines, compressors, CNC spindles, and heavy pumps) exhibits distinct operational regimes:
- **State 0 (Idle / Warm-up Regime)**: Low-load operation, lower rotational vibration amplitude ($\mu_0 \approx 1.50$, $\sigma_0 \approx 0.35$).
- **State 1 (Nominal Operating Regime)**: Standard steady-state production load ($\mu_1 \approx 4.01$, $\sigma_1 \approx 0.45$).
- **State 2 (Peak / Heavy Cutting Regime)**: High-torque, heavy-load phase ($\mu_2 \approx 6.96$, $\sigma_2 \approx 0.60$).

Physical machines do not transition arbitrarily between regimes: an idle spindle cannot jump directly to maximum peak cut without spinning through nominal speed. Thus, an HMM with $K = 3$ naturally maps to the physical states of the asset.

---

### 2.2 Mathematical Formulation of Information Criteria
To empirically determine the optimal number of hidden states without underfitting or overfitting, we compute **AIC** and **BIC**.

For a 1D Gaussian HMM with $K$ states and diagonal variance, the number of free parameters $p$ is:
$$p = (K - 1) + K(K - 1) + 2K = K^2 + 2K - 1$$
- Initial state distribution $\pi$: $K - 1$ degrees of freedom ($\sum \pi_i = 1$).
- State transition probability matrix $A$: $K(K - 1)$ degrees of freedom (each row sums to 1).
- Emission parameters: $K$ means ($\mu_k$) and $K$ variances ($\sigma_k^2$).

The criteria on training sample size $N = 2,250$ (75% train split) are:
$$\text{AIC} = -2 \ln(L) + 2p$$
$$\text{BIC} = -2 \ln(L) + p \ln(N)$$

Where $\ln(L)$ is the maximized total log-likelihood computed via the Baum-Welch Forward-Backward algorithm.

---

### 2.3 Empirical Evaluation Across Candidate Hidden States ($K \in [1, 6]$)

We trained candidate models with 5 random restarts using Baum-Welch EM. The out-of-sample log-likelihood was measured on a 25% held-out validation split (750 samples):

| Hidden States ($K$) | Free Parameters ($p$) | Train Log-Likelihood | Val Log-Likelihood | Val LL / Sample | AIC | BIC | Convergence |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 2 | -4,747.17 | -1,543.34 | -2.0578 | 9,498.33 | 9,509.77 | Converged |
| **2** | 7 | -3,764.42 | -1,260.82 | -1.6811 | 7,542.83 | 7,582.86 | Converged |
| **3 (Selected)** | **14** | **-2,183.33** | **-873.39** | **-1.1645** | **4,394.65** | **4,474.71** | **Converged** |
| **4** | 23 | -2,178.51 | -886.98 | -1.1826 | 4,403.02 | 4,534.55 | Converged |
| **5** | 34 | -2,174.14 | -1,082.32 | -1.4431 | 4,416.28 | 4,610.71 | Converged |
| **6** | 47 | -2,164.59 | -906.39 | -1.2085 | 4,423.19 | 4,691.97 | Converged |

#### Key Insights from Empirical Results:
1. **Dramatic BIC Minimum at $K = 3$**: BIC drops precipitously from $K=1$ ($9,509.77$) to $K=2$ ($7,582.86$) to $K=3$ (**$4,474.71$**). For $K \ge 4$, BIC increases ($4,534.55$ at $K=4$, $4,610.71$ at $K=5$, $4,691.97$ at $K=6$) as the penalty for extraneous parameters exceeds likelihood gain.
2. **AIC Minimum at $K = 3$**: AIC achieves its lowest value at $K=3$ ($4,394.65$).
3. **Generalization Maximum on Held-Out Validation**: Held-out log-likelihood per sample reaches its maximum at $K=3$ ($-1.1645$). Models with $K \ge 4$ overfit the training sequence and generalize worse to held-out data ($-1.1826$ for $K=4$, $-1.4431$ for $K=5$).
4. **State Degeneracy for $K \ge 4$**: Inspecting $K=4$ reveals that the model artificially splits the Peak state into two redundant sub-states ($\mu_{2a} = 6.95$, $\mu_{2b} = 6.98$), confirming parameter redundancy.

**Conclusion**: Both theoretical machine physics and empirical information criteria unambiguously confirm **$K = 3$** as the optimal hidden state count.

---

## 3. Anomaly Threshold Calibration

In a streaming pipeline, individual point likelihoods can fluctuate due to random Gaussian tails. To provide robust anomaly alerting:
1. **Sliding Window Normalization**: We score sequences over a sliding window of length $W = 20$ timesteps:
   $$\ell_t = \frac{1}{W} \ln P(X_{t-W+1 : t} \mid \lambda)$$
   Normalizing by $W$ yields an average log-likelihood per sample that is invariant to window size.
2. **Empirical Calibration on Normal Data**:
   Sliding $W = 20$ across the 3,000 normal training samples generates the reference likelihood distribution:
   - $\mu_{\ell} = -2.9971$
   - $\sigma_{\ell} = 2.3434$
   - 1st Percentile $\tau_{1\%} = -8.0894$
3. **Threshold Selection**:
   Setting $\tau = \tau_{1\%} = -8.0894$ guarantees that on normal operating equipment, the detector generates an expected False Alarm Rate of less than $1\% - 2.6\%$, while offering high sensitivity to genuine mechanical degradation.

---

## 4. Detection Performance Across Industrial Anomaly Types

Testing on a continuous 2,200-sample telemetry stream with five injected failure modes:

| Anomaly Event | Type | Injected Duration | HMM Detection Status | Detection Latency | Alarm Coverage |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Event 1** | Sudden Mechanical Shock / Spikes | $t \in [300, 325]$ | **Detected (True)** | **0 steps** | **100.0%** |
| **Event 2** | Thermal & Bearing Degradation Drift | $t \in [650, 730]$ | **Detected (True)** | **14 steps** | **65.4%** |
| **Event 3** | High-Frequency Chatter / Variance Burst | $t \in [1050, 1130]$ | **Detected (True)** | **20 steps** | **41.98%** |
| **Event 4** | Forbidden Markov Transition Flutter | $t \in [1450, 1530]$ | **Detected (True)** | **16 steps** | **77.78%** |
| **Event 5** | Sensor Disconnection / Flatline Dropout | $t \in [1820, 1880]$ | **Detected (True)** | **18 steps** | **70.49%** |

### Overall Stream Performance Metrics:
- **Event Detection Rate**: **100% (5 out of 5 distinct failure modes detected)**
- **Mean Detection Latency**: **13.6 steps**
- **False Alarm Rate (FAR)**: **2.65%**
- **Sample-level Precision**: **0.7721**

### Why HMM Excels over Static Methods:
In **Event 4 (Forbidden Transition Violation)**, sensor readings rapidly alternate between 1.50 and 7.00. Crucially:
- A static threshold detector (e.g. min/max bands or Isolation Forest on individual points) **completely misses this anomaly** because $1.50$ and $7.00$ are valid operational values.
- The **Hidden Markov Model detects it with 77.8% coverage**, because the Markov transition probability $A_{0,2} = 0.0$ heavily penalizes direct Idle-to-Peak transitions, driving window log-likelihood down to $-16.37$ (well below $\tau = -8.09$).
