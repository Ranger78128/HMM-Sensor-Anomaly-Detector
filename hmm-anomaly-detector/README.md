# Anomaly Detection with Hidden Markov Models (HMM)

A real-time anomaly detection system using Hidden Markov Models (GaussianHMM) to identify unusual patterns, degradation, and state transition failures in continuous industrial machine sensor time-series data.

---

## 🌟 Highlights

- **Probabilistic Sequential Modeling**: Captures both state emission distributions and temporal state transition dynamics using `hmmlearn`.
- **Rigorous Model Selection**: Empirically validates optimal hidden state count ($K=3$) via Bayesian Information Criterion (BIC), Akaike Information Criterion (AIC), and held-out log-likelihood curves.
- **Sliding-Window Scoring**: Normalizes log-likelihood per sample over rolling windows, providing robust sequence-level probability evaluation.
- **Statistically Calibrated Threshold**: Sets the anomaly decision threshold $\tau$ directly from the empirical training log-likelihood distribution.
- **Multi-Failure Mode Detection**: Detects mechanical impact spikes, thermal/bearing drift, high-frequency chatter/variance bursts, sensor dropout flatlines, and **forbidden Markov transition violations** (which static point-based detectors miss).
- **Dual Inference Engine**:
  - Vectorized batch sliding-window processor (>2,200 samples/sec).
  - Point-by-point live telemetry streaming processor (`process_stream_sample`) with rolling buffer for real-time edge/IoT deployment.
- **Publication-Quality Visualizations**: 4-panel anomaly overlay dashboard, sliding-window log-likelihood traces, Viterbi operational regime tracking, and BIC/AIC elbow curves.

---

## 📁 Repository Structure

```
hmm-anomaly-detector/
├── data/
│   ├── data_generator.py                    # Industrial machine sensor simulator
│   ├── machine_sensor_normal.csv            # 3,000 timesteps of clean normal operating data
│   └── machine_sensor_test_stream.csv       # 2,200 timesteps of continuous stream with 5 anomaly types
├── src/
│   ├── __init__.py
│   ├── hmm_detector.py                      # Core HMM Anomaly Detector class
│   ├── model_selection.py                   # AIC, BIC & cross-validation state selection
│   ├── evaluate.py                          # Sample-level and event-level detection evaluation
│   └── visualize.py                         # Multi-panel visualization dashboard
├── models/
│   ├── trained_hmm.joblib                   # Serialized GaussianHMM weights
│   └── detector_config.json                 # Threshold, state parameters, and transition matrix
├── reports/
│   ├── model_selection_summary.md           # In-depth model selection and state rationale report
│   ├── model_selection_table.csv            # Empirical BIC, AIC, and validation likelihoods table
│   ├── event_detection_summary.csv          # Event-by-event detection latency & alarm coverage
│   ├── evaluation_metrics.json              # Full quantitative performance metrics
│   ├── anomaly_detection_overlay.png        # 4-panel anomaly detection dashboard
│   ├── model_selection_aic_bic.png          # AIC & BIC elbow curves
│   └── state_emissions_and_transitions.png  # Learned Gaussian emissions & transition matrix heatmap
├── tests/
│   └── test_detector.py                     # Unit tests for HMM detector and streaming buffer
├── main.py                                  # Complete end-to-end execution pipeline
└── README.md                                # Project documentation
```

---

## ⚙️ Installation & Setup

1. **Navigate to the Directory**:
   ```powershell
   cd C:\Users\KAUSHIK\.gemini\antigravity\scratch\hmm-anomaly-detector
   ```

2. **Install Required Packages**:
   ```bash
   pip install hmmlearn numpy pandas scikit-learn scipy matplotlib seaborn joblib pytest
   ```

---

## 🚀 Quickstart Guide

### 1. Run the Complete End-to-End Pipeline
Execute data verification, model selection, training, sliding-window scoring, evaluation, and dashboard visualization:
```bash
python main.py
```

### 2. Run the Unit Test Suite
Execute the automated test suite testing model fitting, threshold calibration, window scoring, and real-time streaming:
```bash
python -m pytest tests/ -v
```

---

## 🧠 Model Selection: Why $K = 3$ Hidden States?

### 1. Physical Machine Regimes
A rotating machine (e.g. turbine or CNC spindle) operates across three distinct physical regimes:
- **State 0 (Idle / Warm-up)**: Low vibration amplitude ($\mu_0 \approx 1.50$, $\sigma_0 \approx 0.35$).
- **State 1 (Nominal Operating)**: Steady-state production cutting/pumping ($\mu_1 \approx 4.01$, $\sigma_1 \approx 0.45$).
- **State 2 (Peak Load)**: High-torque, heavy cutting load ($\mu_2 \approx 6.96$, $\sigma_2 \approx 0.60$).

Direct transitions between Idle and Peak are physically constrained under normal operation.

### 2. Empirical Information Criteria (BIC & AIC)

$$p = K^2 + 2K - 1 \quad \text{(Free Parameters)}$$
$$\text{AIC} = -2 \ln(L) + 2p \quad | \quad \text{BIC} = -2 \ln(L) + p \ln(N)$$

| Hidden States ($K$) | Free Parameters ($p$) | Train Log-Likelihood | Val Log-Likelihood | AIC | BIC | Convergence |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 2 | -4,747.17 | -1,543.34 | 9,498.33 | 9,509.77 | Converged |
| 2 | 7 | -3,764.42 | -1,260.82 | 7,542.83 | 7,582.86 | Converged |
| **3 (Optimal)** | **14** | **-2,183.33** | **-873.39** | **4,394.65** | **4,474.71** | **Converged** |
| 4 | 23 | -2,178.51 | -886.98 | 4,403.02 | 4,534.55 | Converged |
| 5 | 34 | -2,174.14 | -1,082.32 | 4,416.28 | 4,610.71 | Converged |
| 6 | 47 | -2,164.59 | -906.39 | 4,423.19 | 4,691.97 | Converged |

- **BIC Minimum**: Attained at **$K = 3$** (4,474.71). For $K \ge 4$, the parameter penalty dominates.
- **AIC Minimum**: Attained at **$K = 3$** (4,394.65).
- **Validation Generalization**: Held-out log-likelihood per sample is maximized at $K = 3$ (-1.1645). Models with $K \ge 4$ overfit the training sequence.

---

## 📈 Anomaly Detection Methodology

### 1. Sliding-Window Normalized Likelihood
For a rolling window of length $W = 20$ timesteps:
$$\ell_t = \frac{1}{W} \ln P(X_{t-W+1 : t} \mid \lambda) = \frac{\text{model.score}(X_{t-W+1 : t})}{W}$$

### 2. Threshold Calibration
On the 3,000 normal operating training samples:
- $\mu_{\ell} = -2.9971, \sigma_{\ell} = 2.3434$.
- 1st Percentile: $\tau_{1\%} = -8.0894$.
- A window at timestep $t$ is flagged as anomalous if:
$$\ell_t < \tau$$

### 3. Real-Time Telemetry Streaming Code
```python
from src.hmm_detector import HMMAnomalyDetector

# Load production detector
detector = HMMAnomalyDetector.load("models")
detector.reset_stream()

# Live sample-by-sample processing
is_anomaly, norm_ll, state = detector.process_stream_sample(sensor_reading)

if is_anomaly:
    print(f"🚨 Anomaly Detected! LL={norm_ll:.2f} (Threshold={detector.threshold:.2f}), State={state}")
```

---

## 📊 Performance & Detection Results

| Anomaly Type | Description | Injected Period | Detected | Latency |
| :--- | :--- | :---: | :---: | :---: |
| **Mechanical Shock / Spike** | High-amplitude transient peaks (12 - 14) | $t \in [300, 325]$ | ✅ True | 0 steps |
| **Bearing Thermal Drift** | Baseline shift to uncharacteristic 6.25 | $t \in [650, 730]$ | ✅ True | 14 steps |
| **Resonance Chatter** | High-frequency variance explosion | $t \in [1050, 1130]$ | ✅ True | 20 steps |
| **Forbidden State Violation** | Rapid alternating jumps between Idle & Peak | $t \in [1450, 1530]$ | ✅ True | 16 steps |
| **Sensor Disconnection** | Transducer open-circuit flatline to 0.0 | $t \in [1820, 1880]$ | ✅ True | 18 steps |

- **Sample Precision**: **0.7721**
- **False Alarm Rate (FAR)**: **2.65%**
- **Event Detection Rate**: **100%** (all 5 failure modes detected)
- **Mean Detection Latency**: **13.5 timesteps**
