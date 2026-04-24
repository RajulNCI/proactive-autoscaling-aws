# ML-Based Proactive Autoscaling on AWS
### MSc Cloud Computing — Research in Computing (CA2)
**National College of Ireland | Student: Rajul Dixit | ID: X24197483**

---

## Research Question

How accurately can a supervised Random Forest machine learning model predict cloud workload demand to enable proactive autoscaling on AWS EC2, and to what extent does this outperform the standard AWS CloudWatch reactive baseline in SLA compliance, scaling response time, and resource utilisation efficiency?

---

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| ML Pipeline | Random Forest regression model on synthetic workload data | ✅ Complete |
| Preliminary Results | R²=0.9531, MAE=0.0352, RMSE=0.0454 | ✅ Complete |
| AWS Deployment | RF proactive autoscaling pipeline on EC2 | 🔜 Planned (Capstone) |
| Reactive Baseline | AWS Auto Scaling Groups + CloudWatch alarms | 🔜 Planned (Capstone) |
| JMeter Load Testing | 2-hour comparative load test on both systems | 🔜 Planned (Capstone) |
| Comparative Evaluation | Paired t-test analysis across all metrics | 🔜 Planned (Capstone) |

---

## Preliminary Results

Trained on 3,000 synthetic workload data points modelled on Google Cluster Traces v3 statistical properties (80/20 chronological split).

| Metric | Value | Notes |
|--------|-------|-------|
| R² | 0.9531 | 95.3% of CPU variance explained at t+10 min |
| MAE | 0.0352 | Mean absolute error on normalised CPU [0,1] |
| RMSE | 0.0454 | Root mean squared error on normalised CPU [0,1] |
| Top Feature | roll_mean_5 (0.961) | 5-bucket rolling mean is dominant signal |
| Actual SLA Violations | 54 events | CPU > 0.70 threshold breaches in test set |
| Proactive Triggers | 25 events | Fired before breach — 46.3% coverage |

---

## How to Run

```bash
# Install dependencies
pip install scikit-learn pandas numpy

# Run the pipeline
python MainML.py
```

No external dataset download needed — workload data is generated synthetically inside the script.

---

## Dataset

Synthetic workload data modelled on the statistical properties of
[Google Cluster Traces v3](https://github.com/google/cluster-data) (CC BY 4.0 licence).

- 15,000 raw data points → resampled to 3,000 five-minute buckets
- Sinusoidal baseline CPU utilisation (diurnal pattern)
- Gaussian noise (σ=0.08) representing natural variation
- 200 random burst spike events of magnitude [0.3, 0.6]

---

## ML Pipeline Overview

```
Raw Data Generation (15,000 points)
        ↓
Temporal Resampling (3,000 x 5-min buckets)
        ↓
Feature Engineering (lag t-1 to t-12, rolling mean/std/max)
        ↓
80/20 Chronological Train/Test Split (no shuffling)
        ↓
MinMax Normalisation (fitted on train only)
        ↓
Random Forest Regressor (n_estimators=150, max_depth=15)
        ↓
Evaluation (MAE, RMSE, R²) + Feature Importance
        ↓
Autoscaling Simulation (proactive triggers vs SLA violations)
```

---

## Planned AWS Architecture (Capstone Semester)

```
CloudWatch Metrics (every 5 min)
        ↓
S3 Data Store
        ↓
RF Prediction Engine (t3.micro EC2)  →  10-min ahead CPU forecast
        ↓
Proactive Scaling Decision Engine    →  ASG scale-out if forecast > 65%

Parallel: CloudWatch Reactive Baseline → scale-out if actual CPU > 70%

Both evaluated under identical Apache JMeter 2-hour load test
```

---

## Repository Structure

```
proactive-autoscaling-aws/
├── README.md                        ← This file
├── rf_proactive_autoscaler.py       ← Complete ML pipeline
├── requirements.txt                 ← Dependencies
└── results/
    └── preliminary_results.txt      ← Actual terminal output
```

---

## References

This project is part of the CA2 Research Proposal for the module Research in Computing (MSCCLOUD1_A) at National College of Ireland, 2025–2026.

Key references:
- Dogani et al. (2023) — Auto-scaling taxonomy survey, Computer Communications
- Khaleq & Ra (2024) — Cloud autoscaling problems, Sensors
- Zhang et al. (2020) — ML cloud resource management survey, IEEE Access
- Google Cluster Traces v3 — https://github.com/google/cluster-data

---

## License

MIT License — see LICENSE file for details.
