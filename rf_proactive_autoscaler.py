"""
RF-Based Proactive Autoscaling
"""

import pandas as pd
import numpy as np
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# STEP 1 — SYNTHETIC CLUSTER WORKLOAD DATA (REALISTIC)
# ─────────────────────────────────────────────────────────────
print("Generating realistic cluster workload dataset...")

np.random.seed(42)
n = 15000

time = np.arange(n)

# realistic CPU pattern: wave + noise + spikes
cpu = 0.4 + 0.3 * np.sin(np.linspace(0, 30, n)) + np.random.normal(0, 0.08, n)

# random spikes (simulate load bursts)
spike_indices = np.random.choice(n, size=200, replace=False)
cpu[spike_indices] += np.random.uniform(0.3, 0.6, size=200)

cpu = np.clip(cpu, 0, 1)

df = pd.DataFrame({"timestamp": time, "cpu": cpu})

print(f"Dataset generated: {len(df):,} rows")

# ─────────────────────────────────────────────────────────────
# STEP 2 — PREPROCESSING
# ─────────────────────────────────────────────────────────────
print("\n--- PREPROCESSING ---")

df = df.sort_values("timestamp").reset_index(drop=True)

# 5-minute bucket simulation
df["bucket"] = df["timestamp"] // 5
df_resampled = df.groupby("bucket")["cpu"].mean().reset_index()

# fill missing
full = pd.DataFrame(
    {"bucket": range(df_resampled.bucket.min(), df_resampled.bucket.max() + 1)}
)

df_resampled = full.merge(df_resampled, on="bucket", how="left")
df_resampled["cpu"] = df_resampled["cpu"].interpolate()

print(f"After resampling: {len(df_resampled):,}")

# ─────────────────────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print("\n--- FEATURE ENGINEERING ---")

data = df_resampled.copy()

for lag in [1, 2, 3, 6, 12]:
    data[f"lag_{lag}"] = data["cpu"].shift(lag)

data["roll_mean_5"] = data["cpu"].shift(1).rolling(5).mean()
data["roll_std_10"] = data["cpu"].shift(1).rolling(10).std()
data["roll_max_5"] = data["cpu"].shift(1).rolling(5).max()

data["target"] = data["cpu"].shift(-2)

data = data.dropna().reset_index(drop=True)

features = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "roll_mean_5",
    "roll_std_10",
    "roll_max_5",
]

X = data[features].values
y = data["target"].values

# ─────────────────────────────────────────────────────────────
# STEP 4 — TRAIN TEST SPLIT (time-series safe)
# ─────────────────────────────────────────────────────────────
split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

scaler_X = MinMaxScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = MinMaxScaler()
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# ─────────────────────────────────────────────────────────────
# STEP 5 — MODEL TRAINING
# ─────────────────────────────────────────────────────────────
print("\n--- TRAINING MODEL ---")

rf = RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)

rf.fit(X_train, y_train)

pred = rf.predict(X_test)

pred = scaler_y.inverse_transform(pred.reshape(-1, 1)).ravel()
y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

# ─────────────────────────────────────────────────────────────
# STEP 6 — METRICS
# ─────────────────────────────────────────────────────────────
mae = mean_absolute_error(y_true, pred)
rmse = np.sqrt(mean_squared_error(y_true, pred))
r2 = r2_score(y_true, pred)

print("\n================ RESULTS ================")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")

# ─────────────────────────────────────────────────────────────
# STEP 7 — FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────
print("\n--- FEATURE IMPORTANCE ---")

for f, imp in sorted(zip(features, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"{f:<15} {imp:.3f}")

# ─────────────────────────────────────────────────────────────
# STEP 8 — SIMPLE AUTOSCALING SIMULATION
# ─────────────────────────────────────────────────────────────
print("\n--- AUTOSCALING SIMULATION ---")

threshold = 0.7

violations = np.sum(y_true > threshold)
proactive_hits = np.sum(pred > threshold)

print(f"Actual SLA violations: {violations}")
print(f"Proactive triggers   : {proactive_hits}")

print("\nDONE ✔ Pipeline working successfully")
