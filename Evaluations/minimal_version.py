import pandas as pd
import os
from sktime.forecasting.ttm import TinyTimeMixerForecaster
from sktime.forecasting.base import ForecastingHorizon

# Automatically detect script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "system_metrics.csv")

# Step 1: Load Dataset
df = pd.read_csv(file_path)

# Convert system_time to datetime
df["timestamp"] = pd.to_datetime(df["system_time"], unit="s", origin="unix")
# df["timestamp"] = pd.to_datetime(df["system_time"])
df.set_index("timestamp", inplace=True)  # Use timestamps as index

print("Top 10 values of system_time:")
print(df["system_time"].head(10))  # Print first 10 rows of feature data

features = [
    'chrony_system_time_offset', 'chrony_last_offset', 'chrony_frequency_drift_ppm',
    'chrony_residual_freq_ppm', 'chrony_skew', 'cpu_temp', 'cpu_freq',
    'network_rtt', 'power', 'cpu_load'
]

target = 'chrony_system_time_offset'

# Drop NaN values
df.dropna(subset=features + [target], inplace=True)

# Step 2: Define Training & Forecasting Horizon
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

# Ensure frequency is set
# Try to infer frequency
freq = pd.infer_freq(test_df.index)
print(f"Frecuency choosen: {freq}")

# test_df = test_df.asfreq(freq)
# fh = ForecastingHorizon(test_df.index, is_relative=False, freq=freq)
fh = ForecastingHorizon(X_test.index, is_relative=False, freq=freq)

# Step 3: Initialize TinyTimeMixerForecaster
forecaster = TinyTimeMixerForecaster(
    model_path=None,  # Initialize with random weights
    fit_strategy="full",  # Full training
    config={
        "context_length": 8,
        "prediction_length": 2
    },
    training_args={
        "num_train_epochs": 1,
        "output_dir": "test_output",
        "per_device_train_batch_size": 32,
    },
)

# Step 4: Train and Predict
forecaster.fit(y_train, fh=fh, X=X_train)  # Train

X_test.index = X_test.index.to_period("2s")
print("Top 10 values of X_test:")
print(X_test.head(10))  # Print first 10 rows of feature data

print("\nIndex of X_test:")
print(X_test.index[:10])  # Print first 10 index values

# Run prediction
y_pred = forecaster.predict(fh=fh, X=X_test)

# Print Results
# print(y_pred.head())  # Display some predictions
