import os
import pandas as pd
import torch
import tempfile
from transformers import set_seed, Trainer, TrainingArguments
from tsfm_public import TimeSeriesPreprocessor, get_datasets
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.visualization import plot_predictions
import matplotlib.pyplot as plt
import numpy as np

# Set seed for reproducibility
SEED = 42
set_seed(SEED)

# Path to your dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "system_metrics_preprocessed.csv")

# Load Dataset

timestamp_column = "timestamp"
# Define Features & Target
features = [
    "system_time", "chrony_last_offset", "chrony_frequency_drift_ppm", "chrony_residual_freq_ppm",
    "chrony_skew", "cpu_temp", "cpu_freq", "network_rtt", "power", "cpu_load"
]
target = ["chrony_system_time_offset"]  # Prediction target

df = pd.read_csv(
    file_path,
    parse_dates=[timestamp_column],
)

# Drop NaN values
df.dropna(subset=features + target, inplace=True)

# Model Path (Use the IBM Granite TTM R2 Model)
TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r2"

# Context and Prediction Lengths
CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 10  # Adjustable

# Define Split Configuration (80% train, 10% valid, 10% test)
train_size = int(len(df) * 0.8)
valid_size = int(len(df) * 0.1)
split_config = {
    "train": [0, train_size],
    "valid": [train_size, train_size + valid_size],
    "test": [train_size + valid_size, len(df)],
}

# Column Specifications for Preprocessing
column_specifiers = {
    "timestamp_column": timestamp_column,
    "id_columns": [],
    "target_columns": target,
    "control_columns": features,
}

# Preprocess Data
tsp = TimeSeriesPreprocessor(
    **column_specifiers,
    context_length=CONTEXT_LENGTH,
    prediction_length=PREDICTION_LENGTH,
    scaling=True,
    encode_categorical=False,
    scaler_type="standard",
)

# Load Pre-trained TTM Model
zeroshot_model = get_model(
    TTM_MODEL_PATH,
    context_length=CONTEXT_LENGTH,
    prediction_length=PREDICTION_LENGTH,
    freq_prefix_tuning=False,
    freq=None,
    prefer_l1_loss=False,
    prefer_longer_context=True,
)

# Prepare Datasets
dset_train, dset_valid, dset_test = get_datasets(
    tsp, df, split_config, use_frequency_token=zeroshot_model.config.resolution_prefix_tuning
)

# Zero-Shot Evaluation (No Training)
temp_dir = tempfile.mkdtemp()
zeroshot_trainer = Trainer(
    model=zeroshot_model,
    args=TrainingArguments(
        output_dir=temp_dir,
        per_device_eval_batch_size=64,  # Batch size for evaluation
        seed=SEED,
        report_to="none",
    ),
)

# Evaluate (Zero-Shot Performance)
print("+" * 20, "Test MSE Zero-Shot", "+" * 20)
zeroshot_output = zeroshot_trainer.evaluate(dset_test)
print(zeroshot_output)

# Get Predictions
predictions_dict = zeroshot_trainer.predict(dset_test)
predictions_np = predictions_dict.predictions[0].flatten()  # Convert to 1D

# Extract Full Chrony System Offset (entire dataset)
full_true_values = df["chrony_system_time_offset"].values
full_time_axis = np.arange(len(full_true_values))

# Extract True Values Only for the Test Set
true_values_test = df.iloc[split_config["test"][0]:split_config["test"][1]]["chrony_system_time_offset"].values
# Slice predictions to match the test set
predictions_np = predictions_np[:len(true_values_test)]
# Create time axis for predictions
test_time_axis = np.arange(split_config["test"][0], split_config["test"][0] + len(predictions_np))

# 🔹 Custom Visualization: Full Chrony Offset + Predictions Overlay
plt.figure(figsize=(14, 6))
# Plot Full Chrony Offset as Background
plt.plot(full_time_axis, full_true_values, label="Full Chrony System Offset", color="gray", alpha=0.5)
# Overlay Model Predictions on Test Set
plt.plot(test_time_axis, predictions_np, label="Predicted Values", color="red", linewidth=2, alpha=0.7)
# Overlay Ground Truth in Test Range
plt.plot(test_time_axis, true_values_test, label="True Values (Test)", color="blue", linestyle="dashed")

# Labels and Title
plt.xlabel("Time Steps")
plt.ylabel("System Time Offset")
plt.title("Chrony System Offset: Full vs Predictions (Test)")
plt.legend()
plt.grid()
plt.show()