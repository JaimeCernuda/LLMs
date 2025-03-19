import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Generate the measured series based on the true series
def generate_measured_series(true_series, p, n, e, random_factor=0):
    measured_series = np.copy(true_series)
    num_samples = len(true_series)

    for i in range(0, num_samples, n):
        for j in range(i, min(i + n, num_samples)):
            deviation = p * (j - i) + np.random.uniform(-random_factor, random_factor)
            measured_series[j] += deviation
        if i + n < num_samples:
            measured_series[i + n] = true_series[i + n] + e + np.random.uniform(-random_factor, random_factor)

    return measured_series


# Example usage
np.random.seed(42)  # For reproducibility

true_series = np.linspace(1, 50, 100)  # Example true series
p = 0.05  # Rate of divergence
n = 5  # Number of samples before resynchronization
e = 0.5  # Synchronization error
random_factor = 0.1  # Randomness factor
prediction_length = 5  # Arbitrary length for prediction

measured_series = generate_measured_series(true_series, p, n, e, random_factor)

# Calculate the difference series
diff_series = measured_series - true_series

# Load the pretrained ChronosPipeline model
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-base",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Prepare the context tensor from the diff series
context = torch.tensor(diff_series, dtype=torch.float32)

# Make predictions using the model
forecast = pipeline.predict(context, prediction_length)  # shape [num_series, num_samples, prediction_length]

# Get the forecast values and calculate quantiles
forecast_index = range(len(diff_series), len(diff_series) + prediction_length)
low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

print(median)

# # Reconstruct the predicted series
# predicted_diff = median[:len(forecast_index)]
# predicted_series = measured_series[-len(forecast_index):] - predicted_diff
#
# # Calculate the metrics
# true_values_for_prediction = true_series[-len(forecast_index):]
# mae = mean_absolute_error(true_values_for_prediction, predicted_series)
# rmse = np.sqrt(mean_squared_error(true_values_for_prediction, predicted_series))
#
# # Print the metrics
# print(f"Mean Absolute Error (MAE): {mae}")
# print(f"Root Mean Squared Error (RMSE): {rmse}")
#
# # Visualize the forecast
# plt.figure(figsize=(14, 7))
# plt.plot(range(len(true_series)), true_series, color="blue", label="True Series")
# plt.plot(range(len(measured_series)), measured_series, color="red", linestyle="--", label="Measured Series")
# plt.plot(forecast_index, predicted_series, color="green", linestyle="--", label="Predicted Series")
# plt.fill_between(forecast_index, measured_series[-len(forecast_index):] - low,
#                  measured_series[-len(forecast_index):] - high, color="green", alpha=0.3,
#                  label="80% prediction interval")
# plt.legend()
# plt.xlabel('Sample Index')
# plt.ylabel('Value')
# plt.title('True Series vs Measured Series vs Predicted Series')
# plt.show()

plt.figure(figsize=(8, 4))
plt.plot(diff_series, color="royalblue", label="historical data")
plt.plot(forecast_index, median, color="tomato", label="median forecast")
plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
plt.legend()
plt.grid()
plt.show()