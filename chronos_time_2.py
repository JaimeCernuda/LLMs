import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 600

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


# Function to predict and evaluate
def predict_and_evaluate(true_series, measured_series, test_size, prediction_output_size, test_size_type='percentage'):
    # Determine the split index based on the test_size_type
    if test_size_type == 'percentage':
        split_index = int(len(true_series) * (1 - test_size / 100))
    elif test_size_type == 'absolute':
        split_index = len(true_series) - test_size
    else:
        raise ValueError("Invalid test_size_type. Use 'percentage' or 'absolute'.")

    train_true_series = true_series[:split_index]
    train_measured_series = measured_series[:split_index]
    test_measured_series = measured_series[split_index:]
    test_true_series = true_series[split_index:]

    # Use the same size as test_size for prediction_sample_size
    prediction_sample_size = split_index

    # Calculate the difference series for the training data
    diff_series = train_measured_series - train_true_series

    # Load the pretrained ChronosPipeline model
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-base",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Prepare the context tensor from the training difference series
    context = torch.tensor(diff_series[-prediction_sample_size:], dtype=torch.float32)

    # Make predictions using the model
    forecast = pipeline.predict(context,
                                prediction_output_size)  # shape [num_series, num_samples, prediction_output_size]

    # Get the forecast values and calculate quantiles
    low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

    # Reconstruct the predicted true series for the test data
    predicted_true_series = test_measured_series[:prediction_output_size] - median

    # Calculate the metrics
    true_values_for_prediction = test_true_series[:prediction_output_size]
    mae = mean_absolute_error(true_values_for_prediction, predicted_true_series)
    rmse = np.sqrt(mean_squared_error(true_values_for_prediction, predicted_true_series))

    # Print the metrics
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Visualize the forecast
    plt.figure(figsize=(14, 7))
    plt.plot(range(split_index), true_series[:split_index], color="blue", label="True Series (Training)")
    plt.plot(range(split_index), measured_series[:split_index], color="red", linestyle="--",
             label="Measured Series (Training)")
    plt.plot(range(split_index, split_index + prediction_output_size), predicted_true_series, color="green",
             linestyle="--", label="Predicted Series (Test)")
    plt.plot(range(split_index, split_index + prediction_output_size),
             true_series[split_index:split_index + prediction_output_size], color="blue", linestyle=":",
             label="True Series (Test)")
    plt.fill_between(range(split_index, split_index + prediction_output_size),
                     test_measured_series[:prediction_output_size] - low,
                     test_measured_series[:prediction_output_size] - high, color="green", alpha=0.3,
                     label="80% Prediction Interval")
    plt.legend()
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('True Series vs Measured Series vs Predicted Series')
    plt.show()


# Example usage
np.random.seed(42)  # For reproducibility

true_series = np.linspace(1, 50, 100)  # Example true series
p = 1  # Rate of divergence
n = 5  # Number of samples before resynchronization
e = 0.5  # Synchronization error
random_factor = 1  # Randomness factor

measured_series = generate_measured_series(true_series, p, n, e, random_factor)

# Configure the test size and prediction output size
test_size = 40  # This can be in percentage or absolute value
prediction_output_size = 10  # Number of samples to predict
test_size_type = 'absolute'  # Use 'percentage' or 'absolute'

# Run prediction and evaluation
predict_and_evaluate(true_series, measured_series, test_size, prediction_output_size, test_size_type)
