import pandas as pd
import numpy as np
import os
import time
from sktime.forecasting.ttm import TinyTimeMixerForecaster
from sktime.forecasting.base import ForecastingHorizon
from sktime.performance_metrics.forecasting import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Step 1: Load and Preprocess the Dataset
def load_and_preprocess_data(file_path, features, target):
    df = pd.read_csv(file_path)

    # Convert system_time (Unix timestamp) to datetime
    df["timestamp"] = pd.to_datetime(df["system_time"], unit="s", origin="unix")
    df.set_index("timestamp", inplace=True)  # Use timestamps as index

    # Handle missing values
    df.dropna(subset=features + [target], inplace=True)

    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])
    scaled_df = pd.DataFrame(scaled_features, columns=features, index=df.index)

    # Add target column back
    scaled_df[target] = df[target].values

    return scaled_df, scaler

# Step 2: Evaluate the Model for Different Training Levels
def evaluate_model(file_path, features, target, training_levels, output_csv):
    scaled_df, scaler = load_and_preprocess_data(file_path, features, target)

    results = []

    for training_ratio in training_levels:
        # Define training-test split based on training level
        split_idx = int(len(scaled_df) * training_ratio)
        train_df, test_df = scaled_df.iloc[:split_idx], scaled_df.iloc[split_idx:]

        # Define training and test sets
        X_train, y_train = train_df[features], train_df[target]
        X_test, y_test = test_df[features], test_df[target]

        # Ensure frequency is set
        test_df.index.freq = pd.infer_freq(test_df.index)
        fh = ForecastingHorizon(test_df.index, is_relative=False)

        # Decide fit strategy based on training level
        if training_ratio == 0.0:
            fit_strategy = "zero-shot"
            train_kwargs = {}
        else:
            fit_strategy = "minimal"  # Minimal training
            train_kwargs = {"output_dir": "./tmp"}  # Required for training

        forecaster = TinyTimeMixerForecaster(
            model_path='ibm-granite/granite-timeseries-ttm-r2',
            validation_split=0.0,
            broadcasting=False,
            fit_strategy=fit_strategy
        )

        # Fit the model
        if training_ratio == 0.0:
            forecaster.fit(y_train=np.zeros_like(y_train), X=X_train, fh=fh)  # Zero-shot workaround
        else:
            forecaster.fit(y_train, X=X_train, fh=fh, output_dir="./tmp")  # Training mode, fix output_dir issue


        # Predict
        start_time = time.time()
        y_pred = forecaster.predict(fh=fh, X=X_test)
        end_time = time.time()

        # Compute metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        execution_time = end_time - start_time

        # Store results
        results.append({
            'training_ratio': training_ratio,
            'average_mae': mae,
            'average_rmse': rmse,
            'average_execution_time': execution_time
        })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

    return results_df

# Main Execution
if __name__ == "__main__":
    # Automatically detect script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "system_metrics.csv")

    features = [
        'chrony_system_time_offset', 'chrony_last_offset', 'chrony_frequency_drift_ppm',
        'chrony_residual_freq_ppm', 'chrony_skew', 'cpu_temp', 'cpu_freq',
        'network_rtt', 'power', 'cpu_load'
    ]

    target = 'chrony_system_time_offset'  # Predicting time offset

    training_levels = [0.1, 0.2, 0.4, 0.8]  # Different levels of training

    output_csv = os.path.join(script_dir, "training_evaluation_results.csv")

    # Evaluate the model for different training levels
    results_df = evaluate_model(file_path, features, target, training_levels, output_csv)

    print(results_df)