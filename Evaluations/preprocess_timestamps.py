import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "system_metrics.csv")
out_path = os.path.join(script_dir, "system_metrics_preprocessed.csv")
# Step 1: Read CSV and Convert `system_time` to Datetime
df = pd.read_csv(file_path)

df["timestamp"] = pd.to_datetime(df["system_time"], unit="s", origin="unix")

# Set as index and convert to PeriodIndex
df.set_index("timestamp", inplace=True)
df.index = df.index.to_period("2S")  # Ensure frequency

# Save the preprocessed dataset
df.to_csv(out_path)

print("Preprocessed CSV saved successfully!")
