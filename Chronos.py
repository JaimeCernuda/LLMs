from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline
import csv
import time
import matplotlib.pyplot as plt

model_names = ["amazon/chronos-bolt-tiny", "amazon/chronos-bolt-mini", "amazon/chronos-bolt-small", 
               "amazon/chronos-bolt-base"]
# model_names = ["amazon/chronos-bolt-tiny"]
prediction_lengths = [1, 6, 12, 24]
context_lengths = [50, 100, 200, 400, 800]
iterations = 20

# df = pd.read_csv("https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv")
# passengers = df["#Passengers"]
# base_values = torch.tensor(df["#Passengers"])

df = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly/train.csv")
passengers = df["target"]
base_values = torch.tensor(df["target"])

execution_times = np.zeros((len(context_lengths), len(model_names), len(prediction_lengths)))
total_mae = np.zeros((len(context_lengths), len(model_names), len(prediction_lengths)))
total_rmse = np.zeros((len(context_lengths), len(model_names), len(prediction_lengths)))

for i, name in enumerate(model_names):
  pipeline = BaseChronosPipeline.from_pretrained(name, device_map="auto", torch_dtype=torch.bfloat16,)
  for j, prediction_length in enumerate(prediction_lengths):
    for k, context_length in enumerate(context_lengths):
      assert prediction_length + context_length <= passengers.size
      context = base_values[:context_length]
      true_values = base_values[context_length:context_length + prediction_length]
      mae = 0
      rmse = 0
      start_time = time.time()
      for _ in range(iterations):
        forecast = pipeline.predict(context, prediction_length)  # shape [num_series, num_samples, prediction_length]
        # visualize the forecast
        forecast_index = range(len(df), len(df) + prediction_length)
        low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
        mae += mean_absolute_error(true_values, median)
        rmse += np.sqrt(mean_squared_error(true_values, median))
      end_time = time.time()
      execution_times[k, i, j] = (end_time - start_time)/iterations
      total_mae[k, i, j] = mae/iterations
      total_rmse[k, i, j] = rmse/iterations
      print(f"{i}, {j}, {k}: {execution_times[k, i, j]}")
  del pipeline

for k, context_length in enumerate(context_lengths):
  with open(f"execution_times_{context_length}.csv", "w", newline="") as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(["Pipeline/Length"] + prediction_lengths)  # Header row
      for i, pipeline in enumerate(model_names):
          writer.writerow([pipeline] + list(execution_times[k][i]))

  with open(f"total_mae_{context_length}.csv", "w", newline="") as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(["Pipeline/Length"] + prediction_lengths)  # Header row
      for i, pipeline in enumerate(model_names):
          writer.writerow([pipeline] + list(total_mae[k][i]))

  with open(f"total_rmse_{context_length}.csv", "w", newline="") as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(["Pipeline/Length"] + prediction_lengths)  # Header row
      for i, pipeline in enumerate(model_names):
          writer.writerow([pipeline] + list(total_rmse[k][i]))

  # Plot execution times as a bar graph
  fig, ax = plt.subplots(figsize=(10, 6))
  bar_width = 0.15
  index = np.arange(len(prediction_lengths))

  for i, model_name in enumerate(model_names):
    ax.bar(index + i * bar_width, execution_times[k][i], bar_width, label=model_name)

  ax.set_xlabel('Prediction Length')
  ax.set_ylabel('Execution Time (s)')
  ax.set_title('Execution Time by Model and Prediction Length')
  ax.set_xticks(index + bar_width * (len(model_names) - 1) / 2)
  ax.set_xticklabels(prediction_lengths)
  ax.legend()

  plt.tight_layout()
  plt.savefig(f'time_{context_length}.png')
  plt.close()

  # Plot total MAE as a bar graph
  fig, ax = plt.subplots(figsize=(10, 6))

  for i, model_name in enumerate(model_names):
    ax.bar(index + i * bar_width, total_mae[k][i], bar_width, label=model_name)

  ax.set_xlabel('Prediction Length')
  ax.set_ylabel('Total MAE')
  ax.set_title('Total MAE by Model and Prediction Length')
  ax.set_xticks(index + bar_width * (len(model_names) - 1) / 2)
  ax.set_xticklabels(prediction_lengths)
  ax.legend()

  plt.tight_layout()
  plt.savefig(f'mae_{context_length}.png')
  plt.close()




# plt.figure(figsize=(8, 4))
# plt.plot(df["#Passengers"], color="royalblue", label="historical data")
# plt.plot(forecast_index, median, color="tomato", label="median forecast")
# plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
# plt.legend()
# plt.grid()
# plt.show()