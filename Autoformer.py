from huggingface_hub import hf_hub_download
import torch
from transformers import AutoformerForPrediction

file = hf_hub_download(
    repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
)
batch = torch.load(file)

model = AutoformerForPrediction.from_pretrained("huggingface/autoformer-tourism-monthly")

# during training, one provides both past and future values
# as well as possible additional features
# outputs = model(
#     past_values=batch["past_values"],
#     past_time_features=batch["past_time_features"],
#     past_observed_mask=batch["past_observed_mask"],
#     static_categorical_features=batch["static_categorical_features"],
#     future_values=batch["future_values"],
#     future_time_features=batch["future_time_features"],
# )
#
# loss = outputs.loss
# loss.backward()

# during inference, one only provides past values
# as well as possible additional features
# the model autoregressively generates future values
outputs = model.generate(
    past_values=batch["past_values"],
    past_time_features=batch["past_time_features"],
    past_observed_mask=batch["past_observed_mask"],
    static_categorical_features=batch["static_categorical_features"],
    future_time_features=batch["future_time_features"],
)

mean_prediction = outputs.sequences.mean(dim=1)

import matplotlib.pyplot as plt

# Convert tensors to numpy arrays for plotting
past_values = batch["past_values"].numpy().flatten()
predicted_values = mean_prediction.detach().numpy().flatten()

# Define the range for plotting
time_steps = range(len(past_values))
future_steps = range(len(past_values), len(past_values) + len(predicted_values))

# Plot the actual and predicted values
plt.figure(figsize=(12, 6))
plt.plot(time_steps, past_values, color='blue', label='Actual Values')
plt.plot(future_steps, predicted_values, color='red', label='Predicted Values')
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.title('Time Series Forecasting with Autoformer')
plt.legend()
plt.show()
