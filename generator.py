import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 600

def generate_measured_series(true_series, p, n, e, random_factor=0):
    """
    Generate a measured series based on a true series with sawtooth pattern deviation.

    Parameters:
    - true_series: numpy array of true values
    - p: rate of divergence per sample
    - n: number of samples before resynchronization
    - e: synchronization error
    - random_factor: factor to introduce randomness (default is 0, no randomness)

    Returns:
    - measured_series: numpy array of measured values
    """
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

true_series = np.linspace(1, 100, 1000)  # Example true series
p = 0.2  # Rate of divergence
n = 50  # Number of samples before resynchronization
e = 0.5  # Synchronization error
random_factor = 0.2  # Randomness factor

measured_series = generate_measured_series(true_series, p, n, e, random_factor)

diff_series = measured_series - true_series
# Plotting the true series and measured series
plt.figure(figsize=(14, 7))
plt.plot(true_series, label='True Series', color='blue')
plt.plot(measured_series, label='Measured Series', color='red', linestyle='--')
plt.plot(diff_series, label='Differential', color='green')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Time')
# plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.show()
