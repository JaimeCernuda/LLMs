import time
import csv
import time_metrics
import system_metrics
import datetime
from multiprocessing_monitor import SystemMonitor  # Assuming SystemMonitor is in a separate file

CSV_FILE = "system_metrics.csv"

# ------------------ Function to Flatten Metrics ------------------ #
def flatten_metrics(metrics):
    """Flattens a nested dictionary into a single-level dictionary."""
    flattened = {}
    for key, value in metrics.items():
        if isinstance(value, dict):  # If a function returns multiple values
            flattened.update(value)  # Unpack dictionary into separate columns
        else:
            flattened[key] = value  # Normal key-value
    return flattened

# ------------------ Main Execution ------------------ #
if __name__ == "__main__":
    DURATION = datetime.timedelta(days=1)
    INTERVAL = datetime.timedelta(seconds=1)
    SAMPLES = int(DURATION.total_seconds() / INTERVAL.total_seconds())

    tasks = {
        "system_time": time_metrics.get_system_time,
        "high_precision_time": time_metrics.get_high_precision_time,
        "ntp_data": time_metrics.query_ntp_time,  # Returns both time & RTT as a dictionary
        # "chrome_time": time_metrics.get_chrome_time,
        "chrony_data": time_metrics.get_chrony_data,  # Returns both time & drift as a dictionary
        "cpu_temp": system_metrics.get_average_core_temp,
        "cpu_freq": system_metrics.get_cpu_frequency,
        "cpu_load": system_metrics.get_cpu_load,
        "power": system_metrics.get_current_power,
        "network_rtt": system_metrics.get_network_rtt,
    }

    monitor = SystemMonitor(tasks)
    monitor.start_monitoring()

    try:
        with open(CSV_FILE, mode="a", newline="") as file:
            writer = csv.writer(file)
            # Collect initial metrics to determine column names
            initial_metrics = flatten_metrics(monitor.collect_metrics())
            # Write CSV headers
            writer.writerow(list(initial_metrics.keys()))

            # Start time-based sampling loop
            start_time = time.time()
            for i in range(SAMPLES):
                target_time = start_time + i * INTERVAL.total_seconds()
                # Collect metrics
                metrics = flatten_metrics(monitor.collect_metrics())
                # Write to CSV
                writer.writerow(list(metrics.values()))
                # Sleep precisely until next target time
                if max(0, target_time - time.time()) <= 0: print("")
                time.sleep(max(0, target_time - time.time()))

    finally:
        monitor.stop_monitoring()
