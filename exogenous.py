import time
import psutil
import numpy as np
import pandas as pd
import os
import subprocess
import re
import shutil

# Function to get CPU temperature (Linux)
# def get_cpu_temperature():
#     try:
#         temp_file = "/sys/class/thermal/thermal_zone0/temp"
#         if os.path.exists(temp_file):
#             with open(temp_file, "r") as f:
#                 return int(f.read().strip()) / 1000  # Convert from millidegrees
#         return None
#     except Exception as e:
#         return None

def get_average_core_temp():
    temps = psutil.sensors_temperatures()
    
    if "coretemp" not in temps:
        print("No core temperature data available.")
        return None

    core_temps = [sensor.current for sensor in temps["coretemp"] if "Core" in sensor.label]

    if not core_temps:
        print("No per-core temperatures found.")
        return None

    avg_temp = sum(core_temps) / len(core_temps)
    return avg_temp

# Function to get CPU frequency
def get_cpu_frequency():
    return psutil.cpu_freq().current if psutil.cpu_freq() else None

# Function to get CPU load
def get_cpu_load():
    return psutil.cpu_percent(interval=1)  # 1-second sampling

# Function to get power consumption (Linux hwmon)
# def get_power_draw():
#     try:
#         power_path = "/sys/class/power_supply/BAT0/power_now"
#         if os.path.exists(power_path):
#             with open(power_path, "r") as f:
#                 return int(f.read().strip()) / 1e6  # Convert µW to W
#         return None
#     except Exception as e:
#         return None

def is_ipmi_dcmi_installed():
    """Check if ipmi-dcmi is installed by looking for it in PATH."""
    return shutil.which("ipmi-dcmi") is not None

def get_current_power():
    if not is_ipmi_dcmi_installed():
        print("Error: ipmi-dcmi is not installed or not in PATH.")
        return None

    try:
        # Run the ipmi-dcmi command
        result = subprocess.run(
            ["sudo", "ipmi-dcmi", "--get-system-power-statistics"],
            capture_output=True,
            text=True,
            check=True
        )

        # Extract the "Current Power" value using regex
        match = re.search(r"Current Power\s*:\s*(\d+)\s*Watts", result.stdout)
        if match:
            return int(match.group(1))  # Return power as an integer (Watts)
        else:
            print("Current Power value not found in output.")
            return None
    except FileNotFoundError:
        print("Error: ipmi-dcmi command not found.")
    except subprocess.CalledProcessError as e:
        print(f"Error running ipmi-dcmi: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return None
    
# Function to get network round-trip time (RTT)
def get_network_rtt(host="time.google.com"):
    try:
        result = subprocess.run(["ping", "-c", "1", host], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout
        if "time=" in output:
            time_ms = float(output.split("time=")[-1].split()[0])
            return time_ms
        return None
    except Exception as e:
        return None

# Initialize Data Collection
log_interval = 5  # seconds
duration = 300  # Run for 5 minutes
log_data = []

start_time = time.time()

while (time.time() - start_time) < duration:
    timestamp = time.time()  # Get high-precision time
    cpu_temp = get_cpu_temperature()
    cpu_freq = get_cpu_frequency()
    cpu_load = get_cpu_load()
    power_draw = get_power_draw()
    network_rtt = get_network_rtt()

    # Store collected data
    log_data.append([timestamp, cpu_temp, cpu_freq, cpu_load, power_draw, network_rtt])
    
    print(f"Time: {timestamp}, Temp: {cpu_temp}°C, Freq: {cpu_freq} MHz, Load: {cpu_load}%, Power: {power_draw}W, RTT: {network_rtt}ms")

    time.sleep(log_interval)

# Save data to CSV
df = pd.DataFrame(log_data, columns=["Timestamp", "CPU_Temp", "CPU_Freq", "CPU_Load", "Power_Draw", "Network_RTT"])
df.to_csv("clock_skew_log.csv", index=False)
