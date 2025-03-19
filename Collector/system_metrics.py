import psutil
import shutil
import subprocess
import re

def get_average_core_temp():
    temps = psutil.sensors_temperatures()
    if "coretemp" not in temps:
        return None
    core_temps = [sensor.current for sensor in temps["coretemp"] if "Core" in sensor.label]
    return sum(core_temps) / len(core_temps) if core_temps else None

def get_cpu_frequency():
    return psutil.cpu_freq().current if psutil.cpu_freq() else None

def get_cpu_load():
    return psutil.cpu_percent(interval=1)

def is_ipmi_dcmi_installed():
    return shutil.which("ipmi-dcmi") is not None

def get_current_power():
    if not is_ipmi_dcmi_installed():
        return None
    try:
        result = subprocess.run(
            ["sudo", "ipmi-dcmi", "--get-system-power-statistics"],
            capture_output=True, text=True, check=True
        )
        match = re.search(r"Current Power\\s*:\\s*(\\d+)\\s*Watts", result.stdout)
        return int(match.group(1)) if match else None
    except Exception:
        return None

def get_network_rtt(host="time.google.com"):
    try:
        result = subprocess.run(["ping", "-c", "1", host], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout
        return float(output.split("time=")[-1].split()[0]) if "time=" in output else None
    except Exception:
        return None
