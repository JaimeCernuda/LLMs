import time
import datetime
import subprocess
import ntplib
from selenium import webdriver
from datetime import datetime, timezone

CSV_FILE = "system_metrics.csv"  # Generic CSV file for all tasks

# ------------------ NTP Timing ------------------ #
def query_ntp_time():
    """Fetches time from an NTP server and returns time, RTT, offset, and delay in a comparable format."""
    try:
        ntp_client = ntplib.NTPClient()
        request_time = time.perf_counter()  # Local timestamp before sending request
        response = ntp_client.request("0.ubuntu.pool.ntp.org", version=3)
        response_time = time.perf_counter()  # Local timestamp after receiving response

        return {
            "ntp_time": response.tx_time,  # Server timestamp when it sent the response, in seconds (same as time.time())
            "ntp_rtt": response_time - request_time,  # Measured RTT using perf_counter()
            "ntp_offset": response.offset,  # Estimated clock offset
            "ntp_delay": response.delay,  # Total round-trip delay reported by NTP
        }
    except Exception as e:
        print(f"Warning: NTP query failed ({e})")
        return {"ntp_time": None, "ntp_rtt": None, "ntp_offset": None, "ntp_delay": None}

# ------------------ System & High-Precision Timing ------------------ #

def get_system_time():
    """Returns system wall-clock time."""
    return {"system_time": time.time()}

def get_high_precision_time():
    """Returns high-precision monotonic time."""
    return {"high_precision_time": time.perf_counter()}

# ------------------ Chrome Timing ------------------ #

def get_chrome_time():
    """Fetches Chrome's internal timestamp."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    try:
        driver.get("https://www.google.com")
        chrome_time = driver.execute_script("return performance.timeOrigin")
        return {"chrome_time": chrome_time / 1000}  # Convert ms to seconds
    except Exception as e:
        print(f"Chrome timing failed: {e}")
        return {"chrome_time": None}
    finally:
        driver.quit()

# ------------------ Chrony Timing & Drift ------------------ #

def get_chrony_data():
    """Fetches estimated clock drift and system time from Chrony in the same format as time.time()."""
    try:
        output = subprocess.check_output(["chronyc", "tracking"]).decode()
        metrics = {}

        for line in output.split("\n"):
            if "System time" in line:
                parts = line.split(":")[1].strip().split()
                metrics["chrony_system_time_offset"] = float(parts[0]) * (-1 if "slow" in parts else 1)
            elif "Last offset" in line:
                parts = line.split(":")[1].strip().split()
                metrics["chrony_last_offset"] = float(parts[0]) * (-1 if "slow" in parts else 1)
            elif "Frequency" in line:
                parts = line.split(":")[1].strip().split()
                metrics["chrony_frequency_drift_ppm"] = float(parts[0]) * (-1 if "slow" in parts else 1)
            elif "Residual freq" in line:
                metrics["chrony_residual_freq_ppm"] = float(line.split(":")[1].strip().split()[0])
            elif "Root delay" in line:
                metrics["chrony_root_delay"] = float(line.split(":")[1].strip().split()[0])
            elif "Root dispersion" in line:
                metrics["chrony_root_dispersion"] = float(line.split(":")[1].strip().split()[0])
            elif "Ref time" in line:
                metrics["chrony_last_ntp_sync_time"] = line.split(":")[1].strip()
            elif "Skew" in line:
                metrics["chrony_skew"] = line.split(":")[1].strip()

        return metrics
    except Exception as e: 
        print(f"Chrony fetch failed: {e}")
        return {"chrony_system_time_offset": None, "chrony_last_offset": None, "chrony_frequency_drift_ppm": None, "chrony_residual_freq_ppm": None,
                "chrony_root_delay": None, "chrony_root_dispersion": None, "chrony_last_ntp_sync_time": None, "chrony_skew": None}