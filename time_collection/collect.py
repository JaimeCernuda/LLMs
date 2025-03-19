import os
import time
import csv
import multiprocessing
import subprocess
import ntplib
from selenium import webdriver
from datetime import datetime, timezone
from threading import Event

# Configuration
NTP_SERVER = "pool.ntp.org"
SAMPLES = 100
INTERVAL = 10  # Seconds
OUTPUT_FILE = "clock_jitter_data.csv"

# Global synchronization event
sync_event = Event()

def set_cpu_affinity(core_id):
    """Pin the process to a specific CPU core."""
    try:
        os.sched_setaffinity(0, {core_id})
        print(f"Process pinned to CPU {core_id}")
    except AttributeError:
        print("CPU affinity not supported.")

def set_high_priority():
    """Set the process to higher priority (nice -10)."""
    try:
        os.nice(-10)
        print("Increased process priority.")
    except PermissionError:
        print("Need sudo to change priority.")

def get_ntp_client():
    """Sets up the NTP client."""
    return ntplib.NTPClient()

def query_ntp_time(ntp_client):
    """Fetches time from an NTP server."""
    try:
        request_time = time.perf_counter()
        response = ntp_client.request(NTP_SERVER, version=3)
        response_time = time.perf_counter()

        ntp_time = datetime.utcfromtimestamp(response.tx_time).replace(tzinfo=timezone.utc)
        rtt = response_time - request_time
        return ntp_time.isoformat(), rtt
    except Exception as e:
        print(f"Warning: NTP query failed ({e})")
        return None, None

def measure_system_time(queue):
    """Records system time (time.time())."""
    set_cpu_affinity(1)
    set_high_priority()
    for _ in range(SAMPLES):
        sync_event.wait()
        queue.put(time.time())

def measure_high_precision_time(queue):
    """Records high-precision time (time.perf_counter())."""
    set_cpu_affinity(2)
    set_high_priority()
    for _ in range(SAMPLES):
        sync_event.wait()
        queue.put(time.perf_counter())

def measure_ntp_time(ntp_client, queue):
    """Records NTP time using a pre-initialized client."""
    set_cpu_affinity(3)
    set_high_priority()
    for _ in range(SAMPLES):
        sync_event.wait()
        ntp_time, rtt = query_ntp_time(ntp_client)
        queue.put((ntp_time, rtt))

def get_chrome_time():
    """Fetches Chrome's internal timestamp."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    try:
        driver.get("https://www.google.com")
        chrome_time = driver.execute_script("return performance.timeOrigin")
        return chrome_time / 1000  # Convert ms to seconds
    except Exception as e:
        print(f"Chrome timing failed: {e}")
        return None
    finally:
        driver.quit()

def measure_chrome_time(queue):
    """Records Chrome performance timing."""
    set_cpu_affinity(1)
    set_high_priority()
    for _ in range(SAMPLES):
        sync_event.wait()
        queue.put(get_chrome_time())

def get_chrony_drift():
    """Fetches estimated clock drift from Chrony without modifying system time."""
    try:
        output = subprocess.check_output(["chronyc", "tracking"]).decode()
        for line in output.split("\n"):
            if "System time" in line:
                return line.split(":")[1].strip()
    except Exception as e:
        print(f"Chrony fetch failed: {e}")
        return None

def measure_chrony_drift(queue):
    """Records Chrony’s drift estimation."""
    set_cpu_affinity(2)
    set_high_priority()
    for _ in range(SAMPLES):
        sync_event.wait()
        queue.put(get_chrony_drift())

def main():
    system_time_queue = multiprocessing.Queue()
    high_precision_queue = multiprocessing.Queue()
    ntp_queue = multiprocessing.Queue()
    chrome_queue = multiprocessing.Queue()
    chrony_queue = multiprocessing.Queue()

    ntp_client = get_ntp_client()

    processes = [
        multiprocessing.Process(target=measure_system_time, args=(system_time_queue,)),
        multiprocessing.Process(target=measure_high_precision_time, args=(high_precision_queue,)),
        multiprocessing.Process(target=measure_ntp_time, args=(ntp_client, ntp_queue)),
        multiprocessing.Process(target=measure_chrome_time, args=(chrome_queue,)),
        multiprocessing.Process(target=measure_chrony_drift, args=(chrony_queue,))
    ]

    for p in processes:
        p.start()

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Sample", "System_Time", "High_Precision_Time", "Chrome_Time", "Chrony_Drift", "NTP_Time", "RTT"])

        start_time = time.time()
        for i in range(SAMPLES):
            target_time = start_time + i * INTERVAL
            time.sleep(max(0, target_time - time.time()))

            sync_event.set()
            sync_event.clear()

            sys_time = system_time_queue.get()
            high_prec_time = high_precision_queue.get()
            ntp_time, rtt = ntp_queue.get()
            chrome_time = chrome_queue.get()
            chrony_drift = chrony_queue.get()

            writer.writerow([i + 1, sys_time, high_prec_time, chrome_time, chrony_drift, ntp_time, rtt])
            print(f"[{i+1}/{SAMPLES}] System: {sys_time:.6f} | Perf: {high_prec_time:.6f} | Chrome: {chrome_time} | Chrony: {chrony_drift} | NTP: {ntp_time} | RTT: {rtt:.6f}")

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
