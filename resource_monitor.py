#!/usr/bin/env python3

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import json
import time
import psutil
import subprocess
import sys
import threading
import os


try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except (ImportError, ModuleNotFoundError, pynvml.NVMLError):
    GPU_AVAILABLE = False


def get_gpu_usage():
    if not GPU_AVAILABLE:
        return None
    gpu_data = []
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_data.append({
            "gpu_index": i,
            "memory_total_MB": mem_info.total / 1024**2,
            "memory_used_MB": mem_info.used / 1024**2,
            "memory_free_MB": mem_info.free / 1024**2,
            "gpu_utilization_percent": util.gpu,
            "memory_utilization_percent": util.memory
        })
    return gpu_data


def monitor_resources(peaks: dict, interval=1, stop_event=None):
    # Continuously monitor resources and update peak values
    while stop_event is None or not stop_event.is_set():
        cpu_percent = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()

        # Update CPU and RAM peaks
        peaks["cpu_percent"] = max(peaks["cpu_percent"], cpu_percent)
        peaks["ram_used_MB"] = max(peaks["ram_used_MB"], ram.used / 1024**2)
        peaks["ram_free_MB"] = min(peaks["ram_free_MB"], ram.available / 1024**2)

        # Update GPU peaks
        gpu_data = get_gpu_usage()
        if gpu_data:
            for i, gpu in enumerate(gpu_data):
                gpu_peak = peaks["gpu"].setdefault(i, {
                    "memory_total_MB": gpu["memory_total_MB"],
                    "memory_used_MB": 0,
                    "memory_free_MB": gpu["memory_total_MB"],  # start with max
                    "gpu_utilization_percent": 0,
                    "memory_utilization_percent": 0
                })
                gpu_peak["memory_used_MB"] = max(gpu_peak["memory_used_MB"], gpu["memory_used_MB"])
                gpu_peak["memory_free_MB"] = min(gpu_peak["memory_free_MB"], gpu["memory_free_MB"])
                gpu_peak["gpu_utilization_percent"] = max(gpu_peak["gpu_utilization_percent"], gpu["gpu_utilization_percent"])
                gpu_peak["memory_utilization_percent"] = max(gpu_peak["memory_utilization_percent"], gpu["memory_utilization_percent"])

        time.sleep(interval)


def main():
    if len(sys.argv) < 2:
        print("Usage: ./resource_monitor.py <script_to_run> [args...]")
        sys.exit(1)

    script_to_run = sys.argv[1]
    script_args = sys.argv[2:]

    ram = psutil.virtual_memory()

    # Initialize peaks
    peaks = {
        "cpu_percent": 0.0,
        "ram_total_MB": ram.total / 1024**2,
        "ram_used_MB": 0.0,
        "ram_free_MB": ram.total / 1024**2,
        "gpu": {},
    }

    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_resources,
        args=(peaks, 1, stop_event)  # sample every 1 second
    )
    monitor_thread.start()

    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"

    try:
        process = subprocess.Popen(
            [sys.executable, script_to_run, *script_args],
            env=env
        )
        process.wait()
    finally:
        stop_event.set()
        monitor_thread.join()

        # Convert GPU peaks dict
        gpu_list = []
        for idx, stats in peaks["gpu"].items():
            entry = {"gpu_index": idx, **stats}
            gpu_list.append(entry)

        output = {
            "cpu_percent": peaks["cpu_percent"],
            "ram_total_MB": peaks["ram_total_MB"],
            "ram_used_MB": peaks["ram_used_MB"],
            "ram_free_MB": peaks["ram_free_MB"],
            "gpu": gpu_list if gpu_list else None,
            "timestamp": time.time()
        }

        with open("system_resources.json", "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

        print(f"\nPeak system_resources saved to {os.path.abspath('system_resources.json')}")


if __name__ == "__main__":
    main()