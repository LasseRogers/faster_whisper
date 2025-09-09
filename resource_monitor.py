#!/usr/bin/env python3
import json
import time
import psutil
import subprocess
import sys
import threading
import warnings
import os

warnings.simplefilter(action='ignore', category=FutureWarning)

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except (ImportError, ModuleNotFoundError, pynvml.NVMLError):
    GPU_AVAILABLE = False


def get_gpu_usage():
    # Return GPU memory and utilization for all GPUs
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


def monitor_resources(output_file="system_resources.json", interval=5, stop_event=None):
    # Continuously monitor CPU, RAM, and GPU usage and write to JSON
    while stop_event is None or not stop_event.is_set():
        data = {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_total_MB": psutil.virtual_memory().total / 1024**2,
            "ram_used_MB": psutil.virtual_memory().used / 1024**2,
            "ram_free_MB": psutil.virtual_memory().available / 1024**2,
            "gpu": get_gpu_usage(),
            "timestamp": time.time()
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        time.sleep(interval)


def main():
    if len(sys.argv) < 2:
        print("Usage: ./monitoring_script.py <script_to_run> [args...]")
        sys.exit(1)

    script_to_run = sys.argv[1]
    script_args = sys.argv[2:]

    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_resources,
        args=("system_resources.json", 5, stop_event)
    )
    monitor_thread.start()

    # Run the target script with FutureWarnings ignored
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"  # Ignore warnings in child process

    try:
        process = subprocess.Popen(
            [sys.executable, script_to_run, *script_args],
            env=env
        )
        process.wait()
    finally:
        # Stop monitoring when the script finishes
        stop_event.set()
        monitor_thread.join()

        # Check if the file exists and print appropriate message
        if os.path.exists("system_resources.json"):
            print(f"\nsystem_resources successfully saved to {os.path.abspath('system_resources.json')}")
        else:
            print("\nFailed to generate system_resources.json")

if __name__ == "__main__":
    main()