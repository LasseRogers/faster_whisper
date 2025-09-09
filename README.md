- Dependencies:
  - `faster-whisper`
  - `torch`
  - `psutil`
  - `pynvml` (optional, for GPU monitoring)
  - `librosa`
  - `matplotlib`
  - `pyyaml`

## Usage
```bash
./main.py <input_path> -o <output_dir>: default: /transcriptions -n <limit>: optional, limit number of files processed


## Usage With Resource Monitoring
The `resource_monitor.py` script allows you to monitor system resources while running transcription jobs.
It tracks CPU usage, RAM usage, and GPU memory/utilization (if NVIDIA GPU is available) in real time.

```bash
./resource_monitor.py ./main.py ...
