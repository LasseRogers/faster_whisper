import os
import yaml
import logging
import torch
import json
from faster_whisper import WhisperModel, BatchedInferencePipeline
from typing import List

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # For headless backend
import librosa
import librosa.display
import numpy as np

# Supported audio file extensions
SUPPORTED_EXTENSIONS = ('.mp3', '.mp4', '.wav', '.flac', '.ts', '.mpeg', '.mpeg2')


def load_config(path="config.yaml") -> dict:
    # Load YAML configuration
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_logger(logging_enabled: bool):
    # Setup logger if enabled
    logger = None
    if logging_enabled:
        logger = logging.getLogger("faster_whisper")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


def get_device() -> str:
    # Return "cuda" if available, else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_batched_model(model_size: str, device: str) -> BatchedInferencePipeline:
    # Load Whisper model and wrap it in a batched inference pipeline
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return BatchedInferencePipeline(model=model)


def write_transcription_json(
    # Write transcription segments and metadata to JSON
    audio_file: str,
    segments: list,
    info,
    json_file: str,
    device: str = None,
    transcription_time_sec: float = None
) -> str:

    duration_sec = getattr(info, "duration", 0) if getattr(info, "duration", None) else 0

    # Duration of speech segments transcribed
    speech_duration_sec = sum(seg["end"] - seg["start"] for seg in segments)

    # Duration of audio filtered out by VAD
    non_speech_duration_sec = max(duration_sec - speech_duration_sec, 0)

    # Convert to minutes
    duration_min = duration_sec / 60
    speech_duration_min = speech_duration_sec / 60
    non_speech_duration_min = non_speech_duration_sec / 60
    run_time_min = transcription_time_sec / 60 if transcription_time_sec else None

    # Compute recognition speed
    recognition_speed = None
    if transcription_time_sec and transcription_time_sec > 0:
        recognition_speed = speech_duration_sec / transcription_time_sec

    # Build JSON schema
    data = {
        "audio_file": audio_file,
        "device": device,
        "audio_duration_min": duration_min,
        "non_speech_duration_min": non_speech_duration_min,
        "speech_duration_min": speech_duration_min,
        "run_time_min": run_time_min,
        "recognition_speed": recognition_speed,
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": segments
    }

    # Save JSON
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return json_file


def collect_audio_files(input_path: str, limit: int = None) -> List[str]:
    # Get all audio files from folder or single file
    if os.path.isdir(input_path):
        files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                 if f.lower().endswith(SUPPORTED_EXTENSIONS)]
    elif os.path.isfile(input_path) and input_path.lower().endswith(SUPPORTED_EXTENSIONS):
        files = [input_path]
    else:
        raise FileNotFoundError("No valid audio files found.")
    return files[:limit] if limit else files


def plot_waveform_with_vad(audio_file: str, segments: list, output_dir: str):
    # Load audio
    y, sr = librosa.load(audio_file, sr=None, mono=True)

    if len(y) == 0:
        raise ValueError(f"No audio data found in {audio_file}")

    # Downsample for plotting speed (max 100k points)
    max_points = 100_000
    if len(y) > max_points:
        hop = len(y) // max_points
        y = y[::hop]
        times = (np.arange(len(y)) * hop / sr) / 60.0  # convert to minutes
    else:
        times = (np.arange(len(y)) / sr) / 60.0  # convert to minutes

    # Create plot
    plt.figure(figsize=(12, 4))
    plt.plot(times, y, alpha=0.7, label="Waveform")

    # Sort segments
    segments = sorted(segments, key=lambda s: s["start"])

    # Mark speech (green) and non-speech (red) in minutes
    last_end = 0.0
    for seg in segments:
        if seg["start"] > last_end:
            plt.axvspan(last_end / 60.0, seg["start"] / 60.0, color="red", alpha=0.2)
        plt.axvspan(seg["start"] / 60.0, seg["end"] / 60.0, color="green", alpha=0.3)
        last_end = seg["end"]

    duration_min = times[-1]
    if last_end / 60.0 < duration_min:
        plt.axvspan(last_end / 60.0, duration_min, color="red", alpha=0.2)

    plt.title(f"Waveform with VAD Segments: {os.path.basename(audio_file)}")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Amplitude")

    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    plot_file = os.path.join(output_dir, f"{base_name}_waveform.png")
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()

    return plot_file