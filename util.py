import os
import yaml
import logging
import torch
import json
from faster_whisper import WhisperModel
from typing import List, Dict

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


def load_model(model_size: str, device: str) -> WhisperModel:
    # Load Whisper model with appropriate precision
    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def get_output_file(audio_file: str, output_dir: str = ".") -> str:
    # Get TXT output file path for audio file
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    return os.path.join(output_dir, f"{base_name}.txt")


def write_transcription_json(audio_file: str, segments: List[Dict], info, device: str = None, output_dir: str = ".") -> str:
    # Save transcription and metadata as JSON
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}.json")

    # Compute VAD-filtered duration
    vad_filtered_duration_sec = sum(seg["end"] - seg["start"] for seg in segments)
    duration_min = getattr(info, "duration", 0) / 60 if getattr(info, "duration", None) else None
    vad_filtered_duration_min = vad_filtered_duration_sec / 60

    data = {
        "audio_file": audio_file,
        "device": device,
        "duration": duration_min,
        "vad_filtered_duration": vad_filtered_duration_min,
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": segments
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return output_file


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
