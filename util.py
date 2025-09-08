import os
import yaml
import logging
import torch
import json
from faster_whisper import WhisperModel, BatchedInferencePipeline
from typing import List

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


def get_output_files(audio_file: str, output_dir: str = ".") -> dict:
    # Return both TXT and JSON output file paths for a given audio file
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    return {
        "txt": os.path.join(output_dir, f"{base_name}.txt"),
        "json": os.path.join(output_dir, f"{base_name}.json")
    }

def write_transcription_json(
    audio_file: str,
    segments: list,
    info,
    json_file: str,
    device: str = None,
    transcription_time_sec: float = None
) -> str:

    # Write transcription metadata to a JSON file.
    # Calculate total duration of transcribed segments (speech) in seconds
    vad_filtered_duration_sec = sum(seg["end"] - seg["start"] for seg in segments)

    # Get total audio duration from info object
    duration_sec = getattr(info, "duration", 0) if getattr(info, "duration", None) else 0

    # Convert durations from seconds to minutes
    duration_min = duration_sec / 60
    vad_filtered_duration_min = vad_filtered_duration_sec / 60

    # Remaining audio considered non-speech (or filtered out)
    speech_duration = duration_min - vad_filtered_duration_min

    # Compute recognition speed
    recognition_speed = None
    if transcription_time_sec and transcription_time_sec > 0:
        recognition_speed = speech_duration * 60 / transcription_time_sec  # speech_duration is in min, convert to sec

    # Build JSON data
    data = {
        "audio_file": audio_file,
        "device": device,
        "duration": duration_min,
        "vad_filtered_duration": vad_filtered_duration_min,
        "speech_duration": speech_duration,
        "recognition_speed": recognition_speed,
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": segments
    }

    # Write JSON file
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
