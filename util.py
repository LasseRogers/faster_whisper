import os
import yaml
import logging
import torch
import json
from faster_whisper import WhisperModel


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_logger(logging_enabled):
    logger = None
    if logging_enabled:
        logger = logging.getLogger("faster_whisper")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_size, device):
    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def write_transcription(segments, output_file="transcription.txt"):
    with open(output_file, "w", encoding="utf-8") as f:
        for segment in segments:
            line = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
            print(line)
            f.write(line + "\n")


def get_output_file(audio_file, output_dir="."):
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    return os.path.join(output_dir, f"{base_name}.txt")


def write_transcription_json(audio_file, segments, info, device=None, output_dir="."):
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}.json")

    # Compute VAD-filtered duration from segments
    vad_filtered_duration_sec = sum(seg["end"] - seg["start"] for seg in segments)

    # Convert seconds to minutes
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
