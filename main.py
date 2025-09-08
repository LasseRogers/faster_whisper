#!/usr/bin/env python3

import os
import argparse
import json
from util import (
    load_config,
    setup_logger,
    get_device,
    load_model,
    get_output_file
)

# Supported extensions
SUPPORTED_EXTENSIONS = ('.mp3', '.mp4', '.wav', '.flac', '.ts', '.mpeg', '.mpeg2')

# Setup argument parser
parser = argparse.ArgumentParser(description="Transcribe audio files using Faster Whisper")
parser.add_argument("input_path", help="Audio file or folder containing files")
parser.add_argument("-n", "--limit", type=int, default=None, help="Limit the number of files to transcribe")
parser.add_argument("-o", "--output", type=str, default=None, help="Output directory for text and JSON files")
args = parser.parse_args()

input_path = args.input_path
file_limit = args.limit
output_dir_cli = args.output

# Load configuration
config = load_config()
model_size = config.get("model_size", "large-v3")
beam_size = config.get("beam_size", 5)
language = config.get("language", None)
vad_filter = config.get("vad_filter", False)
logging_enabled = config.get("logging_enabled", False)
output_dir_config = config.get("output_dir", "transcriptions")

# Determine final output directory
output_dir = output_dir_cli if output_dir_cli else output_dir_config
os.makedirs(output_dir, exist_ok=True)

# Setup logging
logger = setup_logger(logging_enabled)

# Determine device
device = get_device()
if logging_enabled:
    logger.info(f"Using device: {device}")

# Load model
model = load_model(model_size, device)

# Gather audio files
if os.path.isdir(input_path):
    audio_files = [
        os.path.join(input_path, f) for f in os.listdir(input_path)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]
elif os.path.isfile(input_path) and input_path.lower().endswith(SUPPORTED_EXTENSIONS):
    audio_files = [input_path]
else:
    print("No valid audio files found.")
    exit(1)

if not audio_files:
    print("No valid audio files found.")
    exit(1)

# Apply limit if specified
if file_limit is not None:
    audio_files = audio_files[:file_limit]

# Transcribe audio
for audio_file in audio_files:
    if logging_enabled:
        logger.info(f"Processing {audio_file}...")

    # Determine output files
    txt_file = get_output_file(audio_file, output_dir)
    json_file = os.path.splitext(txt_file)[0] + ".json"

    segments_data = []

    # Stream segments as they are transcribed
    with open(txt_file, "w", encoding="utf-8") as f:
        segments_generator, info = model.transcribe(
            audio_file,
            beam_size=beam_size,
            language=language,
            vad_filter=vad_filter
        )

        if logging_enabled:
            if language is None:
                logger.info(
                    "Auto-detected language '%s' with probability %f" %
                    (info.language, info.language_probability)
                )
            else:
                logger.info(
                    "Specified language '%s' (detected probability %f)" %
                    (info.language, info.language_probability)
                )

        for segment in segments_generator:
            line = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
            print(line)
            f.write(line + "\n")

            # Save segment data for JSON
            segments_data.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })

    if logging_enabled:
        logger.info(f"Text transcription saved to {txt_file}")

    # Save JSON transcription and metadata
    json_data = {
        "audio_file": audio_file,
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": segments_data
    }

    with open(json_file, "w", encoding="utf-8") as jf:
        json.dump(json_data, jf, ensure_ascii=False, indent=4)

    if logging_enabled:
        logger.info(f"Transcription and metadata saved to {json_file}")
