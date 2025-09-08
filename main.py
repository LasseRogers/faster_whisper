#!/usr/bin/env python3

import os
import argparse
from util import (
    load_config,
    setup_logger,
    get_device,
    load_model,
    get_output_file,
    write_transcription_json
)

SUPPORTED_EXTENSIONS = ('.mp3', '.mp4', '.wav', '.flac', '.ts', '.mpeg', '.mpeg2')

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

output_dir = output_dir_cli if output_dir_cli else output_dir_config
os.makedirs(output_dir, exist_ok=True)

logger = setup_logger(logging_enabled)
device = get_device()
if logging_enabled:
    logger.info(f"Using device: {device}")

model = load_model(model_size, device)

# Collect audio files
if os.path.isdir(input_path):
    audio_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(SUPPORTED_EXTENSIONS)]
elif os.path.isfile(input_path) and input_path.lower().endswith(SUPPORTED_EXTENSIONS):
    audio_files = [input_path]
else:
    print("No valid audio files found.")
    exit(1)

if file_limit is not None:
    audio_files = audio_files[:file_limit]

# Transcription loop
for audio_file in audio_files:
    if logging_enabled:
        logger.info(f"Processing {audio_file}...")

    txt_file = get_output_file(audio_file, output_dir)
    segments_data = []

    with open(txt_file, "w", encoding="utf-8") as f:
        segments_generator, info = model.transcribe(
            audio_file,
            beam_size=beam_size,
            language=language,
            vad_filter=vad_filter
        )

        if logging_enabled:
            lang_msg = "Specified" if language else "Auto-detected"
            logger.info(f"{lang_msg} language '{info.language}' (probability {info.language_probability})")

        for segment in segments_generator:
            line = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
            print(line)
            f.write(line + "\n")
            segments_data.append({"start": segment.start, "end": segment.end, "text": segment.text})

    if logging_enabled:
        logger.info(f"Text transcription saved to {txt_file}")

    json_file = write_transcription_json(audio_file, segments_data, info, device=device, output_dir=output_dir)
    if logging_enabled:
        logger.info(f"Transcription and metadata saved to {json_file}")
