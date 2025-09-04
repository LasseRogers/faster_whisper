#!/usr/bin/env python3

import sys
import yaml
import logging
import torch
from faster_whisper import WhisperModel

# Check if audio file is provided
if len(sys.argv) < 2:
    print("Usage: ./transcribe.py <audio_file>")
    sys.exit(1)

audio_file = sys.argv[1]

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

model_size = config.get("model_size", "large-v3")
beam_size = config.get("beam_size", 5)
language = config.get("language", None)
vad_filter = config.get("vad_filter", False)
logging_enabled = config.get("logging_enabled", False)

# Output file
output_file = "transcription.txt"

# Setup logging
logger = None
if logging_enabled:
    logger = logging.getLogger("faster_whisper")
    logger.setLevel(logging.INFO) # DEBUG for VAD segments

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Terminal handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(output_file, encoding="utf-8", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Print device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if logging_enabled:
    logger.info(f"Using device: {device}")

# Load model
model = WhisperModel(
    model_size,
    device=device,
    compute_type="float16" if device == "cuda" else "int8"
)

# Transcribe audio
segments, info = model.transcribe(
    audio_file,
    beam_size=beam_size,
    language=language,
    vad_filter=vad_filter
)

# Write transcription to the same file
with open(output_file, "a", encoding="utf-8") as f:
    if language is None:
        lang_info = "Auto-detected language '%s' with probability %f\n\n" % (
            info.language, info.language_probability
        )
    else:
        lang_info = "Specified language '%s' (detected probability %f)\n\n" % (
            info.language, info.language_probability
        )
    print(lang_info.strip())
    f.write(lang_info)

    for segment in segments:
        line = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
        #print(line)
        f.write(line + "\n")

print(f"\nTranscription saved to {output_file}")
