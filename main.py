#!/usr/bin/env python3

import sys
import yaml
import logging
import json
import torch
import subprocess
from tqdm import tqdm
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

# Setup logging if enabled
if logging_enabled:
    logging.basicConfig()
    logging.getLogger("faster_whisper").setLevel(logging.INFO) # DEBUG for VAD segments

# Print device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Run model
model = WhisperModel(model_size, device=device, compute_type="float16" if device=="cuda" else "int8")

# Function to get total duration using ffprobe
def get_audio_duration(filename):
    """Return duration of audio/video in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        filename
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(result.stdout.strip())

# Get total duration of audio/video in seconds
total_duration = get_audio_duration(audio_file)

# Transcribe audio
segments, info = model.transcribe(audio_file, beam_size=beam_size, language=language, vad_filter=vad_filter)

# Pprogress bar
print("\nTranscribing audio:")
with tqdm(total=total_duration, unit="s", unit_scale=True, desc="Audio progress") as pbar:
    for segment in segments:
        # update progress to the end of this segment
        pbar.n = min(segment.end, total_duration)
        pbar.refresh()

# Print detected or specified language
if language is None:
    print("Auto-detected language '%s' with probability %f" % (info.language, info.language_probability))
else:
    print("Specified language '%s' (detected probability %f)" % (info.language, info.language_probability))

# Print transcription segments
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

# Save SRT file
srt_file = audio_file.rsplit(".", 1)[0] + ".srt"
with open(srt_file, "w", encoding="utf-8") as f:
    for i, segment in enumerate(segments, start=1):
        f.write(f"{i}\n")
        f.write(f"{segment.start:.3f} --> {segment.end:.3f}\n")
        f.write(f"{segment.text}\n\n")
print(f"SRT saved to {srt_file}")

# Save JSON file
json_file = audio_file.rsplit(".", 1)[0] + ".json"
json_data = {
    "language": info.language,
    "language_probability": info.language_probability,
    "segments": [
        {"start": s.start, "end": s.end, "text": s.text} for s in segments
    ]
}
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=2, ensure_ascii=False)
print(f"JSON saved to {json_file}")
