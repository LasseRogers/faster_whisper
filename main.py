#!/usr/bin/env python3

import os
import argparse
import time
import json
from util import (
    load_config,
    setup_logger,
    get_device,
    load_model,
    get_output_file,
    write_transcription_json,
    collect_audio_files
)

def parse_args():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Transcribe audio files using Faster Whisper")
    parser.add_argument("input_path", help="Audio file or folder containing files")
    parser.add_argument("-n", "--limit", type=int, default=None, help="Limit number of files")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output directory")
    return parser.parse_args()


def transcribe_file(model, audio_file, output_dir, beam_size=5, language=None, vad_filter=False, logger=None, device=None):
    # Determine the output TXT file path
    txt_file = get_output_file(audio_file, output_dir)
    segments_data = []  # Store segment info for JSON

    start_time = time.time()  # Start timing transcription

    with open(txt_file, "w", encoding="utf-8") as f:
        # Transcribe audio using the model
        segments_generator, info = model.transcribe(
            audio_file, beam_size=beam_size, language=language, vad_filter=vad_filter
        )

        # Log detected or specified language
        if logger:
            lang_msg = "Specified" if language else "Auto-detected"
            logger.info(f"{lang_msg} language '{info.language}' (probability {info.language_probability})")

        # Process each segment
        for segment in segments_generator:
            line = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
            #print(line)
            f.write(line + "\n")  # Write segment to TXT file
            segments_data.append({"start": segment.start, "end": segment.end, "text": segment.text})

    # Calculate total transcription time
    transcription_time_sec = time.time() - start_time

    if logger:
        logger.info(f"Text transcription saved to {txt_file}")

    # Save metadata and durations to JSON
    json_file = write_transcription_json(
        audio_file,
        segments_data,
        info,
        device=device,
        output_dir=output_dir,
        transcription_time_sec=transcription_time_sec
    )

    if logger:
        logger.info(f"Transcription and metadata saved to {json_file}")

    # Return paths to TXT and JSON files
    return txt_file, json_file



def main():
    # Main program
    args = parse_args()
    config = load_config()
    output_dir = args.output or config.get("output_dir", "transcriptions")
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(config.get("logging_enabled", False))
    device = get_device()
    if logger:
        logger.info(f"Using device: {device}")

    model = load_model(config.get("model_size", "large-v3"), device)

    # Collect audio files from input path
    audio_files = collect_audio_files(args.input_path, args.limit)

    recognition_speeds = []  # <-- List to store recognition_speed for each file

    # Transcribe each file
    for audio_file in audio_files:
        if logger:
            logger.info(f"Processing {audio_file}...")
        txt_file, json_file = transcribe_file(
            model, audio_file, output_dir,
            beam_size=config.get("beam_size", 5),
            language=config.get("language", None),
            vad_filter=config.get("vad_filter", False),
            logger=logger,
            device=device
        )

        # Read recognition_speed from JSON
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data.get("recognition_speed") is not None:
                recognition_speeds.append(data["recognition_speed"])

    # Compute and print average recognition_speed
    if recognition_speeds:
        avg_speed = sum(recognition_speeds) / len(recognition_speeds)
        print(f"\nAverage recognition speed: {avg_speed:.2f}")
    else:
        print("\nNo recognition speed data available.")

if __name__ == "__main__":
    main()
