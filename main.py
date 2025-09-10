#!/usr/bin/env python3

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import argparse
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from util import (
    load_config,
    setup_logger,
    get_device,
    load_batched_model,
    get_output_files,
    write_transcription_json,
    collect_audio_files,
    plot_waveform_with_vad
)


def parse_args():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Transcribe audio files using Faster Whisper")
    parser.add_argument("input_path", help="Audio file or folder containing files")
    parser.add_argument("-n", "--limit", type=int, default=None, help="Limit number of files")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output directory")
    return parser.parse_args()


def transcribe_file(model, audio_file, output_dir, batch_size=16, language=None, vad_filter=False, logger=None,
                    device=None, config=None):
    # Get both TXT and JSON output paths
    files = get_output_files(audio_file, output_dir)
    txt_file = files["txt"]
    json_file = files["json"]

    # Store segment info for JSON output
    segments_data = []

    # Start timing transcription
    start_time = time.time()

    with open(txt_file, "w", encoding="utf-8") as f:
        # Transcribe audio using batched inference
        segments, info = model.transcribe(
            audio_file,
            batch_size=batch_size,
            language=language,
            vad_filter=vad_filter
        )

        # Log detected or specified language
        if logger:
            lang_msg = "Specified" if language else "Auto-detected"
            logger.info(f"{lang_msg} language '{info.language}' (probability {info.language_probability})")

        # Process each segment
        for segment in segments:
            # Format: [start_time -> end_time] transcribed_text
            line = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
            # Write segment to TXT file
            f.write(line + "\n")
            segments_data.append({"start": segment.start, "end": segment.end, "text": segment.text})

    # Calculate total transcription time
    transcription_time_sec = time.time() - start_time

    if logger:
        logger.info(f"Text transcription saved to {txt_file}")

    # Save metadata to JSON
    write_transcription_json(
        audio_file,
        segments_data,
        info,
        json_file=json_file,
        device=device,
        transcription_time_sec=transcription_time_sec
    )

    if logger:
        logger.info(f"Transcription and metadata saved to {json_file}")

    # Plot waveform with VAD segments if enabled
    if config.get("waveform_plot_enable", False):
        try:
            plot_file = plot_waveform_with_vad(audio_file, segments_data, output_dir)
            if logger:
                logger.info(f"Waveform plot saved to {plot_file}")
        except Exception as e:
            if logger:
                logger.error(f"Failed to plot waveform for {audio_file}: {e}")

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

    # Load Whisper model and settings from config
    model = load_batched_model(config.get("model_size", "large-v3"), device)
    batch_size = config.get("batch_size", 16)
    language = config.get("language", None)
    vad_filter = config.get("vad_filter", False)
    workers = config.get("workers", None)  # Number of parallel workers from config

    # Collect audio files from input path
    audio_files = collect_audio_files(args.input_path, args.limit)

    # Store recognition_speed for each file
    recognition_speeds = []

    # Use ThreadPoolExecutor to process multiple files in parallel
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                transcribe_file,
                model,
                audio_file,
                output_dir,
                batch_size,
                language,
                vad_filter,
                logger,
                device,
                config
            ): audio_file
            for audio_file in audio_files
        }

        for future in as_completed(futures):
            audio_file = futures[future]
            try:
                txt_file, json_file = future.result()
                # Read recognition_speed from JSON
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("recognition_speed") is not None:
                        recognition_speeds.append(data["recognition_speed"])
            except Exception as e:
                if logger:
                    logger.error(f"Failed to process {audio_file}: {e}")

    # Compute and print average recognition_speed
    if recognition_speeds:
        avg_speed = sum(recognition_speeds) / len(recognition_speeds)
        print(f"\nAverage recognition speed: {avg_speed:.2f}")
    else:
        print("\nNo recognition speed data available.")


if __name__ == "__main__":
    main()