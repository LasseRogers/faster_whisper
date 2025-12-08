#!/usr/bin/env python3

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")

import os
import argparse
import time
import json
import multiprocessing as mp
from util import (
    load_config,
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
    parser.add_argument("-g", "--gpus", type=str, default=None, help="Comma-separated GPU IDs (e.g., '0,1,2,3')")
    return parser.parse_args()


def get_available_gpus(gpu_arg=None):
    """Get list of available GPU device IDs"""
    import torch

    if gpu_arg:
        # User specified GPUs
        return [int(g.strip()) for g in gpu_arg.split(',')]

    # Auto-detect all available GPUs
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))

    return [0]  # Fallback to single GPU


def transcribe_file(model, audio_file, output_dir, batch_size=16, language=None, vad_filter=False,
                    device=None, config=None, gpu_id=None):
    # Create a subfolder for this audio file
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    file_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(file_output_dir, exist_ok=True)

    # Get both TXT and JSON output paths
    txt_file = os.path.join(file_output_dir, f"{base_name}.txt")
    json_file = os.path.join(file_output_dir, f"{base_name}.json")

    # Store segment info for JSON output
    segments_data = []

    # Start timing transcription
    start_time = time.time()

    print(f"[GPU {gpu_id}] Processing {os.path.basename(audio_file)}")

    with open(txt_file, "w", encoding="utf-8") as f:
        # Transcribe audio using batched inference
        segments, info = model.transcribe(
            audio_file,
            batch_size=batch_size,
            language=language,
            vad_filter=vad_filter,
            word_timestamps=True
        )

        # Process each segment
        for segment in segments:
            # Format: [start_time -> end_time] transcribed_text
            line = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
            # Write segment to TXT file
            f.write(line + "\n")
            segments_data.append({"start": segment.start, "end": segment.end, "text": segment.text})

    # Calculate total transcription time
    transcription_time_sec = time.time() - start_time

    # Save metadata to JSON
    write_transcription_json(
        audio_file,
        segments_data,
        info,
        json_file=json_file,
        device=device,
        transcription_time_sec=transcription_time_sec
    )

    # Plot waveform with VAD segments if enabled
    if config.get("vad_plot_enable", False):
        try:
            plot_file = plot_waveform_with_vad(audio_file, segments_data, file_output_dir)
            print(f"[GPU {gpu_id}] VAD plot saved to {plot_file}")
        except Exception as e:
            print(f"[GPU {gpu_id}] Failed to produce VAD plot for {audio_file}: {e}")

    # Return paths to TXT and JSON files
    return txt_file, json_file


def gpu_worker(gpu_id, audio_files, output_dir, model_size, batch_size, language, vad_filter, config):
    """Worker process that handles a specific GPU"""
    # Set this process to only see one GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Import here to ensure CUDA initializes after CUDA_VISIBLE_DEVICES is set
    from faster_whisper import WhisperModel, BatchedInferencePipeline

    device = "cuda"

    print(f"[GPU {gpu_id}] Loading model (CUDA_VISIBLE_DEVICES={gpu_id})")

    # Load model
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    batched_model = BatchedInferencePipeline(model=model)

    results = []
    recognition_speeds = []

    for audio_file in audio_files:
        try:
            txt_file, json_file = transcribe_file(
                batched_model,
                audio_file,
                output_dir,
                batch_size,
                language,
                vad_filter,
                device,
                config,
                gpu_id
            )

            # Read recognition_speed from JSON
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("recognition_speed") is not None:
                    recognition_speeds.append(data["recognition_speed"])

            results.append({
                'audio_file': audio_file,
                'txt_file': txt_file,
                'json_file': json_file,
                'error': None
            })

            print(f"[GPU {gpu_id}] ✓ Completed {os.path.basename(audio_file)}")

        except Exception as e:
            print(f"[GPU {gpu_id}] ✗ Failed {os.path.basename(audio_file)}: {e}")
            results.append({
                'audio_file': audio_file,
                'txt_file': None,
                'json_file': None,
                'error': str(e)
            })

    print(f"[GPU {gpu_id}] Finished processing {len(audio_files)} files")

    return {
        'gpu_id': gpu_id,
        'results': results,
        'recognition_speeds': recognition_speeds
    }


def main():
    # Main program
    args = parse_args()
    config = load_config()
    output_dir = args.output or config.get("output_dir", "transcriptions")
    os.makedirs(output_dir, exist_ok=True)

    # Get available GPUs
    gpu_ids = get_available_gpus(args.gpus)

    print(f"Using GPUs: {gpu_ids}")

    # Load settings from config
    model_size = config.get("model_size", "large-v3")
    batch_size = config.get("batch_size", 16)
    language = config.get("language", None)
    vad_filter = config.get("vad_filter", False)

    # Collect audio files from input path
    audio_files = collect_audio_files(args.input_path, args.limit)

    print(f"Found {len(audio_files)} audio files to process")
    print(f"Distributing across {len(gpu_ids)} GPU(s)\n")

    # Distribute files across GPUs (round-robin)
    num_gpus = len(gpu_ids)
    files_per_gpu = [[] for _ in range(num_gpus)]

    for i, audio_file in enumerate(audio_files):
        gpu_idx = i % num_gpus
        files_per_gpu[gpu_idx].append(audio_file)

    # Show distribution
    for i, gpu_id in enumerate(gpu_ids):
        if files_per_gpu[i]:
            print(f"GPU {gpu_id}: {len(files_per_gpu[i])} files")

    print("\nStarting transcription...\n")

    # Store all recognition speeds
    all_recognition_speeds = []

    # Use multiprocessing to spawn one process per GPU
    with mp.Pool(processes=num_gpus) as pool:
        # Create tasks for each GPU
        tasks = []
        for i, gpu_id in enumerate(gpu_ids):
            if files_per_gpu[i]:  # Only if there are files to process
                task = pool.apply_async(
                    gpu_worker,
                    (gpu_id, files_per_gpu[i], output_dir, model_size, batch_size,
                     language, vad_filter, config)
                )
                tasks.append(task)

        # Wait for all tasks to complete and collect results
        for task in tasks:
            try:
                result = task.get()
                all_recognition_speeds.extend(result['recognition_speeds'])
            except Exception as e:
                print(f"Worker process failed: {e}")

    # Compute and print average recognition_speed
    print("\n" + "=" * 50)
    if all_recognition_speeds:
        avg_speed = sum(all_recognition_speeds) / len(all_recognition_speeds)
        print(f"Average recognition speed: {avg_speed:.2f}x realtime")
        print(f"Total files processed: {len(all_recognition_speeds)}")
    else:
        print("No recognition speed data available.")
    print("=" * 50)


if __name__ == "__main__":
    # Required for multiprocessing on Windows and for proper CUDA handling
    mp.set_start_method('spawn', force=True)
    main()