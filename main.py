#!/usr/bin/env python3

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")

import os
import argparse
import time
import json
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    parser.add_argument("-w", "--workers", type=int, default=None,
                         help="Number of workers per GPU, i.e. files processed concurrently on one GPU (overrides config)")
    parser.add_argument("-b", "--beam-size", type=int, default=None,
                         help="Beam size for decoding (overrides config)")
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
                    device=None, config=None, gpu_id=None, beam_size=5, run_settings=None):
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
        # NOTE: faster-whisper / CTranslate2 models are safe to call concurrently
        # from multiple threads on the same model instance - the actual compute
        # happens in C++ and releases the GIL during inference.
        segments, info = model.transcribe(
            audio_file,
            batch_size=batch_size,
            language=language,
            vad_filter=vad_filter,
            word_timestamps=True,
            beam_size=beam_size
        )

        # Process each segment
        for segment in segments:
            # Format: [start_time -> end_time] transcribed_text
            line = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
            # Write segment to TXT file
            f.write(line + "\n")

            # Per-word log-probability info (available since word_timestamps=True)
            words_data = None
            if segment.words:
                words_data = [
                    {
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability
                    }
                    for w in segment.words
                ]

            segments_data.append({
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                # avg_logprob: average log-probability of tokens in this segment
                # (closer to 0 = more confident, more negative = less confident)
                "avg_logprob": segment.avg_logprob,
                # no_speech_prob: model's estimated probability this segment is silence/non-speech
                "no_speech_prob": segment.no_speech_prob,
                # compression_ratio: text compression ratio, high values can indicate repetition/looping
                "compression_ratio": segment.compression_ratio,
                # temperature: sampling temperature actually used for this segment - Whisper
                # retries at a higher temperature when a greedy decode fails a quality check,
                # so a non-zero value here is itself a signal this segment was harder to decode
                "temperature": segment.temperature,
                # seek: internal audio-offset marker used by faster-whisper's decoding loop
                "seek": segment.seek,
                # tokens: raw token IDs before decoding to text - mainly useful for
                # custom re-scoring or low-level debugging
                "tokens": segment.tokens,
                "words": words_data
            })

    # Calculate total transcription time
    transcription_time_sec = time.time() - start_time

    # Save metadata to JSON
    write_transcription_json(
        audio_file,
        segments_data,
        info,
        json_file=json_file,
        device=device,
        transcription_time_sec=transcription_time_sec,
        beam_size=beam_size,
        run_settings=run_settings
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


def _process_one_file(model, audio_file, output_dir, batch_size, language, vad_filter,
                       device, config, gpu_id, beam_size=5, run_settings=None):
    """Wrapper used by the thread pool - handles errors per-file so one
    failure doesn't kill the whole batch, and returns a uniform result dict."""
    try:
        txt_file, json_file = transcribe_file(
            model, audio_file, output_dir, batch_size, language, vad_filter,
            device, config, gpu_id, beam_size, run_settings
        )

        recognition_speed = None
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            recognition_speed = data.get("recognition_speed")

        print(f"[GPU {gpu_id}] ✓ Completed {os.path.basename(audio_file)}")

        return {
            'audio_file': audio_file,
            'txt_file': txt_file,
            'json_file': json_file,
            'error': None,
            'recognition_speed': recognition_speed
        }

    except Exception as e:
        print(f"[GPU {gpu_id}] ✗ Failed {os.path.basename(audio_file)}: {e}")
        return {
            'audio_file': audio_file,
            'txt_file': None,
            'json_file': None,
            'error': str(e),
            'recognition_speed': None
        }


def gpu_worker(gpu_id, audio_files, output_dir, model_size, batch_size, language, vad_filter,
               config, workers_per_gpu=1, beam_size=5, run_settings=None):
    """Worker process that handles a specific GPU.

    Files assigned to this GPU are processed using a thread pool so that
    up to `workers_per_gpu` transcriptions can be in-flight on the GPU at
    once, instead of strictly one-at-a-time.
    """
    # Set this process to only see one GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Import here to ensure CUDA initializes after CUDA_VISIBLE_DEVICES is set
    from faster_whisper import WhisperModel, BatchedInferencePipeline

    device = "cuda"

    print(f"[GPU {gpu_id}] Loading model (CUDA_VISIBLE_DEVICES={gpu_id}, "
          f"workers_per_gpu={workers_per_gpu})")

    # Load model - a single instance shared across threads
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    batched_model = BatchedInferencePipeline(model=model)

    results = []
    recognition_speeds = []

    if workers_per_gpu <= 1:
        # Original sequential behavior
        for audio_file in audio_files:
            result = _process_one_file(
                batched_model, audio_file, output_dir, batch_size, language,
                vad_filter, device, config, gpu_id, beam_size, run_settings
            )
            results.append(result)
            if result['recognition_speed'] is not None:
                recognition_speeds.append(result['recognition_speed'])
    else:
        # Concurrent processing: up to `workers_per_gpu` transcriptions
        # in flight on this GPU at once via threads sharing one model.
        with ThreadPoolExecutor(max_workers=workers_per_gpu) as executor:
            futures = {
                executor.submit(
                    _process_one_file, batched_model, audio_file, output_dir,
                    batch_size, language, vad_filter, device, config, gpu_id,
                    beam_size, run_settings
                ): audio_file
                for audio_file in audio_files
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result['recognition_speed'] is not None:
                    recognition_speeds.append(result['recognition_speed'])

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

    # Number of workers per GPU, i.e. files processed concurrently on one GPU
    workers_per_gpu = args.workers if args.workers is not None \
        else config.get("workers_per_gpu", 1)

    # Beam size for decoding (faster-whisper defaults to 5 if not passed at all;
    # we default to 5 here too so behavior matches faster-whisper's own default
    # unless explicitly overridden in config.yaml or via -b)
    beam_size = args.beam_size if args.beam_size is not None \
        else config.get("beam_size", 5)

    # Snapshot of every effective setting used for this run (accounting for any
    # CLI overrides, not just the raw config.yaml) - stored in each output JSON
    # so every transcript is self-documenting about what produced it.
    run_settings = {
        "model_size": model_size,
        "batch_size": batch_size,
        "language": language,
        "vad_filter": vad_filter,
        "vad_plot_enable": config.get("vad_plot_enable", False),
        "workers_per_gpu": workers_per_gpu,
        "beam_size": beam_size,
        "output_dir": output_dir,
        "gpu_ids": gpu_ids,
    }

    # Collect audio files from input path
    audio_files = collect_audio_files(args.input_path, args.limit)

    print(f"Found {len(audio_files)} audio files to process")
    print(f"Distributing across {len(gpu_ids)} GPU(s), "
          f"{workers_per_gpu} worker(s) per GPU\n")

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

    # Start timing total processing
    total_start_time = time.time()

    # Store all recognition speeds and per-GPU stats
    all_recognition_speeds = []
    gpu_stats = {}

    # Use multiprocessing to spawn one process per GPU
    with mp.Pool(processes=num_gpus) as pool:
        # Create tasks for each GPU
        tasks = []
        for i, gpu_id in enumerate(gpu_ids):
            if files_per_gpu[i]:  # Only if there are files to process
                task = pool.apply_async(
                    gpu_worker,
                    (gpu_id, files_per_gpu[i], output_dir, model_size, batch_size,
                     language, vad_filter, config, workers_per_gpu, beam_size, run_settings)
                )
                tasks.append(task)

        # Wait for all tasks to complete and collect results
        for task in tasks:
            try:
                result = task.get()
                gpu_id = result['gpu_id']
                speeds = result['recognition_speeds']

                # Store per-GPU statistics
                if speeds:
                    avg_gpu_speed = sum(speeds) / len(speeds)
                    min_gpu_speed = min(speeds)
                    max_gpu_speed = max(speeds)

                    gpu_stats[gpu_id] = {
                        'avg_speed': avg_gpu_speed,
                        'min_speed': min_gpu_speed,
                        'max_speed': max_gpu_speed,
                        'num_files': len(speeds)
                    }

                all_recognition_speeds.extend(speeds)
            except Exception as e:
                print(f"Worker process failed: {e}")

    # Get total time
    total_elapsed_time = time.time() - total_start_time
    total_minutes = total_elapsed_time / 60

    # Print per-GPU statistics
    print("\n" + "=" * 70)
    print("PER-GPU PERFORMANCE STATISTICS")
    print("=" * 70)

    if gpu_stats:
        for gpu_id in sorted(gpu_stats.keys()):
            stats = gpu_stats[gpu_id]
            print(f"GPU {gpu_id}:")
            print(f"  Files processed: {stats['num_files']}")
            print(f"  Average speed:   {stats['avg_speed']:.2f}x realtime")
            print(f"  Min speed:       {stats['min_speed']:.2f}x realtime")
            print(f"  Max speed:       {stats['max_speed']:.2f}x realtime")
            print()

    # Compute and print overall statistics
    print("=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)
    if all_recognition_speeds:
        avg_speed = sum(all_recognition_speeds) / len(all_recognition_speeds)
        min_speed = min(all_recognition_speeds)
        max_speed = max(all_recognition_speeds)

        print(f"Average recognition speed: {avg_speed:.2f}x realtime")
        print(f"Min recognition speed:     {min_speed:.2f}x realtime")
        print(f"Max recognition speed:     {max_speed:.2f}x realtime")
        print(f"Total files processed:     {len(all_recognition_speeds)}")

        # Calculate effective parallel throughput
        if len(gpu_stats) > 1:
            effective_throughput = avg_speed * len(gpu_stats)
            print(f"Effective parallel throughput: {effective_throughput:.2f}x realtime ({len(gpu_stats)} GPUs)")
    else:
        print("No recognition speed data available.")

    print("=" * 70)
    print(f"Total processing time: {total_minutes:.2f} minutes ({total_elapsed_time:.1f} seconds)")
    print("=" * 70)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()