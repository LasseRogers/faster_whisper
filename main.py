#!/usr/bin/env python3

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")

import os
import argparse
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from util import (
    load_config,
    collect_audio_files,
    get_available_gpus,
    transcribe_file,
    write_failed_files_json,
    add_speed_stats_to_json
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
    parser.add_argument("-m", "--model-size", type=str, default=None,
                         help="Whisper model size, e.g. 'large-v3', 'medium', 'small' (overrides config)")
    parser.add_argument("-bs", "--batch-size", type=int, default=None,
                         help="Batch size for batched inference (overrides config)")
    parser.add_argument("-l", "--language", type=str, default=None,
                         help="Language code, e.g. 'da', 'en' (overrides config)")

    # Booleans: use a mutually exclusive on/off pair per flag so the CLI can
    # both enable and disable something the config turned on, with "not passed
    # at all" as a distinct third state that falls back to config.yaml
    vad_group = parser.add_mutually_exclusive_group()
    vad_group.add_argument("--vad-filter", dest="vad_filter", action="store_true", default=None,
                            help="Enable VAD filtering (overrides config)")
    vad_group.add_argument("--no-vad-filter", dest="vad_filter", action="store_false",
                            help="Disable VAD filtering (overrides config)")

    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument("--vad-plot", dest="vad_plot_enable", action="store_true", default=None,
                             help="Enable VAD waveform plots (overrides config)")
    plot_group.add_argument("--no-vad-plot", dest="vad_plot_enable", action="store_false",
                             help="Disable VAD waveform plots (overrides config)")

    return parser.parse_args()


def _process_one_file(model, audio_file, output_dir, batch_size, language, vad_filter,
                       device, gpu_id, run_settings=None):
    """Wrapper used by the thread pool - handles errors per-file so one
    failure doesn't kill the whole batch, and returns a uniform result dict."""
    try:
        txt_file, json_file, recognition_speed, run_time_min = transcribe_file(
            model, audio_file, output_dir, batch_size, language, vad_filter,
            device, gpu_id, run_settings
        )

        print(f"[GPU {gpu_id}] ✓ Completed {os.path.basename(audio_file)}")

        return {
            'audio_file': audio_file,
            'txt_file': txt_file,
            'json_file': json_file,
            'error': None,
            'recognition_speed': recognition_speed,
            'run_time_min': run_time_min
        }

    except Exception as e:
        print(f"[GPU {gpu_id}] ✗ Failed {os.path.basename(audio_file)}: {e}")
        return {
            'audio_file': audio_file,
            'txt_file': None,
            'json_file': None,
            'error': str(e),
            'recognition_speed': None,
            'run_time_min': None
        }


def gpu_worker(gpu_id, audio_files, output_dir, model_size, batch_size, language, vad_filter,
               workers_per_gpu=1, run_settings=None):
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
    run_times = []

    if workers_per_gpu <= 1:
        # Original sequential behavior
        for audio_file in audio_files:
            result = _process_one_file(
                batched_model, audio_file, output_dir, batch_size, language,
                vad_filter, device, gpu_id, run_settings
            )
            results.append(result)
            if result['recognition_speed'] is not None:
                recognition_speeds.append(result['recognition_speed'])
            if result['run_time_min'] is not None:
                run_times.append(result['run_time_min'])
    else:
        # Concurrent processing: up to `workers_per_gpu` transcriptions
        # in flight on this GPU at once via threads sharing one model.
        with ThreadPoolExecutor(max_workers=workers_per_gpu) as executor:
            futures = {
                executor.submit(
                    _process_one_file, batched_model, audio_file, output_dir,
                    batch_size, language, vad_filter, device, gpu_id, run_settings
                ): audio_file
                for audio_file in audio_files
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result['recognition_speed'] is not None:
                    recognition_speeds.append(result['recognition_speed'])
                if result['run_time_min'] is not None:
                    run_times.append(result['run_time_min'])

    print(f"[GPU {gpu_id}] Finished processing {len(audio_files)} files")

    return {
        'gpu_id': gpu_id,
        'results': results,
        'recognition_speeds': recognition_speeds,
        'run_times': run_times
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

    # Load settings from config, with CLI flags taking precedence when provided
    model_size = args.model_size if args.model_size is not None \
        else config.get("model_size", "large-v3")
    batch_size = args.batch_size if args.batch_size is not None \
        else config.get("batch_size", 16)
    language = args.language if args.language is not None \
        else config.get("language", None)
    vad_filter = args.vad_filter if args.vad_filter is not None \
        else config.get("vad_filter", False)

    # Number of workers per GPU, i.e. files processed concurrently on one GPU
    workers_per_gpu = args.workers if args.workers is not None \
        else config.get("workers_per_gpu", 1)

    # Beam size for decoding (faster-whisper defaults to 5 if not passed at all;
    # we default to 5 here too so behavior matches faster-whisper's own default
    # unless explicitly overridden in config.yaml or via -b)
    beam_size = args.beam_size if args.beam_size is not None \
        else config.get("beam_size", 5)

    vad_plot_enable = args.vad_plot_enable if args.vad_plot_enable is not None \
        else config.get("vad_plot_enable", False)

    # Snapshot of every effective setting used for this run (accounting for any
    # CLI overrides, not just the raw config.yaml) - stored in each output JSON
    # so every transcript is self-documenting about what produced it.
    run_settings = {
        "model_size": model_size,
        "batch_size": batch_size,
        "language": language,
        "vad_filter": vad_filter,
        "vad_plot_enable": vad_plot_enable,
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

    # Store all recognition speeds, run times, per-file results, and per-GPU stats
    all_recognition_speeds = []
    all_run_times = []
    all_results = []
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
                     language, vad_filter, workers_per_gpu, run_settings)
                )
                tasks.append(task)

        # Wait for all tasks to complete and collect results
        for task in tasks:
            try:
                result = task.get()
                gpu_id = result['gpu_id']
                speeds = result['recognition_speeds']
                run_times = result['run_times']

                # Store per-GPU statistics
                if speeds:
                    gpu_stats[gpu_id] = {
                        'avg_speed': sum(speeds) / len(speeds),
                        'min_speed': min(speeds),
                        'max_speed': max(speeds),
                        'num_files': len(speeds)
                    }

                all_recognition_speeds.extend(speeds)
                all_run_times.extend(run_times)
                for r in result['results']:
                    r['gpu_id'] = gpu_id
                all_results.extend(result['results'])
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
    overall_speed_stats = None
    if all_recognition_speeds:
        overall_speed_stats = {
            "avg_speed": sum(all_recognition_speeds) / len(all_recognition_speeds),
            "min_speed": min(all_recognition_speeds),
            "max_speed": max(all_recognition_speeds),
            "num_files": len(all_recognition_speeds)
        }
        if all_run_times:
            overall_speed_stats["run_time_min"] = total_minutes

        print(f"Average recognition speed: {overall_speed_stats['avg_speed']:.2f}x realtime")
        print(f"Min recognition speed:     {overall_speed_stats['min_speed']:.2f}x realtime")
        print(f"Max recognition speed:     {overall_speed_stats['max_speed']:.2f}x realtime")
        print(f"Total files processed:     {overall_speed_stats['num_files']}")

        # Calculate effective parallel throughput
        if len(gpu_stats) > 1:
            effective_throughput = overall_speed_stats['avg_speed'] * len(gpu_stats)
            overall_speed_stats['effective_parallel_throughput'] = effective_throughput
            print(f"Effective parallel throughput: {effective_throughput:.2f}x realtime ({len(gpu_stats)} GPUs)")
    else:
        print("No recognition speed data available.")

    # Report any files that failed to transcribe
    failed_results = [r for r in all_results if r['error']]
    if failed_results:
        print("=" * 70)
        print(f"FAILED FILES ({len(failed_results)})")
        print("=" * 70)
        for r in failed_results:
            print(f"  {os.path.basename(r['audio_file'])}: {r['error']}")

        # Persist to disk too, so this list survives beyond the console output
        failed_file = write_failed_files_json(failed_results, output_dir)
        print(f"\nFailed files list saved to {failed_file}")

    print("=" * 70)
    print(f"Total processing time: {total_minutes:.2f} minutes ({total_elapsed_time:.1f} seconds)")
    print("=" * 70)

    # Patch the same per-GPU and overall speed stats shown above directly into
    # each successful file's own transcription JSON, rather than a separate
    # summary file - these stats aren't known until every file is done, so
    # this necessarily runs as a pass after the main transcription loop.
    successful_results = [r for r in all_results if r['json_file']]
    for r in successful_results:
        add_speed_stats_to_json(
            r['json_file'],
            gpu_speed_stats=gpu_stats.get(r['gpu_id']),
            overall_speed_stats=overall_speed_stats
        )
    if successful_results:
        print(f"Speed stats added to {len(successful_results)} transcription JSON file(s)")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()