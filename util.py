import os
import yaml
import json
import time
import shutil
from typing import List

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # For headless backend
import librosa
import librosa.display
import numpy as np

# Supported audio file extensions
SUPPORTED_EXTENSIONS = ('.mp3', '.mp4', '.wav', '.flac', '.ts', '.mpeg', '.mpeg2')


def load_config(path="config.yaml") -> dict:
    # Load YAML configuration
    with open(path, "r") as f:
        return yaml.safe_load(f)


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
                    device=None, gpu_id=None, run_settings=None):
    run_settings = run_settings or {}
    beam_size = run_settings.get("beam_size", 5)

    # Create a subfolder for this audio file
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    file_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(file_output_dir, exist_ok=True)

    try:
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

        # Save metadata to JSON - recognition_speed comes back directly from
        # this call, no need to reopen the file we just wrote.
        json_file, recognition_speed = write_transcription_json(
            audio_file,
            segments_data,
            info,
            json_file=json_file,
            device=device,
            transcription_time_sec=transcription_time_sec,
            run_settings=run_settings
        )

        # Plot waveform with VAD segments if enabled
        if run_settings.get("vad_plot_enable", False):
            try:
                plot_file = plot_waveform_with_vad(audio_file, segments_data, file_output_dir)
                print(f"[GPU {gpu_id}] VAD plot saved to {plot_file}")
            except Exception as e:
                print(f"[GPU {gpu_id}] Failed to produce VAD plot for {audio_file}: {e}")

        # Return paths to TXT and JSON files, plus recognition_speed
        return txt_file, json_file, recognition_speed

    except Exception:
        # Transcription failed partway through - remove the subfolder we
        # created so failed files don't leave behind an empty (or partial)
        # directory. The error itself is re-raised so _process_one_file's
        # error handling still records it normally.
        shutil.rmtree(file_output_dir, ignore_errors=True)
        raise


def write_transcription_json(
    # Write transcription segments and metadata to JSON
    audio_file: str,
    segments: list,
    info,
    json_file: str,
    device: str = None,
    transcription_time_sec: float = None,
    run_settings: dict = None
) -> tuple:

    duration_sec = getattr(info, "duration", 0) if getattr(info, "duration", None) else 0
    duration_after_vad_sec = getattr(info, "duration_after_vad", None)

    # Duration of audio filtered out by VAD - based on faster-whisper's own
    # duration_after_vad value
    non_speech_duration_sec = max(duration_sec - duration_after_vad_sec, 0) \
        if duration_after_vad_sec is not None else None

    # Convert to minutes
    duration_min = duration_sec / 60
    duration_after_vad_min = duration_after_vad_sec / 60 if duration_after_vad_sec is not None else None
    non_speech_duration_min = non_speech_duration_sec / 60 if non_speech_duration_sec is not None else None

    # Compute recognition speed, based on faster-whisper's own duration_after_vad
    recognition_speed = None
    if transcription_time_sec and transcription_time_sec > 0 and duration_after_vad_sec is not None:
        recognition_speed = duration_after_vad_sec / transcription_time_sec

    # Compute an overall average log-probability across all segments,
    # weighted by segment duration (longer segments count more toward the average)
    avg_logprob_overall = None
    weighted_segments = [
        (seg["end"] - seg["start"], seg["avg_logprob"])
        for seg in segments
        if seg.get("avg_logprob") is not None and (seg["end"] - seg["start"]) > 0
    ]
    if weighted_segments:
        total_weight = sum(w for w, _ in weighted_segments)
        avg_logprob_overall = sum(w * lp for w, lp in weighted_segments) / total_weight

    # Build JSON schema
    data = {
        "audio_file": audio_file,
        "device": device,
        "audio_duration_min": duration_min,
        "non_speech_duration_min": non_speech_duration_min,
        # duration_after_vad_min: reported directly by faster-whisper's VAD step
        # (converted to minutes to match the other duration fields here)
        "duration_after_vad_min": duration_after_vad_min,
        # recognition_speed for this individual file is not stored at the top
        # level - instead it gets folded into the "gpu" and "overall"
        # aggregate stats added later, via add_speed_stats_to_json
        # run_settings: full snapshot of every effective config value used for
        # this run (model_size, batch_size, language, vad_filter, beam_size,
        # workers_per_gpu, output_dir, gpu_ids), accounting for any
        # CLI overrides - not just the raw config.yaml. Makes each JSON
        # self-documenting about exactly what settings produced it.
        "run_settings": run_settings,
        "language": info.language,
        "language_probability": info.language_probability,
        # duration-weighted average log-probability across all segments in this file
        "avg_logprob_overall": avg_logprob_overall,
        "segments": segments
    }

    # Save JSON
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    # Return recognition_speed alongside the file path so callers don't need
    # to reopen and re-parse the JSON we just wrote to get this value back -
    # used later to build the aggregate "gpu"/"overall" speed_stats.
    return json_file, recognition_speed


def write_failed_files_json(failed_results: list, output_dir: str) -> str:
    """Write a persistent record of any files that failed to transcribe,
    so the failures survive even if the console output is lost (closed
    terminal, redirected background job, etc.)."""
    failed_file = os.path.join(output_dir, "failed_files.json")

    data = [
        {
            "audio_file": r["audio_file"],
            "error": r["error"]
        }
        for r in failed_results
    ]

    with open(failed_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return failed_file


def add_speed_stats_to_json(json_file: str, gpu_speed_stats: dict, overall_speed_stats: dict) -> None:
    """Patch an already-written transcription JSON with the aggregate speed
    stats for the run (per-GPU and overall min/max/avg), matching what's
    printed to the console. These aren't known until every file is done, so
    this necessarily runs as a second pass after the main transcription loop
    - unlike the write-then-reread we removed earlier, this is a deliberate
    update rather than reading back a value we already had in memory."""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["speed_stats"] = {
        "gpu": gpu_speed_stats,
        "overall": overall_speed_stats
    }

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def collect_audio_files(input_path: str, limit: int = None) -> List[str]:
    # Get all audio files from folder or single file
    if os.path.isdir(input_path):
        files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                 if f.lower().endswith(SUPPORTED_EXTENSIONS)]
    elif os.path.isfile(input_path) and input_path.lower().endswith(SUPPORTED_EXTENSIONS):
        files = [input_path]
    else:
        raise FileNotFoundError("No valid audio files found.")
    return files[:limit] if limit else files


def plot_waveform_with_vad(audio_file: str, segments: list, output_dir: str):
    # Load audio
    y, sr = librosa.load(audio_file, sr=None, mono=True)

    if len(y) == 0:
        raise ValueError(f"No audio data found in {audio_file}")

    # Downsample for plotting speed (max 100k points)
    max_points = 100_000
    if len(y) > max_points:
        hop = len(y) // max_points
        y = y[::hop]
        times = (np.arange(len(y)) * hop / sr) / 60.0  # convert to minutes
    else:
        times = (np.arange(len(y)) / sr) / 60.0  # convert to minutes

    # Create plot
    plt.figure(figsize=(12, 4))
    plt.plot(times, y, alpha=0.7, label="Waveform")

    # Sort segments
    segments = sorted(segments, key=lambda s: s["start"])

    # Mark speech (green) and non-speech (red) in minutes
    last_end = 0.0
    for seg in segments:
        if seg["start"] > last_end:
            plt.axvspan(last_end / 60.0, seg["start"] / 60.0, color="red", alpha=0.2)
        plt.axvspan(seg["start"] / 60.0, seg["end"] / 60.0, color="green", alpha=0.3)
        last_end = seg["end"]

    duration_min = times[-1]
    if last_end / 60.0 < duration_min:
        plt.axvspan(last_end / 60.0, duration_min, color="red", alpha=0.2)

    plt.title(f"Waveform with VAD Segments: {os.path.basename(audio_file)}")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Amplitude")

    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    plot_file = os.path.join(output_dir, f"{base_name}_waveform.png")
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()

    return plot_file