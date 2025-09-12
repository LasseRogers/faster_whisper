#! /usr/bin/env python

import subprocess
from pathlib import Path
import argparse
import json

def get_danish_audio_stream(file_path):
    """
    Returns the index of the Danish audio stream if it exists, else None
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index:stream_tags=language",
        "-of", "json",
        str(file_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error reading metadata for {file_path}: {result.stderr}")
        return None

    try:
        metadata = json.loads(result.stdout)
        streams = metadata.get("streams", [])
        for stream in streams:
            tags = stream.get("tags", {})
            if tags.get("language", "").lower() == "dan":
                return stream.get("index")
    except json.JSONDecodeError:
        print(f"Failed to parse metadata for {file_path}")
        return None

    return None

def down_sample(path_to_file, output_dir):
    """
    Downsample audio to 16K mono WAV
    """
    path_to_file = Path(path_to_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fileout = output_dir / f"{path_to_file.stem}.wav"

    stream_index = get_danish_audio_stream(path_to_file)
    if stream_index is not None:
        ffmpeg_cmd = [
            "ffmpeg", "-v", "quiet", "-y",
            "-i", str(path_to_file),
            "-map", f"0:{stream_index}",
            "-ac", "1",
            "-ar", "16k",
            str(fileout)
        ]
    else:
        ffmpeg_cmd = [
            "ffmpeg", "-v", "quiet", "-y",
            "-i", str(path_to_file),
            "-ac", "1",
            "-ar", "16k",
            str(fileout)
        ]

    print(f"Processing: {path_to_file.name}")
    ffmpeg_out = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

    if ffmpeg_out.returncode == 0:
        print(f"Success: {fileout}")
        return fileout
    else:
        print(f"Failed to convert: {path_to_file}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Downsample audio to 16K mono WAV.")
    parser.add_argument("input", help="Input file or folder")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process (folder only)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path.cwd() / "downsampled"

    if input_path.is_file():
        down_sample(input_path, output_dir)
    elif input_path.is_dir():
        files = list(input_path.glob("*.*"))
        if args.limit:
            files = files[:args.limit]

        for file in files:
            if file.is_file():
                down_sample(file, output_dir)
    else:
        print("Input path is invalid. Must be a file or folder.")

if __name__ == "__main__":
    main()
