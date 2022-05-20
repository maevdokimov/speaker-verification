## MUSAN dataset from https://openslr.org/17/

import argparse
import json
from pathlib import Path

import sox


def _depth_two_audio_files(base_path, allowed_suffixes=[".wav", ".m4a"]):
    files = []

    for subdir in base_path.iterdir():
        if subdir.is_file():
            continue

        for wav_file in subdir.iterdir():
            if wav_file.suffix not in allowed_suffixes:
                continue

            files.append(wav_file)

    return files


def create_noise_manifest(root_path, output_manifest_path):
    noise_path, music_path = root_path / "noise", root_path / "music"

    noise_files = _depth_two_audio_files(noise_path)
    music_files = _depth_two_audio_files(music_path)

    with open(output_manifest_path, "w") as file:
        for wav_file in noise_files + music_files:
            wav_duration = sox.file_info.duration(wav_file)
            dumped_json = json.dumps(
                {"audio_filename": str(wav_file), "duration": wav_duration, "offset": 0, "text": ""}
            )

            file.write(dumped_json)
            file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", type=Path, required=True)
    parser.add_argument("--output-manifest-path", type=Path, required=True)
    args = parser.parse_args()

    create_noise_manifest(args.root_path, args.output_manifest_path)
