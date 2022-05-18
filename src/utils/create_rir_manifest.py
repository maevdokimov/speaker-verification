## RIR dataset from https://openslr.org/28/

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import sox


def compute_uniform_hash(str_name):
    encoded_str = str_name.encode()
    hash_digest = hashlib.sha512(encoded_str).digest()

    return int.from_bytes(hash_digest, "big")


def create_rir_manifest(root_path, output_manifest_path, p):
    if p <= 0.0 or p > 1.0:
        raise ValueError("Parameter p should be 0 < p <= 1")
    max_hash_size, total_hash_size = Decimal(str(p)).as_integer_ratio()

    base_path = root_path / "simulated_rirs"

    with open(output_manifest_path, "w") as file:
        for room_type in base_path.iterdir():
            if not room_type.is_dir():
                continue

            for room in room_type.iterdir():
                if not room.name.startswith("Room"):
                    continue

                for room_wav in room.iterdir():
                    relative_stable_name = str(room_wav.relative_to(room_wav.parents[2]))
                    hash_name = compute_uniform_hash(relative_stable_name)

                    if hash_name % total_hash_size < max_hash_size:
                        wav_duration = sox.file_info.duration(room_wav)
                        dumped_json = json.dumps(
                            {"audio_filename": str(room_wav), "duration": wav_duration, "offset": 0, "text": ""}
                        )

                        file.write(dumped_json)
                        file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", type=Path, required=True)
    parser.add_argument("--output-manifest-path", type=Path, required=True)
    parser.add_argument("--p", default=0.1)
    args = parser.parse_args()

    create_rir_manifest(args.root_path, args.output_manifest_path, args.p)
