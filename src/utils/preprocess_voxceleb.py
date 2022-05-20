import argparse
import json
from itertools import chain
from pathlib import Path

import sox
from sklearn.model_selection import StratifiedShuffleSplit

from src.utils.create_musan_noise_manifest import _depth_two_audio_files

SEED = 42


def preprocess_voxceleb(
    voxceleb1_path: Path,
    voxceleb1_test_path: Path,
    voxceleb2_path: Path,
    voxceleb_manifest_path: Path,
    dev_p: float,
    id: int,
):
    voxceleb1_speakers = list((voxceleb1_path / "wav").iterdir())
    voxceleb2_speakers = list((voxceleb2_path / "dev" / "aac").iterdir())
    voxceleb1_test_speakers = list((voxceleb1_test_path / "wav").iterdir())

    voxceleb1_audio = list(chain.from_iterable([_depth_two_audio_files(spk) for spk in voxceleb1_speakers]))
    voxceleb2_audio = list(chain.from_iterable([_depth_two_audio_files(spk) for spk in voxceleb2_speakers]))
    voxceleb1_test_audio = list(chain.from_iterable([_depth_two_audio_files(spk) for spk in voxceleb1_test_speakers]))

    voxceleb_train_audio = sorted(voxceleb1_audio + voxceleb2_audio)
    train_labels = list(map(lambda x: x.parents[id].name, voxceleb_train_audio))
    test_labels = list(map(lambda x: x.parents[id].name, voxceleb1_test_audio))

    print(f"Unique train speakers: {len(set(train_labels))}")

    split = StratifiedShuffleSplit(n_splits=1, test_size=dev_p, random_state=SEED)
    for _train_idx, _dev_idx in split.split(train_labels, train_labels):
        with open(voxceleb_manifest_path / "train.json", "w") as train_file:
            for idx in _train_idx:
                audio_path = voxceleb_train_audio[idx]
                dumped_json = json.dumps(
                    {
                        "audio_filename": str(audio_path),
                        "duration": sox.file_info.duration(audio_path),
                        "offset": 0,
                        "label": train_labels[idx],
                    }
                )
                train_file.write(dumped_json)
                train_file.write("\n")

        with open(voxceleb_manifest_path / "dev.json", "w") as dev_file:
            for idx in _dev_idx:
                audio_path = voxceleb_train_audio[idx]
                dumped_json = json.dumps(
                    {
                        "audio_filename": str(audio_path),
                        "duration": sox.file_info.duration(audio_path),
                        "offset": 0,
                        "label": train_labels[idx],
                    }
                )
                dev_file.write(dumped_json)
                dev_file.write("\n")

        with open(voxceleb_manifest_path / "test.json", "w") as test_file:
            for idx in range(len(voxceleb1_test_audio)):
                audio_path = voxceleb1_test_audio[idx]
                dumped_json = json.dumps(
                    {
                        "audio_filename": str(audio_path),
                        "duration": sox.file_info.duration(audio_path),
                        "offset": 0,
                        "label": test_labels[idx],
                    }
                )
                test_file.write(dumped_json)
                test_file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--voxceleb1-path", type=Path, required=True)
    parser.add_argument("--voxceleb1-test-path", type=Path, required=True)
    parser.add_argument("--voxceleb2-path", type=Path, required=True)
    parser.add_argument("--voxceleb-manifest-path", type=Path, required=True)
    parser.add_argument("--dev-p", type=float, default=0.1)
    parser.add_argument("--id", type=int, default=1)
    args = parser.parse_args()

    preprocess_voxceleb(
        args.voxceleb1_path,
        args.voxceleb1_test_path,
        args.voxceleb2_path,
        args.voxceleb_manifest_path,
        args.dev_p,
        args.id,
    )
