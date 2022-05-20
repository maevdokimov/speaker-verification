import argparse
import multiprocessing as mp
import warnings
from itertools import chain
from pathlib import Path

import librosa
import soundfile as sf
from tqdm import trange

from src.utils.create_musan_noise_manifest import _depth_two_audio_files

warnings.filterwarnings("ignore")


def convert_audiofile(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    file_path.unlink()
    sf.write(str(file_path.with_suffix(".wav")), audio, sample_rate)


def convert_voxceleb2(voxceleb2_path: Path, num_workers: int):
    voxceleb2_speakers = list((voxceleb2_path / "dev" / "aac").iterdir())
    voxceleb2_audio = list(chain.from_iterable([_depth_two_audio_files(spk) for spk in voxceleb2_speakers]))
    voxceleb2_target_audio = [p for p in voxceleb2_audio if p.suffix == ".m4a"]

    num_workers = min(num_workers, mp.cpu_count())
    with mp.Pool(processes=num_workers) as pool:
        for i in trange(0, len(voxceleb2_target_audio), num_workers):
            pool.map(convert_audiofile, voxceleb2_target_audio[i : i + num_workers])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--voxceleb2-path", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=32)
    args = parser.parse_args()

    convert_voxceleb2(args.voxceleb2_path, args.num_workers)
