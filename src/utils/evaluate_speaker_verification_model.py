import argparse
import importlib
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.common.parts.preprocessing import collections
from omegaconf import OmegaConf, omegaconf
from tqdm import trange

from src.utils.embedding_storage import EmbeddingStorage

DEVICE = torch.device("cuda:0")
REMOVE_KEYS = ["train_ds", "optim", "sched"]
VALIDATION_KEYS = ["validation_ds"]


def parse_target(target: str):
    target_parts = target.split(".")

    return ".".join(target_parts[:-1]), target_parts[-1]


def load_nemo_checkpoint(ckpt_path: Path, config_override_path: Optional[Path], return_validation_params: bool):
    d = torch.load(ckpt_path, map_location=DEVICE)
    conf = d["hyper_parameters"]

    if config_override_path is not None:
        config_override = OmegaConf.load(config_override_path)
        conf = OmegaConf.merge(conf, config_override)

    validation_params = {}
    with omegaconf.open_dict(conf):
        for key in REMOVE_KEYS:
            if key in conf.keys():
                conf.pop(key)

        for key in VALIDATION_KEYS:
            if key in conf.keys():
                validation_params[key] = conf.pop(key)

    target_prefix, target_module = parse_target(conf.target)
    cls = getattr(importlib.import_module(target_prefix), target_module)
    model = cls(conf)
    model.load_state_dict(d["state_dict"])
    model.to(DEVICE)
    model.eval()

    if return_validation_params:
        return model, validation_params

    return model


def evaluate_manifest(model: pl.LightningModule, manifest_path: Path, validation_config: Dict):
    validation_data_config = validation_config["validation_ds"]
    validation_data_config.dataset.manifest_filepath = str(manifest_path)
    validation_data_config.dataset.min_duration = None
    validation_data_config.dataset.max_duration = None
    validation_data_config.dataloader_params.batch_size = 128
    validation_data_config.dataloader_params.num_workers = 8
    print(OmegaConf.to_yaml(validation_data_config))
    model.setup_validation_data(validation_data_config)

    trainer = pl.Trainer(gpus=1)
    trainer.validate(model)


@torch.no_grad()
def compute_embeddings(
    model: pl.LightningModule, manifest_path: Path, validation_config: Dict, output_embeddings: Path
):
    embedding_storage = EmbeddingStorage(device=DEVICE, normalized=True)

    validation_data_config = validation_config["validation_ds"]
    file_collection = collections.ASRSpeechLabel(manifests_files=str(manifest_path))
    featurizer = WaveformFeaturizer(
        sample_rate=validation_data_config.sample_rate, int_values=validation_data_config.get("int_values", False)
    )

    for i in trange(len(file_collection), desc="Computing embeddings"):
        sample = file_collection[i]
        input_signal = featurizer.process(sample.audio_file, offset=sample.offset, duration=sample.duration)
        input_signal = input_signal.unsqueeze(0).to(DEVICE)
        input_signal_length = torch.tensor([input_signal.shape[1]], dtype=torch.long, device=DEVICE)

        _, embedding = model(input_signal=input_signal, input_signal_length=input_signal_length)
        embedding_storage[sample.audio_file] = embedding

    embedding_storage.save_embeddings(output_embeddings)


def evaluate_eer(input_embeddings: Path, input_eer_file: Path):
    embedding_storage = EmbeddingStorage.load_embeddings(input_embeddings, DEVICE)

    scores, labels = [], []

    def _compute_score(_emb1, _emb2):
        return torch.dot(_emb1.squeeze(), _emb2.squeeze()).item()

    with open(input_eer_file, "r") as in_file:
        for line in in_file.readlines():
            line = line.strip()
            label, path1, path2 = line.split(" ")

            emb1 = embedding_storage[path1]
            emb2 = embedding_storage[path2]
            scores.append(_compute_score(emb1, emb2))
            labels.append(int(label))

    scores, labels = np.array(scores), np.array(labels)

    same_id_scores = scores[labels == 1]
    diff_id_scores = scores[labels == 0]
    thresh = np.linspace(np.min(diff_id_scores), np.max(same_id_scores), 1000)
    thresh = np.expand_dims(thresh, 1)
    fr_matrix = same_id_scores < thresh
    fa_matrix = diff_id_scores >= thresh
    fr_rate = np.mean(fr_matrix, 1)
    fa_rate = np.mean(fa_matrix, 1)

    thresh_idx = np.argmin(np.abs(fa_rate - fr_rate))
    eer = (fr_rate[thresh_idx] + fa_rate[thresh_idx]) / 2

    print(f"EER: {eer}")


def run_evaluation(args: argparse.Namespace):
    if args.evaluation_mode == "classification":
        print("Evaluating classification")

        model, validation_configs = load_nemo_checkpoint(args.model_ckpt_path, args.override_config_path, True)
        print("Successfully loaded speaker verification model")

        evaluate_manifest(model, args.input_manifest, validation_configs)
    elif args.evaluation_mode == "embeddings":
        print(f"Computing embeddings for {args.input_manifest}")

        model, validation_configs = load_nemo_checkpoint(args.model_ckpt_path, args.override_config_path, True)
        print("Successfully loaded speaker verification model")

        compute_embeddings(model, args.input_manifest, validation_configs, args.output_embeddings)
    elif args.evaluation_mode == "eer":
        print("Evaluating EER")

        evaluate_eer(args.input_embeddings, args.input_eer_file)
    else:
        print(f"Unknown option {args.evaluation_mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation-mode",
        choices=["embeddings", "classification", "eer"],
        required=True,
        help="Whether to compute EER from file with pairs or to classification" "metrics from manifest file",
    )
    parser.add_argument(
        "--model-ckpt-path", type=Path, help="Model checkpoint. Can be any checkpoint from model checkpoint"
    )
    parser.add_argument("--input-manifest", type=Path, help="Input manifest file")
    parser.add_argument("--input-embeddings", type=Path, help="Input file with EmbeddingStorage")
    parser.add_argument("--input-eer-file", type=Path, help="Input file with line structure: {0,1} path1 path2")
    parser.add_argument("--output-embeddings", type=Path, help="Where to save EmbeddingStorage")
    parser.add_argument("--override-config-path", type=Path, help="Config to override model params")

    args = parser.parse_args()

    run_evaluation(args)
