from typing import Dict, Optional

import torch
from hydra.utils import instantiate
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.common.metrics import TopKClassificationAccuracy
from nemo.core.classes import ModelPT
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import *
from nemo.utils import logging
from omegaconf import DictConfig
from pytorch_lightning import Trainer


class SpeakerClassificationModel(ModelPT):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        self.preprocessor = instantiate(cfg.preprocessor)
        self.encoder = instantiate(cfg.encoder)
        self.decoder = instantiate(cfg.decoder)
        self.loss = instantiate(cfg.loss)

        self._accuracy = TopKClassificationAccuracy(top_k=[1])

    def __setup_dataloader_from_config(self, config: Optional[Dict]):
        if "augmentor" in config:
            augmentor = process_augmentations(config["augmentor"])
        else:
            augmentor = None

        featurizer = WaveformFeaturizer(
            sample_rate=config.sample_rate, int_values=config.get("int_values", False), augmentor=augmentor
        )

        dataset = instantiate(config.dataset, featurizer=featurizer)
        collate_fn = getattr(dataset, config.collate_fn)
        return torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, **config.dataloader_params)

    def setup_training_data(self, cfg):
        self._train_dl = self.__setup_dataloader_from_config(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self.__setup_dataloader_from_config(cfg)

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_signal": NeuralType(("B", "T"), AudioSignal()),
            "input_signal_length": NeuralType(tuple("B"), LengthsType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "logits": NeuralType(("B", "D"), LogitsType()),
            "embs": NeuralType(("B", "D"), AcousticEncodedRepresentation()),
        }

    @typecheck()
    def forward(self, input_signal, input_signal_length):
        processed_signal, processed_signal_len = self.preprocessor(
            input_signal=input_signal,
            length=input_signal_length,
        )

        encoded, length = self.encoder(audio_signal=processed_signal, length=processed_signal_len)
        logits, embs = self.decoder(encoder_outputs=encoded, encoder_lengths=length)
        return logits, embs

    def training_step(self, batch, batch_idx):
        audio_signal, audio_signal_len, labels, _ = batch
        logits, _ = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        loss = self.loss(logits=logits, labels=labels)

        self.log("loss", loss)

        self._accuracy(logits=logits, labels=labels)
        top_k = self._accuracy.compute()
        self._accuracy.reset()
        for i, top_i in enumerate(top_k):
            self.log(f"training_batch_accuracy_top@{i}", top_i)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        audio_signal, audio_signal_len, labels, _ = batch
        logits, _ = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        loss_value = self.loss(logits=logits, labels=labels)
        acc_top_k = self._accuracy(logits=logits, labels=labels)
        correct_counts, total_counts = self._accuracy.correct_counts_k, self._accuracy.total_counts_k

        return {
            "val_loss": loss_value,
            "val_correct_counts": correct_counts,
            "val_total_counts": total_counts,
            "val_acc_top_k": acc_top_k,
        }

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        correct_counts = torch.stack([x["val_correct_counts"] for x in outputs]).sum(axis=0)
        total_counts = torch.stack([x["val_total_counts"] for x in outputs]).sum(axis=0)

        self._accuracy.correct_counts_k = correct_counts
        self._accuracy.total_counts_k = total_counts
        topk_scores = self._accuracy.compute()
        self._accuracy.reset()

        logging.info("val_loss: {:.3f}".format(val_loss_mean))
        self.log("val_loss", val_loss_mean)
        for top_k, score in zip(self._accuracy.top_k, topk_scores):
            self.log("val_epoch_accuracy_top@{}".format(top_k), score)

        return {
            "val_loss": val_loss_mean,
            "val_acc_top_k": topk_scores,
        }

    @classmethod
    def list_available_models(cls):
        return []
