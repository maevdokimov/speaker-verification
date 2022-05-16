import pytorch_lightning as pl
from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager

from src.models.speaker_classification import SpeakerClassificationModel


@hydra_runner(config_path="conf", config_name="resnet34")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    model = SpeakerClassificationModel(cfg=cfg.model, trainer=trainer)

    lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    device_monitor = pl.callbacks.gpu_stats_monitor.GPUStatsMonitor()
    trainer.callbacks.extend([lr_logger, epoch_time_logger, device_monitor])

    trainer.fit(model)


if __name__ == "__main__":
    main()
