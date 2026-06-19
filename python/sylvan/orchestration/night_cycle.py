"""Night cycle orchestration."""

from __future__ import annotations

from pathlib import Path

from ..config import SylvanConfig
from ..control.trainer import ControllerTrainer
from ..training.trainer import WorldModelTrainer


def run_night_training(config: SylvanConfig, run_dir: Path) -> dict[str, object]:
    trainer = WorldModelTrainer(config)
    return trainer.train(run_dir)


def run_controller_training(
    config: SylvanConfig,
    run_dir: Path,
    *,
    world_model_checkpoint: Path,
) -> dict[str, object]:
    trainer = ControllerTrainer(config)
    return trainer.train(run_dir, world_model_checkpoint=world_model_checkpoint)
