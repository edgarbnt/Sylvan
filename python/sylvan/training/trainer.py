"""Training entrypoint for the world model."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from ..buffer.dataset import ReplaySequenceDataset, collate_sequence_samples
from ..buffer.replay_window import select_replay_window
from ..config import SylvanConfig
from ..device import resolve_torch_device
from ..models.world_model import WorldModelV0
from .checkpointing import load_checkpoint, save_checkpoint
from .loops import train_epoch, validate_epoch


class WorldModelTrainer:
    def __init__(self, config: SylvanConfig) -> None:
        self.config = config
        self.device, self.device_reason = resolve_torch_device()
        print(f"[Python] WorldModelTrainer device: {self.device} | {self.device_reason}")

    def build_model(self) -> WorldModelV0:
        env = self.config.env
        train = self.config.train
        return WorldModelV0(
            proprio_dim=env.proprio_dim,
            action_dim=env.action_dim,
            metrics_dim=env.metrics_dim,
            hidden_dim=train.hidden_dim,
            latent_dim=train.latent_dim,
        ).to(self.device)

    def train(self, run_dir: Path, checkpoint_name: str = "world_model_v0.pt") -> dict[str, object]:
        replay_window = select_replay_window(
            self.config.paths.replay_buffer_dir,
            current_run_dir=run_dir,
            window_size=self.config.train.replay_window_size,
        )
        dataset = ReplaySequenceDataset(replay_window, self.config.train.sequence_length)
        if len(dataset) == 0:
            raise ValueError(f"No sequences available in replay buffer run {run_dir}")

        val_size = len(dataset) // 5
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            collate_fn=collate_sequence_samples,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            collate_fn=collate_sequence_samples,
        )

        model = self.build_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.train.learning_rate)

        history: list[dict[str, object]] = []
        checkpoint_path = self.config.paths.checkpoints_dir / checkpoint_name
        best_checkpoint_path = self.config.paths.checkpoints_dir / "world_model_v0.best.pt"
        warm_started = False

        if checkpoint_path.exists():
            payload = load_checkpoint(checkpoint_path, model, optimizer)
            warm_started = True
            print(
                "[Python] Warm-start World Model from %s (epoch %s)"
                % (checkpoint_path, payload.get("epoch", "?"))
            )

        best_val_loss: float | None = None
        if best_checkpoint_path.exists():
            best_payload = torch.load(best_checkpoint_path, map_location="cpu", weights_only=True)
            best_metrics = best_payload.get("metrics", {})
            if isinstance(best_metrics, dict) and "loss" in best_metrics:
                best_val_loss = float(best_metrics["loss"])

        for epoch in range(self.config.train.epochs):
            train_metrics = train_epoch(model, train_loader, optimizer, self.device)
            val_metrics = validate_epoch(model, val_loader, self.device)
            metrics = {"train": train_metrics, "validation": val_metrics}
            history.append({"epoch": epoch, **metrics})
            print(f"[Python] Night Training World Model | Epoch {epoch+1}/{self.config.train.epochs} | Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            save_checkpoint(
                destination=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
            )
            if best_val_loss is None or val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(
                    destination=best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=val_metrics,
                )

        return {
            "run_dir": str(run_dir),
            "replay_window": [str(path) for path in replay_window],
            "checkpoint_path": str(checkpoint_path),
            "best_checkpoint_path": str(best_checkpoint_path),
            "warm_started": warm_started,
            "config": asdict(self.config),
            "history": history,
        }
