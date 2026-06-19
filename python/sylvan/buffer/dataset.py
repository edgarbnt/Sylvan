"""PyTorch dataset adapter for the replay buffer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .reader import iter_episodes
from .schema import Transition
from ..constants import LOCOMOTION_METRIC_KEYS


@dataclass(slots=True)
class SequenceSample:
    proprio: torch.Tensor
    vision: torch.Tensor
    energy: torch.Tensor
    metrics: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_proprio: torch.Tensor
    next_vision: torch.Tensor
    next_energy: torch.Tensor
    next_metrics: torch.Tensor
    done: torch.Tensor


class ReplaySequenceDataset(Dataset[SequenceSample]):
    def __init__(self, run_dir: Path | list[Path], sequence_length: int) -> None:
        self.sequence_length = sequence_length
        self.run_dirs = self._normalize_run_dirs(run_dir)
        self.samples: list[list[Transition]] = []

        for one_run_dir in self.run_dirs:
            for episode in iter_episodes(one_run_dir):
                if len(episode) < sequence_length:
                    continue
                for index in range(0, len(episode) - sequence_length + 1):
                    self.samples.append(episode[index : index + sequence_length])

    @staticmethod
    def _normalize_run_dirs(run_dir: Path | list[Path]) -> list[Path]:
        if isinstance(run_dir, list):
            return run_dir
        return [run_dir]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SequenceSample:
        sequence = self.samples[index]
        return SequenceSample(
            proprio=torch.tensor(
                [step.obs.proprio for step in sequence], dtype=torch.float32
            ),
            vision=torch.tensor(
                [step.obs.vision for step in sequence], dtype=torch.float32
            ),
            energy=torch.tensor(
                [step.obs.energy for step in sequence], dtype=torch.float32
            ),
            metrics=torch.tensor(
                [[step.obs.metrics.get(key, 0.0) for key in LOCOMOTION_METRIC_KEYS] for step in sequence],
                dtype=torch.float32,
            ),
            action=torch.tensor(
                [step.action for step in sequence], dtype=torch.float32
            ),
            reward=torch.tensor(
                [step.reward for step in sequence], dtype=torch.float32
            ),
            next_proprio=torch.tensor(
                [step.next_obs.proprio for step in sequence], dtype=torch.float32
            ),
            next_vision=torch.tensor(
                [step.next_obs.vision for step in sequence], dtype=torch.float32
            ),
            next_energy=torch.tensor(
                [step.next_obs.energy for step in sequence], dtype=torch.float32
            ),
            next_metrics=torch.tensor(
                [
                    [step.next_obs.metrics.get(key, 0.0) for key in LOCOMOTION_METRIC_KEYS]
                    for step in sequence
                ],
                dtype=torch.float32,
            ),
            done=torch.tensor([step.done for step in sequence], dtype=torch.float32),
        )


def collate_sequence_samples(samples: list[SequenceSample]) -> SequenceSample:
    return SequenceSample(
        proprio=torch.stack([sample.proprio for sample in samples], dim=0),
        vision=torch.stack([sample.vision for sample in samples], dim=0),
        energy=torch.stack([sample.energy for sample in samples], dim=0),
        metrics=torch.stack([sample.metrics for sample in samples], dim=0),
        action=torch.stack([sample.action for sample in samples], dim=0),
        reward=torch.stack([sample.reward for sample in samples], dim=0),
        next_proprio=torch.stack([sample.next_proprio for sample in samples], dim=0),
        next_vision=torch.stack([sample.next_vision for sample in samples], dim=0),
        next_energy=torch.stack([sample.next_energy for sample in samples], dim=0),
        next_metrics=torch.stack([sample.next_metrics for sample in samples], dim=0),
        done=torch.stack([sample.done for sample in samples], dim=0),
    )
