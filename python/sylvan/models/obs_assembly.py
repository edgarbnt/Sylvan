"""Assemble the world-model observation vector [proprio ++ vision ++ energy].

ONE place that builds the WM obs so TRAINING (buffer/loops) and INFERENCE (planner/
server) normalise energy identically. The WM is "hungry" only if energy is in its
observation; energy is normalised to [0,1] so its 1 dim isn't drowned by the 94 proprio
dims (and isn't a raw 0..100 value that dominates the MSE).
"""

from __future__ import annotations

import torch

from ..constants import DEFAULT_MAX_ENERGY

ENERGY_IDX = -1  # energy is the LAST column of the assembled obs


def assemble_wm_obs(
    proprio: torch.Tensor,
    vision: torch.Tensor,
    energy: torch.Tensor,
    *,
    max_energy: float = DEFAULT_MAX_ENERGY,
) -> torch.Tensor:
    """proprio [...,P], vision [...,V], energy [...] (raw 0..max_energy) -> [...,P+V+1].

    Energy is normalised to [0,1] and appended LAST.
    """
    energy_norm = (energy / max_energy).unsqueeze(-1)
    return torch.cat([proprio, vision, energy_norm], dim=-1)
