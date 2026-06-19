"""Dataset for the Phase 4 command-space world model.

Reads the per-step `wm` ground-truth block ridden alongside the policy transitions
(SYLVAN_WM_COLLECT=1): command (vx, omega), torso pose (x, z, yaw) and the real food
radar at decision time. Targets are built from CONSECUTIVE rows' decision-time (`t0`)
snapshots — physics integrates BETWEEN Godot frames, so the within-row t1 snapshot is
the same physics state as t0 and must not be used as a target.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..models.command_wm import DISPLACEMENT_SCALE, EAT_SAMPLE_WEIGHT

# RÉTINE étage 2 : si 1, l'obs WM utilise la rétine (144) au lieu du radar (12) → obs_dim 277. cf wm_dataset._obs_at.
_WM_USE_RETINA = os.environ.get("SYLVAN_WM_USE_RETINA", "0") == "1"
# 🅑 : horizon du label 'repas imminent' (eat_soon) pour la perte auxiliaire food-aware du WM.
_EAT_SOON_K = int(os.environ.get("SYLVAN_EAT_SOON_K", "20"))


@dataclass(slots=True)
class CommandSequenceSample:
    obs: torch.Tensor           # [T, obs_dim]  proprio ++ radar ++ energy(0..1)
    command: torch.Tensor       # [T, 2]
    next_obs: torch.Tensor      # [T, obs_dim]
    displacement: torch.Tensor  # [T, 3]  body-frame (d_fwd, d_lat, d_yaw) * DISPLACEMENT_SCALE
    done: torch.Tensor          # [T]
    eat_weight: torch.Tensor    # [T]  1 + EAT_SAMPLE_WEIGHT on eat transitions
    eat_soon: torch.Tensor      # [T]  1 si un repas survient dans les K prochains pas (cible food-aware 🅑)
    torso: torch.Tensor         # [T+1, 3]  world (x, z, yaw) chain, for open-loop eval


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def load_wm_episode(path: Path) -> dict[str, torch.Tensor] | None:
    """One episode JSONL -> aligned tensors (length N-1: the last row has no t+1 target)."""
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            # RÉTINE étage 2 : SYLVAN_WM_USE_RETINA=1 → l'obs WM = proprio ++ RÉTINE(144) ++ énergie (277)
            # au lieu de proprio ++ radar(12) ++ énergie (145). Le WM perçoit alors les rayons couleur BRUTS
            # (vrai JEPA, oracle radar mort dans le WM aussi). Flag explicite → rétro-compatible (WM hex_v2).
            _key = "retina0" if _WM_USE_RETINA else "radar0"
            if "wm" not in r or not r["wm"].get(_key):
                return None
            rows.append(r)
    if len(rows) < 2:
        return None
    obs, cmd, nxt, disp, done, eatw, torso = [], [], [], [], [], [], []
    # ate au pas de transition t (= rows[t+1] a mangé) → eat_soon[t] = un repas dans [t, t+K)
    ate_flags = [1.0 if rows[t + 1]["wm"].get("ate", 0.0) > 0.0 else 0.0 for t in range(len(rows) - 1)]
    eat_soon = [1.0 if any(ate_flags[t:t + _EAT_SOON_K]) else 0.0 for t in range(len(rows) - 1)]

    def _obs_at(r: dict) -> list[float]:
        return list(r["obs"]["proprio"]) + list(r["wm"][_key]) + [r["obs"]["energy"] / 100.0]

    for t in range(len(rows) - 1):
        r, rn = rows[t], rows[t + 1]
        x0, z0, yaw0 = r["wm"]["torso0"]
        x1, z1, yaw1 = rn["wm"]["torso0"]
        dxw, dzw = x1 - x0, z1 - z0
        # Body frame at t: forward = +z rotated by yaw (yaw = atan2(f.x, f.z)).
        s, c = math.sin(yaw0), math.cos(yaw0)
        obs.append(_obs_at(r))
        cmd.append(r["wm"]["cmd"])
        nxt.append(_obs_at(rn))
        disp.append([
            (dxw * s + dzw * c) * DISPLACEMENT_SCALE,
            (dxw * c - dzw * s) * DISPLACEMENT_SCALE,
            _wrap_angle(yaw1 - yaw0) * DISPLACEMENT_SCALE,
        ])
        done.append(1.0 if r["done"] else 0.0)
        eatw.append(1.0 + (EAT_SAMPLE_WEIGHT if rn["wm"].get("ate", 0.0) > 0.0 else 0.0))
        torso.append([x0, z0, yaw0])
    torso.append(list(rows[-1]["wm"]["torso0"]))
    return {
        "obs": torch.tensor(obs, dtype=torch.float32),
        "command": torch.tensor(cmd, dtype=torch.float32),
        "next_obs": torch.tensor(nxt, dtype=torch.float32),
        "displacement": torch.tensor(disp, dtype=torch.float32),
        "done": torch.tensor(done, dtype=torch.float32),
        "eat_weight": torch.tensor(eatw, dtype=torch.float32),
        "eat_soon": torch.tensor(eat_soon, dtype=torch.float32),
        "torso": torch.tensor(torso, dtype=torch.float32),
    }


def list_wm_episodes(run_dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for d in run_dirs:
        paths.extend(sorted(Path(d).glob("*.jsonl")))
    return paths


class CommandSequenceDataset(Dataset[CommandSequenceSample]):
    def __init__(self, episode_paths: list[Path], sequence_length: int, stride: int = 4) -> None:
        self.sequence_length = sequence_length
        self.windows: list[tuple[int, int]] = []
        self.episodes: list[dict[str, torch.Tensor]] = []
        skipped = 0
        for path in episode_paths:
            ep = load_wm_episode(path)
            if ep is None:
                skipped += 1
                continue
            n = ep["obs"].shape[0]
            if n < sequence_length:
                continue
            ep_idx = len(self.episodes)
            self.episodes.append(ep)
            for start in range(0, n - sequence_length + 1, stride):
                self.windows.append((ep_idx, start))
        if skipped:
            print(f"[wm_dataset] {skipped} episodes sans bloc wm ignorés")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> CommandSequenceSample:
        ep_idx, start = self.windows[index]
        ep = self.episodes[ep_idx]
        end = start + self.sequence_length
        return CommandSequenceSample(
            obs=ep["obs"][start:end],
            command=ep["command"][start:end],
            next_obs=ep["next_obs"][start:end],
            displacement=ep["displacement"][start:end],
            done=ep["done"][start:end],
            eat_weight=ep["eat_weight"][start:end],
            eat_soon=ep["eat_soon"][start:end],
            torso=ep["torso"][start : end + 1],
        )


def collate_command_samples(samples: list[CommandSequenceSample]) -> CommandSequenceSample:
    return CommandSequenceSample(
        obs=torch.stack([s.obs for s in samples], dim=0),
        command=torch.stack([s.command for s in samples], dim=0),
        next_obs=torch.stack([s.next_obs for s in samples], dim=0),
        displacement=torch.stack([s.displacement for s in samples], dim=0),
        done=torch.stack([s.done for s in samples], dim=0),
        eat_weight=torch.stack([s.eat_weight for s in samples], dim=0),
        eat_soon=torch.stack([s.eat_soon for s in samples], dim=0),
        torso=torch.stack([s.torso for s in samples], dim=0),
    )
