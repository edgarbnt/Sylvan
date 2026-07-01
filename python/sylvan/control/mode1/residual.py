"""Chemin obs-résidu PARTAGÉ (serve_mode1 / serve_mode1_collect / serve_planner_command).

Le résidu gelé (hexapod_v2) EXIGE une obs byte-identique : proprio[132] ++ [vx, ω, 0×10]. Cette
construction était copiée VERBATIM dans trois serveurs ; on l'extrait ici pour garantir l'identité
(une divergence d'un octet = un résidu qui reçoit une obs hors-distribution). Inclut le nan_to_num +
clamp final (garde-fou identique partout)."""

from __future__ import annotations

import torch

VISION_DIM = 12  # slot vision de l'obs-résidu : [vx, ω, 0×10]


def residual_action(residual, proprio: list, vx: float, om: float) -> list:
    """proprio[132] ++ [vx, ω, 0×10] → résidu gelé → action[18] (nan_to_num + clamp[-1,1]).

    `residual` = GaussianActorCritic gelé ; `residual.mean(obs)` = action déterministe (moyenne).
    Renvoie une list[float] (déjà sanitizée : jamais de NaN/inf) prête pour la réponse TCP."""
    vision = [float(vx), float(om)] + [0.0] * (VISION_DIM - 2)
    res_in = torch.tensor(list(proprio) + vision, dtype=torch.float32).unsqueeze(0)
    action = residual.mean(res_in)[0]
    action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    return [float(v) for v in action.tolist()]
