"""Saillance de PULSION — « à quoi ressemble ce qui compte pour cette pulsion ? », appris.

Jumeau générique de `DangerSaliency` (`python/scripts/train_danger_saliency.py`), qui est le
seul module de perception marqué `pur` dans la carte d'archi et qui a dissous la clé-apparence
« danger = vert ». La forme est identique — c'est délibéré : elle a été jugée, et la version
SOMME a été testée et a ÉCHOUÉ (crédit partiel à la mauvaise couleur, ρ̂ figé). Négatif banké,
ne pas y revenir.

    P(conséquence bientôt | rétine) = σ( b + max_{rayons touchants k}  s(rgb_k) · g(d_k) )

- `s(rgb)` — MLP 3→16→1 sur **la couleur SEULE, jamais la distance**. C'est la propriété
  clé : ce qu'elle apprend de près vaut à TOUTE distance, ce qui manquait à tous les
  remplacements réfutés (§5 de `docs/design_perception_pure_faim.md`), qui lisaient
  (depth, RGB) ensemble et se faisaient piéger par les 39,9 % de rayons d'arbres partageant
  ce volume.
- `g(d)` — portée apprise σ((ρ−d)/τ), en MÈTRES.
- max-pooling — la conséquence a UNE source (Multiple Instance Learning, Ilse et al. 2018).
- prior de parcimonie sur `s` — « rien ne compte sans preuve vécue » : les apparences jamais
  contraintes retombent à zéro.

Ce qui change par rapport au danger : SEULEMENT l'étiquette (consommation vécue au lieu de
dégâts vécus) et le nom de la pulsion. Une tête par pulsion — faim, soif, danger — d'où ce
module générique plutôt qu'un troisième copier-coller.

⚠️ Ce module ne contient AUCUNE couleur codée-main. C'est tout l'objet du chantier.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

N_RAY = 36
RETINA_DIM = 144
RETINA_RANGE_M = 10.0  # portée du raycast (perception.gd MAX_RANGE)
TOUCH_MAX = 0.999  # d >= 0.999 = le rayon ne touche rien
SAL_THR = 0.5  # seuil de lecture de s — PINNÉ, jamais fitté (comme le danger)
LAMBDA_S = 0.01  # prior de parcimonie — constante de conception


class DriveSaliency(nn.Module):
    """rétine(144) → P(la pulsion sera satisfaite/heurtée bientôt), et s(rgb) par rayon."""

    def __init__(self, hidden: int = 16, rho_init: float = 1.5) -> None:
        super().__init__()
        self.app = nn.Sequential(nn.Linear(3, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.rho = nn.Parameter(torch.tensor(rho_init))
        self.tau_raw = nn.Parameter(torch.tensor(0.5))
        self.bias = nn.Parameter(torch.tensor(-3.0))

    def s(self, rgb: torch.Tensor) -> torch.Tensor:
        """[..., 3] -> (0,1). L'APPARENCE seule — la distance n'entre jamais ici."""
        return torch.sigmoid(self.app(rgb).squeeze(-1))

    def g(self, dist_m: torch.Tensor) -> torch.Tensor:
        """Portée apprise. Prend des MÈTRES, pas la profondeur normalisée."""
        tau = nn.functional.softplus(self.tau_raw) + 0.05
        return torch.sigmoid((self.rho - dist_m) / tau)

    def rho_hat(self) -> float:
        """Distance où g vaut 0,5 — la portée que la tête a inférée du vécu."""
        return float(self.rho)

    def parts(self, retina: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """[B, 144] -> (logits [B], s [B, 36], touch [B, 36])."""
        r = retina.view(-1, N_RAY, 4)
        d, rgb = r[..., 0], r[..., 1:]
        touch = d < TOUCH_MAX
        s = self.s(rgb)
        logits = self.bias + (s * self.g(d * RETINA_RANGE_M) * touch.float()).amax(dim=-1)
        return logits, s, touch

    def tick_logits(self, retina: torch.Tensor) -> torch.Tensor:
        return self.parts(retina)[0]

    @torch.no_grad()
    def ray_scores(self, retina: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[..., 144] -> (s [..., 36], touch [..., 36]). Le lecteur de DÉPLOIEMENT.

        C'est ce que `slot_head` consommera à la place du cosinus codé-main : un score
        d'appartenance par rayon, dans [0,1], sur lequel on seuille à SAL_THR.
        """
        shape = retina.shape[:-1]
        r = retina.reshape(-1, N_RAY, 4)
        d, rgb = r[..., 0], r[..., 1:]
        return self.s(rgb).reshape(*shape, N_RAY), (d < TOUCH_MAX).reshape(*shape, N_RAY)


def save_drive_saliency(model: DriveSaliency, path: str | Path, drive: str, **extra) -> None:
    """Checkpoint plat, JSON-compatible (chargé avec weights_only=True)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "form": "drive_saliency_v1",
            "drive": drive,
            "thr": SAL_THR,
            "rho_hat": model.rho_hat(),
            **extra,
        },
        p,
    )


def load_drive_saliency(path: str | Path) -> tuple[DriveSaliency, dict]:
    ck = torch.load(str(path), map_location="cpu", weights_only=True)
    m = DriveSaliency()
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck
