"""EgomotionHead — proprio[132] → (dyaw, dfwd, dlat).

Petite tête apprise (1-hidden-layer MLP) qui prédit l'ego-motion en un pas depuis le vecteur
proprioceptif 132-d.  La normalisation d'entrée (μ, σ) est stockée dans le checkpoint pour
que `predict()` soit auto-contenu (aucun état externe requis).

Convention ego-motion (identique à egomotion_from_torso dans diag_slot_memory_drift.py) :
  dyaw  (rad)  = wrap(yaw1 − yaw0)
  dfwd  (m)    = dx·sin(yaw0) + dz·cos(yaw0)   — projection vers l'avant
  dlat  (m)    = dx·cos(yaw0) − dz·sin(yaw0)   — projection vers la droite

Usage:
    from sylvan.models.egomotion_head import EgomotionHead, load_egomotion_head
    head = load_egomotion_head("data/checkpoints/egomotion_head/best.pt")
    dyaw, dfwd, dlat = head.predict(proprio_list_132)
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn

__all__ = ["EgomotionHead", "load_egomotion_head"]


class EgomotionHead(nn.Module):
    """MLP proprio[132] → (dyaw, dfwd, dlat).

    La normalisation d'entrée est un buffer persistant stocké dans le checkpoint,
    ce qui rend `predict()` entièrement auto-contenu.
    """

    PROPRIO_DIM = 132
    OUTPUT_DIM = 3       # dyaw, dfwd, dlat
    HIDDEN = 128

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.PROPRIO_DIM, self.HIDDEN),
            nn.SiLU(),
            nn.Linear(self.HIDDEN, self.HIDDEN),
            nn.SiLU(),
            nn.Linear(self.HIDDEN, self.OUTPUT_DIM),
        )
        # Normalisation entrée : initialisée à identité, remplie au moment de l'entraînement
        self.register_buffer("mu_x", torch.zeros(self.PROPRIO_DIM))
        self.register_buffer("sd_x", torch.ones(self.PROPRIO_DIM))

    # ------------------------------------------------------------------
    # Inférence
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 132) tenseur brut.  Retourne (B, 3) = (dyaw, dfwd, dlat)."""
        xn = (x - self.mu_x) / self.sd_x
        return self.net(xn)

    @torch.no_grad()
    def predict(self, proprio: List[float]) -> Tuple[float, float, float]:
        """Interface déploiement.  proprio = liste de 132 floats.
        Retourne (dyaw, dfwd, dlat) en (rad, m, m)."""
        if len(proprio) != self.PROPRIO_DIM:
            raise ValueError(
                f"EgomotionHead.predict attend {self.PROPRIO_DIM} floats, reçu {len(proprio)}"
            )
        self.eval()
        x = torch.tensor(proprio, dtype=torch.float32).unsqueeze(0)
        out = self.forward(x)
        dyaw, dfwd, dlat = out[0].tolist()
        return float(dyaw), float(dfwd), float(dlat)


# ---------------------------------------------------------------------------
# Helpers checkpoint
# ---------------------------------------------------------------------------

def save_egomotion_head(head: EgomotionHead, path: str) -> None:
    """Sauvegarde le module + ses buffers de normalisation."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": head.state_dict()}, path)


def load_egomotion_head(path: str) -> EgomotionHead:
    """Charge un EgomotionHead depuis un checkpoint.  Prêt pour `predict()`."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    head = EgomotionHead()
    head.load_state_dict(ck["model"])
    head.eval()
    return head
