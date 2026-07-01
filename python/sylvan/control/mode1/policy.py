# python/sylvan/control/mode1/policy.py
import math

import torch
import torch.nn as nn
from torch.distributions import Normal

from sylvan.control.mode1.obs import N_RAYS
TOK = 2 + N_RAYS  # niveau, valence, 36 profondeurs couleur-gatées

# Bornes de log_std (MÊME convention que sylvan/control/ppo/policy.py) : plancher = anti-collapse
# d'exploration ("agent gelé"), plafond = anti-bruit-fou en début de RL. Définies ici pour garder
# ce module autonome (DriveSymmetricPolicy ne dépend pas de l'infra PPO).
LOG_STD_FLOOR = math.log(0.05)
LOG_STD_CEIL = math.log(2.0)

def map_action(mean: torch.Tensor) -> torch.Tensor:
    """mean[...,0]→vx∈[0.55,0.75] ; mean[...,1]→ω∈[-0.6,0.6] (régime propre, design §2.5).
    Mapping LINÉAIRE + clamp : les bornes (vx=0.75, ω=±0.6) sont ATTEIGNABLES à sortie finie
    (vx=0.75 à mean=+1 ; ω=±0.6 à mean=±1) — contrairement à sigmoid/tanh qui ne touchent jamais
    leurs asymptotes (le planner sature à ±0.6 / vx>0.7, que le BC ne pouvait pas reproduire).
    Clamp = cohérent avec le bornage d'action de l'infra PPO existante."""
    vx = (0.65 + 0.10 * mean[..., 0:1]).clamp(0.55, 0.75)
    om = (0.6 * mean[..., 1:2]).clamp(-0.6, 0.6)
    return torch.cat([vx, om], dim=-1)

class DriveSymmetricPolicy(nn.Module):
    """proprio + N tokens-pulsion → encodeur PARTAGÉ par token → pooling invariant (mean) → tronc → (vx,ω).
    Aucun slot 'énergie'/'soif' en dur : ajouter une pulsion = un token de plus, MÊMES poids (design §2.3)."""
    def __init__(self, proprio_dim=132, ray_dim: int = N_RAYS, hidden=128, action_dim=2):
        super().__init__()
        tok_dim = 2 + ray_dim
        self.token_enc = nn.Sequential(nn.Linear(tok_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.proprio_enc = nn.Sequential(nn.Linear(proprio_dim, hidden), nn.SiLU())
        self.trunk = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, action_dim))
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))  # utilisé au RL (Phase 2)

    def forward(self, proprio, tokens):
        # proprio:[B,132] ; tokens:[B,D,TOK] (D variable). Pooling mean = invariant par permutation des drives.
        h_tok = self.token_enc(tokens).mean(dim=1)          # [B,hidden]
        h = torch.cat([self.proprio_enc(proprio), h_tok], -1)  # [B,2*hidden]
        return self.trunk(h)                                  # [B,action_dim] (mean non-bornée)

    @torch.no_grad()
    def act(self, proprio, tokens):
        return map_action(self.forward(proprio, tokens))

    @torch.no_grad()
    def sample(self, proprio, tokens, generator=None):
        """Échantillonne la commande RAW z ~ Normal(mean, std) (Phase 2 RL, collecte on-policy).

        La distribution de la politique porte sur z = sortie BRUTE de forward() (PRÉ-map_action) ;
        map_action est l'ACTIONNEUR déterministe appliqué APRÈS le tirage (commande actionnée =
        map_action(z)). On ne clampe PAS z ici : le bornage (vx∈[0.55,0.75], ω∈[±0.6]) est fait en
        aval par map_action. Renvoie (z[...,2], logprob[...]). Mêmes maths que ppo.policy.sample →
        le ratio PPO vaut exactement 1 au premier pas de chaque itération."""
        mean = self.forward(proprio, tokens)                       # z-mean BRUT (non borné)
        std = self.log_std.clamp(LOG_STD_FLOOR, LOG_STD_CEIL).exp()
        eps = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
        z = mean + std * eps
        logprob = Normal(mean, std).log_prob(z).sum(-1)
        return z, logprob
