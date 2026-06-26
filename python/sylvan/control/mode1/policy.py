# python/sylvan/control/mode1/policy.py
import torch
import torch.nn as nn

N_RAYS = 36
TOK = 2 + N_RAYS  # niveau, valence, 36 profondeurs couleur-gatées

def map_action(mean: torch.Tensor) -> torch.Tensor:
    """mean[...,0]→vx∈[0.55,0.75] ; mean[...,1]→ω∈[-0.6,0.6] (régime propre, design §2.5)."""
    vx = 0.55 + 0.10 * torch.sigmoid(mean[..., 0:1])
    om = 0.6 * torch.tanh(mean[..., 1:2])
    return torch.cat([vx, om], dim=-1)

class DriveSymmetricPolicy(nn.Module):
    """proprio + N tokens-pulsion → encodeur PARTAGÉ par token → pooling invariant (mean) → tronc → (vx,ω).
    Aucun slot 'énergie'/'soif' en dur : ajouter une pulsion = un token de plus, MÊMES poids (design §2.3)."""
    def __init__(self, proprio_dim=132, hidden=128, action_dim=2):
        super().__init__()
        self.token_enc = nn.Sequential(nn.Linear(TOK, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
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
