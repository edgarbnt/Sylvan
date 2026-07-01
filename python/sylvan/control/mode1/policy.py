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
    Aucun slot 'énergie'/'soif' en dur : ajouter une pulsion = un token de plus, MÊMES poids (design §2.3).

    Phase 2 (RL) ajoute une TÊTE DE VALEUR drive-symétrique (`value_head`) qui consomme la MÊME feature
    partagée que l'acteur → le critique généralise lui aussi à une nouvelle pulsion (Gate-S). Un
    ENCODAGE-OBS-EMPAQUETÉ (`pack_obs`/`unpack_obs` + `evaluate_actions`/`value`/`mean` empaquetés) permet
    de réutiliser `ppo.update.ppo_update` SANS le modifier : obs plate = cat([proprio, tokens.flatten])."""
    def __init__(self, proprio_dim=132, ray_dim: int = N_RAYS, hidden=128, action_dim=2):
        super().__init__()
        tok_dim = 2 + ray_dim
        self.proprio_dim = int(proprio_dim)
        self.tok_dim = int(tok_dim)                          # 38 : largeur d'un token (niveau, valence, 36 depths)
        self.action_dim = int(action_dim)
        self.token_enc = nn.Sequential(nn.Linear(tok_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.proprio_enc = nn.Sequential(nn.Linear(proprio_dim, hidden), nn.SiLU())
        self.trunk = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, action_dim))
        # Critique drive-symétrique : consomme la feature partagée (2*hidden), pas le proprio brut → il
        # généralise à une nouvelle pulsion exactement comme l'acteur (design §2.3, Gate-S).
        self.value_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))  # utilisé au RL (Phase 2)

    def load_state_dict(self, state_dict, strict: bool = False, assign: bool = False):
        """`strict=False` PAR DÉFAUT (rétro-compat). Les checkpoints BC (Phase 1) et les serveurs de
        déploiement (serve_mode1 / serve_mode1_collect) chargent `state_dict` SANS `value_head` (le
        critique est un ajout Phase 2) — un chargement strict lèverait 'Missing key value_head.*'. Les
        serveurs n'utilisent QUE l'acteur (`forward`/`sample`), un critique fraîchement initialisé y est
        inoffensif. Un checkpoint Phase-2 (avec value_head) recharge exactement, les clés supplémentaires
        étant simplement présentes des deux côtés."""
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    # ------------------------------------------------------------------ #
    # Feature partagée (acteur ET critique la consomment → tout est drive-symétrique)
    # ------------------------------------------------------------------ #
    def _features(self, proprio, tokens):
        # proprio:[*,proprio_dim] ; tokens:[*,D,TOK]. Pooling mean = invariant par permutation des drives.
        h_tok = self.token_enc(tokens).mean(dim=-2)              # [*,hidden]  (moyenne sur l'axe des drives)
        return torch.cat([self.proprio_enc(proprio), h_tok], -1)  # [*,2*hidden]

    def forward(self, proprio, tokens):
        return self.trunk(self._features(proprio, tokens))       # [*,action_dim] (mean non-bornée)

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

    # ------------------------------------------------------------------ #
    # Interface EMPAQUETÉE (obs plate) : réutilise ppo_update tel quel.
    # obs = cat([proprio[.,proprio_dim], tokens.reshape(.,D*TOK)], -1)  → largeur proprio_dim + D*TOK.
    # ------------------------------------------------------------------ #
    def pack_obs(self, proprio, tokens):
        """proprio[*,P] + tokens[*,D,TOK] → obs plate [*,P+D*TOK]."""
        flat = tokens.reshape(*tokens.shape[:-2], tokens.shape[-2] * tokens.shape[-1])
        return torch.cat([proprio, flat], dim=-1)

    def unpack_obs(self, obs):
        """obs plate [*,P+D*TOK] → (proprio[*,P], tokens[*,D,TOK]). D INFÉRÉ de la largeur."""
        proprio = obs[..., : self.proprio_dim]
        rest = obs[..., self.proprio_dim :]
        d = rest.shape[-1] // self.tok_dim
        tokens = rest.reshape(*rest.shape[:-1], d, self.tok_dim)
        return proprio, tokens

    def evaluate_actions(self, obs, action):
        """Log-prob, entropie, valeur de `action` sous la politique COURANTE, depuis une obs EMPAQUETÉE.
        `action` = z BRUT (pré-map_action) → la gaussienne porte sur z, log_prob(z) est exacte (cohérent
        avec `sample`). Consommé par ppo_update (snapshot behavior → old_log_prob ; puis update)."""
        proprio, tokens = self.unpack_obs(obs)
        feat = self._features(proprio, tokens)
        mean = self.trunk(feat)
        std = self.log_std.clamp(LOG_STD_FLOOR, LOG_STD_CEIL).exp()
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.value_head(feat).squeeze(-1)
        return log_prob, entropy, value

    def value(self, obs):
        """Valeur V(obs empaquetée) → [B]."""
        proprio, tokens = self.unpack_obs(obs)
        return self.value_head(self._features(proprio, tokens)).squeeze(-1)

    def mean(self, obs):
        """Mean action BRUTE (pré-map_action) depuis une obs empaquetée → [B,action_dim]."""
        proprio, tokens = self.unpack_obs(obs)
        return self.trunk(self._features(proprio, tokens))

    def mean_std(self) -> float:
        """Écart-type moyen d'exploration (borné) — signal de santé lu par ppo_update."""
        return float(self.log_std.clamp(LOG_STD_FLOOR, LOG_STD_CEIL).exp().mean().item())
