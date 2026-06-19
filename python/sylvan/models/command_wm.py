"""Phase 4: world model in COMMAND space.

The action is the 2-D command (vx, omega) the JEPA planner will emit — not the 12 raw
joint torques. The motor base (CPG + frozen residual) is part of the ENVIRONMENT here:
the model learns the command -> real motion map of the whole embodied stack.

Key design choice: the model predicts the per-step torso DISPLACEMENT in the BODY frame
(d_fwd, d_lat, d_yaw), not an absolute world position — the obs carries no world position,
so an absolute target is unlearnable, while the body-frame delta is translation/rotation
invariant and integrates into a trajectory at eval/planning time.
"""

from __future__ import annotations

import os

import torch
from torch import nn
import torch.nn.functional as F

from .encoders import ProprioEncoder
from .heads import DoneHead, MetricsPredictionHead, ProprioPredictionHead
from .rssm import SimpleRSSM

COMMAND_DIM = 2          # (vx, omega)
DISPLACEMENT_DIM = 3     # (d_fwd, d_lat, d_yaw) in the body frame at t
DISPLACEMENT_SCALE = 100.0  # raw deltas are ~1e-2 m / 4e-3 rad per step; scale to ~O(1) for the MSE
# eat transitions are rare; upweight them so energy jumps are learned. Overridable (SYLVAN_EAT_SAMPLE_WEIGHT)
# for the eat-rich retrain toward 🅑 (the WM must learn the eat-bump it currently captures at only ~4%).
EAT_SAMPLE_WEIGHT = float(os.environ.get("SYLVAN_EAT_SAMPLE_WEIGHT", "30.0"))

# Loss weights. The Phase-4 (Dreamer-like) defaults below reproduce the validated wm_command_v2.
# Phase B (JEPA-ification) shifts weight off the reconstruction terms (proprio, radar) toward the
# latent-prediction term — see BLUEPRINT §13. displacement/energy/done are abstract readouts (the
# planner's signal, the cost, the milestone), NOT input reconstruction, so they stay anchored.
DEFAULT_LOSS_WEIGHTS: dict[str, float] = {
    "latent": 1.0,        # ✅ the JEPA path: predict the next obs's encoding (stop-grad target)
    "proprio": 1.0,       # ❌ generative input reconstruction (94-d)
    "radar": 5.0,         # ❌ generative input reconstruction (12-d), also the planner's perception readout
    "energy": 20.0,       # abstract cost readout — anchor
    "displacement": 10.0, # abstract body-frame readout — the milestone + planner signal — anchor
    "done": 1.0,          # abstract readout — anchor
}


class CommandWorldModel(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        proprio_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        predictor_arch: str = "shallow",
        with_food_head: bool = False,
    ) -> None:
        super().__init__()
        # obs = proprio ++ food radar ++ energy. In CPG mode the POLICY's vision channel carries
        # the command, so the radar here comes from the wm ground-truth block, not obs.vision.
        self.obs_dim = obs_dim
        self.proprio_dim = proprio_dim
        self.predictor_arch = predictor_arch
        self.encoder = ProprioEncoder(obs_dim, hidden_dim, latent_dim)
        self.rssm = SimpleRSSM(latent_dim, COMMAND_DIM, hidden_dim)
        # Phase B (BLUEPRINT §13): the latent predictor is the JEPA path. "deep" muscles it
        # (extra layer + LayerNorm) — the asymmetric predictor that, with the stop-grad target,
        # holds BYOL-style non-collapse. "shallow" reproduces wm_command_v2 exactly (default).
        if predictor_arch == "deep":
            self.encoded_predictor = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
        else:
            self.encoded_predictor = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
        self.obs_head = ProprioPredictionHead(latent_dim, obs_dim)
        # latent + command -> body-frame displacement; reuses the generic [latent ⊕ action] MLP head.
        self.displacement_head = MetricsPredictionHead(latent_dim, COMMAND_DIM, DISPLACEMENT_DIM)
        self.done_head = DoneHead(latent_dim)
        # Tête AUXILIAIRE 'repas imminent' (🅑, 2026-06-19) — N'EST PAS sauvée dans le checkpoint : elle ne
        # sert qu'à FORCER le latent RÊVÉ (dream) à transporter la bouffe pendant l'entraînement (le rêve
        # propageait le mouvement mais pas la nourriture → coût-latent inexploitable, cf diag_value_direct).
        # À l'inférence on utilise la ValueHead séparée. Optionnelle → structure de checkpoint inchangée.
        if with_food_head:
            self.food_head = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
            )

    def dream_latents(self, obs0: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:
        """Rollout FREE-RUNNING (open-loop) AVEC gradient → latents rêvés [B, T, latent_dim].
        Récurrence IDENTIQUE à rollout_open_loop (ce que le planner/gate utilisent) : après le pas 0,
        l'entrée est la propre prédiction encodée du modèle. Sert la perte auxiliaire 'food-aware'."""
        batch_size, horizon, _ = commands.shape
        hidden = torch.zeros(batch_size, self.rssm.gru.hidden_size, device=obs0.device, dtype=obs0.dtype)
        obs_input = self.encoder(obs0)
        lats = []
        for t in range(horizon):
            hidden = self.rssm.gru(torch.cat((obs_input, commands[:, t]), dim=-1), hidden)
            latent = self.rssm.to_latent(hidden)
            lats.append(latent)
            obs_input = self.encoded_predictor(latent)
        return torch.stack(lats, dim=1)

    def forward(
        self,
        obs: torch.Tensor,
        commands: torch.Tensor,
        *,
        scheduled_sampling_prob: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        encoded_obs = self.encoder(obs)
        latents, hidden = self.rssm(
            encoded_obs,
            commands,
            encoded_predictor=self.encoded_predictor,
            scheduled_sampling_prob=scheduled_sampling_prob,
        )
        return {
            "latents": latents,
            "hidden": hidden,
            "predicted_next_encoded": self.encoded_predictor(latents),
            "predicted_next_obs": self.obs_head(latents),
            "predicted_displacement": self.displacement_head(latents, commands),
            "predicted_done_logits": self.done_head(latents),
        }

    @torch.no_grad()
    def rollout_open_loop(
        self,
        obs0: torch.Tensor,
        commands: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Dream forward from a single real observation under a command sequence.

        obs0: [B, obs_dim]; commands: [B, T, 2]. Mirrors the free-running branch of the
        RSSM's scheduled sampling exactly: after step 0 the obs input is the model's own
        encoded prediction, never a real observation.
        """
        batch_size, horizon, _ = commands.shape
        hidden = torch.zeros(batch_size, self.rssm.gru.hidden_size, device=obs0.device, dtype=obs0.dtype)
        obs_input = self.encoder(obs0)
        disps, obs_preds, done_logits, latents = [], [], [], []
        for t in range(horizon):
            hidden = self.rssm.gru(torch.cat((obs_input, commands[:, t]), dim=-1), hidden)
            latent = self.rssm.to_latent(hidden)
            disps.append(self.displacement_head(latent, commands[:, t]))
            obs_preds.append(self.obs_head(latent))
            done_logits.append(self.done_head(latent))
            latents.append(latent)
            obs_input = self.encoded_predictor(latent)
        return {
            "predicted_displacement": torch.stack(disps, dim=1),   # [B, T, 3]
            "predicted_next_obs": torch.stack(obs_preds, dim=1),   # [B, T, obs_dim]
            "predicted_done_logits": torch.stack(done_logits, dim=1),
            "predicted_latents": torch.stack(latents, dim=1),      # [B, T, latent_dim] — dreamed RSSM latents (critic probe)
        }


def compute_command_wm_losses(
    outputs: dict[str, torch.Tensor],
    *,
    next_obs: torch.Tensor,
    displacement: torch.Tensor,
    done: torch.Tensor,
    eat_weight: torch.Tensor,
    model: CommandWorldModel,
    proprio_dim: int,
    weights: dict[str, float] | None = None,
    latent_loss_mode: str = "mse",
    vicreg_var: float = 0.0,
    vicreg_cov: float = 0.0,
    vicreg_gamma: float = 1.0,
) -> dict[str, torch.Tensor]:
    w = {**DEFAULT_LOSS_WEIGHTS, **(weights or {})}
    pred_obs = outputs["predicted_next_obs"]
    with torch.no_grad():
        target_encoded = model.encoder(next_obs)
    # "cosine" (Phase B): scale-invariant latent loss = 2-2·cos. Removes the magnitude-shrink
    # collapse cheat that raw MSE permits (shrinking both embeddings drives MSE→0 for free).
    if latent_loss_mode == "cosine":
        pred_n = F.normalize(outputs["predicted_next_encoded"], dim=-1)
        tgt_n = F.normalize(target_encoded, dim=-1)
        latent_loss = F.mse_loss(pred_n, tgt_n)
    else:
        latent_loss = F.mse_loss(outputs["predicted_next_encoded"], target_encoded)
    proprio_loss = F.mse_loss(pred_obs[..., :proprio_dim], next_obs[..., :proprio_dim])
    radar_loss = F.mse_loss(pred_obs[..., proprio_dim:-1], next_obs[..., proprio_dim:-1])
    energy_se = (pred_obs[..., -1] - next_obs[..., -1]) ** 2
    energy_loss = (eat_weight * energy_se).sum() / eat_weight.sum()
    disp_loss = F.mse_loss(outputs["predicted_displacement"], displacement)
    done_loss = F.binary_cross_entropy_with_logits(outputs["predicted_done_logits"], done)
    # VICReg (Phase B step 2) on the RSSM latents — the representation eff_rank watches.
    # Weights 0 → terms vanish exactly → 1.2 behavior unchanged (reversible).
    vic_var, vic_cov = vicreg_terms(outputs["latents"], gamma=vicreg_gamma)
    total = (
        w["latent"] * latent_loss
        + w["proprio"] * proprio_loss
        + w["radar"] * radar_loss
        + w["energy"] * energy_loss
        + w["displacement"] * disp_loss
        + w["done"] * done_loss
        + vicreg_var * vic_var
        + vicreg_cov * vic_cov
    )
    return {
        "loss": total,
        "latent": latent_loss.detach(),
        "proprio": proprio_loss.detach(),
        "radar": radar_loss.detach(),
        "energy": energy_loss.detach(),
        "displacement": disp_loss.detach(),
        "done": done_loss.detach(),
        "vic_var": vic_var.detach(),
        "vic_cov": vic_cov.detach(),
    }


def vicreg_terms(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg variance + covariance regularizers on a batch of latents [..., L] (Bardes/LeCun 2022).

    variance: hinge mean(relu(gamma - std_per_dim)) — forces every dim to vary ≥ gamma → kills the
      lazy 'all latents equal' solution that makes our latent-prediction loss trivially zero.
    covariance: sum of squared off-diagonal covariances / L — decorrelates dims → spreads info,
      raises eff_rank. Both push the representation OFF the ~5-active-dim regime measured at 1.2.
    """
    z = z.reshape(-1, z.shape[-1])
    n = z.shape[0]
    std = torch.sqrt(z.var(dim=0) + eps)
    var_loss = torch.relu(gamma - std).mean()
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.t() @ zc) / max(1, n - 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = off.pow(2).sum() / z.shape[-1]
    return var_loss, cov_loss


@torch.no_grad()
def representation_health(latents: torch.Tensor) -> dict[str, float]:
    """Anti-collapse metrics on a batch of latents [..., L] — a METRIC, never a loss.

    These are VICReg's variance/covariance quantities used purely for observation BEFORE we
    add them as a Phase-B objective (BLUEPRINT §13 step 2): if the JEPA weight-shift collapses
    the representation, lat_std craters toward 0 and eff_rank toward 1.
    """
    z = latents.reshape(-1, latents.shape[-1]).float()
    std = z.std(dim=0)
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.t() @ zc) / max(1, z.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    s = float(eig.sum())
    eff_rank = (s * s) / (float(eig.pow(2).sum()) + 1e-12) if s > 0 else 0.0
    off = cov - torch.diag(torch.diag(cov))
    return {
        "lat_std": float(std.mean()),
        "lat_std_min": float(std.min()),
        "eff_rank": eff_rank,            # participation ratio; L=128 if perfectly spread, 1 if collapsed
        "offdiag": float(off.abs().mean()),
    }
