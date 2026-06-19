"""Model-free PPO controller (milestone J0).

A GROUNDED control layer: the policy is trained on REAL Godot rollouts and the
REAL reward, decoupled from the world model. This is the blueprint's answer to the
"trained 100% in imagination" pathology (BLUEPRINT.md §6/§8) — prove balance is
learnable on the real body before re-introducing the JEPA world model for planning.
"""

from .policy import GaussianActorCritic
from .stochastic_server import serve_stochastic_policy
from .rollout import RolloutBatch, build_rollout_batch, compute_gae
from .update import PPOConfig, ppo_update

__all__ = [
    "GaussianActorCritic",
    "serve_stochastic_policy",
    "RolloutBatch",
    "build_rollout_batch",
    "compute_gae",
    "PPOConfig",
    "ppo_update",
]
