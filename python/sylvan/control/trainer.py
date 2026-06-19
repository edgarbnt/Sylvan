"""Controller trainer for locomotion Phase 2."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..buffer.dataset import ReplaySequenceDataset, collate_sequence_samples
from ..buffer.replay_window import select_replay_window
from ..config import SylvanConfig
from ..device import resolve_torch_device
from ..models.world_model import WorldModelV0
from ..training.checkpointing import load_checkpoint, save_checkpoint
from .controller import LocomotionController
from .imagined_rollouts import imagine_rollout


class ControllerTrainer:
    def __init__(self, config: SylvanConfig) -> None:
        self.config = config
        self.device, self.device_reason = resolve_torch_device()
        print(f"[Python] ControllerTrainer device: {self.device} | {self.device_reason}")
        # Bumped: active-balance phase (effort/forward-vel removed from reward,
        # in-episode perturbations, lighter action regularisation). Invalidates
        # old controller checkpoints and resets the stable score.
        # v2: added a LEADING balance term (horizontal COM-drift speed penalty) so
        # the per-step reward is action-dependent within the imagination horizon
        # (v1 was flat -> actor had no gradient, loss pinned at the upright cap).
        # v3: J1 showed v2 still too weak over the trusted ~30-step horizon (slow
        # drift, fall only ~step 76). Added the dominant accumulating term: COM
        # offset from the feet-midpoint (see reward_manager.gd active_balance_v3).
        self.objective_version = "active_balance_v3"

    def _build_world_model(self) -> WorldModelV0:
        env = self.config.env
        train = self.config.train
        model = WorldModelV0(
            proprio_dim=env.proprio_dim,
            action_dim=env.action_dim,
            metrics_dim=env.metrics_dim,
            hidden_dim=train.hidden_dim,
            latent_dim=train.latent_dim,
        ).to(self.device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def _build_controller(self) -> LocomotionController:
        controller_cfg = self.config.controller
        return LocomotionController(
            input_dim=self.config.train.latent_dim,
            hidden_dim=controller_cfg.hidden_dim,
            action_dim=self.config.env.action_dim,
        ).to(self.device)

    def train(self, run_dir: Path, *, world_model_checkpoint: Path) -> dict[str, object]:
        replay_window = select_replay_window(
            self.config.paths.replay_buffer_dir,
            current_run_dir=run_dir,
            window_size=self.config.controller.replay_window_size,
        )
        dataset = ReplaySequenceDataset(replay_window, sequence_length=1)
        if len(dataset) == 0:
            raise ValueError(f"No controller samples available in replay buffer run {run_dir}")

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            collate_fn=collate_sequence_samples,
        )

        world_model = self._build_world_model()
        load_checkpoint(world_model_checkpoint, world_model)

        controller = self._build_controller()
        optimizer = torch.optim.Adam(
            controller.parameters(), lr=self.config.controller.learning_rate
        )
        checkpoint_path = self.config.paths.checkpoints_dir / "controller_v0.pt"
        latest_checkpoint_path = self.config.paths.checkpoints_dir / "controller_v0.latest.pt"
        warm_started = False

        if checkpoint_path.exists():
            checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            checkpoint_metrics = checkpoint_payload.get("metrics", {})
            checkpoint_objective_version = "unknown"
            if isinstance(checkpoint_metrics, dict):
                checkpoint_objective_version = checkpoint_metrics.get("objective_version", "unknown")

            if checkpoint_objective_version == self.objective_version:
                payload = load_checkpoint(checkpoint_path, controller, optimizer)
                warm_started = True
                print(
                    "[Python] Warm-start Controller from %s (epoch %s)"
                    % (checkpoint_path, payload.get("epoch", "?"))
                )
            else:
                print(
                    "[Python] Controller objective changed (%s -> %s). Starting a fresh controller for this regime."
                    % (checkpoint_objective_version, self.objective_version)
                )

        history: list[dict[str, float]] = []
        for epoch in range(self.config.controller.epochs):
            epoch_actor_loss = 0.0
            epoch_critic_loss = 0.0
            batch_count = 0
            for batch in dataloader:
                if batch_count >= self.config.controller.imagination_batches:
                    break
                optimizer.zero_grad(set_to_none=True)
                initial_proprio = batch.proprio[:, 0].to(self.device)
                initial_metrics = batch.metrics[:, 0].to(self.device)
                # targets shape: [horizon, batch_size]
                # values shape: [horizon, batch_size]
                targets, values, raw_actions = imagine_rollout(
                    controller,
                    world_model,
                    proprio=initial_proprio,
                    metrics=initial_metrics,
                    horizon=self.config.controller.imagined_horizon,
                    discount=self.config.controller.discount,
                    action_noise=self.config.controller.imagination_noise,
                )
                
                # Action regularization penalties to prevent policy saturation and rapid changes (bang-bang control)
                action_sat = torch.mean(torch.square(raw_actions))
                action_smooth = torch.mean(torch.square(raw_actions[1:] - raw_actions[:-1]))
                
                # Lightened (was 0.1 each): heavy regularisation pushed the policy
                # toward "do nothing", a passive-standing local optimum. Keep a small
                # smoothness term to avoid pure bang-bang control.
                action_sat_coef = 0.02
                action_smooth_coef = 0.02

                # Actor maximizes target value directly (sum over time steps) and minimizes action penalties
                actor_loss = -targets.mean() + (action_sat_coef * action_sat) + (action_smooth_coef * action_smooth)
                
                # Critic minimizes MSE between predicted value state and the bootstrapped target
                critic_loss = F.mse_loss(values, targets.detach())
                
                loss = actor_loss + critic_loss
                loss.backward()
                optimizer.step()
                epoch_actor_loss += float(actor_loss.detach().cpu())
                epoch_critic_loss += float(critic_loss.detach().cpu())
                batch_count += 1

            history.append(
                {
                    "epoch": float(epoch),
                    "actor_loss": epoch_actor_loss / max(1, batch_count),
                    "critic_loss": epoch_critic_loss / max(1, batch_count),
                }
            )
            print(f"[Python] Night Training Controller | Epoch {epoch+1}/{self.config.controller.epochs} | Actor Loss: {history[-1]['actor_loss']:.4f} | Critic Loss: {history[-1]['critic_loss']:.4f}")

        final_metrics = history[-1] if history else {}
        final_metrics["objective_version"] = self.objective_version
        save_checkpoint(
            destination=latest_checkpoint_path,
            model=controller,
            optimizer=optimizer,
            epoch=self.config.controller.epochs,
            metrics=final_metrics,
        )
        save_checkpoint(
            destination=checkpoint_path,
            model=controller,
            optimizer=optimizer,
            epoch=self.config.controller.epochs,
            metrics=final_metrics,
        )

        return {
            "checkpoint_path": str(checkpoint_path),
            "latest_checkpoint_path": str(latest_checkpoint_path),
            "replay_window": [str(path) for path in replay_window],
            "warm_started": warm_started,
            "objective_version": self.objective_version,
            "history": history,
        }
