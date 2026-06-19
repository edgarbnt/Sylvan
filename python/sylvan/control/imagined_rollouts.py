"""Imagined rollouts through the learned world model."""

from __future__ import annotations

import torch


def imagine_rollout(
    controller,
    world_model,
    *,
    proprio: torch.Tensor,
    metrics: torch.Tensor,
    horizon: int,
    discount: float,
    action_noise: float = 0.1,
    return_rewards: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Imagine trajectory through World Model and compute correct value-bootstrapped returns.

    Uses Dreamer-style TD-lambda returns for robust, mathematically correct gradients.

    With ``return_rewards=True`` the raw per-step predicted-reward sequence
    ``[horizon, batch]`` is appended to the returned tuple. This is purely
    additive — the default 3-tuple unpacking used by training is unchanged — and
    is consumed by the honest imagination->reality transfer eval (see
    ``sylvan.evaluation.transfer``), which calls this with ``action_noise=0.0``.
    """
    rewards = []
    values = []
    states = []
    dones = []
    raw_actions = []
    
    with torch.no_grad():
        encoded = world_model.encoder(proprio.unsqueeze(1))
        state = encoded[:, 0]
        hidden = None

    # Step 1: Rollout trajectory through the dynamics model (rssm/world_model)
    for _ in range(horizon):
        raw_action = controller.act(state)
        raw_actions.append(raw_action)
        # Sylvan has action noise during imagination to prevent deterministic policy saturation
        action = torch.clamp(raw_action + torch.randn_like(raw_action) * action_noise, -1.0, 1.0)
        
        outputs = world_model(
            proprio=None,
            actions=action.unsqueeze(1),
            initial_hidden=hidden,
            encoded_obs=state.unsqueeze(1),
        )
        
        hidden = outputs["hidden"]
        state = outputs["predicted_next_encoded"][:, 0]
        
        predicted_reward = outputs["predicted_reward"][:, 0]
        predicted_done = torch.sigmoid(outputs["predicted_done_logits"][:, 0])
        
        rewards.append(predicted_reward)
        values.append(controller.value(state))
        dones.append(predicted_done)
        states.append(state)

    # Convert lists to tensors: shape [horizon, batch_size]
    rewards_t = torch.stack(rewards, dim=0)
    values_t = torch.stack(values, dim=0)
    dones_t = torch.stack(dones, dim=0)
    
    # Step 2: Compute Bellman target values bootstrapped backwards from the end (TD-lambda or TD-0)
    # G_t = r_t + discount * (1 - done_t) * G_{t+1}
    # G_H = V(s_H)
    horizon_steps = rewards_t.shape[0]
    batch_size = rewards_t.shape[1]
    
    targets = torch.zeros_like(rewards_t)
    
    # Terminal value prediction bootstrap
    last_value = values_t[-1].detach()
    
    for t in reversed(range(horizon_steps)):
        reward = rewards_t[t]
        done = dones_t[t]
        # We bootstrap the next step target
        last_value = reward + discount * (1.0 - done) * last_value
        targets[t] = last_value
        
    # We return the target sum (mean over horizon for actor loss), state value predictions, and the raw actions.
    # This aligns the Actor target directly with Bellman Equation returns.
    # Actor maximizes: targets.mean(dim=0)
    # Critic minimizes MSE between predicted values and bellman targets.
    raw_actions_t = torch.stack(raw_actions, dim=0)
    if return_rewards:
        return targets, values_t, raw_actions_t, rewards_t
    return targets, values_t, raw_actions_t
