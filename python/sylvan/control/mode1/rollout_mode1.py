"""Mode-1 (Task 8) : buffer JSONL espace-commande → RolloutBatch PPO.

Le buffer est produit par `scripts.serve_mode1_collect` : une ligne = une MACRO-transition (cadence de
replan), schéma
    {"proprio":[132],"retina":[144],"energy","thirst",
     "command_raw":[z0,z1],"command_act":[vx,om],"reward","steps","done","truncated"}.
Un épisode = lignes CONTIGUËS ; une ligne avec `done` OU `truncated` = True TERMINE l'épisode. Il n'y a
PAS de next_obs stocké (contrairement au buffer per-tick de `ppo.rollout`).

On reconstruit les tokens-pulsion par transition via `build_tokens` (perception rétine → tokens[D,38]),
on EMPAQUETTE l'obs (cat([proprio, tokens.flatten])) à la convention de `DriveSymmetricPolicy`, on
recalcule `old_log_prob`/`values` sous la politique BEHAVIOR (celle qui a collecté ; le z stocké EST le z
échantillonné → exact), puis GAE(λ) par épisode avec le bon bootstrap terminal/tronqué.

Bootstrap (documenté) :
  - transitions INTÉRIEURES : V[t+1] vient de la même séquence de `values` (géré par compute_gae) ;
  - DERNIÈRE transition d'un épisode :
      * `done`      → last_value = 0                          (fin réelle : pas de futur) ;
      * `truncated` → last_value = V(obs[-1])  (self-bootstrap : le schéma n'a pas de next_obs ; la valeur
                      de l'état COURANT est un proxy proche de V(next_obs) sur une macro-transition) ;
      * épisode PENDANT (buffer coupé sans terminal) → traité comme tronqué (bootstrap), même logique.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from sylvan.control.mode1.obs import build_tokens
from sylvan.control.ppo.rollout import RolloutBatch, compute_gae


def _iter_lines(buffer_dir: Path):
    """Rend les lignes JSON de tous les part-*.jsonl, triés par nom puis dans l'ordre du fichier."""
    for path in sorted(Path(buffer_dir).glob("part-*.jsonl")):
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                yield json.loads(raw)


def _split_episodes(lines: list[dict]) -> list[list[dict]]:
    """Découpe en épisodes : une ligne `done` OU `truncated` termine (inclus) l'épisode courant.
    Un reliquat sans terminal (buffer coupé) forme un dernier épisode 'pendant' (traité tronqué en aval)."""
    episodes: list[list[dict]] = []
    cur: list[dict] = []
    for ln in lines:
        cur.append(ln)
        if bool(ln.get("done")) or bool(ln.get("truncated")):
            episodes.append(cur)
            cur = []
    if cur:  # reliquat non terminé → épisode pendant
        episodes.append(cur)
    return episodes


def build_rollout_batch_mode1(
    buffer_dir: Path,
    behavior_policy,
    *,
    gamma: float,
    lam: float,
    device: str = "cpu",
) -> tuple[RolloutBatch | None, dict[str, float]]:
    lines = list(_iter_lines(Path(buffer_dir)))
    episodes = _split_episodes(lines)

    obs_chunks, act_chunks, olp_chunks, adv_chunks, ret_chunks, val_chunks = ([] for _ in range(6))
    n_done = n_truncated = 0
    episode_step_lengths: list[int] = []   # ticks survécus par épisode (survie = Σ steps)
    total_reward = 0.0

    for ep in episodes:
        # -- Empaqueter l'obs par transition (rétine → tokens → cat) --------------------------------
        obs_rows = []
        for tr in ep:
            proprio, tokens, _meta = build_tokens(
                {"proprio": tr["proprio"], "retina": tr["retina"],
                 "energy": tr["energy"], "thirst": tr["thirst"]}
            )
            obs_rows.append(behavior_policy.pack_obs(proprio, tokens))
        obs = torch.stack(obs_rows).to(device=device, dtype=torch.float32)          # [T, P+D*38]
        acts = torch.tensor([tr["command_raw"] for tr in ep], dtype=torch.float32, device=device)  # z BRUT
        rews = torch.tensor([float(tr["reward"]) for tr in ep], dtype=torch.float32, device=device)
        # done PAR transition : seule la dernière peut être vraie (frontière d'épisode)
        dones = torch.tensor([1.0 if bool(tr.get("done")) else 0.0 for tr in ep],
                             dtype=torch.float32, device=device)

        with torch.no_grad():
            old_log_prob, _ent, values = behavior_policy.evaluate_actions(obs, acts)
            last = ep[-1]
            if bool(last.get("done")):
                last_value = torch.zeros((), dtype=torch.float32, device=device)   # fin réelle
                n_done += 1
            else:
                # tronqué OU épisode pendant : self-bootstrap sur l'état courant (pas de next_obs au schéma)
                last_value = behavior_policy.value(obs[-1:])[0]
                if bool(last.get("truncated")):
                    n_truncated += 1

        adv, ret = compute_gae(rews, values, dones, last_value, gamma=gamma, lam=lam)
        obs_chunks.append(obs); act_chunks.append(acts); olp_chunks.append(old_log_prob)
        adv_chunks.append(adv); ret_chunks.append(ret); val_chunks.append(values)

        episode_step_lengths.append(int(sum(int(tr.get("steps", 1)) for tr in ep)))
        total_reward += float(rews.sum().item())

    num_transitions = int(sum(c.shape[0] for c in obs_chunks))
    stats = {
        "n_transitions": float(num_transitions),
        "n_episodes": float(len(episodes)),
        "n_done": float(n_done),
        "n_truncated": float(n_truncated),
        "mean_episode_steps": float(sum(episode_step_lengths) / len(episode_step_lengths))
        if episode_step_lengths else 0.0,
        "mean_reward": float(total_reward / num_transitions) if num_transitions else 0.0,
    }
    if num_transitions == 0:
        return None, stats

    advantages = torch.cat(adv_chunks)
    # std population (unbiased=False) : défini pour N=1 (→0, pas de NaN) ; en vrai run N≫1 → identique.
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    batch = RolloutBatch(
        obs=torch.cat(obs_chunks),
        actions=torch.cat(act_chunks),
        old_log_prob=torch.cat(olp_chunks),
        advantages=advantages,
        returns=torch.cat(ret_chunks),
        values=torch.cat(val_chunks),
    )
    return batch, stats
