"""Offline sanity for the Phase-5 CommandPlanner — no Godot.

On held-out WM episodes, at steps where food is actually sensed, run the planner from the
real observation and check the chosen command STEERS TOWARD the food: omega sign should match
the food's egocentric bearing (food on the right → turn right), and vx should be forward. Also
report the planner's own predicted closing of distance. Catches sign/scoring bugs before any
network plumbing.

Usage: python -m scripts.sanity_command_planner --checkpoint ../data/checkpoints/wm_command_v1/wm_best.pt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from sylvan.buffer.wm_dataset import load_wm_episode
from sylvan.control.planning.command_planner import (
    CommandPlanner,
    CommandPlanConfig,
    food_xz_from_radar,
)
from sylvan.models.command_wm import CommandWorldModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--max-steps-per-ep", type=int, default=40)
    args = ap.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    meta = payload["meta"]
    model = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"])
    model.load_state_dict(payload["model"])
    model.eval()
    planner = CommandPlanner(model, CommandPlanConfig())

    episodes = [load_wm_episode(Path(p)) for p in meta["val_episodes"]]
    episodes = [e for e in episodes if e is not None]

    n_food = 0
    n_correct_sign = 0
    n_forward = 0
    closes = []  # predicted (food_dist - pred_min_dist): positive = planner expects to close in
    for ep in episodes:
        obs_all = ep["obs"]  # [N, obs_dim]; radar = obs[:, proprio:-1]
        proprio_dim = meta["proprio_dim"]
        n = obs_all.shape[0]
        for t in range(0, min(n, args.max_steps_per_ep)):
            radar = obs_all[t, proprio_dim:-1].tolist()
            food = food_xz_from_radar(radar)
            if food is None:
                continue
            n_food += 1
            res = planner.plan(obs_all[t], radar)
            vx, om = res["command"]
            fx, _fz = res["food"]  # fx = right component; >0 → food on the right
            # Convention: food on the right (fx>0) should give omega... check empirically which sign
            # turns right. We record agreement of sign(om) with sign(fx); report both signs' rates.
            if fx != 0.0 and om != 0.0 and (om > 0) == (fx > 0):
                n_correct_sign += 1
            if vx > 0:
                n_forward += 1
            if res.get("pred_min_dist") is not None:
                closes.append(res["food_dist"] - res["pred_min_dist"])

    print(f"décisions avec bouffe perçue : {n_food}")
    if n_food:
        print(f"vx forward (>0)            : {100*n_forward/n_food:.0f}%")
        print(f"sign(omega)==sign(food.x)  : {100*n_correct_sign/n_food:.0f}%  "
              f"(une des deux orientations doit dominer → le planner tourne du bon côté)")
        if closes:
            cl = sorted(closes)
            med = cl[len(cl)//2]
            print(f"fermeture distance prédite : médiane {med:+.2f}m "
                  f"(food_dist - pred_min_dist ; >0 = il prévoit de s'approcher)")
            print(f"  fraction où il prévoit de s'approcher : {100*sum(c>0 for c in closes)/len(closes):.0f}%")


if __name__ == "__main__":
    main()
