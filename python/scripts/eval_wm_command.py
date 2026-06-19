"""Phase 4 milestone: open-loop prediction accuracy of the command-space world model.

For held-out episodes (the val split saved in the checkpoint), encode ONE real observation,
dream forward 50/100 steps under the TRUE command sequence (never re-reading real obs),
integrate the predicted body-frame displacements into a world trajectory, and compare with
the real torso chain.

GO criteria (Phase 5): median final position error < 0.5 m @ 50 steps (and < 1.2 m @ 100),
median final yaw error < 20 deg.

Usage: python -m scripts.eval_wm_command --checkpoint data/checkpoints/wm_command_v1/wm_best.pt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from sylvan.buffer.wm_dataset import load_wm_episode
from sylvan.models.command_wm import DISPLACEMENT_SCALE, CommandWorldModel


def integrate(disp: torch.Tensor, pose0: torch.Tensor) -> torch.Tensor:
    """disp [T,3] scaled body-frame deltas + pose0 [3] (x, z, yaw) -> poses [T+1, 3]."""
    poses = [pose0.clone()]
    x, z, yaw = (float(v) for v in pose0)
    for t in range(disp.shape[0]):
        d_fwd, d_lat, d_yaw = (float(v) / DISPLACEMENT_SCALE for v in disp[t])
        s, c = math.sin(yaw), math.cos(yaw)
        x += d_fwd * s + d_lat * c
        z += d_fwd * c - d_lat * s
        yaw += d_yaw
        poses.append(torch.tensor([x, z, yaw]))
    return torch.stack(poses)


def yaw_err_deg(a: float, b: float) -> float:
    d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(math.degrees(d))


def main() -> None:
    ap = argparse.ArgumentParser(description="Open-loop accuracy of the command-space WM.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 10, 50, 100])
    ap.add_argument("--window-stride", type=int, default=50)
    args = ap.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    meta = payload["meta"]
    model = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                              predictor_arch=meta.get("predictor_arch", "shallow"))
    model.load_state_dict(payload["model"])
    model.eval()

    episodes = [load_wm_episode(Path(p)) for p in meta["val_episodes"]]
    episodes = [e for e in episodes if e is not None]
    print(f"[eval] {len(episodes)} épisodes val | horizons {args.horizons}")

    hmax = max(args.horizons)
    results: dict[int, dict[str, list[float]]] = {
        h: {"pos": [], "yaw": [], "still": [], "travel": []} for h in args.horizons
    }
    for ep in episodes:
        n = ep["obs"].shape[0]
        for start in range(0, n - hmax, args.window_stride):
            obs0 = ep["obs"][start : start + 1]
            cmds = ep["command"][start : start + hmax].unsqueeze(0)
            out = model.rollout_open_loop(obs0, cmds)
            pred = integrate(out["predicted_displacement"][0], ep["torso"][start])
            true = ep["torso"][start : start + hmax + 1]
            for h in args.horizons:
                dpos = float(torch.linalg.norm(pred[h, :2] - true[h, :2]))
                still = float(torch.linalg.norm(true[h, :2] - true[0, :2]))  # baseline: "ne bouge pas"
                results[h]["pos"].append(dpos)
                results[h]["yaw"].append(yaw_err_deg(float(pred[h, 2]), float(true[h, 2])))
                results[h]["still"].append(still)
                results[h]["travel"].append(still)

    def q(v: list[float], p: float) -> float:
        s = sorted(v)
        return s[min(len(s) - 1, int(p * len(s)))]

    print(f"\n{'h':>4} | {'pos med':>8} {'pos p90':>8} | {'yaw med':>8} {'yaw p90':>8} | {'trajet med':>10} | n")
    for h in args.horizons:
        r = results[h]
        print(
            f"{h:>4} | {q(r['pos'],0.5):>7.3f}m {q(r['pos'],0.9):>7.3f}m | "
            f"{q(r['yaw'],0.5):>7.1f}° {q(r['yaw'],0.9):>7.1f}° | {q(r['travel'],0.5):>9.3f}m | {len(r['pos'])}"
        )
    if 50 in results:
        ok = q(results[50]["pos"], 0.5) < 0.5 and q(results[50]["yaw"], 0.5) < 20.0
        print(f"\nJALON @50 pas (pos<0.5m, yaw<20°): {'ATTEINT ✅' if ok else 'PAS ENCORE ❌'}")


if __name__ == "__main__":
    main()
