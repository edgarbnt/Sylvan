"""Hold-a-constant-pose diagnostic (the decisive body-viability test).

Since the action->angle re-centring fix, action=0 maps to the NEUTRAL standing
pose for EVERY joint (neutral_angle=0), so the constant standing action is simply
all-zeros. QUADRUPED (2026-06-08): the vector is now 12-d (4 legs × [hip_x, hip_z, knee_x])
and proprio is 94-d. This test confirms the body is a holdable upright equilibrium before we
spend compute training. (Verified: the quad stands 401/400, fall 0%, action=0×12.)

This script serves that constant action and measures survival. It tranches the
last fork:
  - If HOLD-NEUTRAL stands (survives the horizon) => the body IS balanceable; the
    whole failure was that exploration/reward was centred on a crouch and the
    policy never had a reason to find the standing action. Clean fix: re-centre
    the action->angle map so action=0 = neutral pose.
  - If HOLD-NEUTRAL still falls => morphology problem (COM/feet/mass): the upright
    pose is not even a holdable equilibrium with the available motor authority.

Usage (from python/):
    python3 -m scripts.diagnose_hold --episodes 3 --max-steps 400
    python3 -m scripts.diagnose_hold --action "0,0,0,0,0,0,0,0,0,0,0,0"
"""

from __future__ import annotations

import argparse
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sylvan.config import SylvanConfig
from sylvan.buffer.reader import iter_episodes
from sylvan.control.policy_server import _PolicyRequestHandler, _PolicyTCPServer
from sylvan.orchestration.day_cycle import prepare_day_run, run_godot_day

# Neutral standing action: action=0 = neutral pose for every joint (straight legs).
# QUADRUPED = 12 DOF (4 legs × [hip_x, hip_z, knee_x]).
NEUTRAL = [0.0] * 12


class _ConstService:
    def __init__(self, action: list[float]) -> None:
        self.action = action

    def predict(self, payload: dict[str, object]) -> list[float]:
        return list(self.action)


@contextmanager
def _serve(action, *, host: str) -> Iterator[dict[str, object]]:
    server = _PolicyTCPServer((host, 0), _PolicyRequestHandler, inference_service=_ConstService(action))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        h, p = server.server_address
        yield {"host": h, "port": p}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _measure(config, action, *, label):
    run_dir = prepare_day_run(config, run_name=f"diag_hold_{label}")
    for f in Path(run_dir).glob("*.jsonl"):  # clear stale episodes (fixed run name)
        f.unlink()
    with _serve(action, host=config.godot.policy_host) as srv:
        run_godot_day(
            config,
            run_dir,
            policy_server_host=srv["host"],
            policy_server_port=srv["port"],
            exploration_noise_initial=0.0,
            exploration_noise_final=0.0,
            collector_mode="policy_server",
        )
    episodes = [ep for ep in iter_episodes(Path(run_dir)) if ep]
    if not episodes:
        return {"label": label, "n": 0, "mean_len": 0.0, "fall_rate": 0.0}
    lengths = [len(ep) for ep in episodes]
    falls = sum(1 for ep in episodes if ep[-1].done)
    n = len(episodes)
    return {
        "label": label,
        "n": n,
        "mean_len": sum(lengths) / n,
        "min_len": min(lengths),
        "max_len": max(lengths),
        "fall_rate": falls / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Hold-a-constant-pose viability test.")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--action", default=None, help="Comma-separated 18-d action; default = neutral stand.")
    args = ap.parse_args()

    config = SylvanConfig()
    config.day.num_episodes = args.episodes
    config.env.max_episode_steps = args.max_steps

    action = NEUTRAL if args.action is None else [float(x) for x in args.action.split(",")]
    print(f"[hold] action = {action}")
    print(f"[hold] {args.episodes} episodes, max_steps={args.max_steps}, no crutch\n")

    r = _measure(config, action, label="NEUTRAL")
    print(
        f"[hold] NEUTRAL | mean_len={r['mean_len']:.1f} "
        f"(min={r.get('min_len',0)} max={r.get('max_len',0)}) "
        f"fall={r['fall_rate']*100:.0f}% (n={r['n']})"
    )

    print("\n=== VERDICT ===")
    survived = r["fall_rate"] < 1.0 or r["mean_len"] >= 0.95 * args.max_steps
    if survived:
        print(
            f"→ La pose neutre TIENT (len {r['mean_len']:.0f}/{args.max_steps}). Le corps EST "
            "equilibrable. Fix propre: re-centrer le mapping action->angle (action=0 = pose neutre) "
            "et/ou initialiser la moyenne de la politique sur l'action neutre."
        )
    else:
        base = 81.0
        if r["mean_len"] > 1.5 * base:
            print(
                f"→ La pose neutre tient BIEN plus longtemps que le fall passif (~81): "
                f"{r['mean_len']:.0f}. Les actions ONT de l'autorite quand on commande la bonne pose "
                "— le probleme etait le centrage du mapping, pas la morphologie."
            )
        else:
            print(
                f"→ Meme en pose neutre le corps tombe (~{r['mean_len']:.0f}). C'est la MORPHOLOGIE "
                "(COM/pieds/masse/autorite moteur): la pose debout n'est pas un equilibre tenable. "
                "Il faut corriger le corps avant tout RL."
            )


if __name__ == "__main__":
    main()
