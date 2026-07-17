"""JUGE 2-BRAS du Gate-capacité (docs/design_gate_capacite.md §gates, PRÉ-ENREGISTRÉ) :
« survit à un changement de monde », prouvé en vies via un swap d'apparence food en cours de vie.

Lit le BC log par-tick (SYLVAN_BC_LOG, `scripts.build_typed_slots`-style : obs.energy/thirst/health
par ligne) des DEUX bras (contrôle = WM typé STATIQUE + swap ; appris = WM typé + re-mesure
périodique + swap), MÊME swap dans les deux bras (T=700, grâce 200, fenêtre tardive [900,fin]).

Le harnais de collecte (SYLVAN_COLLECTOR_MODE=policy_server) n'envoie JAMAIS de message "reset"
TCP entre deux vies (vérifié, python/sylvan/control/remeasure.py) : les 24 vies d'un run tombent
dans UN SEUL fichier ep_0000.jsonl. Ce module SEGMENTE donc lui-même les vies, à partir du
GODOT.LOG (vérité-terrain : `[Godot] Episode N | Step S | ...]`, écrit toutes les 10 pas par
`collect_critic_corpus_kin.sh`) plutôt que par saut de drive (LIFE_JUMP) : un décès par CHUTE
(agent_instance.has_fallen) ne fait pas nécessairement sauter énergie/soif/santé de plus de
LIFE_JUMP au respawn suivant → l'heuristique par saut FUSIONNE parfois deux vies consécutives
en une seule (mesuré sur un run réel : 22 segments détectés au lieu de 24, avec 2 « vies »
> 5000 pas — impossible, SYLVAN_MAX_EPISODE_STEPS=3000). Le godot.log donne l'index d'épisode
EXACT ; la longueur de chaque épisode (dernier Step vu +1) somme au total du BC log à ±1 pas/10
pas près (granularité du log) — négligeable devant les fenêtres de 700/900 pas. Puis classe
chaque REPAS (relief énergie) par fenêtre (pré-swap [0,T) vs tardive [T+grâce,fin]) selon son
tick DANS LA VIE (pas global).

Usage :
    PYTHONPATH=python env_pytorch_3.12/bin/python -m scripts.judge_gate_capacite \
        --control data/replay_buffer/critic_kin_gcctl1 data/replay_buffer/critic_kin_gcctl2 \
        --learned data/replay_buffer/critic_kin_gclrn1 data/replay_buffer/critic_kin_gclrn2
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

from scripts.build_typed_slots import RELIEF
from scripts.train_danger_saliency import LIFE_JUMP

_EP_RE = re.compile(r"\[Godot\] Episode (\d+) \| Step (\d+) \| Energy: [\d.]+ \| Thirst: [\d.]+ \| Health: [\d.]+")

SWAP_TICK = 700     # pas DANS LA VIE où food bascule d'apparence (déclaré, cf food_manager.gd)
GRACE = 200         # pas de grâce après le swap avant que la fenêtre tardive ne commence
LATE_START = SWAP_TICK + GRACE   # 900

G_SWAP_CONTROL_MAX_RATIO = 0.3    # contrôle DOIT s'effondrer : taux tardif <= 0.3x taux pré-swap
G_CAPACITE_MIN_RATIO = 0.6        # appris DOIT récupérer : taux tardif >= 0.6x taux pré-swap
KILL_MIN_PRE_TICKS = 2000         # sous ce total de pas-vécus en pré-swap poolés : pas assez de
                                   # signal pour juger (INCONCLUSIF, pas un verdict forcé)


def _open(path: Path):
    if path.exists():
        return open(path, errors="ignore")
    gz = path.with_suffix(path.suffix + ".gz")
    if gz.exists():
        return gzip.open(gz, "rt", errors="ignore")
    raise FileNotFoundError(path)


def _episode_lengths_from_godot_log(path: Path) -> list[int] | None:
    """Longueur (en pas) de chaque épisode, reconstruite depuis les lignes périodiques `[Godot]
    Episode N | Step S | ...` (vérité-terrain, écrites tous les 10 pas par
    collect_critic_corpus_kin.sh). Dernier Step vu par épisode +1 = estimation (±9 pas de
    granularité — négligeable devant les fenêtres 700/900). None si le fichier est absent."""
    if not path.exists():
        return None
    last_step: dict[int, int] = {}
    with open(path, errors="ignore") as f:
        for line in f:
            m = _EP_RE.search(line)
            if m:
                ep, step = int(m.group(1)), int(m.group(2))
                last_step[ep] = max(last_step.get(ep, 0), step)
    if not last_step:
        return None
    n_ep = max(last_step) + 1
    return [last_step.get(e, 0) + 1 for e in range(n_ep)]


def _life_segments(run_dir: Path) -> list[list[tuple[float, float, float]]]:
    """Un run_dir -> liste de VIES, chaque vie = liste de (energy, thirst, health) par tick DANS
    LA VIE. Segmente par LONGUEUR reconstruite depuis godot.log (vérité-terrain -- un décès par
    CHUTE peut ne pas faire sauter les drives de plus de LIFE_JUMP, fusionnant deux vies sous
    l'ancienne heuristique). Fallback SAUT (LIFE_JUMP) si godot.log est absent (usage hors de ce
    harnais -- moins fiable, à ne pas préférer quand godot.log existe)."""
    points: list[tuple[float, float, float]] = []
    for ep_path in sorted(run_dir.glob("ep_*.jsonl*")):
        stem = ep_path.name.split(".")[0]
        with _open(run_dir / (stem + ".jsonl")) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                obs = r["obs"]
                points.append((float(obs["energy"]), float(obs["thirst"]), float(obs["health"])))

    lengths = _episode_lengths_from_godot_log(run_dir / "godot.log")
    if lengths is not None:
        lives: list[list[tuple[float, float, float]]] = []
        i = 0
        for length in lengths:
            lives.append(points[i:i + length])
            i += length
        if i < len(points):    # reliquat (granularité du log) -> rattaché à la dernière vie
            lives[-1].extend(points[i:])
        return [life for life in lives if life]

    # fallback : segmentation heuristique par saut de drive (moins fiable, cf docstring)
    lives, cur, prev = [], [], None
    for pt in points:
        if prev is not None and max(pt[0] - prev[0], pt[1] - prev[1], pt[2] - prev[2]) > LIFE_JUMP:
            lives.append(cur)
            cur = []
        cur.append(pt)
        prev = pt
    if cur:
        lives.append(cur)
    return lives


def _windowed_rate(lives: list[list[tuple[float, float, float]]]) -> dict:
    """Poole les REPAS (relief énergie) et les pas-vécus par fenêtre sur toutes les vies."""
    meals_pre = ticks_pre = meals_late = ticks_late = 0
    for life in lives:
        n = len(life)
        ticks_pre += min(SWAP_TICK, n)
        ticks_late += max(0, n - LATE_START)
        for k in range(1, n):
            if life[k][0] - life[k - 1][0] <= RELIEF:
                continue
            if k < SWAP_TICK:
                meals_pre += 1
            elif k >= LATE_START:
                meals_late += 1
    rate_pre = meals_pre / ticks_pre if ticks_pre else float("nan")
    rate_late = meals_late / ticks_late if ticks_late else float("nan")
    return {"n_lives": len(lives), "meals_pre": meals_pre, "ticks_pre": ticks_pre,
            "meals_late": meals_late, "ticks_late": ticks_late,
            "rate_pre": rate_pre, "rate_late": rate_late,
            "ratio": (rate_late / rate_pre) if rate_pre else float("nan")}


def _arm_stats(run_dirs: list[Path]) -> dict:
    per_seed = [_windowed_rate(_life_segments(d)) for d in run_dirs]
    pooled_lives = sum(s["n_lives"] for s in per_seed)
    meals_pre = sum(s["meals_pre"] for s in per_seed)
    ticks_pre = sum(s["ticks_pre"] for s in per_seed)
    meals_late = sum(s["meals_late"] for s in per_seed)
    ticks_late = sum(s["ticks_late"] for s in per_seed)
    rate_pre = meals_pre / ticks_pre if ticks_pre else float("nan")
    rate_late = meals_late / ticks_late if ticks_late else float("nan")
    return {"per_seed": per_seed, "n_lives": pooled_lives,
            "meals_pre": meals_pre, "ticks_pre": ticks_pre,
            "meals_late": meals_late, "ticks_late": ticks_late,
            "rate_pre": rate_pre, "rate_late": rate_late,
            "ratio": (rate_late / rate_pre) if rate_pre else float("nan")}


def judge(control_dirs: list[Path], learned_dirs: list[Path]) -> dict:
    control = _arm_stats(control_dirs)
    learned = _arm_stats(learned_dirs)
    noise_margin = abs(control["per_seed"][0]["rate_late"] - control["per_seed"][-1]["rate_late"]) \
        if len(control["per_seed"]) > 1 else 0.0
    inconclusive = control["ticks_pre"] < KILL_MIN_PRE_TICKS or learned["ticks_pre"] < KILL_MIN_PRE_TICKS
    g_swap_control = (not inconclusive) and control["ratio"] <= G_SWAP_CONTROL_MAX_RATIO
    g_capacite = (not inconclusive) and learned["ratio"] >= G_CAPACITE_MIN_RATIO \
        and (learned["rate_late"] - control["rate_late"]) > noise_margin
    return {"control": control, "learned": learned, "noise_margin": noise_margin,
            "inconclusive": inconclusive, "g_swap_control": g_swap_control,
            "g_capacite": g_capacite, "pass": g_swap_control and g_capacite}


def _fmt_arm(name: str, s: dict) -> str:
    lines = [f"[gate-capacite] {name} : {s['n_lives']} vies poolées"]
    for i, seed_s in enumerate(s["per_seed"], start=1):
        lines.append(f"  seed{i}: {seed_s['n_lives']} vies, pré {seed_s['meals_pre']}/{seed_s['ticks_pre']}"
                      f" (taux={seed_s['rate_pre']:.5f}), tardif {seed_s['meals_late']}/{seed_s['ticks_late']}"
                      f" (taux={seed_s['rate_late']:.5f}), ratio={seed_s['ratio']:.3f}")
    lines.append(f"  POOLÉ: pré {s['meals_pre']}/{s['ticks_pre']} (taux={s['rate_pre']:.5f}), "
                 f"tardif {s['meals_late']}/{s['ticks_late']} (taux={s['rate_late']:.5f}), "
                 f"ratio={s['ratio']:.3f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", nargs="+", required=True)
    ap.add_argument("--learned", nargs="+", required=True)
    args = ap.parse_args()
    res = judge([Path(p) for p in args.control], [Path(p) for p in args.learned])
    print(_fmt_arm("BRAS CONTRÔLE (statique)", res["control"]))
    print(_fmt_arm("BRAS APPRIS (re-mesure)", res["learned"]))
    print(f"\n[gate-capacite] marge de bruit (contrôle seed1 vs seed2, taux tardif) = "
          f"{res['noise_margin']:.5f}")
    if res["inconclusive"]:
        print(f"[gate-capacite] ⚠️ INCONCLUSIF : moins de {KILL_MIN_PRE_TICKS} pas-vécus pré-swap "
              f"poolés dans un bras -> pas assez de signal pour juger.")
    print(f"[gate-capacite] G-swap-control (contrôle s'effondre, ratio <= {G_SWAP_CONTROL_MAX_RATIO}) : "
          f"{'✅' if res['g_swap_control'] else '❌'} (ratio={res['control']['ratio']:.3f})")
    print(f"[gate-capacite] G-capacité (appris récupère, ratio >= {G_CAPACITE_MIN_RATIO} ET "
          f"tardif appris - tardif contrôle > bruit) : {'✅' if res['g_capacite'] else '❌'} "
          f"(ratio={res['learned']['ratio']:.3f}, Δtardif={res['learned']['rate_late'] - res['control']['rate_late']:.5f})")
    print(f"\n[gate-capacite] {'✅✅✅ PASS' if res['pass'] else '❌ ÉCHEC'} — "
          f"{'la re-mesure périodique (embryon jour/nuit) est promue' if res['pass'] else 'négatif à commiter, diagnostiquer sur trace'}")


if __name__ == "__main__":
    main()
