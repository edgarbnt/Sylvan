"""Phase C du chantier P5 — RE-TRAIN DÉ-CONTAMINÉ des têtes dg (docs/design_purete_hjepa.md §P5).

Les têtes vivantes (douleur̂ v3, P̂ repas, P̂mort) ont des LABELS purs (vécus) mais des FEATURES
dg1/dg2 calculées à travers la lunette clé-apparence green_points. Ici on les ré-entraîne avec
la lunette SAILLANCE (apprise de la conséquence) comme source des features.

Mécanique (G-feat, phase A) : sur les ticks de décision où lunette saillance ≡ lunette verte
(même cardinal ET Hausdorff ≤ 0.05 m — ≥99 % exigé), les dg1/dg2 LOGGÉS sont EXACTEMENT les dg
de la lunette apprise (identité des points perçus) → la « recomputation » est l'identité ; les
décisions DIVERGENTES sont écartées et comptées (pas de reconstruction de repère). Les trainers
VIVANTS sont relancés TELS QUELS sur les runs filtrés (parité train/déploiement).

GATES DE PARITÉ (§P5, pré-enregistrés) :
  - pain′  : AUC CV-4 ≥ 0.874 (vivant 0.894 − 0.02) ;
  - P̂′     : |auc_cv − 0.683| ≤ 0.02 (le vivant jugé 45/8) ;
  - P̂mort′ : AUC CV-4 ≥ 0.819 (vivant 0.839 − 0.02) ET monotonie santé ;
  - G-consist lunette+marges : replay offline EXACT — coût = longueur + W·[(ρ̂−dg1)⁺+(ρ̂−dg2)⁺]
    (longueur = coût_loggé − W·intrusion_loggée ; dg loggés = dg-saillance par identité), remise
    composée recalculée avec les têtes ′ ; bascule ≤ 1.2× l'analytique-vert.
    ⚠️ limite déclarée : les candidats rejoués sont ceux du LOG (tangents posés lunette verte,
    marge 1.4) — le placement tangent vivant (ρ̂+0.4) n'est pas rejouable offline, le juge le mesure.

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python -m scripts.decontaminate_heads [--stage all]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch

from scripts.train_danger_saliency import DangerSaliency, _hausdorff, saliency_points
from scripts.train_sprint_critic import (DEATH_RUNS, DEFAULT_RUNS, SprintCritic, _pain_of,
                                         load_corpus, simulate_choice, sprint_inputs)
from scripts.train_waypoint_pain import PainCritic, _open_text, _text_path
from sylvan.control.waypoint_layer import WaypointConfig, green_points

PAIN_RUNS = ["data/replay_buffer/critic_kin_wpx1", "data/replay_buffer/critic_kin_wpx2",
             "data/replay_buffer/critic_kin_wpx3"]          # corpus du pain_v3 vivant (meta runs)
ALL_RUNS = PAIN_RUNS + DEATH_RUNS                            # wpx∪g24∪spx∪judge∪pure (13 runs)
DECONT_PREFIX = "data/replay_buffer/decont_"
SAL_CKPT = "data/checkpoints/danger_saliency/saliency_best.pt"
_CFG = WaypointConfig()
_INTR_EPS = 0.02


def _load_saliency(path: str) -> tuple[DangerSaliency, float, float]:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = DangerSaliency()
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, float(ck["thr"]), float(ck["rho_hat"])


def _retinas_at(run: Path, ticks: set[int]) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for i, line in enumerate(_open_text(run / "ep_0000.jsonl")):
        if i in ticks:
            try:
                out[i] = json.loads(line)["wm"]["retina0"]
            except json.JSONDecodeError:
                continue
    return out


def stage_filter(sal: DangerSaliency, thr: float) -> None:
    """Matérialise les runs filtrés : decisions.jsonl restreint aux ticks lunette-identiques,
    symlinks vers le BC/godot.log du run source. Idempotent, chemins scопés (pas de glob)."""
    for run in ALL_RUNS:
        src = Path(run)
        dst = Path(DECONT_PREFIX + src.name.removeprefix("critic_kin_"))
        dst.mkdir(parents=True, exist_ok=True)
        decs = [json.loads(line) for line in _open_text(src / "decisions.jsonl")]
        rets = _retinas_at(src, {d["tick"] for d in decs})
        kept, dropped = [], 0
        for d in decs:
            ret = rets.get(d["tick"])
            if ret is None:
                dropped += 1
                continue
            gp, sp = green_points(ret), saliency_points(sal, ret, thr)
            if len(gp) == len(sp) and _hausdorff(gp, sp) <= 0.05:
                kept.append(d)
            else:
                dropped += 1
        with open(dst / "decisions.jsonl", "w") as f:
            for d in kept:
                f.write(json.dumps(d) + "\n")
        for name in ("ep_0000.jsonl", "godot.log"):
            p = _text_path(src / name)
            link = dst / p.name
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(p.resolve(), link)
        print(f"[decont] {src.name}: {len(kept)} décisions gardées, {dropped} écartées "
              f"({100 * dropped / max(len(decs), 1):.1f}%)")


def stage_train() -> None:
    """Relance les trainers VIVANTS tels quels sur les runs filtrés (mêmes seeds/iters)."""
    def dr(runs: list[str]) -> list[str]:
        return [DECONT_PREFIX + Path(r).name.removeprefix("critic_kin_") for r in runs]

    env = dict(os.environ, PYTHONPATH="python")
    cmds = [
        [sys.executable, "-m", "scripts.train_waypoint_pain", "--runs", *dr(PAIN_RUNS),
         "--out", "data/checkpoints/waypoint_pain_decont"],
        [sys.executable, "-m", "scripts.train_sprint_critic", "--runs", *dr(DEFAULT_RUNS),
         "--pain", "data/checkpoints/waypoint_pain_decont/pain_best.pt",
         "--out", "data/checkpoints/sprint_critic_decont"],
        [sys.executable, "-m", "scripts.train_sprint_critic", "--head", "death",
         "--runs", *dr(DEATH_RUNS),
         "--pain", "data/checkpoints/waypoint_pain_decont/pain_best.pt",
         "--out", "data/checkpoints/sprint_death_decont"],
    ]
    for cmd in cmds:
        print(f"[decont] → {' '.join(cmd[2:6])} …", flush=True)
        subprocess.run(cmd, check=True, env=env)


def _sal_replay_choice(r: dict, rho: float, critic: SprintCritic, pain: PainCritic,
                       kappa: float, drain: float, restore: float) -> bool | None:
    """Rejoue decide() sous la lunette saillance : coût = longueur + W·(ρ̂−dg)⁺ par leg, remise
    composée (têtes ′). Longueur EXACTE = coût_loggé − W·intr_loggée. None si intr NaN (skip)."""
    intr_g = r["intr_all"]
    if any(v != v for v in intr_g):
        return None
    feats = r["feats_all"]
    dg1 = [f[7] * 10.0 for f in feats]
    dg2 = [f[8] * 10.0 for f in feats]
    intr_s = [max(0.0, rho - a) + max(0.0, rho - b) for a, b in zip(dg1, dg2)]
    length = [c - _CFG.block_weight * i for c, i in zip(r["costs"], intr_g)]
    costs = [ln + _CFG.block_weight * i for ln, i in zip(length, intr_s)]
    if critic is not None and r["target"] in ("food", "water"):
        pains = _pain_of(pain, feats)
        x = sprint_inputs(feats, (r["e"], r["t"], r["h"]), pains)
        drive = r["e"] if r["target"] == "food" else r["t"]
        ben = min(restore, 100.0 - drive) / drain
        with torch.no_grad():
            p = critic.p(x)
        costs = [c - min(_CFG.block_weight * intr_s[i],
                         0.02 * max(0.0, float(p[i]) * ben - kappa * pains[i] * 100.0))
                 if intr_s[i] > 0.0 else c
                 for i, c in enumerate(costs)]
    best_i = min(range(1, len(costs)), key=lambda i: costs[i])
    chosen = best_i if costs[best_i] < costs[0] * (1.0 - _CFG.hysteresis) else 0
    return intr_s[chosen] > _INTR_EPS


def stage_gates(rho: float) -> None:
    pk = torch.load("data/checkpoints/waypoint_pain_decont/pain_best.pt",
                    map_location="cpu", weights_only=True)
    sk = torch.load("data/checkpoints/sprint_critic_decont/sprint_best.pt",
                    map_location="cpu", weights_only=True)
    dk = torch.load("data/checkpoints/sprint_death_decont/death_best.pt",
                    map_location="cpu", weights_only=True)
    g_pain = pk["auc_cv"] >= 0.874
    g_p = abs(sk["auc_cv"] - 0.683) <= 0.02
    g_death = dk["auc_cv"] >= 0.819 and bool(dk["mono_health"])

    # G-consist lunette+marges : paires consécutives intra-poursuite des 6 runs analytiques.
    pain = PainCritic()
    pain.load_state_dict(pk["state_dict"])
    pain.eval()
    critic = SprintCritic()
    critic.load_state_dict(sk["state_dict"])
    critic.eval()
    rows = load_corpus([DECONT_PREFIX + Path(r).name.removeprefix("critic_kin_")
                        for r in DEFAULT_RUNS])
    by_run: dict[str, list[dict]] = {}
    for r in rows:
        by_run.setdefault(r["run"], []).append(r)
    flips_g = flips_s = n_pairs = 0
    kw = dict(rho=rho, critic=critic, pain=pain, kappa=float(sk["kappa_data"]),
              drain=float(sk["drain"]), restore=float(sk["restore"]))
    for seq in by_run.values():
        seq.sort(key=lambda r: r["tick"])
        for a, b in zip(seq, seq[1:]):
            if a["target"] != b["target"] or b["tick"] - a["tick"] > 60:
                continue
            sa, sb = _sal_replay_choice(a, **kw), _sal_replay_choice(b, **kw)
            if sa is None or sb is None:
                continue
            n_pairs += 1
            flips_g += simulate_choice(a, None, None) != simulate_choice(b, None, None)
            flips_s += sa != sb
    rate_g = flips_g / max(n_pairs, 1)
    rate_s = flips_s / max(n_pairs, 1)
    g_consist = rate_s <= 1.2 * rate_g + 1e-9

    print(f"\n[decont] === GATES DE PARITÉ (pré-enregistrés §P5) ===")
    print(f"[decont] pain′   : AUC {pk['auc_cv']:.3f} ≥ 0.874 → {'✅' if g_pain else '❌'} "
          f"(vivant 0.894 ; mono={pk['monotone']})")
    print(f"[decont] P̂′      : AUC {sk['auc_cv']:.3f} ∈ 0.683±0.02 → {'✅' if g_p else '❌'} "
          f"(G-mono trainer : santé {sk['q_by_health_deep']}, prof {sk['q_by_depth_wounded']})")
    print(f"[decont] P̂mort′  : AUC {dk['auc_cv']:.3f} ≥ 0.819, mono={dk['mono_health']} → "
          f"{'✅' if g_death else '❌'}")
    print(f"[decont] G-consist lunette+ρ̂ : {n_pairs} paires | vert {100 * rate_g:.1f}% vs "
          f"saillance {100 * rate_s:.1f}% (≤1.2×) → {'✅' if g_consist else '❌'}")
    verdict = g_pain and g_p and g_death and g_consist
    print(f"[decont] {'✅ PARITÉ TENUE → phase D (juge closed-loop) licenciée' if verdict else '❌ GATE ÉCHOUÉ → diagnostiquer sur trace (budget re-train : 1 par tête)'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saliency", default=SAL_CKPT)
    ap.add_argument("--stage", choices=("all", "filter", "train", "gates"), default="all")
    args = ap.parse_args()
    sal, thr, rho = _load_saliency(args.saliency)
    if args.stage in ("all", "filter"):
        stage_filter(sal, thr)
    if args.stage in ("all", "train"):
        stage_train()
    if args.stage in ("all", "gates"):
        stage_gates(rho)


if __name__ == "__main__":
    main()
