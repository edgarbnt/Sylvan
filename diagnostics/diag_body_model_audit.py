"""AUDIT DE PÉREMPTION — Phase 1 : le MODÈLE DU CORPS interne du planner est-il encore le bon ?

CE QUE CE DIAG N'EST PAS. Ce n'est pas un A/B de préférence. Les constantes auditées ici encodent
un FAIT PHYSIQUE MESURABLE du corps (vitesse, vitesse de virage, métabolisme, durée de vie). Il y a
une bonne réponse, et une valeur fausse est un BUG, pas un arbitrage de réglage. D'où : verdict
GRATUIT sur corpus existants, zéro run (CLAUDE.md §1 : gater le cher derrière le pas-cher).

POURQUOI MAINTENANT. `far_align` a montré qu'une constante calibrée sur le corps à PATTES et jamais
revue après le pivot cinématique peut devenir activement nuisible. Deux constantes du modèle du
corps ne sont overridées par AUCUN harnais (vérifié 2026-07-21) — elles tournent donc partout sur
leur défaut d'origine :
    nominal_speed  = 0.02   m/pas   (command_planner.py:101)
    surv_turn_rate = 0.015  rad/pas (command_planner.py:121, commenté « hexapode ~25-50°/s »)

OÙ ELLES AGISSENT (command_planner.py) :
    turn_f = |bearing| / surv_turn_rate      -> nb de pas imaginés pour se réorienter
    t_atteinte = distance / nominal_speed    -> puis coût = t_atteinte x resource_drain
  Donc `nominal_speed` trop GRAND => le planner se croit plus rapide qu'il n'est => il SOUS-ESTIME
  le coût métabolique d'aller loin. `surv_turn_rate` trop PETIT => il SUR-ESTIME le coût de tourner
  => il préfère ce qui est déjà devant. Les deux biaisent la DÉCISION, pas seulement l'estimation.

CLI :
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_body_model_audit.py \
        --runs data/replay_buffer/A data/replay_buffer/B [--selfcheck]
"""
from __future__ import annotations

import argparse
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guards import CONSUME_JUMP, TELEPORT_M, _reset_ticks, _ticks  # noqa: E402

TOL = 0.15          # au-delà de 15 % d'écart declaré/mesuré : PÉRIMÉ
OMEGA_MAX = 0.55    # |omega| commandé au-delà duquel on considère un virage « plein régime »
                    # (grille de candidats du planner : -0.6 -0.3 0 +0.3 +0.6)

# Valeurs DÉCLARÉES dans le code, avec la façon dont elles sont servies en vrai.
DECLARED = {
    "nominal_speed":   (0.02,   "m/pas",   "command_planner.py:101 — JAMAIS overridé"),
    "surv_turn_rate":  (0.015,  "rad/pas", "command_planner.py:121 — JAMAIS overridé"),
    "resource_drain":  (0.0005, "/pas",    "défaut code 0.0016, harnais 0.0005"),
    "resource_restore": (0.4,   "niveau",  "défaut code 0.5, harnais 0.4"),
}
# ⚠️ `surv_horizon` (3000) a d'abord été audité ici puis RETIRÉ des verdicts (2026-07-21) : c'est un
# PLAFOND de la simulation imaginée, pas une prédiction de durée de vie. Le comparer à la durée de
# vie mesurée produisait un faux « PÉRIMÉ x0.67 » — un plafond plus large que les vies typiques est
# normal, voire souhaitable. Il n'a pas de vérité-terrain mesurable => il relève des PRÉFÉRENCES
# (catégorie b, décidable seulement par A/B), pas du modèle du corps. La distribution des vies reste
# affichée comme CONTEXTE.


def _wrap(a: float) -> float:
    """Différence d'angle ramenée dans [-pi, pi] — sans ça, un passage ±pi fabrique un faux virage."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def measure_body_model(run: str) -> dict:
    T = _ticks(run)
    if not T:
        raise SystemExit(f"corpus vide : {run}")
    E = [float(t["obs"]["energy"]) for t in T]
    TH = [float(t["obs"]["thirst"]) for t in T]
    resets = _reset_ticks(E, TH)

    disp: list[float] = []
    dyaw_all: list[float] = []
    dyaw_fast: list[float] = []
    for i in range(1, len(T)):
        if i in resets:
            continue
        pa, pb = T[i - 1]["wm"].get("torso0"), T[i]["wm"].get("torso0")
        if not pa or not pb:
            continue
        d = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
        if d >= TELEPORT_M:                      # respawn résiduel : jamais du mouvement
            continue
        disp.append(d)
        dy = abs(_wrap(pb[2] - pa[2]))
        if dy < math.pi / 2:                     # garde anti-téléportation angulaire
            dyaw_all.append(dy)
            cmd = T[i - 1]["wm"].get("cmd")
            if cmd and abs(float(cmd[1])) >= OMEGA_MAX:
                dyaw_fast.append(dy)

    def drain(series: list[float]) -> float:
        drops = [a - b for a, b in zip(series, series[1:]) if 0 < a - b < 1.0]
        return st.median(drops) if drops else float("nan")

    def restore(series: list[float]) -> float:
        j = [series[i] - series[i - 1] for i in range(1, len(series))
             if series[i] - series[i - 1] > CONSUME_JUMP and i not in resets]
        return st.median(j) if j else float("nan")

    lives, start = [], 0
    for r in sorted(resets):
        lives.append(r - start)
        start = r
    lives.append(len(T) - start)
    lives = [x for x in lives if x > 0]

    moving = [d for d in disp if d > 1e-6]
    turning = [d for d in dyaw_all if d > 1e-6]
    return {
        "run": run, "ticks": len(T),
        "speed": st.median(moving) if moving else float("nan"),
        "turn_median": st.median(turning) if turning else float("nan"),
        "turn_fullrate": st.median(dyaw_fast) if dyaw_fast else float("nan"),
        "turn_p99": (sorted(turning)[int(0.99 * (len(turning) - 1))] if turning else float("nan")),
        "n_fast": len(dyaw_fast),
        "drain_e_raw": drain(E), "drain_t_raw": drain(TH),
        "restore_e_raw": restore(E), "restore_t_raw": restore(TH),
        "life_median": st.median(lives) if lives else float("nan"),
        "life_max": max(lives) if lives else float("nan"),
        "lives": len(lives),
    }


def verdict(name: str, measured: float) -> str:
    dec, unit, note = DECLARED[name]
    if not (measured == measured):                                     # NaN
        return f"  {name:17s} déclaré {dec:<8g} {unit:8s} | MESURE IMPOSSIBLE ({note})"
    ratio = measured / dec if dec else float("inf")
    stale = abs(measured - dec) > TOL * max(abs(measured), 1e-12)
    tag = f"🚨 PÉRIMÉ (x{ratio:.2f})" if stale else f"OK (x{ratio:.2f})"
    return (f"  {name:17s} déclaré {dec:<8g} {unit:8s} | mesuré {measured:<9.4g} | {tag}\n"
            f"    {note}")


def _report(m: dict) -> None:
    print(f"\n=== {os.path.basename(m['run'].rstrip('/'))}  ({m['ticks']} ticks, {m['lives']} vies)")
    print(verdict("nominal_speed", m["speed"]))
    print(verdict("surv_turn_rate", m["turn_fullrate"]))
    print(f"    virage : médian {m['turn_median']:.4f} | plein régime (|ω|>={OMEGA_MAX}) "
          f"{m['turn_fullrate']:.4f} (n={m['n_fast']}) | p99 {m['turn_p99']:.4f} rad/pas")
    print(verdict("resource_drain", m["drain_e_raw"] / 100.0))
    print(f"    drain brut énergie {m['drain_e_raw']:.4g}/pas, soif {m['drain_t_raw']:.4g}/pas "
          f"(jauges 0-100 -> normalisé /100)")
    print(verdict("resource_restore", m["restore_e_raw"] / 100.0))
    print(f"    restore ABSORBÉ énergie {m['restore_e_raw']:.4g} (plafond 100 -> le nominal est "
          f"écrêté, ne jamais déclarer le nominal)")
    print(f"  surv_horizon      déclaré 3000     pas      | CONTEXTE SEULEMENT (pas un fait mesurable)\n"
          f"    vies mesurées : médiane {m['life_median']:.0f} pas, max {m['life_max']:.0f} pas — un "
          f"PLAFOND de simulation\n    plus large que les vies typiques est normal ; à trancher en "
          f"catégorie (b), par A/B, pas ici.")


def _selfcheck() -> None:
    """Corpus SYNTHÉTIQUE à vérité connue : vitesse, virage et drain imposés."""
    import json
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix="bodymodel_")
    try:
        rows, e, yaw = [], 100.0, 0.0
        for k in range(3000):
            e -= 0.05
            yaw = _wrap(yaw + 0.019)                 # virage IMPOSÉ 0.019 rad/pas, passe par ±pi
            rows.append({"obs": {"energy": e, "thirst": 90.0},
                         "wm": {"torso0": [0.013 * k, 0.0, yaw], "cmd": [0.8, 0.6]}})
        with open(os.path.join(tmp, "ep_0000.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        m = measure_body_model(tmp)
        assert abs(m["speed"] - 0.013) < 1e-6, f"vitesse : {m['speed']}"
        assert abs(m["turn_fullrate"] - 0.019) < 1e-6, f"virage plein régime : {m['turn_fullrate']}"
        assert abs(m["drain_e_raw"] - 0.05) < 1e-6, f"drain : {m['drain_e_raw']}"
        # la garde d'enroulement doit tenir : sans _wrap, le passage ±pi donnerait ~6.28 rad/pas
        assert m["turn_p99"] < 0.1, f"enroulement ±pi non géré : p99={m['turn_p99']}"
        print("selfcheck OK — vitesse, virage plein régime, drain et enroulement ±pi corrects")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck()
        return
    if not a.runs:
        ap.error("--runs requis (ou --selfcheck)")
    for r in a.runs:
        _report(measure_body_model(r))


if __name__ == "__main__":
    main()
