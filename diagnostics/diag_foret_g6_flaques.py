"""G6 GRATUIT — LES FLAQUES : l'incertitude de l'eau est-elle OBSERVABLE et GRADUELLE ?

PÉRIMÈTRE. Aucune collecte retenue, aucun entraînement. On lance une courte babillage-eau, on LIT ce
que le monde rapporte, on jette le run. Quelques minutes.

POURQUOI (design_foret_complete.md §2.12 + §2.12bis). L'eau revient sous forme de FLAQUES dispersées
à disponibilité VARIABLE — c'est la variabilité qui porte la valeur d'apprentissage, PAS la 2ᵉ
pulsion (l'arbitrage faim/soif est déjà tranché par un coût analytique, et le critique d'arbitrage a
échoué). La règle §2.12bis, déjà PAYÉE par une mesure : l'incertitude doit être OBSERVABLE et
GRADUELLE, jamais instantanée et cachée. La relocalisation périssable SAUTE (aléatoire, invisible au
WM déterministe, anomalie A4) = MAUVAIS format. Une flaque RÉTRÉCIT en douceur = BON format.

CE QUE VÉRIFIE LA SONDE, sur ce que le monde SERT réellement (log [puddle], §6bis) :
  T1 VARIE ....... amplitude du niveau (max - min) >= 0,50 : la flaque sèche ET se remplit vraiment.
  T2 GRADUEL ..... plus gros pas de niveau par tick <= 0,05 : aucun saut. Contraste : une
                   relocalisation ferait un pas ~0,85 ; le seuil est 17x plus bas.
  T3 CHOIX ....... désynchronisation moyenne >= 0,15 : au même instant certaines flaques sont
                   pleines, d'autres sèches -> il y a un CHOIX (sinon toutes sèchent ensemble = pas
                   de décision, juste une famine périodique).
  T4 DÉFAUT ...... sans SYLVAN_WATER_PUDDLE_PERIOD, AUCUN log flaque -> bit-identique.

PERCEPTION. Le signal d'incertitude est la TAILLE de la flaque (empreinte rétine), pas sa couleur :
en rétrécissant, MOINS de rayons la touchent, mais chaque rayon reste bleu. La sonde le vérifie —
l'eau (0.20,0.50,0.95) reste dans le cône eau (cos bleu > 0,55) quelle que soit sa taille, car le
cosinus est invariant d'échelle. Donc la flaque reste « de l'eau » en séchant, et c'est bien son
EMPREINTE qui varie, de façon perceptible.

CE QUE LA SONDE NE DIT PAS : que l'entité RETIENNE les flaques et anticipe leur cycle — mémoire +
retrain, hors périmètre. Elle établit que le monde produit une incertitude du BON format (§2.12bis).

CLI :
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_foret_g6_flaques.py
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_foret_g6_flaques.py --selfcheck
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from sylvan.world import BOSQUETS_V2  # noqa: E402

GODOT = os.path.join(ROOT, "tools", "godot", "godot")

# L'eau, telle que main.gd la configure (Color(0.2, 0.5, 0.95)) — constante moteur, notée ici.
WATER_ALBEDO = np.array([0.20, 0.50, 0.95])
QUERY_BLUE = np.array([0.0, 0.0, 1.0])
SLOT_THRESHOLD = 0.55

PERIOD = 300
RANGE_MIN = 0.50
STEP_MAX = 0.05
DESYNC_MIN = 0.15
JUMP_REF = 0.85     # ce que ferait une relocalisation (le mauvais format), pour le contraste

# `[puddle] WATER : cycle 300 ticks | niveau MESURE 0.15..1.00 | plus gros pas/tick 0.0089 (graduel)
#  | desync moyen 0.301 (choix) | boire si >= 0.40`
RE_PUDDLE = re.compile(
    r"\[puddle\] WATER : cycle (\d+) ticks \| niveau MESURE ([\d.]+)\.\.([\d.]+) \| "
    r"plus gros pas/tick ([\d.]+) .*? desync moyen ([\d.]+)")


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _run(period: int) -> str:
    run_dir = "/tmp/foret_g6"
    os.system(f"rm -rf {run_dir}")
    e = dict(os.environ)
    e.update(BOSQUETS_V2.to_env())
    e.update({
        "SYLVAN_WATER_COUNT": "3", "SYLVAN_WATER_PATCHES": "3", "SYLVAN_THIRST_DRAIN": "0.05",
        "SYLVAN_WATER_PUDDLE_DRINK": "0.4",
        "SYLVAN_COLLECT": "1", "SYLVAN_WM_COLLECT": "1", "SYLVAN_COLLECTOR_MODE": "babbling",
        "SYLVAN_CPG": "1", "SYLVAN_RESIDUAL_GAIN": "0.0", "SYLVAN_TURN_FADE": "0",
        "SYLVAN_WM_VX_MIN": "0.55", "SYLVAN_WM_VX_MAX": "0.75", "SYLVAN_WM_WMAX": "0.6",
        "SYLVAN_POLICY_EXPLORATION_STD_INITIAL": "0", "SYLVAN_POLICY_EXPLORATION_STD_FINAL": "0",
        "SYLVAN_REFLEX_STRENGTH": "0", "SYLVAN_ASSIST_RATIO": "0",
        "SYLVAN_NUM_EPISODES": "2", "SYLVAN_MAX_EPISODE_STEPS": "600",
        "SYLVAN_SEED": "1", "SYLVAN_RUN_DIR": run_dir,
        # homéostasie ON : les flaques (comme les proies) tiquent via try_consume, sous les pulsions —
        # exactement le régime d'une collecte réelle. La DÉSACTIVER figerait le monde.
    })
    if period > 0:
        e["SYLVAN_WATER_PUDDLE_PERIOD"] = str(period)
    else:
        e.pop("SYLVAN_WATER_PUDDLE_PERIOD", None)
    p = subprocess.run([GODOT, "--path", os.path.join(ROOT, "godot"), "--headless"],
                       env=e, capture_output=True, text=True, timeout=600)
    out = p.stdout + p.stderr
    os.system(f"rm -rf {run_dir}")
    for fatal in ("Parse Error", "Failed to load script"):
        if fatal in out:
            first = next((ln for ln in out.splitlines() if fatal in ln), fatal)
            raise SystemExit(f"[period={period}] Godot n'a PAS chargé — mesure invalide.\n  {first}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()

    print(f"MONDE : {BOSQUETS_V2.name} + eau en flaques (3 flaques, cycle {PERIOD} ticks)")

    # PERCEPTION : la flaque reste « de l'eau » en rétrécissant (cosinus invariant d'échelle).
    cb_full = _cos(WATER_ALBEDO, QUERY_BLUE)
    cb_small = _cos(WATER_ALBEDO * 0.15, QUERY_BLUE)
    rgb = tuple(round(float(x), 2) for x in WATER_ALBEDO)
    print(f"\n  PERCEPTION : eau {rgb} → cos bleu {cb_full:.3f} (pleine) / "
          f"{cb_small:.3f} (à 15 %) — {'reste EAU' if cb_small > SLOT_THRESHOLD else 'SORT du cône eau !'}")
    perc_ok = cb_full > SLOT_THRESHOLD and cb_small > SLOT_THRESHOLD

    on = _run(PERIOD)
    m = RE_PUDDLE.findall(on)
    if not m:
        raise SystemExit("aucune ligne [puddle] servie — le mécanisme n'a pas tourné (mesure invalide)")
    cyc, lo, hi, step, desync = (int(m[-1][0]), float(m[-1][1]), float(m[-1][2]),
                                 float(m[-1][3]), float(m[-1][4]))
    rng = hi - lo
    print(f"\n  SERVI (log [puddle], §6bis) : cycle {cyc} ticks | niveau {lo:.2f}..{hi:.2f} "
          f"(amplitude {rng:.2f}) | plus gros pas/tick {step:.4f} | desync {desync:.3f}")

    off = _run(0)
    off_clean = "[puddle]" not in off

    fails = []
    if not perc_ok:
        fails.append(f"T-perception : l'eau à 15 % de taille sort du cône eau (cos bleu {cb_small:.3f})")
    if rng < RANGE_MIN:
        fails.append(f"T1 varie : amplitude {rng:.2f} < {RANGE_MIN} — la flaque ne sèche pas assez")
    if step > STEP_MAX:
        fails.append(f"T2 graduel : plus gros pas {step:.4f} > {STEP_MAX} — ce n'est pas graduel")
    if desync < DESYNC_MIN:
        fails.append(f"T3 choix : desync {desync:.3f} < {DESYNC_MIN} — les flaques sèchent ensemble")
    if not off_clean:
        fails.append("T4 défaut : le mode OFF émet quand même un log flaque (pas bit-identique)")

    print("\n=== LECTURE ===")
    print(f"  varie      amplitude {rng:.2f} (seuil {RANGE_MIN})           → {'OK' if rng>=RANGE_MIN else 'ÉCHEC'}")
    print(f"  graduel    plus gros pas {step:.4f} (seuil {STEP_MAX} ; un SAUT ferait ~{JUMP_REF}) "
          f"→ {'OK' if step<=STEP_MAX else 'ÉCHEC'}  [{JUMP_REF/max(step,1e-9):.0f}x sous le saut]")
    print(f"  choix      desync {desync:.3f} (seuil {DESYNC_MIN})            → {'OK' if desync>=DESYNC_MIN else 'ÉCHEC'}")
    print(f"  défaut     mode OFF sans log flaque              → {'OK' if off_clean else 'ÉCHEC'}")

    print("\n=== VERDICT ===")
    if fails:
        for f in fails:
            print(f"  ✗ {f}")
        print("  G6 FLAQUES = ÉCHEC")
        return 1
    print("  G6 FLAQUES = PASS — l'eau varie, GRADUELLEMENT (aucun saut), en désynchronisé (un choix),")
    print("  et reste perceptible comme eau en séchant. Incertitude du BON format (§2.12bis).")
    print("  ⚠️ NON MESURÉ ICI : que l'entité RETIENNE les flaques et anticipe le cycle (mémoire + retrain).")
    return 0


def selfcheck() -> int:
    # Le cosinus est invariant d'échelle : une flaque qui rétrécit reste « de l'eau ».
    cb = _cos(WATER_ALBEDO, QUERY_BLUE)
    cb2 = _cos(WATER_ALBEDO * 0.1, QUERY_BLUE)
    assert abs(cb - cb2) < 1e-6 and cb > SLOT_THRESHOLD, (cb, cb2)
    print(f"  [ok] eau : cos bleu {cb:.3f} > 0.55, invariant d'échelle (pleine == à 10 %)")

    # Le parseur lit la ligne [puddle].
    line = ("[puddle] WATER : cycle 300 ticks | niveau MESURE 0.15..1.00 | plus gros pas/tick 0.0089 "
            "(graduel) | desync moyen 0.301 (choix) | boire si >= 0.40")
    m = RE_PUDDLE.findall(line)
    assert m and m[0] == ("300", "0.15", "1.00", "0.0089", "0.301"), m
    print("  [ok] le parseur lit la ligne [puddle] émise par food_manager.gd")

    # Le niveau analytique : cosinus surélevé sur une période -> pas maxi = pi*(1-floor)/period, borné.
    period, floor = 300, 0.15
    lvls = [floor + (1 - floor) * 0.5 * (1 - math.cos(2 * math.pi * t / period)) for t in range(period)]
    max_step = max(abs(lvls[t + 1] - lvls[t]) for t in range(period - 1))
    assert max_step < STEP_MAX, max_step
    assert (max(lvls) - min(lvls)) > RANGE_MIN
    print(f"  [ok] niveau analytique (période {period}) : amplitude {max(lvls)-min(lvls):.2f}, "
          f"plus gros pas {max_step:.4f} < {STEP_MAX} — graduel par construction")

    # Désynchronisation : à un instant donné, 3 flaques déphasées ont des niveaux différents.
    def lvl(i, t, n=3):
        frac = ((t / period) + i / n) % 1.0
        return floor + (1 - floor) * 0.5 * (1 - math.cos(2 * math.pi * frac))
    spread = np.std([lvl(i, 50) for i in range(3)])
    assert spread > 0.1, spread
    print(f"  [ok] désync analytique : 3 flaques déphasées, écart-type {spread:.3f} > 0.1 — un choix existe")

    assert os.path.exists(GODOT), GODOT
    print("  [ok] binaire Godot présent")
    print("SELFCHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
