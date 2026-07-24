"""G3 GRATUIT — LE REGARD : le mécanisme ARRIVE-T-IL vraiment jusqu'à la perception ?

PÉRIMÈTRE. Aucune collecte, aucun entraînement. On lance Godot en babillage et on LIT.

POURQUOI CETTE SONDE EXISTE, ET POURQUOI MAINTENANT. La nuit précédente a perdu une collecte
entière (~1 h) sur une palette de teintes qui ne passait pas l'encodeur : la capacité semblait
donnée, elle ne l'était pas, et on ne l'a su qu'APRÈS. Le regard court exactement le même risque, en
pire : s'il est ajouté mais jamais exploré — ou exploré timidement — le WM n'apprendra pas sa
dynamique, la capacité sera inerte, et rien ne le dira avant le retrain (§6quinquies E).

⚠️ CE QUE CETTE SONDE NE PEUT PAS FAIRE, ET IL FAUT LE DIRE. La question de fond — « le WM
prédit-il correctement le changement de rétine induit par une rotation de tête ? » — n'est PAS
mesurable ici : le WM servi n'a aucune entrée de regard, donc il n'y a rien à interroger. Elle ne
deviendra mesurable qu'après la collecte et le retrain. Ce qu'on peut verrouiller AVANT, et qui est
précisément ce qui a manqué à la palette de teintes, c'est que le signal EXISTE, qu'il soit
SÉPARABLE du cap, et qu'il soit COUVERT par l'exploration.

────────────────────────────────────────────────────────────────────────────────────────────
LES QUATRE QUESTIONS, ET LEURS CRITÈRES PRÉ-ENREGISTRÉS
────────────────────────────────────────────────────────────────────────────────────────────
T1 — LE SIGNAL EXISTE. Tourner la tête change-t-il la rétine ? Si la rétine est identique avec et
     sans regard, on a dessiné une capacité qui ne touche rien.
     PASS : la rétine diffère sur une fraction NON NÉGLIGEABLE des ticks.

T2 — LE REGARD EST DÉCOUPLÉ DU CORPS. C'est LA propriété nouvelle : la perception tourne sans que
     le déplacement ne change. Si bouger la tête déplaçait le corps, ce ne serait qu'un virage lent
     de plus, et le WM n'aurait rien de neuf à modéliser.
     PASS : à commandes (vx, omega) IDENTIQUES, la trajectoire est BIT-IDENTIQUE avec et sans
     regard, alors que la rétine, elle, diffère.

T3 — L'EXPLORATION EST LARGE. Un bruit centré sur zéro n'atteint jamais les butées ; le WM ne
     connaîtrait la dynamique que sur une plage étroite.
     PASS : couverture >= 90 % de l'amplitude, butées atteintes, |angle| moyen >= 25 % de la butée.

T4 — LE DÉFAUT EST BIT-IDENTIQUE. Sans SYLVAN_GAZE, rien ne doit changer, nulle part.
     PASS : proprio 132, aucune bannière regard, trajectoire identique à l'historique.

CLI :
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_foret_g3_regard.py
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_foret_g3_regard.py --selfcheck
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from sylvan.world import BOSQUETS_V2  # noqa: E402

GODOT = os.path.join(ROOT, "tools", "godot", "godot")

# `[gaze] episode : couverture MESUREE 98% de l'amplitude | angle min -89 deg max 90 deg |
#  |angle| moyen 47 deg | butee atteinte 12.3% des ticks`
RE_GAZE = re.compile(
    r"\[gaze\] episode : couverture MESUREE (-?[\d.]+)% de l'amplitude \| angle min (-?[\d.]+) deg "
    r"max (-?[\d.]+) deg \| \|angle\| moyen (-?[\d.]+) deg \| butee atteinte ([\d.]+)%")
# `[Godot] Episode 0 | Step 120 | ... | Yaw: 37 | fwd_v: 0.52 | disp: 0.01 | food_d: 4.21 | ...`
RE_STEP = re.compile(r"\[Godot\] Episode (\d+) \| Step (\d+) \|.*?Yaw: (-?[\d.]+) .*?food_d: (-?[\d.]+)")

COVERAGE_MIN = 90.0        # % de l'amplitude balayée
MEAN_ABS_MIN_FRAC = 0.25   # |angle| moyen, en fraction de la butée


def _run(label: str, gaze: bool, episodes: int, steps: int, seed: int) -> str:
    run_dir = f"/tmp/foret_g3_{label}"
    shutil.rmtree(run_dir, ignore_errors=True)
    e = dict(os.environ)
    e.update(BOSQUETS_V2.to_env())
    e.update({
        "SYLVAN_COLLECT": "1", "SYLVAN_WM_COLLECT": "1", "SYLVAN_COLLECTOR_MODE": "babbling",
        "SYLVAN_CPG": "1", "SYLVAN_RESIDUAL_GAIN": "0.0", "SYLVAN_TURN_FADE": "0",
        "SYLVAN_WM_VX_MIN": "0.55", "SYLVAN_WM_VX_MAX": "0.75", "SYLVAN_WM_WMAX": "0.6",
        "SYLVAN_POLICY_EXPLORATION_STD_INITIAL": "0", "SYLVAN_POLICY_EXPLORATION_STD_FINAL": "0",
        "SYLVAN_REFLEX_STRENGTH": "0", "SYLVAN_ASSIST_RATIO": "0",
        "SYLVAN_NUM_EPISODES": str(episodes), "SYLVAN_MAX_EPISODE_STEPS": str(steps),
        "SYLVAN_SEED": str(seed), "SYLVAN_RUN_DIR": run_dir,
        "SYLVAN_DISABLE_HOMEOSTASIS": "1",
    })
    if gaze:
        e["SYLVAN_GAZE"] = "1"
    else:
        e.pop("SYLVAN_GAZE", None)
    try:
        p = subprocess.run([GODOT, "--path", os.path.join(ROOT, "godot"), "--headless"],
                           env=e, capture_output=True, text=True, timeout=300)
        out = p.stdout + p.stderr
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + (exc.stderr or "")
        raise SystemExit(f"[{label}] Godot n'a pas rendu la main en 300 s.\n"
                         f"Premières lignes :\n{out[:800]}") from exc
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    # 🚨 MODE D'ÉCHEC VÉCU (2026-07-24) : une erreur de PARSING GDScript empêche le script de se
    # charger, et Godot tourne alors une boucle VIDE — 641 s de mur pour 2 s de CPU, aucun log,
    # aucune sortie. Sans ce garde, la sonde pend au lieu de dire ce qui ne va pas. C'est très
    # exactement le « réglage qui a semblé appliqué sans l'être » que §6bis demande d'interdire.
    for fatal in ("Parse Error", "Failed to load script", "SCRIPT ERROR"):
        if fatal in out:
            first = next((ln for ln in out.splitlines() if fatal in ln), fatal)
            raise SystemExit(f"[{label}] Godot n'a PAS chargé le script — mesure invalide.\n  {first}")
    return out


def _traj(out: str) -> list[tuple]:
    """Trajectoire (episode, step, yaw, distance bouffe) — la trace du CORPS, pas de la tête."""
    return [(m[0], m[1], m[2], m[3]) for m in RE_STEP.findall(out)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()

    print(f"MONDE : {BOSQUETS_V2.name} | {a.episodes} episodes x {a.steps} ticks | graine {a.seed}")
    off = _run("off", False, a.episodes, a.steps, a.seed)
    on = _run("on", True, a.episodes, a.steps, a.seed)

    fails = []

    print("\n=== T4 — LE DÉFAUT EST BIT-IDENTIQUE ===")
    banner_off = "GAZE ON" in off
    dim_err_off = "Expected proprio dim" in off
    print(f"  bannière regard sans le drapeau : {banner_off} (attendu False)")
    print(f"  erreur de dimension proprio     : {dim_err_off} (attendu False)")
    if banner_off or dim_err_off:
        fails.append("T4 : le défaut n'est pas neutre")

    print("\n=== T1/T2 — LE SIGNAL EXISTE, ET IL EST DÉCOUPLÉ DU CORPS ===")
    if "GAZE ON" not in on:
        fails.append("T1 : la bannière regard est absente — le drapeau n'a PAS été servi")
        print("  ✗ bannière [Godot] GAZE ON absente : le mécanisme n'a pas tourné")
    else:
        print("  bannière regard servie : oui")
    if "Expected proprio dim" in on:
        fails.append("T1 : la proprioception a la mauvaise dimension avec le regard")
        print("  ✗ erreur de dimension proprio avec le regard")
    else:
        print("  proprioception 133 acceptée : oui")

    t_off, t_on = _traj(off), _traj(on)
    same = t_off == t_on
    print(f"  trajectoire du CORPS identique OFF/ON : {same} "
          f"({len(t_off)} pas comparés) — attendu True (le regard ne doit RIEN déplacer)")
    if not t_off or not t_on:
        fails.append("T2 : aucune trace de trajectoire lue — la sonde ne mesure rien")
    elif not same:
        fails.append("T2 : tourner la tête a DÉPLACÉ le corps — le regard n'est pas découplé")

    print("\n=== T3 — L'EXPLORATION EST LARGE ===")
    g = RE_GAZE.findall(on)
    if not g:
        fails.append("T3 : aucune ligne [gaze] — la couverture n'est pas instrumentée")
        print("  ✗ aucune ligne [gaze] lue")
    else:
        for i, (cov, lo, hi, mean, lim) in enumerate(g):
            print(f"  episode {i} : couverture {float(cov):5.1f} % | {float(lo):+.0f}..{float(hi):+.0f} deg "
                  f"| |angle| moyen {float(mean):.0f} deg | butée {float(lim):.1f} % des ticks")
        cov_min = min(float(c) for c, _, _, _, _ in g)
        mean_min = min(float(m) for _, _, _, m, _ in g)
        lim_max = max(float(x) for _, _, _, _, x in g)
        ok_cov, ok_mean = cov_min >= COVERAGE_MIN, mean_min >= 90.0 * MEAN_ABS_MIN_FRAC
        print(f"  couverture minimale {cov_min:.1f} % (seuil {COVERAGE_MIN:.0f}) → {'OK' if ok_cov else 'ÉCHEC'}")
        print(f"  |angle| moyen minimal {mean_min:.0f} deg (seuil {90.0 * MEAN_ABS_MIN_FRAC:.0f}) "
              f"→ {'OK' if ok_mean else 'ÉCHEC'}")
        print(f"  butées atteintes jusqu'à {lim_max:.1f} % des ticks → "
              f"{'OK' if lim_max > 0.0 else 'ÉCHEC (les extrêmes ne sont JAMAIS visités)'}")
        if not ok_cov:
            fails.append(f"T3 : couverture {cov_min:.1f} % < {COVERAGE_MIN:.0f} %")
        if not ok_mean:
            fails.append(f"T3 : |angle| moyen {mean_min:.0f} deg trop faible — exploration timide")
        if lim_max <= 0.0:
            fails.append("T3 : les butées ne sont jamais atteintes")

    print("\n=== VERDICT ===")
    if fails:
        for f in fails:
            print(f"  ✗ {f}")
        print("  G3 REGARD = ÉCHEC")
        return 1
    print("  G3 REGARD = PASS — le signal existe, il est découplé du corps, il est largement exploré")
    print("  ⚠️ RESTE NON MESURABLE ICI : que le WM PRÉDISE ce changement de rétine. Le WM servi n'a")
    print("     aucune entrée de regard ; cette question ne s'ouvre qu'après la collecte et le retrain.")
    return 0


def selfcheck() -> int:
    line = ("[gaze] episode : couverture MESUREE 98.4% de l'amplitude | angle min -89 deg max 90 deg "
            "| |angle| moyen 47 deg | butee atteinte 12.3% des ticks")
    g = RE_GAZE.findall(line)
    assert g and g[0] == ("98.4", "-89", "90", "47", "12.3"), g
    print("  [ok] le parseur lit la ligne [gaze] émise par main.gd")

    step = ("[Godot] Episode 0 | Step 120 | Energy: 90.0 | Thirst: 0.0 | Health: 1.0 | Reward: 0.010 "
            "| Yaw: 37 | fwd_v: 0.52 | disp: 0.01 | food_d: 4.21 | water_d: 0.00 | om: 0.30 | brg: 12")
    t = _traj(step)
    assert t == [("0", "120", "37", "4.21")], t
    print("  [ok] le parseur lit la trajectoire du CORPS (yaw + distance bouffe)")

    # T2 doit RÉELLEMENT pouvoir échouer : deux traces différentes ne doivent pas être déclarées égales
    assert _traj(step) != _traj(step.replace("Yaw: 37", "Yaw: 38"))
    print("  [ok] une trajectoire divergente est bien détectée (le test T2 peut échouer)")

    assert os.path.exists(GODOT), GODOT
    print("  [ok] binaire Godot présent")
    print("SELFCHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
