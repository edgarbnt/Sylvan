"""DENSITÉ — le monde est-il si dense qu'un repas n'y vaut plus rien ? (analyseur PUR, zéro entraînement)

POURQUOI CETTE SONDE (2026-07-28). Le visuel a montré une entité qui meurt de soif énergie pleine ;
en réparant ça (l'eau était confinée dans un disque de 7 m) une seconde anomalie est apparue, bien
plus grave. Sur les 10 repas observés, l'entité mangeait à énergie **74** et n'encaissait que **18**
points sur les 84 servis — 21 % de rendement — parce que la jauge plafonne à 100. Il y a de la
nourriture tous les 2 m : le planner se recharge en permanence au lieu de chasser, jette 79 % de
chaque repas, et il lui faudrait ~53 repas par vie pour un budget de trajet qui n'en porte que ~12.

MÉCANISME PROPOSÉ, ET IL EST FALSIFIABLE : la DENSITÉ cause le gaspillage. Plus la nourriture est
proche, plus l'entité mange tôt, moins chaque repas vaut. Si c'est vrai, éclaircir le monde doit
faire TOMBER l'énergie-au-repas et MONTER le gain encaissé. Si le rendement est plat sur un balayage
de densité 7x, le mécanisme est réfuté et la densité n'est pas le levier — il faudra chercher
ailleurs, et les 21 % n'étaient qu'un artefact du WM hors-distribution.

CE QUE CETTE SONDE N'EST PAS. Elle ne juge pas la compétence de l'entité : le WM servi a été
entraîné sur l'ANCIENNE dynamique, il est hors-distribution partout. Les absolus sont donc des
bornes, pas des chiffres finaux. Ce qui transfère, c'est la COMPARAISON entre densités : même WM,
même politique, même corps, même graine — seule la densité change. C'est un A/B, pas une mesure.

CRITÈRES PRÉ-ENREGISTRÉS (écrits AVANT la collecte, PRINCIPE N°1) :
  H1 GRIGNOTAGE ... l'énergie-au-repas médiane doit BAISSER d'au moins 15 points entre la densité la
                    plus forte et la plus faible. C'est la prédiction directe du mécanisme.
  H2 RENDEMENT .... le gain médian encaissé par repas doit être >= 1,5x plus grand au plus épars.
  KILL ............ si l'énergie-au-repas varie de moins de 8 points sur tout le balayage, le
                    mécanisme est RÉFUTÉ : STOP, ne pas éclaircir le monde, chercher la vraie cause.
  DÉCISION ........ parmi les densités dont le trajet/repas tient sous le budget par événement,
                    retenir celle qui donne la meilleure survie médiane. Une densité qui gagne en
                    rendement mais dont l'entité ne trouve plus la nourriture n'est pas un progrès.

CLI :
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_foret_densite.py \
        data/replay_buffer/dens_25 data/replay_buffer/dens_60 data/replay_buffer/dens_180
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_foret_densite.py --selfcheck
"""

from __future__ import annotations

import argparse
import os
import re
import statistics as st
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from sylvan.critic_corpus import load_bc_corpus, meal_flags  # noqa: E402
from sylvan.world import FORET_V1  # noqa: E402

H1_DROP_PTS = 15.0        # baisse d'énergie-au-repas exigée entre le plus dense et le plus épars
H2_GAIN_RATIO = 1.5       # facteur d'amélioration exigé sur le gain encaissé
KILL_SPREAD_PTS = 8.0     # sous cet écart total, le mécanisme est réfuté
GAUGE_MAX = 100.0         # plafond de la jauge (homeostasis.gd) — la source du gaspillage


def _density_of(run: str) -> int:
    """Densité lue dans le NOM du corpus (dens_<n>). Le harnais la met là ; on ne la devine pas."""
    m = re.search(r"dens_?(\d+)", os.path.basename(str(run).rstrip("/")))
    if not m:
        raise SystemExit(f"nom de corpus sans densité lisible : {run} (attendu ...dens_<n>)")
    return int(m.group(1))


def measure(run: str) -> dict:
    """Rendement métabolique d'une densité, lu du corpus — jamais d'un oracle."""
    _obs, e, _cmd, bounds = load_bc_corpus(run)
    ate = meal_flags(e, bounds)
    idx = torch.nonzero(ate).flatten().tolist()
    before = [float(e[i - 1]) for i in idx]
    gain = [float(e[i] - e[i - 1]) for i in idx]
    lives = [b - a for a, b in zip(bounds[:-1], bounds[1:])]
    # L'énergie au DERNIER tick de chaque vie : basse = la jauge a tué, haute = fin de l'épisode.
    ends = [float(e[b - 1]) for b in bounds[1:] if b - 1 < len(e)]
    return {
        "run": run, "density": _density_of(run), "ticks": len(e), "lives": len(lives),
        "meals": len(idx), "meals_per_life": len(idx) / max(1, len(lives)),
        "energy_at_meal": st.median(before) if before else float("nan"),
        "gain": st.median(gain) if gain else float("nan"),
        "survival": st.median(lives) if lives else 0.0,
        "end_energy": st.median(ends) if ends else float("nan"),
    }


def render(rows: list[dict], restore: float) -> int:
    """Verdict contre les critères pré-enregistrés. Rend 0 si le mécanisme tient, 1 sinon."""
    rows = sorted(rows, key=lambda r: -r["density"])
    print(f"\n=== RENDEMENT MÉTABOLIQUE PAR DENSITÉ (repas servi = {restore:.0f} pts, "
          f"jauge plafonnée à {GAUGE_MAX:.0f}) ===\n")
    print(f"  {'bosquets':>8} {'vies':>5} {'repas/vie':>10} {'énergie AU repas':>17} "
          f"{'gain encaissé':>14} {'rendement':>10} {'survie méd':>11}")
    for r in rows:
        yield_pct = 100.0 * r["gain"] / restore if restore > 0 else float("nan")
        print(f"  {r['density']:>8} {r['lives']:>5} {r['meals_per_life']:>10.1f} "
              f"{r['energy_at_meal']:>17.0f} {r['gain']:>14.0f} {yield_pct:>9.0f} % "
              f"{r['survival']:>11.0f}")

    dense, sparse = rows[0], rows[-1]
    drop = dense["energy_at_meal"] - sparse["energy_at_meal"]
    spread = max(r["energy_at_meal"] for r in rows) - min(r["energy_at_meal"] for r in rows)
    ratio = sparse["gain"] / dense["gain"] if dense["gain"] > 0 else float("inf")

    print(f"\n  H1 grignotage : énergie-au-repas {dense['energy_at_meal']:.0f} "
          f"({dense['density']} bosquets) -> {sparse['energy_at_meal']:.0f} "
          f"({sparse['density']}) = {drop:+.0f} pts "
          f"[{'PASS' if drop >= H1_DROP_PTS else 'ÉCHEC'}, exigé >= {H1_DROP_PTS:.0f}]")
    print(f"  H2 rendement  : gain {dense['gain']:.0f} -> {sparse['gain']:.0f} pts = {ratio:.2f}x "
          f"[{'PASS' if ratio >= H2_GAIN_RATIO else 'ÉCHEC'}, exigé >= {H2_GAIN_RATIO:.2f}x]")

    if spread < KILL_SPREAD_PTS:
        print(f"\n  🛑 KILL : l'énergie-au-repas ne varie que de {spread:.0f} pts sur tout le "
              f"balayage (< {KILL_SPREAD_PTS:.0f}). Le mécanisme du grignotage est RÉFUTÉ — la "
              "densité n'est PAS le levier. Ne pas éclaircir le monde ; chercher ailleurs.")
        return 1

    ok = drop >= H1_DROP_PTS and ratio >= H2_GAIN_RATIO
    best = max(rows, key=lambda r: (r["survival"], r["gain"]))
    print(f"\n  {'✅ MÉCANISME CONFIRMÉ' if ok else '⚠️  MÉCANISME PARTIEL'} — "
          f"meilleure survie à {best['density']} bosquets "
          f"({best['survival']:.0f} ticks, gain {best['gain']:.0f} pts/repas).")
    if not ok:
        print("     Partiel = la densité agit mais pas assez pour justifier seule le ré-échelonnage."
              "\n     Ne pas conclure au-delà : rapporter les deux critères tels quels (§2).")
    return 0 if ok else 1


def selfcheck() -> int:
    """Les critères se déclenchent-ils dans le bon sens ? On fabrique les trois cas."""
    def row(d: int, at: float, g: float, s: float) -> dict:
        return {"run": f"dens_{d}", "density": d, "ticks": 0, "lives": 3, "meals": 9,
                "meals_per_life": 3.0, "energy_at_meal": at, "gain": g, "survival": s,
                "end_energy": 0.0}

    assert _density_of("data/replay_buffer/dens_25") == 25
    assert _density_of("dens_180/") == 180
    print("  [ok] la densité est lue du nom du corpus, pas devinée")

    confirmed = [row(180, 74, 18, 400), row(25, 20, 80, 2600)]
    assert render(confirmed, 84.0) == 0
    print("  [ok] cas CONFIRMÉ : chute de 54 pts et gain 4,4x -> verdict 0")

    flat = [row(180, 70, 30, 400), row(25, 66, 31, 420)]
    assert render(flat, 84.0) == 1
    print("  [ok] cas PLAT (écart 4 pts) -> KILL, le mécanisme est réfuté")

    weak = [row(180, 70, 30, 400), row(25, 48, 36, 500)]
    assert render(weak, 84.0) == 1
    print("  [ok] cas PARTIEL : H1 passe (22 pts) mais H2 échoue (1,2x) -> pas de conclusion")
    print("\nSELFCHECK PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="*", help="corpus, un par densité, nommés ...dens_<n>")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    if len(a.runs) < 2:
        raise SystemExit("il faut au moins DEUX densités pour qu'un A/B veuille dire quelque chose")
    return render([measure(r) for r in a.runs], FORET_V1.restore_per_item)


if __name__ == "__main__":
    raise SystemExit(main())
