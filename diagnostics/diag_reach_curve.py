"""COURBE D'ATTEINTE vs DISTANCE — l'instrument de jugement canonique de l'audit de péremption.

POURQUOI CE FICHIER EXISTE (2026-07-21). Cette courbe a servi à rendre le verdict far_align
(atteinte [4,6) m : 47,4 -> 69,9 % sans l'échafaudage) mais elle avait été calculée en python
INLINE, jamais persistée : chaque comparaison suivante l'aurait ré-implémentée à la main. C'est
exactement le motif d'erreur que `guards.py` existe pour tuer (compteurs ad-hoc divergents).

POURQUOI CETTE MÉTRIQUE ET PAS LA SURVIE. Le budget métabolique par cycle est ~= 0 : la survie est
une marche aléatoire à dérive nulle, donc un instrument AVEUGLE (dominé par la variance, n en
dizaines). La courbe d'atteinte a un n en MILLIERS et mesure le BUT directement (§2 : mesurer le
but, pas le proxy).

DÉFINITION (explicite, pour qu'elle soit critiquable) :
  - Une OPPORTUNITÉ = un tick où le planner a une cible (`plan.target`) à distance d, DEVANT
    l'entité. Le champ `plan` est loggé 1 tick sur 10 (replan-every) -> ~5 400 opportunités par
    corpus de 54 k ticks.
  - CONDITIONNEMENT DEVANT (cos_bearing > 0, soit |bearing| <= 90°) : la rétine est à 360° mais le
    corps est forward-only ; compter les cibles DERRIÈRE gonfle artificiellement le déficit
    (artefact mesuré et corrigé le 2026-07-21). `--front 0` pour le désactiver.
  - ATTEINTE = la ressource visée est CONSOMMÉE avant l'échéance `slack * d / vitesse`, dans la
    MÊME vie. L'échéance est proportionnelle à la distance -> la métrique est équitable entre
    bandes (sinon les bandes lointaines seraient pénalisées deux fois).
  - La VITESSE est MESURÉE sur le corpus (`guards.measured_constants`), jamais déclarée : c'est
    une constante déclarée fausse d'un facteur 2 qui a fabriqué le faux « 1,88x d'inefficacité ».

CONVENTIONS DE REPÈRE (vérifiées sur corpus, pas supposées) :
  - `wm.torso0` = [x, z, yaw] — l'indice 2 est le YAW (borné a ±pi), PAS une hauteur.
  - `plan.food` / `plan.water` = coordonnées EGO ; l'indice 1 est la composante AVANT
    (même convention que le planner : `cos_brg = slot[...,1] / dist`, command_planner.py:555).

CLI :
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_reach_curve.py \
        --runs data/replay_buffer/A data/replay_buffer/B [--labels off on] [--selfcheck]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guards import CONSUME_JUMP, _reset_ticks, _ticks, measured_constants, sanity  # noqa: E402

BANDS: tuple[float, ...] = (0.0, 2.0, 4.0, 6.0, 8.0)
SLACK = 3.0          # échéance = slack x temps en ligne droite. 3x = généreux mais fini.
MIN_N_BAND = 500     # en dessous : SOUS-PUISSANT, on ne tranche pas (auto-correction n°7)
EFFECT_PTS = 5.0     # écart minimal (points de %) pour qu'un effet compte


def _lives(E: list[float], TH: list[float]) -> list[tuple[int, int]]:
    """Découpe le corpus en vies sur la signature de respawn mesurée (cf. guards._reset_ticks)."""
    resets = sorted(_reset_ticks(E, TH))
    bounds, start = [], 0
    for r in resets:
        bounds.append((start, r))
        start = r
    bounds.append((start, len(E)))
    return [(a, b) for a, b in bounds if b > a]


def _consume_ticks(series: list[float], resets: set[int], lo: int, hi: int) -> list[int]:
    """Ticks de CONSOMMATION réelle dans [lo, hi) : saut de jauge, respawns EXCLUS."""
    return [i for i in range(max(lo, 1), hi)
            if series[i] - series[i - 1] > CONSUME_JUMP and i not in resets]


def reach_curve(runs: str | list[str], *, slack: float = SLACK, front: bool = True,
                bands: tuple[float, ...] = BANDS) -> dict:
    """Courbe d'atteinte, POOLÉE sur plusieurs corpus (les critères exigent >= 2 seeds).

    La vitesse est mesurée PAR corpus : deux seeds n'ont pas exactement la même (0.0100 vs 0.0097
    mesuré), et l'échéance doit utiliser la vitesse du corpus dont vient l'opportunité.
    """
    run_list = [runs] if isinstance(runs, str) else list(runs)
    if not run_list:
        raise SystemExit("aucun corpus fourni")
    hit = [0] * (len(bands) - 1)
    tot = [0] * (len(bands) - 1)
    behind, n_ticks, n_lives, speeds = 0, 0, 0, []

    for run in run_list:
        T = _ticks(run)
        if not T:
            raise SystemExit(f"corpus vide : {run}")
        E = [float(t["obs"]["energy"]) for t in T]
        TH = [float(t["obs"]["thirst"]) for t in T]
        resets = _reset_ticks(E, TH)
        speed = measured_constants(run, T)["speed"]
        if not (speed > 0):
            raise SystemExit(f"vitesse mesurée invalide ({speed}) sur {run}")
        speeds.append(speed)
        lives = _lives(E, TH)
        n_ticks += len(T)
        n_lives += len(lives)

        for lo, hi in lives:
            eat = {"food": _consume_ticks(E, resets, lo, hi),
                   "water": _consume_ticks(TH, resets, lo, hi)}
            for i in range(lo, hi):
                plan = T[i].get("plan")
                if not plan:
                    continue
                tgt = plan.get("target")
                pos = plan.get(tgt) if tgt in ("food", "water") else None
                if not pos:
                    continue
                d = math.hypot(pos[0], pos[1])
                if d <= 0:
                    continue
                if front and (pos[1] / d) <= 0.0:   # composante AVANT négative = cible derrière
                    behind += 1
                    continue
                b = next((k for k in range(len(bands) - 1)
                          if bands[k] <= d < bands[k + 1]), None)
                if b is None:
                    continue                        # hors de la plage auditée
                tot[b] += 1
                if any(i < c <= i + slack * d / speed for c in eat[tgt]):
                    hit[b] += 1

    return {
        "run": " + ".join(os.path.basename(r.rstrip("/")) for r in run_list),
        "runs": run_list, "speed": min(speeds), "speed_max": max(speeds),
        "ticks": n_ticks, "lives": n_lives, "behind_skipped": behind,
        "bands": [{"lo": bands[k], "hi": bands[k + 1], "n": tot[k],
                   "reach_pct": (100.0 * hit[k] / tot[k]) if tot[k] else float("nan")}
                  for k in range(len(bands) - 1)],
    }


def compare(a: dict, b: dict, *, label_a: str, label_b: str) -> list[str]:
    """Verdict PRÉ-ENREGISTRÉ par bande. Ne tranche JAMAIS sous MIN_N_BAND."""
    out = []
    for ba, bb in zip(a["bands"], b["bands"]):
        n = min(ba["n"], bb["n"])
        band = f"[{ba['lo']:.0f},{ba['hi']:.0f})m"
        if n < MIN_N_BAND:
            out.append(f"  {band}: SOUS-PUISSANT (n min={n} < {MIN_N_BAND}) — pas de verdict")
            continue
        delta = bb["reach_pct"] - ba["reach_pct"]
        tag = ("EFFET" if abs(delta) >= EFFECT_PTS else "dans le bruit")
        out.append(f"  {band}: {label_a} {ba['reach_pct']:5.1f}%  ->  {label_b} "
                   f"{bb['reach_pct']:5.1f}%   Δ={delta:+5.1f} pts  [{tag}]  n={ba['n']}/{bb['n']}")
    return out


def _report(r: dict) -> None:
    print(f"\n=== {r['run']}")
    for run in r["runs"]:                              # la garde tourne sur CHAQUE corpus poolé
        s = sanity(run)
        if s["anomalies"]:
            print(f"  🚨 CORPUS SUSPECT -> VERDICT NUL, pas un résultat "
                  f"({os.path.basename(run.rstrip('/'))}) : {s['anomalies']}")
    print(f"  vitesse MESURÉE {r['speed']:.4f}-{r['speed_max']:.4f} m/tick | {r['ticks']} ticks "
          f"| {r['lives']} vies | {r['behind_skipped']} opportunités DERRIÈRE écartées")
    for b in r["bands"]:
        print(f"    [{b['lo']:.0f},{b['hi']:.0f})m : {b['reach_pct']:5.1f} %   (n={b['n']})")


def _selfcheck() -> None:
    """Corpus SYNTHÉTIQUE à vérité connue, un scénario par bande, pour tester la SÉMANTIQUE.

    [2,4) approche 3 m puis consomme          -> DOIT être ~100 %
    [4,6) reste à 5 m, consomme TROP TARD     -> DOIT être 0 % (teste l'ÉCHÉANCE, le coeur)
    [6,8) reste à 7 m, ne consomme jamais     -> DOIT être 0 %
    Piège évité : dans une v1 les cycles « non atteints » approchaient quand même de 0, ce qui
    peuplait les bandes proches d'opportunités ratées et rendait le test ininterprétable.
    """
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix="reachcurve_")
    try:
        rows: list[dict] = []
        e, k = 100.0, 0

        def step(dist: float, *, plan: bool = True) -> None:
            nonlocal e, k
            e -= 0.005                                 # jamais sous 15 -> jamais pris pour un respawn
            row: dict = {"obs": {"energy": e, "thirst": 90.0},
                         "wm": {"torso0": [0.01 * k, 0.0, 0.0]}}
            if plan and k % 10 == 0:                   # `plan` loggé 1 tick sur 10, comme en vrai
                row["plan"] = {"target": "food", "food": [0.0, dist]}
            rows.append(row)
            k += 1

        def eat() -> None:
            nonlocal e, k
            e += 40.0
            rows.append({"obs": {"energy": e, "thirst": 90.0},
                         "wm": {"torso0": [0.01 * k, 0.0, 0.0]}})
            k += 1

        for _ in range(3):                             # [2,4) : approche puis consomme
            for j in range(300):
                step(3.0 - 0.01 * j)
            eat()
        # [4,6) : opportunités groupées AU DÉBUT, repas 2000 ticks plus tard = HORS échéance pour
        # TOUTES (échéance = slack x 5 / 0.01 = 1500 ticks). Piège évité : avec les opportunités
        # étalées jusqu'au repas, les plus tardives tombent DANS l'échéance (mesuré : 71 %) et le
        # test ne prouve plus rien.
        for _ in range(300):
            step(5.0)
        for _ in range(2000):
            step(5.0, plan=False)
        eat()
        for _ in range(600):                           # [6,8) : jamais atteint
            step(7.0)

        with open(os.path.join(tmp, "ep_0000.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        r = reach_curve(tmp)
        got = {(b["lo"], b["hi"]): b for b in r["bands"]}
        near, late, far = got[(2.0, 4.0)], got[(4.0, 6.0)], got[(6.0, 8.0)]
        assert abs(r["speed"] - 0.01) < 1e-6, f"vitesse mesurée fausse : {r['speed']}"
        assert near["n"] and late["n"] and far["n"], f"bandes vides : {r}"
        assert near["reach_pct"] > 90.0, f"proche devrait etre atteint : {near}"
        assert late["reach_pct"] < 10.0, f"repas HORS ÉCHÉANCE compté comme atteint : {late}"
        assert far["reach_pct"] < 10.0, f"lointain ne devrait PAS etre atteint : {far}"

        # l'échéance doit être la SEULE raison de l'échec en [4,6) : un slack énorme le rattrape
        r_slack = reach_curve(tmp, slack=20.0)
        late2 = {(b["lo"], b["hi"]): b for b in r_slack["bands"]}[(4.0, 6.0)]
        assert late2["reach_pct"] > 90.0, f"slack=20 devrait rattraper le repas tardif : {late2}"

        # la garde DERRIÈRE doit écarter les cibles à composante avant négative
        for row in rows:
            if "plan" in row:
                row["plan"]["food"] = [row["plan"]["food"][0], -row["plan"]["food"][1]]
        with open(os.path.join(tmp, "ep_0000.jsonl"), "w") as f:
            for r2 in rows:
                f.write(json.dumps(r2) + "\n")
        rb = reach_curve(tmp)
        assert all(b["n"] == 0 for b in rb["bands"]), f"le filtre DEVANT ne filtre pas : {rb}"
        assert rb["behind_skipped"] > 0
        print("selfcheck OK — atteinte proche >90 %, lointaine <10 %, filtre DEVANT actif")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], help="corpus rapportés un par un")
    ap.add_argument("--a", nargs="*", default=[], help="bras A (poolé sur les seeds)")
    ap.add_argument("--b", nargs="*", default=[], help="bras B (poolé sur les seeds)")
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--slack", type=float, default=SLACK)
    ap.add_argument("--front", type=int, default=1, help="1 = ne compter que les cibles DEVANT")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck()
        return
    if bool(a.a) != bool(a.b):
        ap.error("--a et --b vont par paire")
    if not a.runs and not a.a:
        ap.error("--runs, ou --a/--b, ou --selfcheck")

    for r in a.runs:
        _report(reach_curve(r, slack=a.slack, front=bool(a.front)))
    if a.a:
        ra = reach_curve(a.a, slack=a.slack, front=bool(a.front))
        rb = reach_curve(a.b, slack=a.slack, front=bool(a.front))
        _report(ra)
        _report(rb)
        la, lb = (a.labels + ["A", "B"])[:2]
        print(f"\n=== COMPARAISON {la} -> {lb} "
              f"(effet si |Δ| >= {EFFECT_PTS} pts, verdict si n >= {MIN_N_BAND})")
        for line in compare(ra, rb, label_a=la, label_b=lb):
            print(line)


if __name__ == "__main__":
    main()
