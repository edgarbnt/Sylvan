"""SONDE G0 — le corpus porte-t-il le signal du SPRINT ? (gratuit : 0 run, 0 train)

Chantier critique-sprint (docs/design_critique_sprint.md). Joint decisions.jsonl ↔ flux BC par
tick, segmente en POURSUITES (conventions v3 partagées avec le trainer), classe chaque décision :
  - cross  : le candidat CHOISI croise le vert (intrusion>0 — sprints oracle, traversées ε,
             directs d'hystérésis) ;
  - refuse : le direct est bloqué mais le choix contourne (refus de sprint) ;
  - clear  : rien de bloqué (hors sujet pour le sprint).
et mesure l'issue vécue de la poursuite : repas obtenu (remontée drive >+5, gain OBSERVÉ),
dégâts payés (creux de santé), mort, pas restants.

MESURES GRATUITES qui PINNENT le label du trainer (design §label) :
  - drain énergie MESURÉ (pente médiane) → valeur d'un repas en pas = gain_observé/drain ;
  - κ_data monde-v2 = médiane(pas restants aux décisions)/100 ;
  - fraction des traversées mortes avec santé<50 à la décision (choix de la variante plancher-mort).

GATE G0 (PRÉ-ENREGISTRÉ, design_critique_sprint.md) :
  - ≥100 décisions-traversée labellisées ET ≥100 refus-bloqués tenus ;
  - contraste directionnel : U̅(cross) > U̅(refuse) sur les buckets sains-affamés ET l'inverse
    sur les buckets blessés (NA avant la collecte ε : le corpus g24 n'a pas de sprint blessé).

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_sprint_corpus.py \
      [--runs data/replay_buffer/critic_kin_g24as1 ...] [--selfcheck]
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics as st
from pathlib import Path

from scripts.train_waypoint_pain import (_drives_series, _open_text, _text_path,
                                         health_series, pursuit_end)
from sylvan.control.waypoint_layer import WaypointConfig

_CFG = WaypointConfig()          # W (block_weight) et green_margin du scoreur analytique vivant
_LEN_CAP = 20.0                  # cap de la feature longueur (candidate_features idx 6)
_INTR_EPS = 0.02                 # marge d'arrondi (costs round(3), feats round(4))


def route_intrusions(d: dict) -> list[float]:
    """Intrusion EXACTE par candidat, reconstruite de costs − longueur (corpus à coûts analytiques).

    NaN si la longueur sature le cap de la feature (rare : spawns 2-8 m). Ne PAS utiliser sur un
    corpus collecté avec un scoreur appris (costs ≠ analytique)."""
    out = []
    for c, f in zip(d["costs"], d["feats"]):
        length = f[6] * _LEN_CAP
        if f[6] >= 0.9995:
            out.append(float("nan"))
        else:
            out.append(max(0.0, (c - length) / _CFG.block_weight))
    return out


def load_sprint_decisions(run: Path) -> list[dict]:
    """→ une entrée par décision labellisable : état (e,t,h), classe cross/refuse/clear, issue."""
    df, gl = run / "decisions.jsonl", run / "godot.log"
    if _text_path(df) is None or _text_path(gl) is None:
        print(f"[sprint] ⚠️ {run} incomplet (decisions.jsonl/godot.log) — ignoré")
        return []
    decs = [json.loads(line) for line in _open_text(df)]
    es, ts, hs = _drives_series(run)
    gticks, gvals, gstarts = health_series(gl)
    ep_bounds = gstarts[1:] + [gticks[-1] + 10]
    bc_health = not all(math.isnan(h) for h in hs)

    def h_at(t: int) -> float:
        if bc_health and t < len(hs):
            return hs[t]
        i = bisect.bisect_left(gticks, t)          # fallback log Godot (échantillonné /10)
        return gvals[min(i, len(gvals) - 1)]

    out = []
    for i, d in enumerate(decs):
        t0 = d["tick"]
        if t0 >= len(es):                           # queue de log au-delà du BC : rare, ignoré
            continue
        b = bisect.bisect_right(ep_bounds, t0)
        end = ep_bounds[b] if b < len(ep_bounds) else gticks[-1] + 10
        drv = es if d["target"] == "food" else ts
        t1 = pursuit_end(decs, i, drv, end)
        if t1 <= t0 + 20:                           # fenêtre vide (parité trainer) : sautée
            continue
        gain, got = 0.0, False
        for t in range(t0 + 1, min(t1 + 1, len(drv))):
            if drv[t] > drv[t - 1] + 5.0:
                gain, got = drv[t] - drv[t - 1], True
                break
        h0 = h_at(t0)                               # baseline dégâts (jointure exacte)
        hmin = min(h_at(t) for t in range(t0, min(t1 + 1, len(es)))) if bc_health else \
            min(h_at(t0), h_at(t1))
        # corpus post-Phase-A : intrusion exacte + drives VUS par l'étage sont loggés (additif) ;
        # anciens corpus : reconstruction costs−longueur + jointure tick.
        intr = d.get("intr") or route_intrusions(d)
        e0, t0v, h0v = d["drives"] if d.get("drives") else (es[t0], ts[t0], h0)
        chosen_i, direct_i = intr[d["chosen"]], intr[0]
        cls = ("cross" if chosen_i == chosen_i and chosen_i > _INTR_EPS else
               "refuse" if direct_i == direct_i and direct_i > _INTR_EPS else "clear")
        out.append({
            "run": run.name, "tick": t0, "target": d["target"], "explore": bool(d["explore"]),
            "cls": cls, "intr_chosen": chosen_i, "intr_direct": direct_i,
            "e": e0, "t": t0v, "h": h0v, "d_tg": d["feats"][0][3] * 10.0,
            "got": got, "gain": gain, "dmg": max(0.0, h0 - hmin),
            "died": t1 >= end - 2, "left": max(end - 1 - t0, 0), "steps": t1 - t0,
        })
    return out


def measured_drain(runs: list[Path]) -> float:
    """Drain énergie/pas MESURÉ : médiane des baisses tick-à-tick (les remontées = repas/respawns)."""
    drops = []
    for run in runs:
        es, _, _ = _drives_series(run)
        drops += [es[t - 1] - es[t] for t in range(1, len(es))
                  if 0.0 < es[t - 1] - es[t] < 1.0]
    return st.median(drops) if drops else float("nan")


def bucket_table(rows: list[dict], kappa: float, drain: float, death_floor: bool = False) -> dict:
    """U̅ cross vs refuse par bucket santé×énergie, sur les décisions à DIRECT BLOQUÉ."""
    def U(r: dict) -> float:
        if death_floor and r["died"]:
            return -kappa * 100.0
        return r["gain"] / drain - kappa * r["dmg"]

    blocked = [r for r in rows if r["intr_direct"] == r["intr_direct"] and r["intr_direct"] > _INTR_EPS]
    buckets = {
        "sain-affamé  (h>60, e<50)": lambda r: r["h"] > 60 and r["e"] < 50,
        "sain-repu    (h>60, e≥50)": lambda r: r["h"] > 60 and r["e"] >= 50,
        "blessé       (h≤60)      ": lambda r: r["h"] <= 60,
    }
    table = {}
    for name, pred in buckets.items():
        sub = [r for r in blocked if pred(r)]
        cr = [U(r) for r in sub if r["cls"] == "cross"]
        rf = [U(r) for r in sub if r["cls"] == "refuse"]
        table[name] = (len(cr), st.mean(cr) if cr else float("nan"),
                       len(rf), st.mean(rf) if rf else float("nan"))
    return table


def report(rows: list[dict], drain: float) -> None:
    kappa = st.median([r["left"] for r in rows]) / 100.0
    meals = [r["gain"] for r in rows if r["got"]]
    print(f"\n[sprint] === MESURES (pinnent le label du trainer) ===")
    print(f"[sprint] drain mesuré = {drain:.4f}/pas → valeur repas méd = "
          f"{st.median(meals) / drain:.0f} pas (gain méd {st.median(meals):.1f})" if meals else
          f"[sprint] drain mesuré = {drain:.4f}/pas — AUCUN repas observé")
    print(f"[sprint] κ_data (monde v2) = médiane(left {st.median([r['left'] for r in rows]):.0f})/100 "
          f"= {kappa:.1f} pas/dégât")
    cross = [r for r in rows if r["cls"] == "cross"]
    refuse = [r for r in rows if r["cls"] == "refuse"]
    dead_lowh = [r for r in cross if r["died"] and r["h"] < 50]
    frac = len(dead_lowh) / max(len(cross), 1)
    print(f"[sprint] traversées mortes avec santé<50 à la décision : {len(dead_lowh)}/{len(cross)} "
          f"({100 * frac:.0f}%) → variante plancher-mort {'RETENUE' if frac > 0.10 else 'NON (linéaire suffit)'}")

    print(f"\n[sprint] === CLASSES (par run) ===")
    for run in sorted({r["run"] for r in rows}):
        sub = [r for r in rows if r["run"] == run]
        n = {c: sum(1 for r in sub if r["cls"] == c) for c in ("cross", "refuse", "clear")}
        ex = sum(1 for r in sub if r["explore"])
        print(f"[sprint]   {run:>20} : {len(sub):>4} déc | cross {n['cross']:>3} | refuse {n['refuse']:>3} "
              f"| clear {n['clear']:>4} | ε {ex}")
    if cross:
        qs = st.quantiles([r["intr_chosen"] for r in cross], n=4)
        print(f"[sprint] profondeur d'intrusion des traversées (contraste modulation) : "
              f"q1/méd/q3 = {qs[0]:.2f}/{qs[1]:.2f}/{qs[2]:.2f} m")

    for death_floor in (False, True):
        tag = "plancher-mort" if death_floor else "linéaire"
        print(f"\n[sprint] === U̅ cross vs refuse (direct BLOQUÉ, U {tag}, pas de vie) ===")
        for name, (nc, uc, nr, ur) in bucket_table(rows, kappa, drain, death_floor).items():
            print(f"[sprint]   {name} : cross n={nc:>3} U̅={uc:>7.0f} | refuse n={nr:>3} U̅={ur:>7.0f}")

    print(f"\n[sprint] === GATE G0 (pré-enregistré) ===")
    ok_n = len(cross) >= 100 and len(refuse) >= 100
    tb = bucket_table(rows, kappa, drain)
    nc, uc, nr, ur = tb["sain-affamé  (h>60, e<50)"]
    ok_dir = nc >= 20 and nr >= 20 and uc > ur
    ncb, ucb, nrb, urb = tb["blessé       (h≤60)      "]
    blesse = "NA (collecte ε requise)" if ncb < 20 else ("OK" if ucb < urb else "CONTREDIT")
    print(f"[sprint] volumes : cross {len(cross)} (gate ≥100 : {'OK' if len(cross) >= 100 else 'NON'}) | "
          f"refuse {len(refuse)} (≥100 : {'OK' if len(refuse) >= 100 else 'NON'})")
    print(f"[sprint] direction sains-affamés : U̅cross={uc:.0f} vs U̅refuse={ur:.0f} → "
          f"{'OK' if ok_dir else 'NON'}")
    print(f"[sprint] renversement blessés : {blesse}")
    print(f"[sprint] G0 (hors volet blessés) : {'✅' if ok_n and ok_dir else '❌'}")


def selfcheck() -> None:
    """Corpus synthétique 1 vie : 1 sprint payant (repas+dégâts) + 1 refus → classes/issues connues."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        run = Path(td) / "synth"
        run.mkdir()
        n = 400
        es = [70.0 - 0.05 * t for t in range(n)]
        hs = [100.0] * n
        for t in range(120, 150):                       # morsure pendant le sprint
            hs[t] = 100.0 - 0.5 * (t - 119)
        for t in range(150, n):
            hs[t] = hs[149]
        es[160:] = [es[159] + 40.0 - 0.05 * (t - 160) for t in range(160, n)]  # repas à t=160
        with open(run / "ep_0000.jsonl", "w") as f:
            for t in range(n):
                f.write(json.dumps({"obs": {"energy": es[t], "thirst": 60.0, "health": hs[t]}}) + "\n")
        with open(run / "godot.log", "w") as f:
            for t in range(0, n, 10):
                f.write(f"[Godot] Episode 0 | Step {t} | Energy: {es[t]:.1f} | Thirst: 60.0 | "
                        f"Health: {hs[t]:.1f} | Reward: 0\n")
        # feats idx6 = longueur/20 ; costs = longueur + W·intr → décision 1 : direct bloqué CHOISI
        # (cross, intr 0.4) ; décision 2 : direct bloqué, détour choisi (refuse).
        d1 = {"tick": 100, "target": "food", "chosen": 0, "explore": False, "n_greens": 3,
              "costs": [round(3.0 + _CFG.block_weight * 0.4, 3), 6.0],
              "feats": [[0, 0, 0, 0.3, 0, 1, 3.0 / 20, 0.06, 0.06, 1], [0] * 3 + [0.3, 0, 1, 6.0 / 20, 0.5, 0.5, 0]]}
        d2 = {"tick": 200, "target": "food", "chosen": 1, "explore": False, "n_greens": 3,
              "costs": [round(2.0 + _CFG.block_weight * 0.3, 3), 5.0],
              "feats": [[0, 0, 0, 0.2, 0, 1, 2.0 / 20, 0.07, 0.07, 1], [0] * 3 + [0.2, 0, 1, 5.0 / 20, 0.5, 0.5, 0]]}
        with open(run / "decisions.jsonl", "w") as f:
            f.write(json.dumps(d1) + "\n" + json.dumps(d2) + "\n")
        rows = load_sprint_decisions(run)
        assert len(rows) == 2, rows
        r1, r2 = rows
        assert r1["cls"] == "cross" and abs(r1["intr_chosen"] - 0.4) < 0.01, r1
        assert r1["got"] and abs(r1["gain"] - 40.0) < 1.0, r1                 # repas vu à t=160 < fin poursuite
        assert abs(r1["dmg"] - 15.0) < 1.5, r1                                # creux de santé 100→85
        assert r2["cls"] == "refuse" and not r2["got"], r2                    # aucune remontée après t=160 → poursuite sans repas
        drain = measured_drain([run])
        assert abs(drain - 0.05) < 0.005, drain
    print("[selfcheck] OK — classes, jointure, repas, creux santé, drain")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[
        "data/replay_buffer/critic_kin_g24as1", "data/replay_buffer/critic_kin_g24as2",
        "data/replay_buffer/critic_kin_g24bs1", "data/replay_buffer/critic_kin_g24bs2"])
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    runs = [Path(r) for r in args.runs]
    rows = []
    for run in runs:
        rows += load_sprint_decisions(run)
    if not rows:
        print("[sprint] aucun corpus lisible")
        return
    report(rows, measured_drain(runs))


if __name__ == "__main__":
    main()
