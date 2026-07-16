"""G-GAP — la sonde de LICENCE du critique-waypoint (docs/design_critique_waypoint.md, gate 1).

QUESTION (HIQL fig.8 transposée chez nous) : à cet étage, des CHOIX différents mènent-ils à des
ISSUES mesurablement différentes ? Si non, il n'y a rien à apprendre ici non plus → KILL du
chantier (négatif à commiter). Critère PRÉ-ENREGISTRÉ : écart médian de survie-après-décision
entre choix « à-travers-vert » et choix « dégagés » **> 100 pas** (≫ l'erreur réseau ~2e-4×3000
= 0.6 pas — l'écart d'action qui a tué le critique niveau-bas était de ~0.05 pas).

MÉTHODE (gratuite, corpus exploratoire) : jointure décisions (SYLVAN_WP_LOG, tick global) ↔ flux
BC (issue vécue = pas jusqu'à la fin de la vie ; censure exclue comme load_lived). Les décisions
EXPLORÉES (ε-uniformes) fournissent le contrefactuel que l'argmin déterministe ne donne jamais.

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_waypoint_gap.py \
      --runs data/replay_buffer/critic_kin_wpx1 data/replay_buffer/critic_kin_wpx2
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

DEATH_LVL = 3.0          # drive < 3/100 à la fin du segment = mort (convention load_lived : 0.03)
MIN_LIFE = 150           # vies plus courtes = artefacts de découpe, ignorées


def load_run(d: str, run_idx: int = 0) -> list[dict]:
    """→ décisions du run, enrichies de l'issue vécue (steps_after, mort/censure)."""
    bc = Path(d) / "ep_0000.jsonl"
    dec_f = Path(d) / "decisions.jsonl"
    if not bc.exists() or not dec_f.exists():
        print(f"  ⚠️ {d} : fichiers manquants, run ignoré")
        return []
    drives = []
    for line in open(bc, errors="ignore"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        drives.append((float(r["obs"]["energy"]), float(r["obs"]["thirst"])))
    # vies : coupe quand énergie ET soif remontent d'un coup (respawn) — même convention que load()
    bounds = [0]
    for i in range(1, len(drives)):
        if drives[i][0] > drives[i - 1][0] + 20 and drives[i][1] > drives[i - 1][1] + 20:
            bounds.append(i)
    bounds.append(len(drives))
    life_of = {}
    lives = []
    for li in range(len(bounds) - 1):
        s, e = bounds[li], bounds[li + 1]
        if e - s < MIN_LIFE:
            continue
        death = min(drives[e - 1]) < DEATH_LVL
        lives.append({"start": s, "end": e, "death": death})
        for t in range(s, e):
            life_of[t] = len(lives) - 1
    out = []
    for line in open(dec_f):
        rec = json.loads(line)
        li = life_of.get(rec["tick"])
        if li is None:
            continue
        life = lives[li]
        rec["steps_after"] = life["end"] - rec["tick"]
        rec["death"] = life["death"]
        rec["energy"], rec["thirst"] = drives[rec["tick"]]
        rec["life_key"] = (run_idx, life["start"])       # identifiant de vie UNIQUE inter-runs
        out.append(rec)
    print(f"  {d} : {len(out)} décisions jointes, {len(lives)} vies "
          f"({sum(l['death'] for l in lives)} mortes)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["data/replay_buffer/critic_kin_wpx1", "data/replay_buffer/critic_kin_wpx2"])
    args = ap.parse_args()
    decs = [d for ri, r in enumerate(args.runs) for d in load_run(r, ri)]
    expl = [d for d in decs if d["explore"] and d["death"]]          # explorées, vies NON censurées
    print(f"\ntotal={len(decs)} décisions | explorées={sum(d['explore'] for d in decs)} "
          f"({100 * sum(d['explore'] for d in decs) / max(len(decs), 1):.0f}%) | "
          f"explorées-non-censurées={len(expl)}")
    if len(expl) < 40:
        print("KILL(données) : trop peu de décisions explorées non-censurées — allonger la collecte.")
        return

    # ── LE GAP : issue selon la DÉGAGURE du choix commis (dg = min distance vert→segments, brute,
    #    feats[7]/[8] ×10 m). À-travers-vert (<0.5 m) vs dégagé (>1.5 m). ──
    def dg_min(d: dict) -> float:
        f = d["feats"][d["chosen"]]
        return 10.0 * min(f[7], f[8])

    through = [d["steps_after"] for d in expl if dg_min(d) < 0.5]
    clear = [d["steps_after"] for d in expl if dg_min(d) > 1.5]
    print(f"\n  BUT (survie-après-décision, décisions EXPLORÉES, vies mortes)")
    print(f"  choix à-travers-vert (<0.5 m) : n={len(through):>4} méd={st.median(through) if through else float('nan'):>6.0f} pas")
    print(f"  choix dégagés       (>1.5 m) : n={len(clear):>4} méd={st.median(clear) if clear else float('nan'):>6.0f} pas")
    if not through or not clear:
        print("KILL(données) : une des deux classes est vide — ε trop faible ou monde sans contraste.")
        return
    gap = st.median(clear) - st.median(through)
    print(f"  GAP = {gap:+.0f} pas   (gate pré-enregistré : > +100)")

    # proxy morts-danger : parmi les vies mortes, la DERNIÈRE décision avant la mort était-elle
    # à-travers-vert ? (sur-représentation attendue si le choix cause la mort)
    last_by_life: dict[tuple[int, int], dict] = {}
    for d in decs:
        if d["death"]:
            key = d["life_key"]
            if key not in last_by_life or d["tick"] > last_by_life[key]["tick"]:
                last_by_life[key] = d
    lasts = list(last_by_life.values())
    lt = sum(1 for d in lasts if dg_min(d) < 0.5)
    print(f"  dernière décision avant mort à-travers-vert : {lt}/{len(lasts)} vies "
          f"(base : {100 * len(through) / len(expl):.0f}% des décisions explorées)")

    if gap > 100:
        print(f"\nPASS : le choix de waypoint change l'issue de {gap:+.0f} pas — l'étage est APPRENABLE "
              f"(écart ≫ erreur réseau). → entraîner la correction (G-res, G-rank).")
    else:
        print(f"\nKILL : gap {gap:+.0f} ≤ 100 pas — les issues ne dépendent pas assez du choix. "
              f"NE PAS entraîner ; commiter le négatif et re-diagnostiquer le monde/l'étage.")


if __name__ == "__main__":
    main()
