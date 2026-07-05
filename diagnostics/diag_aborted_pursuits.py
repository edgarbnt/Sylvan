"""diag_aborted_pursuits — pourquoi la config RECORD (slot-2 × survie) abandonne-t-elle encore
51% (5+5) / 82% (1+1) de ses poursuites VRAIES (cible du planner) ?

DÉCOMPOSITION par run avorté (offline, buffers _slot2b déjà sur disque) :
  - interrompu_par_conso : l'AUTRE ressource a été consommée pendant/juste après le run →
    ré-arbitrage légitime post-consommation, PAS un échec (artefact de la métrique « avortée »).
  - flicker : run ultra-court (<= 2 replans) → battement de cible quand les scores des deux plans
    sont quasi égaux — coût du switching quasi nul mais bruit de mesure/commitment.
  - perdu_de_vue : la cible n'était plus visible à la fin du run → dropout de perception (portée).
  - mort : l'épisode se termine (mort/cap) pendant le run.
  - abandon_en_vue : cible visible, pas de conso, run > 2 replans → le VRAI échec à creuser.
+ longueurs médianes des runs convertis vs avortés (un vrai échec dure ; un flicker non).

CRITÈRES PRÉ-ENREGISTRÉS :
  - Si interrompu+flicker >= 60% des avortées → le « 51/82% » est surtout un ARTEFACT de métrique →
    corriger la métrique, pas le comportement.
  - Si abandon_en_vue >= 40% → vrai déficit de committment/conversion → prochain chantier perf dense.
  - Si perdu_de_vue domine en 1+1 → confirme CHERCHER comme chantier épars.

Lancer :  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_aborted_pursuits.py \
              [--files data/replay_buffer/hesit_probe_55_slot2b_surv/ep_0000.jsonl ...] [--selfcheck]
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_plan_target_switches import AFTER_WIN, split_eps  # noqa: E402
from diag_none_visibility import load_replans  # noqa: E402  (target + drives + visibilité par replan)

FLICKER_MAX = 2      # run <= 2 replans = battement


def consumptions(ep: list[dict]) -> list[tuple[int, str]]:
    ev = []
    for i in range(1, len(ep)):
        if 5.0 < ep[i]["e"] - ep[i - 1]["e"] < 45.0:
            ev.append((i, "food"))
        if 5.0 < ep[i]["t"] - ep[i - 1]["t"] < 45.0:
            ev.append((i, "water"))
    return ev


def runs_of(ep: list[dict]) -> list[tuple[int, int, str]]:
    runs: list[tuple[int, int, str]] = []
    for i, r in enumerate(ep):
        tg = r["target"]
        if tg == "none":
            continue
        if runs and runs[-1][2] == tg:
            runs[-1] = (runs[-1][0], i, tg)
        else:
            runs.append((i, i, tg))
    return runs


def analyze(eps: list[list[dict]]) -> dict:
    cats = {"converti": 0, "interrompu_par_conso": 0, "flicker": 0, "perdu_de_vue": 0,
            "mort": 0, "abandon_en_vue": 0}
    len_conv, len_abandon = [], []
    for ep in eps:
        ev = consumptions(ep)
        runs = runs_of(ep)
        for ri, (a, b, tg) in enumerate(runs):
            own = any(a <= i <= b + AFTER_WIN and c == tg for i, c in ev)
            if own:
                cats["converti"] += 1
                len_conv.append(b - a + 1)
                continue
            if ri == len(runs) - 1 and b >= len(ep) - 2:
                cats["mort"] += 1                      # run coupé par la fin d'épisode
                continue
            other = any(a <= i <= b + AFTER_WIN and c != tg for i, c in ev)
            if other:
                cats["interrompu_par_conso"] += 1
                continue
            if (b - a + 1) <= FLICKER_MAX:
                cats["flicker"] += 1
                continue
            vis_key = "red" if tg == "food" else "blue"
            if not any(r[vis_key] for r in ep[max(a, b - 2):b + 1]):
                cats["perdu_de_vue"] += 1
                continue
            cats["abandon_en_vue"] += 1
            len_abandon.append(b - a + 1)
    return {"cats": cats, "len_conv": len_conv, "len_abandon": len_abandon}


def selfcheck() -> None:
    mk = lambda tg, e, t, red=True, blue=True: {"target": tg, "e": e, "t": t, "red": red, "blue": blue,
                                                "food": None, "water": None, "ret_food": None, "ret_water": None}
    # converti (conso food en fin de run) + interrompu (conso water pendant run food) + flicker
    ep = ([mk("food", 50, 50)] * 6 + [mk("food", 90, 50)]          # conso food → converti
         + [mk("water", 90, 50)] * 4 + [mk("water", 90, 48), mk("food", 90, 88)]  # conso water fin → converti
         + [mk("water", 88, 86)] * 1 + [mk("food", 86, 84)] * 8 + [mk("none", 84, 82)] * 4)
    m = analyze([ep])
    assert m["cats"]["converti"] == 2, m
    assert m["cats"]["flicker"] >= 1, m
    print("[selfcheck] OK — converti/flicker classés")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+",
                    default=["data/replay_buffer/hesit_probe_55_slot2b_surv/ep_0000.jsonl",
                             "data/replay_buffer/hesit_probe_11_slot2b_surv/ep_0000.jsonl"])
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    for f in args.files:
        if not Path(f).exists():
            print(f"{f} absent")
            continue
        m = analyze(split_eps(load_replans(f)))
        c = m["cats"]
        n_ab = sum(v for k, v in c.items() if k not in ("converti",))
        tot = sum(c.values())
        print(f"\n=== {f} ===")
        print(f"runs={tot} convertis={c['converti']} ({100*c['converti']/max(1,tot):.0f}%) | avortés={n_ab}")
        for k in ("interrompu_par_conso", "flicker", "perdu_de_vue", "mort", "abandon_en_vue"):
            print(f"  {k:22s}: {c[k]:>4} = {100*c[k]/max(1,n_ab):.0f}% des avortés")
        if m["len_conv"]:
            print(f"  longueur méd runs convertis {st.median(m['len_conv']):.0f} replans"
                  + (f" | abandons-en-vue {st.median(m['len_abandon']):.0f}" if m["len_abandon"] else ""))
    print("\n(critères : interrompu+flicker>=60% → artefact métrique ; abandon_en_vue>=40% → vrai chantier ;"
          "\n perdu_de_vue dominant en 1+1 → CHERCHER confirmé)")


if __name__ == "__main__":
    main()
