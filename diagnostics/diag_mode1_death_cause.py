"""diag_mode1_death_cause — le plateau de survie est-il un problème de DÉCISION, de MOTEUR, ou de PERCEPTION ?

Question owner : le mur de survie n'est-il pas AMPLIFIÉ par un corps lent/peu fluide (trajets longs →
plus de drain en transit → marge d'erreur mince) plutôt que par l'arbitrage ?

Test GRATUIT (read-only sur les buffers gate2b déjà collectés ; ne perturbe PAS le run en cours) :
pour chaque épisode qui MEURT (done=True), on reconstruit — depuis la RÉTINE (color-gating exact de
obs._color_gated_depths, RED=bouffe/BLUE=eau) — la trajectoire de la ressource QUI A TUÉ dans la fenêtre
avant la mort, et on classe la mort :
  - 'introuvable'       : la ressource qui tue n'était PAS visible → PERCEPTION/exploration.
  - 'en_route_lent'     : elle était visible et l'agent S'EN APPROCHAIT mais a manqué de temps → MOTEUR/marge.
  - 'campe_sur_autre'   : elle était visible mais l'agent était COLLÉ à l'autre ressource (drive satisfait) → DÉCISION myope.
  - 'erre'              : visible, ni approchée ni campé → DÉCISION/exploration.

+ Sonde 2 (trajet vs drain), EMPIRIQUE (aucune constante externe) : taux de rapprochement médian
(Δdepth/macro-pas quand l'agent approche) × budget-énergie-en-macropas → « portée atteignable » en
unités-depth, comparée à la distance typique des ressources. Portée >> distance → le corps a de la marge
(PAS le mur) ; portée ≈ distance → serré (moteur-amplifié plausible).

Lancer : PYTHONPATH=python env_pytorch_3.12/bin/python diag_mode1_death_cause.py \
             [--glob 'data/checkpoints/mode1_ppo_gate2b/iter_*/buffer'] [--window 10] [--selfcheck]
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from pathlib import Path

from sylvan.control.mode1.obs import _color_gated_depths, RED, BLUE
from sylvan.control.mode1.rollout_mode1 import _split_episodes

# Régime gate2b (= gate1) : drain 0.05/pas physique, 10 pas physiques par macro-transition (replan-every).
DRAIN_PER_STEP = 0.05
STEPS_PER_MACRO = 10
CLOSE_DEPTH = 0.20      # « collé » à une ressource : profondeur perçue < 0.20
VISIBLE = 0.999         # depth < 0.999 = un rayon a touché la ressource


def _read_lines(path: Path) -> list[dict]:
    """Lecteur ROBUSTE (une ligne JSON éventuellement incomplète = iter en cours d'écriture → skip)."""
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                continue  # ligne à moitié écrite (buffer live)
    return out


def _nearest(retina, color) -> float:
    """Profondeur du rayon le plus proche de cette couleur (1.0 = rien de visible)."""
    return min(_color_gated_depths(retina, color))


def classify_death(ep: list[dict], window: int) -> str:
    last = ep[-1]
    e, t = float(last["energy"]), float(last["thirst"])
    killed_by_food = e <= t            # la faim tue si energy est le drive le plus bas à la mort
    kill_color, other_color = (RED, BLUE) if killed_by_food else (BLUE, RED)

    win = ep[-window:] if len(ep) >= window else ep
    kill_d = [_nearest(tr["retina"], kill_color) for tr in win]
    other_d = [_nearest(tr["retina"], other_color) for tr in win]

    vis = [d for d in kill_d if d < VISIBLE]
    if not vis:
        return "introuvable"
    # tendance d'approche : moyenne 1ʳᵉ moitié vs 2ᵉ moitié (sur toute la fenêtre, non-vis = 1.0 = loin)
    h = len(kill_d) // 2 or 1
    first, second = st.mean(kill_d[:h]), st.mean(kill_d[h:])
    if second < first - 0.03:          # se rapproche nettement
        return "en_route_lent"
    if st.median(other_d) < CLOSE_DEPTH:   # collé à l'AUTRE ressource (drive satisfait)
        return "campe_sur_autre"
    return "erre"


def closing_rate(episodes: list[list[dict]]) -> tuple[float, float]:
    """Taux de rapprochement médian (Δdepth>0 entre macro-pas consécutifs, sur la ressource la plus proche)
    et distance médiane des ressources visibles (unités-depth). Empirique, sans constante externe."""
    deltas, dists = [], []
    for ep in episodes:
        prev = None
        for tr in ep:
            near = min(_nearest(tr["retina"], RED), _nearest(tr["retina"], BLUE))
            if near < VISIBLE:
                dists.append(near)
                if prev is not None and prev < VISIBLE:
                    d = prev - near            # >0 = s'est rapproché depuis le macro-pas précédent
                    if d > 0:
                        deltas.append(d)
            prev = near
    rate = st.median(deltas) if deltas else 0.0
    dist = st.median(dists) if dists else float("nan")
    return rate, dist


def selfcheck() -> None:
    # rétine synthétique : rayon 0 = rouge proche (depth 0.2), rayon 1 = bleu loin (depth 0.5)
    retina = [1.0, 0.0, 0.0, 0.0] * 36
    retina[0:4] = [0.2, 0.9, 0.1, 0.1]   # rouge proche
    retina[4:8] = [0.5, 0.1, 0.1, 0.9]   # bleu moyen
    assert abs(_nearest(retina, RED) - 0.2) < 1e-9, _nearest(retina, RED)
    assert abs(_nearest(retina, BLUE) - 0.5) < 1e-9, _nearest(retina, BLUE)
    # épisode mort de faim (energy bas) collé à l'eau (bleu proche), bouffe visible mais pas approchée
    def line(e, t, food_d, water_d):
        ret = [1.0, 0.0, 0.0, 0.0] * 36
        ret[0:4] = [food_d, 0.9, 0.1, 0.1]
        ret[4:8] = [water_d, 0.1, 0.1, 0.9]
        return {"energy": e, "thirst": t, "retina": ret, "done": True}
    ep = [line(20, 90, 0.6, 0.1)] * 6 + [line(2, 88, 0.6, 0.1)]
    assert classify_death(ep, 10) == "campe_sur_autre", classify_death(ep, 10)
    print("[selfcheck] OK — _nearest (RED/BLUE), classify_death (campe_sur_autre)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/checkpoints/mode1_ppo_gate2b/iter_*/buffer")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return

    all_eps: list[list[dict]] = []
    for d in sorted(glob.glob(args.glob)):
        for p in sorted(Path(d).glob("part-*.jsonl")):
            all_eps.extend(_split_episodes(_read_lines(p)))

    deaths = [ep for ep in all_eps if ep and bool(ep[-1].get("done"))]
    truncs = [ep for ep in all_eps if ep and bool(ep[-1].get("truncated"))]
    print(f"glob={args.glob}")
    print(f"épisodes={len(all_eps)} | morts(done)={len(deaths)} | tronqués(cap)={len(truncs)}")
    if not deaths:
        print("aucune mort dans l'échantillon (que des troncations ?) → survie ~saturée, moteur PAS le mur.")
        return

    cats = {"introuvable": 0, "en_route_lent": 0, "campe_sur_autre": 0, "erre": 0}
    by_hunger = 0
    for ep in deaths:
        cats[classify_death(ep, args.window)] += 1
        if float(ep[-1]["energy"]) <= float(ep[-1]["thirst"]):
            by_hunger += 1
    n = len(deaths)
    print(f"\nmorts par FAIM={by_hunger}/{n}  par SOIF={n-by_hunger}/{n}")
    print("\nCAUSE DE MORT (BUT = décision vs moteur vs perception) :")
    print(f"{'catégorie':<20}{'n':>6}{'%':>8}   interprétation")
    label = {"campe_sur_autre": "DÉCISION (myopie)", "erre": "DÉCISION/explo",
             "en_route_lent": "MOTEUR/marge", "introuvable": "PERCEPTION/explo"}
    for k in ("campe_sur_autre", "erre", "en_route_lent", "introuvable"):
        print(f"{k:<20}{cats[k]:>6}{100*cats[k]/n:>7.0f}%   {label[k]}")

    rate, dist = closing_rate(all_eps)
    budget_macro = 100.0 / (DRAIN_PER_STEP * STEPS_PER_MACRO)   # macro-pas pour épuiser un drive plein
    reach = rate * budget_macro
    print(f"\nSONDE 2 — trajet vs drain (empirique, unités-depth) :")
    print(f"  taux de rapprochement médian = {rate:.4f} depth/macro-pas")
    print(f"  budget = {budget_macro:.0f} macro-pas (drain {DRAIN_PER_STEP}/pas × {STEPS_PER_MACRO})")
    print(f"  → portée atteignable ≈ {reach:.2f} depth  |  distance ressource médiane = {dist:.3f} depth")
    marge = reach / dist if dist and dist == dist else float("inf")
    print(f"  → marge portée/distance ≈ {marge:.1f}×  ({'AMPLE → corps PAS le mur' if marge > 3 else 'SERRÉ → moteur plausible'})")

    print("\n--- VERDICT ---")
    dec = cats["campe_sur_autre"] + cats["erre"]
    if dec >= 0.5 * n and cats["en_route_lent"] < 0.3 * n:
        print(f"DÉCISION domine ({100*dec/n:.0f}% campe/erre vs {100*cats['en_route_lent']/n:.0f}% en-route) "
              "→ le mur est l'ARBITRAGE (job de Mode-1), pas le corps. Leviers décision (pain-shaping/explo).")
    elif cats["en_route_lent"] >= 0.4 * n:
        print(f"MOTEUR-amplifié ({100*cats['en_route_lent']/n:.0f}% meurent en s'approchant) → le corps lent "
              "RÉTRÉCIT la marge. Mais vitesse hexapod_v2 = plafond atteint → décision d'abord, corps = gros redesign.")
    elif cats["introuvable"] >= 0.4 * n:
        print(f"PERCEPTION/explo domine ({100*cats['introuvable']/n:.0f}% ne voient pas la ressource) → explorer/scanner.")
    else:
        print(f"Mixte : décision={100*dec/n:.0f}% moteur={100*cats['en_route_lent']/n:.0f}% "
              f"perception={100*cats['introuvable']/n:.0f}% → croiser avec la sonde 2 (marge portée/distance).")


if __name__ == "__main__":
    main()
