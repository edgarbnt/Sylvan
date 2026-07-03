"""diag_none_visibility — l'errance (replans sans-cible) est-elle AVEUGLE (H1 : rien dans la
rétine → il faut CHERCHER) ou VOYANTE (H2/H3 : ressource visible mais pas de traction / perception
qui lâche) ? + sur les poursuites avortées : la cible était-elle encore visible à l'abandon ?

CONTEXTE (2026-07-04). Sonde cible-planner (a374263) : 69-89% des replans n'approchent RIEN, et le
plancher de survie en monde 1+1 a fait INFÉRER « perception-bound » — jamais mesuré. Ce diag croise,
sur les buffers DÉJÀ sur disque (hesit_probe_{55,11,55_surv,11_surv}), la cible déclarée du planner
(`plan.target`) et la rétine du même tick (`_color_gated_depths` : un rayon rouge/bleu touche-t-il ?).

MESURES par buffer :
  1. part AVEUGLE des replans sans-cible (ni rouge ni bleu visibles) → tranche H1 ;
  2. part VOYANTE des sans-cible (≥1 ressource visible) → tranche H2/H3 ;
  3. poursuites avortées : cible encore VISIBLE à l'abandon (→ H2, le plan cesse de fermer) vs
     DÉCROCHÉE (plus visible sur les derniers replans → H3/H1-en-route) ;
  + visibilité de fond (fraction des replans où rouge/bleu visibles) = contexte du monde.

CRITÈRES PRÉ-ENREGISTRÉS (avant le run) :
  - H1 CONFIRMÉE si aveugle >= 60% des sans-cible (surtout en 1+1) → chantier = CHERCHER + mémoire.
  - H2/H3 si voyant-mais-inerte domine (>= 60%) → retour coût/engagement (H2) ou perception (H3),
    départagés par la mesure 3 (décrochage majoritaire → H3).
  - Mixte → chiffrer les tranches, attaquer la plus grosse.

Lancer :  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_none_visibility.py [--selfcheck]

RÉSULTAT (2026-07-04) :
- H1 RÉFUTÉE NET : part aveugle = 0% PARTOUT (rouge visible 78-100%, bleu 88-97% — la rétine
  360°/10 m couvre ~toute l'arène, même en 1+1) ; 100% des avortées ont leur cible EN VUE à
  l'abandon. L'errance n'est pas un problème de visibilité → CHERCHER n'est PAS le chantier.
- SECTION COORDS (la suite décisive) : erreur ‖position crue − estimation rétine-argmin‖ —
  **BOUFFE (slot appris) : médiane 1.2-1.9 m (5+5) et 2.3-4.3 m (1+1), p90 4.3-7.9 m** vs
  **EAU (EMA oracle) : ~0.85 m partout**. Le slot, entraîné sur des mondes 1-bouffe-sans-eau,
  est HORS-DISTRIBUTION en multi-ressource → le planner poursuit des POSITIONS FANTÔMES pour la
  bouffe. Unifie tout : voyant-mais-inerte, avortées-en-vue, morts de FAIM avec boissons OK
  (eau fiable/bouffe fantôme), densité 5+5 qui masque (un fantôme tombe plus souvent près d'un
  vrai item). Cause racine du plateau = QUALITÉ DES COORDS BOUFFE, ni coût ni visibilité ni moteur.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_forage_hesitation import RESPAWN_JUMP  # noqa: E402
from diag_mode1_death_cause import VISIBLE  # noqa: E402
from sylvan.control.mode1.obs import BLUE, RED, _color_gated_depths  # noqa: E402

BUFFERS = {
    "designed 5+5": "data/replay_buffer/hesit_probe_55/ep_0000.jsonl",
    "designed 1+1": "data/replay_buffer/hesit_probe_11/ep_0000.jsonl",
    "survie   5+5": "data/replay_buffer/hesit_probe_55_surv/ep_0000.jsonl",
    "survie   1+1": "data/replay_buffer/hesit_probe_11_surv/ep_0000.jsonl",
}
ABORT_TAIL = 3      # replans de fin de run inspectés pour « encore visible à l'abandon ? »


def load_replans(path: str) -> list[dict]:
    """Lignes de replan (avec `plan`) → {target, e, t, red_vis, blue_vis}."""
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            p = r.get("plan")
            if p is None:
                continue
            ret = r["wm"]["retina0"]
            rows.append({
                "target": p.get("target", "none"),
                "e": float(r["obs"]["energy"]), "t": float(r["obs"]["thirst"]),
                "red": min(_color_gated_depths(ret, RED)) < VISIBLE,
                "blue": min(_color_gated_depths(ret, BLUE)) < VISIBLE,
                "food": p.get("food"), "water": p.get("water"),
                "ret_food": _retina_pos(ret, RED), "ret_water": _retina_pos(ret, BLUE),
            })
    return rows


RETINA_RANGE_M = 10.0   # MAX_RANGE (perception.gd)


def _retina_pos(ret: list[float], color: str) -> tuple[float, float] | None:
    """Estimation rétine-argmin de la position (x_right, z_fwd) du plus proche item de la couleur."""
    d = _color_gated_depths(ret, color)
    m = min(d)
    if m >= VISIBLE:
        return None
    k = d.index(m)
    b = 2.0 * math.pi * k / 36
    return (m * RETINA_RANGE_M * math.sin(b), m * RETINA_RANGE_M * math.cos(b))


def coords_error(eps: list[list[dict]]) -> dict:
    """Erreur ‖crue − rétine‖ par ressource (bouffe = slot appris, eau = EMA oracle)."""
    ef, ew = [], []
    for ep in eps:
        for r in ep:
            for cru, est, acc in ((r["food"], r["ret_food"], ef), (r["water"], r["ret_water"], ew)):
                if cru is not None and est is not None:
                    acc.append(math.hypot(cru[0] - est[0], cru[1] - est[1]))
    med = lambda v: float(np.median(v)) if v else float("nan")
    p90 = lambda v: float(np.percentile(v, 90)) if v else float("nan")
    return {"food_med": med(ef), "food_p90": p90(ef), "water_med": med(ew), "n": len(ef)}


def split_eps(rows: list[dict]) -> list[list[dict]]:
    eps, cur = [], []
    for r in rows:
        if cur and (r["e"] - cur[-1]["e"] > RESPAWN_JUMP or r["t"] - cur[-1]["t"] > RESPAWN_JUMP):
            eps.append(cur)
            cur = []
        cur.append(r)
    if cur:
        eps.append(cur)
    return [e for e in eps if len(e) > 10]


def target_runs(ep: list[dict]) -> list[tuple[int, int, str]]:
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


def consumed_during(ep: list[dict], a: int, b: int, color: str) -> bool:
    key = "e" if color == "food" else "t"
    hi = min(len(ep) - 1, b + ABORT_TAIL)
    return any(5.0 < ep[i][key] - ep[i - 1][key] < 45.0 for i in range(a + 1, hi + 1))


def analyze(eps: list[list[dict]]) -> dict:
    none_blind = none_seeing = 0
    n_replans = n_none = 0
    red_vis = blue_vis = 0
    ab_dropped = ab_visible = 0
    for ep in eps:
        for r in ep:
            n_replans += 1
            red_vis += r["red"]
            blue_vis += r["blue"]
            if r["target"] == "none":
                n_none += 1
                if r["red"] or r["blue"]:
                    none_seeing += 1
                else:
                    none_blind += 1
        runs = target_runs(ep)
        for a, b, tg in runs[:-1]:                       # dernier run = fin d'épisode, pas un abandon
            if consumed_during(ep, a, b, tg):
                continue                                 # pas avortée
            vis_key = "red" if tg == "food" else "blue"
            tail = ep[max(a, b - ABORT_TAIL + 1): b + 1]
            if any(r[vis_key] for r in tail):
                ab_visible += 1                          # encore visible à l'abandon → H2 (traction)
            else:
                ab_dropped += 1                          # décrochée de la rétine → H3/H1-en-route
    return {"n": n_replans, "none": n_none, "blind": none_blind, "seeing": none_seeing,
            "red_vis": red_vis, "blue_vis": blue_vis,
            "ab_dropped": ab_dropped, "ab_visible": ab_visible}


def selfcheck() -> None:
    ret_none = [1.0, 0.0, 0.0, 0.0] * 36
    ret_red = list(ret_none); ret_red[0:4] = [0.3, 0.9, 0.1, 0.1]
    mk = lambda tg, ret, e=50.0, t=50.0: {"target": tg, "e": e, "t": t,
                                          "red": min(_color_gated_depths(ret, RED)) < VISIBLE,
                                          "blue": min(_color_gated_depths(ret, BLUE)) < VISIBLE}
    # épisode : 5 none-aveugles, 5 none-voyants, puis un run food avorté ENCORE VISIBLE
    ep = [mk("none", ret_none)] * 5 + [mk("none", ret_red)] * 5 \
        + [mk("food", ret_red)] * 4 + [mk("water", ret_red)] * 12
    m = analyze([ep])
    assert m["none"] == 10 and m["blind"] == 5 and m["seeing"] == 5, m
    assert m["ab_visible"] == 1 and m["ab_dropped"] == 0, m
    # run food avorté DÉCROCHÉ (rétine vide sur la fin) → dropped
    ep2 = [mk("food", ret_red)] * 4 + [mk("food", ret_none)] * 4 + [mk("water", ret_red)] * 12
    m2 = analyze([ep2])
    assert m2["ab_dropped"] == 1 and m2["ab_visible"] == 0, m2
    print("[selfcheck] OK — aveugle/voyant, avortée-visible vs décrochée")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return

    print(f"{'buffer':<14}{'replans':>8}{'sans-cible':>11}{'AVEUGLE':>9}{'VOYANT':>8}"
          f"{'vis.rouge':>10}{'vis.bleu':>9}{'avort.décro':>12}{'avort.vis':>10}")
    agg: dict[str, dict] = {}
    for name, path in BUFFERS.items():
        if not Path(path).exists():
            print(f"{name:<14} (absent)")
            continue
        m = analyze(split_eps(load_replans(path)))
        agg[name] = m
        nn = max(1, m["none"])
        nab = max(1, m["ab_dropped"] + m["ab_visible"])
        print(f"{name:<14}{m['n']:>8}{100*m['none']/m['n']:>10.0f}%{100*m['blind']/nn:>8.0f}%"
              f"{100*m['seeing']/nn:>7.0f}%{100*m['red_vis']/m['n']:>9.0f}%{100*m['blue_vis']/m['n']:>8.0f}%"
              f"{100*m['ab_dropped']/nab:>11.0f}%{100*m['ab_visible']/nab:>9.0f}%")

    print("\n=== SECTION COORDS : erreur ‖crue − rétine‖ (bouffe=slot appris, eau=EMA oracle) ===")
    for name, path in BUFFERS.items():
        if not Path(path).exists():
            continue
        c = coords_error(split_eps(load_replans(path)))
        print(f"{name:<14} bouffe méd={c['food_med']:.2f}m p90={c['food_p90']:.2f}m | "
              f"eau méd={c['water_med']:.2f}m (n={c['n']})")

    print("\n--- VERDICT (critères pré-enregistrés) ---")
    for name, m in agg.items():
        nn = max(1, m["none"])
        blind = m["blind"] / nn
        nab = m["ab_dropped"] + m["ab_visible"]
        drop = m["ab_dropped"] / nab if nab else float("nan")
        lead = ("H1 (AVEUGLE → CHERCHER+mémoire)" if blind >= 0.60
                else ("H2/H3 (VOYANT-inerte → coût/perception)" if blind <= 0.40 else "MIXTE"))
        print(f"{name}: aveugle={100*blind:.0f}% → {lead} | avortées décrochées={100*drop:.0f}% "
              f"({'H3 perception lâche' if nab and drop >= 0.6 else 'H2 traction' if nab and drop <= 0.4 else 'mixte'})")


if __name__ == "__main__":
    main()
