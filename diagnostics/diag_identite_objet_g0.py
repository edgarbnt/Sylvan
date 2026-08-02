"""G0 GRATUIT — la bascule silencieuse de cible coûte-t-elle quelque chose ?

Gates pré-inscrits dans `docs/design_identite_objet.md`. Zéro Godot, zéro entraînement.

LE FAIT DE DÉPART `[MESURÉ: diag_perception_honnete_g0.py --question une-proie]` : 85 % des ticks, la
position lue par le slot est réellement occupée par une proie. L'entité n'HALLUCINE pas — elle
regarde une VRAIE proie, mais pas forcément la même d'un instant à l'autre. Son repère se recalcule
à neuf chaque tick, sans notion de continuité.

⚠️ ORDRE DES GATES : le plus TUANT d'abord. On demande « est-ce que ça COÛTE ? » AVANT « est-ce
faisable ? ». La réserve est écrite au design : dans ce monde, la théorie du régime optimal a déjà
montré que prendre N'IMPORTE QUELLE proie est optimal (278 pts/1000 pas contre 125 pour le meilleur
type seul). Les bascules pourraient donc être fréquentes et SANS COÛT — auquel cas le chantier meurt
ici, et c'est un négatif propre.

  G0-1  fréquence des bascules >= 20 % des approches sous 3 m
        ET écart de réussite (sans bascule − avec bascule) >= 15 points
        🛑 STOP si bascules < 10 % OU écart < 5 points
  G0-2  dans >= 80 % des éclipses, le déplacement de la proie < 1/3 de l'espacement inter-bosquets
        ⇒ ré-identification par simple continuité, ZÉRO apprentissage
  G0-3  >= 50 % des paires de proies simultanées ont des teintes distinctes

⚠️ `food_rel0` est un ORACLE D'ÉVAL. La détection de bascule, elle, se fait sur ce que l'entité LIT
   (le slot), pas sur la vérité — c'est bien son basculement à elle qu'on mesure.

Usage :
    PYTHONPATH=python SYLVAN_RETINA_FOV_DEG=120 env_pytorch_3.12/bin/python \
        diagnostics/diag_identite_objet_g0.py
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st
import sys

import torch

sys.path.insert(0, "diagnostics")
from diag_drive_corpus import FOOD_HUES, RETINA_RANGE_M, TOUCH_MAX
from sylvan.models.command_wm import CommandWorldModel

SWITCH_M = 2.0        # saut de la position LUE en un tick => changement d'objet (proie : 0,023 m/tick)
NEAR_M = 3.0          # une « approche »
PATCH_SPACING_M = 8.7  # espacement MINIMUM entre bosquets (SYLVAN_FOOD_PATCH_SPACING)

BAR_FREQ, BAR_COST = 0.20, 15.0
STOP_FREQ, STOP_COST = 0.10, 5.0


def world(rel: tuple[float, float], t: list[float]) -> tuple[float, float]:
    c, s = math.cos(t[2]), math.sin(t[2])
    return (t[0] + rel[1] * s + rel[0] * c, t[1] + rel[1] * c - rel[0] * s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=[f"data/replay_buffer/sp2_{a}{s}" for a in ("ref", "on") for s in (1, 2, 3)])
    ap.add_argument("--wm", default="data/checkpoints/wm_foret_v2_slot/wm_best.pt")
    args = ap.parse_args()

    payload = torch.load(args.wm, map_location="cpu", weights_only=False)
    wm = CommandWorldModel.from_checkpoint(payload)
    wm.eval()
    wm.requires_grad_(False)
    pal = torch.tensor(FOOD_HUES)
    pal = pal / pal.norm(dim=-1, keepdim=True)

    n_app = n_sw = ok_sw = ok_no = tot_sw = tot_no = 0
    eclipses: list[float] = []
    pair_tot = pair_distinct = 0

    for run in args.runs:
        for f in sorted(glob.glob(f"{run}/*.jsonl")):
            R, T, D, ATE = [], [], [], []
            for line in open(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                w, ob = r.get("wm"), r.get("obs")
                if not w:
                    continue
                ret = w.get("retina0")
                t = w.get("torso0") or (ob or {}).get("torso")
                fr = w.get("food_rel0") or [0.0, 0.0, 0.0]
                if not ret or len(ret) != 144 or not t:
                    continue
                R.append(ret)
                T.append(t)
                D.append(math.hypot(fr[0], fr[1]))
                ATE.append(float(w.get("ate", 0.0)) > 0.5)
            if len(R) < 30:
                continue
            X = torch.tensor(R)
            with torch.no_grad():
                P = wm.slot_encoder.positions(X)[:, 0, :]
            sw = [world((float(P[i, 0]), float(P[i, 1])), T[i]) for i in range(len(T))]
            switch = [False] + [math.hypot(sw[i][0] - sw[i - 1][0], sw[i][1] - sw[i - 1][1]) > SWITCH_M
                                for i in range(1, len(sw))]

            # --- G0-1 : approches AVEC vs SANS bascule
            i = 0
            while i < len(D):
                if D[i] < NEAR_M:
                    j, has_sw, ate = i, False, False
                    while j < len(D) and D[j] < NEAR_M:
                        has_sw = has_sw or switch[j]
                        ate = ate or ATE[j]
                        j += 1
                    n_app += 1
                    if has_sw:
                        n_sw += 1
                        tot_sw += 1
                        ok_sw += int(ate)
                    else:
                        tot_no += 1
                        ok_no += int(ate)
                    i = j
                else:
                    i += 1

            # --- G0-2 : durée des éclipses (la cible n'est plus lue au même endroit)
            run_len = 0
            for i in range(1, len(sw)):
                if switch[i]:
                    if run_len:
                        eclipses.append(run_len)
                    run_len = 0
                else:
                    run_len += 1

            # --- G0-3 : deux proies simultanées ont-elles des teintes distinctes ?
            r3 = X.view(-1, 36, 4)
            dd, cc = r3[..., 0], r3[..., 1:]
            ccn = cc / (cc.norm(dim=-1, keepdim=True) + 1e-6)
            cos = ccn @ pal.T
            hit = (cos.amax(-1) > 0.98) & (dd < TOUCH_MAX)
            which = cos.argmax(-1)
            for k in range(len(r3)):
                h = which[k][hit[k]]
                if h.numel() >= 2:
                    pair_tot += 1
                    pair_distinct += int(h.unique().numel() >= 2)

    print(f"{n_app} approches sous {NEAR_M} m\n")
    print("G0-1 — LA BASCULE COÛTE-T-ELLE QUELQUE CHOSE ?")
    freq = n_sw / max(1, n_app)
    r_sw = 100 * ok_sw / max(1, tot_sw)
    r_no = 100 * ok_no / max(1, tot_no)
    cost = r_no - r_sw
    print(f"  approches contenant une bascule : {100 * freq:.1f} %  (barre {100 * BAR_FREQ:.0f} %)")
    print(f"  réussite SANS bascule : {r_no:.1f} %  (n={tot_no})")
    print(f"  réussite AVEC bascule : {r_sw:.1f} %  (n={tot_sw})")
    print(f"  écart = {cost:+.1f} points  (barre {BAR_COST:.0f})")
    ok1 = freq >= BAR_FREQ and cost >= BAR_COST
    stop1 = freq < STOP_FREQ or cost < STOP_COST
    print(f"  {'✅' if ok1 else ('🛑' if stop1 else '~')} G0-1\n")

    print("G0-2 — LA GÉOMÉTRIE SUFFIRAIT-ELLE ?")
    if eclipses:
        q = st.quantiles(eclipses, n=100)
        print(f"  {len(eclipses)} segments sans bascule · durée méd={st.median(eclipses):.0f} ticks "
              f"· q90={q[89]:.0f}")
        # déplacement de la proie pendant une éclipse de cette durée
        dur = st.quantiles(eclipses, n=100)[89]
        disp = 0.023 * dur
        print(f"  déplacement d'une proie sur q90 ({dur:.0f} ticks) = {disp:.2f} m")
        print(f"  espacement MINIMUM entre bosquets = {PATCH_SPACING_M} m "
              f"→ seuil d'ambiguïté = {PATCH_SPACING_M / 3:.2f} m")
        ok2 = disp < PATCH_SPACING_M / 3
        print(f"  {'✅' if ok2 else '❌'} G0-2 — {'suivi trivial, zéro apprentissage' if ok2 else 'apparence nécessaire'}\n")
    else:
        print("  (aucune éclipse détectée)\n")

    print("G0-3 — DEUX PROIES SIMULTANÉES SONT-ELLES DISTINGUABLES ?")
    if pair_tot:
        p = 100 * pair_distinct / pair_tot
        print(f"  {pair_tot} ticks avec >=2 proies visibles · teintes distinctes : {p:.1f} %  (barre 50 %)")
    else:
        print("  (jamais deux proies visibles simultanément)")

    print("\n" + "=" * 74)
    if stop1:
        print("🛑 G0-1 STOP pré-enregistré — la bascule de cible ne coûte pas assez.")
        print("   Changer de proie en cours de route ne fait pas échouer l'approche : dans ce")
        print("   monde, prendre N'IMPORTE QUELLE proie est optimal (théorie du régime optimal,")
        print("   mesurée le 2026-08-02). L'identité d'objet n'achèterait rien ICI.")
        print("   ⇒ négatif propre, ne pas payer G1. C'était la réserve écrite au design.")
    elif ok1:
        print("✅ G0-1 PASSÉ — la bascule coûte. Le chantier est licencié, voir G0-2/G0-3 pour")
        print("   savoir s'il faut apprendre ou s'il suffit de suivre.")
    else:
        print("~ G0-1 ZONE GRISE — coût réel mais sous la barre. Décision owner.")


if __name__ == "__main__":
    main()
