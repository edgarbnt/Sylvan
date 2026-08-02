"""Le monde est-il SURVIVABLE ? — mesurer la viabilité du test AVANT de juger l'entité.

⭐ RÉSULTAT (2026-08-02) : `foret_v1` est INVIVABLE PAR CONSTRUCTION. Tenir en vie exige
~50 m de trajet par 1000 pas ; le corps en parcourt 47. Aucune entité, même parfaite, ne peut
y survivre — et c'est le calcul le PLUS FAVORABLE (espacement minimum entre bosquets, ligne
droite, zéro détour, zéro arbre, proie immobile alors qu'elle fuit à 0,023 m/pas).

POURQUOI CE DIAGNOSTIC EXISTE. Toute une session a été passée à chercher pourquoi l'entité
meurt : on a soupçonné la perception, la visée, l'indécision, la fréquence de replanification,
l'agilité du corps. Trois sondes coûteuses (replan 60, agilité x2, agilité x4) n'ont RIEN bougé,
et l'A/B perception apprise-contre-codée-main n'a rien pu départager. C'était normal : les deux
bras butaient sur une paroi commune EN AVAL. La règle §2 du CLAUDE.md — mesurer la viabilité du
test avant de juger l'agent — aurait dû être appliquée d'abord. Ce script la rend systématique.

CE QU'IL MESURE (uniquement depuis un corpus déjà collecté, zéro Godot, zéro entraînement) :
  1. l'économie de CHAQUE pulsion : réserve de départ, consommation par pas, apport d'une prise
     ⇒ combien de prises par 1000 pas sont NÉCESSAIRES ;
  2. ce que l'entité réalise vraiment ⇒ le déficit, pulsion par pulsion ;
  3. le BUDGET DE TRAJET : mètres parcourables par 1000 pas contre mètres exigés par les
     déplacements entre ressources ⇒ le verdict de viabilité.

LE DÉSÉQUILIBRE TROUVÉ, à garder en tête : la soif descend EXACTEMENT aussi vite que la faim
(0,154/pas) mais une boisson ne rend que 40 quand un repas rend 84. L'eau exige donc DEUX FOIS
plus de trajets pour le même bénéfice — 3,87 boissons/1000 pas contre 1,84 repas. C'est cette
asymétrie qui rend le monde insoluble, et elle tue : soif 26 morts contre faim 6.

⚠️ NE PAS « CORRIGER » EN RENDANT L'ENTITÉ MEILLEURE : il n'y a rien à améliorer tant que le
budget est négatif. Et ne pas non plus élargir les rayons de capture (§2, fausse solution). Le
levier honnête est l'ÉCONOMIE DU MONDE — apport d'une boisson, taux de consommation, ou densité
des points d'eau — jusqu'à ce que le budget redevienne réalisable AVEC une marge.

Usage :
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_viabilite_monde.py \
        --runs data/replay_buffer/replan10
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st

GAIN_MIN = 5.0  # saut de jauge qui signe une PRISE (repas / boisson)
PATCH_SPACING_M = 8.7  # SYLVAN_FOOD_PATCH_SPACING — espacement MINIMUM servi (max 18,5)


def scan(runs: list[str]) -> list[dict]:
    eps = []
    for run in runs:
        for f in sorted(glob.glob(f"{run}/*.jsonl")):
            e = {"energy": [], "thirst": [], "step": [], "n": 0}
            for line in open(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ob, wm = r.get("obs"), r.get("wm")
                if not ob or "energy" not in ob:
                    continue
                e["energy"].append(float(ob["energy"]))
                e["thirst"].append(float(ob.get("thirst", 0.0)))
                t = (wm or {}).get("torso0") or ob.get("torso")
                e["step"].append(t)
                e["n"] += 1
            if e["n"] > 50:
                eps.append(e)
    return eps


def economy(eps: list[dict], key: str) -> dict:
    drain, gain, start = [], [], []
    takes = []
    for e in eps:
        v = e[key]
        start.append(v[0])
        n_take = 0
        for i in range(1, len(v)):
            d = v[i] - v[i - 1]
            if d > GAIN_MIN:
                gain.append(d)
                n_take += 1
            elif d < 0:
                drain.append(-d)
        takes.append(1000 * n_take / len(v))
    if not drain or not gain:
        return {}
    D, G = st.mean(drain), st.median(gain)
    return {
        "start": st.median(start), "drain": D, "gain": G,
        "autonomy": st.median(start) / D, "buys": G / D,
        "need": 1000 * D / G, "got": st.median(takes),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["data/replay_buffer/replan10"])
    ap.add_argument("--spacing", type=float, default=PATCH_SPACING_M)
    args = ap.parse_args()
    eps = scan(args.runs)
    if not eps:
        print("❌ aucun épisode exploitable")
        return
    print(f"{len(eps)} vies\n")

    need_total = 0.0
    for key, label in (("energy", "FAIM  (manger)"), ("thirst", "SOIF  (boire)")):
        r = economy(eps, key)
        if not r:
            continue
        need_total += r["need"]
        deficit = r["got"] < r["need"]
        print(f"{label}")
        print(f"   réserve de départ {r['start']:.0f} → {r['autonomy']:.0f} pas d'autonomie")
        print(f"   consommation {r['drain']:.4f}/pas · une prise rend {r['gain']:.0f} "
              f"→ achète {r['buys']:.0f} pas")
        print(f"   NÉCESSAIRE {r['need']:.2f} / 1000 pas   ·   RÉALISÉ {r['got']:.2f}"
              f"   {'❌ DÉFICIT' if deficit else '✅'}")
        print()

    # Budget de trajet — le verdict.
    sp = []
    for e in eps:
        prev = None
        for t in e["step"]:
            if t and prev:
                sp.append(math.hypot(t[0] - prev[0], t[1] - prev[1]))
            prev = t or prev
    if not sp:
        print("(pas de position enregistrée — relancer avec SYLVAN_WM_COLLECT=1)")
        return
    v = st.median([s for s in sp if s > 1e-4])
    avail = 1000 * v
    required = need_total * args.spacing
    print("BUDGET DE TRAJET par 1000 pas")
    print(f"   parcourable          : {avail:.0f} m  (vitesse médiane {v:.4f} m/pas)")
    print(f"   exigé par les prises : {required:.0f} m  "
          f"({need_total:.2f} trajets x {args.spacing} m d'espacement MINIMUM)")
    ratio = required / avail
    print()
    if ratio > 1.0:
        print(f"🛑 MONDE INVIVABLE : il faudrait {ratio:.1f}x le trajet physiquement possible.")
        print("   Aucune entité, même parfaite, ne peut y survivre. Tout jugement de la")
        print("   perception, du planner ou du corps mesuré dans ce monde est SANS OBJET :")
        print("   les bras d'un A/B butent tous sur cette paroi, donc rien ne se départage.")
        print("   Le levier est l'ÉCONOMIE DU MONDE, pas l'entité (et surtout pas les rayons")
        print("   de capture — §2, fausse solution).")
    elif ratio > 0.7:
        print(f"⚠️  MARGE NULLE : {ratio:.0%} du trajet possible est déjà exigé, sans compter")
        print("   les détours, les arbres, la fuite des proies ni les erreurs de visée.")
    else:
        print(f"✅ viable : {ratio:.0%} du budget de trajet est exigé, il reste de la marge.")


if __name__ == "__main__":
    main()
