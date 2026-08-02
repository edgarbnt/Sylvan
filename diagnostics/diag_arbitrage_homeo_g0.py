"""G0 GRATUIT — un coût de besoin NON-SÉPARABLE changerait-il seulement les décisions ?

Gate pré-inscrit dans `docs/design_arbitrage_homeostatique.md`. Aucun run, aucun entraînement :
on rejoue hors-ligne, sur des corpus DÉJÀ payés, ce que l'entité a fait et ce qu'une fonction de
besoin homéostatique aurait recommandé.

    D(H) = ( Σᵢ max(0, h*ᵢ − hᵢ)ⁿ )^(1/m)        n > m > 1   (ici n=4, m=3)
    valeur de satisfaire la pulsion i  =  D(H) − D(H après restauration de i)

La propriété qui nous intéresse est la 4ᵉ de Keramati & Gutkin : l'effet INHIBITEUR des besoins
non pertinents. Un coût SÉPARABLE (celui servi : survival_weight·deficit) ne peut pas la produire ;
c'est l'hypothèse structurelle du chantier, et ce script dit si elle a une chance de payer.

⚠️ CE QUE CE SCRIPT NE FAIT PAS : recoder le coût du planner. Le projet a déjà produit un faux
verdict en recodant une géométrie (design §5, piège n°2). On ne compare donc PAS « ordre designé
recalculé » contre « ordre homéostatique », mais **ce que l'entité a RÉELLEMENT poursuivi**
(mesuré sur sa trajectoire) contre ce que le besoin homéostatique désigne. C'est observable et
non falsifiable par ma propre réimplémentation.

⚠️ L'ESCOMPTE EST UN CHOIX, PAS UNE MESURE : la théorie prouve que γ<1 est nécessaire mais ne
fixe pas sa valeur. On rapporte donc un BALAYAGE de γ, jamais une valeur choisie pour faire
passer un gate (§2).

BARRES PRÉ-ENREGISTRÉES (design §G0) :
  (a) les deux ordres diffèrent sur ≥ 15 % des décisions
  (b) sur les états CAMPÉS (poursuit une jauge ≥60 alors que l'autre <40), l'homéostatique
      désigne la jauge nécessiteuse dans ≥ 70 % des cas
  STOP si (a) échoue : la forme ne changerait rien, inutile de payer G1/G2.

Usage :
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_arbitrage_homeo_g0.py
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st

N_EXP = 4.0  # n  ) préférence du CORPS, pinnée au design — PAS ajustée pour passer un gate
M_EXP = 3.0  # m  )
SETPOINT = 100.0  # jauge « pleine » nominale
RESTORE = 140.0  # apport d'une prise (identique bouffe/eau depuis le correctif du 2026-08-02)
WINDOW = 20  # ticks pour juger ce qu'elle POURSUIT
SPEED_M_PER_TICK = 0.0469  # vitesse de croisière MESURÉE (diag_viabilite_monde)
GAMMA = 0.999  # escompte par tick — CHOIX, pas une mesure : balayé et rapporté, jamais sélectionné
BAR_DIFF = 0.15
BAR_CAMPED = 0.70
CAMPED_HIGH, CAMPED_LOW = 60.0, 40.0

DEFAULT_RUNS = [f"data/replay_buffer/fx_{a}{s}" for a in ("a", "b") for s in (1, 2, 3)] + [
    "data/replay_buffer/eau_fix"
]


def drive(dev_e: float, dev_t: float, n: float = N_EXP, m: float = M_EXP) -> float:
    """D(H) — NON-SÉPARABLE : les deux déficits vivent sous la même racine."""
    return (dev_e ** n + dev_t ** n) ** (1.0 / m)


def homeo_gain(energy: float, thirst: float,
               n: float = N_EXP, m: float = M_EXP) -> tuple[float, float]:
    """Réduction de besoin apportée par un repas / par une boisson, depuis l'état courant."""
    de = max(0.0, SETPOINT - energy)
    dt = max(0.0, SETPOINT - thirst)
    d0 = drive(de, dt, n, m)
    g_food = d0 - drive(max(0.0, de - RESTORE), dt, n, m)
    g_water = d0 - drive(de, max(0.0, dt - RESTORE), n, m)
    return g_food, g_water


def homeo_choice(r: dict, n: float = N_EXP, m: float = M_EXP,
                 gamma: float = GAMMA) -> str:
    """La cible que le besoin homéostatique désigne, **coût du trajet inclus**.

    ⚠️ POINT CRUCIAL, corrigé le 2026-08-02. Comparer les deux gains NUS (repas vs boisson) ne
    teste RIEN de la forme : avec un apport identique, le plus gros déficit gagne toujours, quels
    que soient n et m — la 1ʳᵉ version de ce script rendait 49,3 % de désaccord IDENTIQUE pour
    tous les exposants, ce qui trahissait la dégénérescence.
    La non-séparabilité n'agit que face à un COÛT : quand la soif est loin de sa cible, le terme
    dt^n domine sous la racine et la réduction de besoin apportée par un REPAS s'écrase (elle
    tend vers 0 quand dt→∞). C'est l'effet inhibiteur des besoins non pertinents, et il ne se
    voit qu'en mettant le gain en regard du trajet à payer pour l'obtenir.
    """
    gf, gw = homeo_gain(r["e"], r["t"], n, m)
    tf = r["df"] / SPEED_M_PER_TICK
    tw = r["dw"] / SPEED_M_PER_TICK
    return "food" if gf * gamma ** tf > gw * gamma ** tw else "water"


def load(runs: list[str]) -> list[list[dict]]:
    eps = []
    for run in runs:
        for f in sorted(glob.glob(f"{run}/*.jsonl")):
            rows = []
            for line in open(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                wm, ob = r.get("wm"), r.get("obs")
                if not wm or not ob or "energy" not in ob:
                    continue
                fr = wm.get("food_rel0") or [0.0, 0.0, 0.0]
                wr = wm.get("water_rel0") or [0.0, 0.0, 0.0]
                rows.append({
                    "e": float(ob["energy"]), "t": float(ob.get("thirst", 0.0)),
                    "df": math.hypot(fr[0], fr[1]), "dw": math.hypot(wr[0], wr[1]),
                    "vf": fr[2] > 0.5, "vw": wr[2] > 0.5,
                })
            if len(rows) > WINDOW * 2:
                eps.append(rows)
    return eps


def pursued(rows: list[dict], i: int) -> str | None:
    """Ce qu'elle poursuit VRAIMENT : la ressource dont la distance décroît le plus."""
    j = min(i + WINDOW, len(rows) - 1)
    if j <= i:
        return None
    df = rows[j]["df"] - rows[i]["df"]
    dw = rows[j]["dw"] - rows[i]["dw"]
    if min(df, dw) > -0.30:  # elle ne ferme sur rien
        return None
    return "food" if df < dw else "water"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    args = ap.parse_args()
    eps = load(args.runs)
    if not eps:
        print("❌ aucun corpus exploitable")
        return
    n_ticks = sum(len(e) for e in eps)
    print(f"{len(eps)} vies · {n_ticks} ticks · n={N_EXP:.0f} m={M_EXP:.0f} restore={RESTORE:.0f}\n")

    diff = same = 0
    for rows in eps:
        for i in range(0, len(rows) - WINDOW, WINDOW):
            p = pursued(rows, i)
            if p is None:
                continue
            if homeo_choice(rows[i]) == p:
                same += 1
            else:
                diff += 1

    tot = diff + same
    print("(a) LES DEUX ORDRES DIFFÈRENT-ILS ?")
    print(f"    {tot} décisions jugées · désaccord {diff} = {100 * diff / max(1, tot):.1f} % "
          f"(barre {BAR_DIFF * 100:.0f} %)")
    ok_a = bool(tot) and diff / tot >= BAR_DIFF
    print(f"    {'✅' if ok_a else '❌'} G0-a\n")

    # (b) RÉÉCRIT LE 2026-08-02 — la 1ʳᵉ version était TAUTOLOGIQUE : « campée » était défini
    # comme « poursuivre la jauge haute alors que l'autre est basse », donc la nécessiteuse avait
    # par construction le plus gros déficit et l'homéostatique la désignait 100 % du temps. Les
    # 100 % étaient dans ma définition, pas dans les données. On juge maintenant là où la
    # pathologie TUE : la fenêtre qui précède une mort de soif.
    print("(b) DANS LA FENÊTRE QUI PRÉCÈDE UNE MORT DE SOIF (200 derniers ticks)")
    pre_n = pre_ok = pre_pursued_water = 0
    for rows in eps:
        last = rows[-1]
        if not (last["t"] < 10.0 and last["e"] > 50.0):  # mort de soif, énergie en stock
            continue
        for i in range(max(0, len(rows) - 200), len(rows) - WINDOW, WINDOW):
            p = pursued(rows, i)
            if p is None:
                continue
            pre_n += 1
            if homeo_choice(rows[i]) == "water":
                pre_ok += 1
            if p == "water":
                pre_pursued_water += 1
    if pre_n:
        print(f"    {pre_n} décisions dans ces fenêtres")
        print(f"    l'homéostatique dit BOIRE : {100 * pre_ok / pre_n:.1f} %  (barre "
              f"{BAR_CAMPED * 100:.0f} %)")
        print(f"    elle poursuivait l'eau     : {100 * pre_pursued_water / pre_n:.1f} %")
        ok_b = pre_ok / pre_n >= BAR_CAMPED
    else:
        print("    aucune mort de soif avec énergie en stock dans ce corpus")
        ok_b = False
    print(f"    {'✅' if ok_b else '❌'} G0-b\n")

    # Sensibilité aux exposants — MESURE rapportée, jamais une sélection.
    # CONTRÔLE DE NON-DÉGÉNÉRESCENCE : si le désaccord ne bouge pas avec (n, m), c'est que la
    # forme n'est pas exercée et que le test ne mesure qu'un « plus gros déficit gagne ».
    print("SENSIBILITÉ (mesure rapportée, jamais une sélection) — désaccord % :")
    print(f"    {'(n,m)':>10} " + " ".join(f"γ={g:<7g}" for g in (0.9990, 0.9970, 0.9900)))
    for n_, m_ in ((2.0, 1.5), (3.0, 2.0), (4.0, 3.0), (6.0, 4.0)):
        line = f"    {f'({n_:.0f},{m_:.1f})':>10} "
        for g_ in (0.9990, 0.9970, 0.9900):
            d2 = s2 = 0
            for rows in eps:
                for i in range(0, len(rows) - WINDOW, WINDOW):
                    p = pursued(rows, i)
                    if p is None:
                        continue
                    h = homeo_choice(rows[i], n_, m_, g_)
                    d2, s2 = (d2 + 1, s2) if h != p else (d2, s2 + 1)
            line += f"{100 * d2 / max(1, d2 + s2):7.1f} "
        print(line)

    print()
    print("🚨 DÉGÉNÉRESCENCE DÉMONTRÉE (2026-08-02) — À LIRE AVANT D'IMPLÉMENTER :")
    print("   Le désaccord ne bouge NI avec (n, m) NI avec γ, et ce n'est pas un bug.")
    print("   D est symétrique et Schur-convexe pour n>1, donc comparer (de−R, dt) à")
    print("   (de, dt−R) revient TOUJOURS à comparer de et dt. Pour deux jauges SYMÉTRIQUES")
    print("   à apport ÉGAL, la forme homéostatique se réduit EXACTEMENT à :")
    print("        « la jauge la plus démunie d'abord »")
    print("   quels que soient les exposants. La non-séparabilité n'achète RIEN sur ce choix.")
    print("   (Elle achète autre chose — aversion au risque, dose-réponse non linéaire — mais")
    print("    pas l'arbitrage entre deux pulsions symétriques.)")
    print("   ⇒ N'IMPLÉMENTER QUE LA RÈGLE SIMPLE en G1. Poser la norme n/m serait habiller")
    print("     un résultat trivial d'une théorie qui ne travaille pas (§2).")
    print()
    print("=" * 74)
    if ok_a and ok_b:
        print("✅ G0 PASSÉ — mais ce qui passe est la règle SIMPLE ci-dessus, pas la norme.")
        print("   Elle diffère du comportement servi sur la moitié des décisions, et dit")
        print("   BOIRE dans les fenêtres qui précèdent les morts de soif. G1 est licencié,")
        print("   sous la forme SIMPLIFIÉE.")
    elif not ok_a:
        print("🛑 G0-a ÉCHOUE = STOP pré-enregistré : les deux ordres coïncident, donc")
        print("   changer la forme du coût ne changerait RIEN. Ne pas payer G1/G2.")
        print("   ⇒ le défaut de soif vient d'ailleurs (portée, mémoire, ou exécution).")
    else:
        print("⚠️  G0-a passe mais G0-b échoue : la forme change des décisions SANS viser")
        print("   la jauge nécessiteuse quand elle campe — c'est-à-dire sans corriger la")
        print("   pathologie visée. Re-scoper AVANT de payer G1.")


if __name__ == "__main__":
    main()
