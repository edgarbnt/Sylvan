"""GATE « la place existe » — la zone nocive cree-t-elle un cout que l'entite aveugle ne peut eviter ?

Lit les logs Godot des deux bras (hazard OFF vs ON, memes graines) et evalue les CRITERES
PRE-ENREGISTRES (docs/etat_critique.md, ecrits AVANT le run) :

  SUCCES (place prouvee) :
    (a) AVEUGLEMENT : l'entite entre dans la zone sur >= 50% des vies (elle ne devie pas, elle
        ne PEUT pas : le cout inne n'a aucun terme de danger).
    (b) COUT REEL : ON fait clairement plus mal que OFF -- sante mediane a la mort plus basse,
        OU >= 1 mort par danger sur 12, OU >= 20 de degats-danger par vie (mediane).
  KILL/REGLAGE : entree < 20% (mauvais placement) ou ON ~ OFF (danger trop faible). Ratés de
    REGLAGE, pas de concept -- on le dit.

Le vrai livrable = le BASELINE AVEUGLE (sante perdue / morts par danger) que l'entite
percevante+decidante devra battre a l'etape suivante.

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_hazard_gate.py \
      --off /tmp/gate_godot_off.log --on /tmp/gate_godot_on.log
"""

from __future__ import annotations

import argparse
import re
import statistics as st

EP = re.compile(r"Episode (\d+) \| Step (\d+) .*?Energy: ([\d.]+) .*?Thirst: ([\d.]+) .*?Health: ([\d.]+)")
HAZ = re.compile(r"\[hazard\] ep (\d+) : entr\S+=(\w+) pas_dans_zone=(\d+) d\S+g\S+ts=([\d.]+)")


def parse_lives(path: str) -> list[dict]:
    """-> une entree par vie : etat FINAL + cause de mort + LE BUT (temps de survie, repas, boissons).

    Repas/boissons comptes via les REMONTEES de niveau entre 2 echantillons (log tous les 10 pas) :
    l'energie draine 0.05/pas (-0.5/echantillon) et ne REMONTE qu'en mangeant (+40) -> toute hausse
    = un repas. Idem soif = une boisson. (2 repas dans la meme fenetre de 10 pas = 1 compte : rare,
    sous-compte assume.) C'est LE BUT (forage), pas le proxy cause-de-mort (lecon 2026-07-16 : en
    monde marginal, 'eviter' peut juste changer la CAUSE de mort -- juger sur survie+forage)."""
    last: dict[int, tuple] = {}
    prev: dict[int, tuple] = {}                  # (energy, thirst) du sample precedent par episode
    meals: dict[int, int] = {}
    drinks: dict[int, int] = {}
    for line in open(path, errors="ignore"):
        m = EP.search(line)
        if m:
            ep = int(m.group(1))
            e, t = float(m.group(3)), float(m.group(4))
            if ep in prev:
                if e > prev[ep][0] + 0.01:
                    meals[ep] = meals.get(ep, 0) + 1
                if t > prev[ep][1] + 0.01:
                    drinks[ep] = drinks.get(ep, 0) + 1
            prev[ep] = (e, t)
            last[ep] = (int(m.group(2)), e, t, float(m.group(5)))
    lives = []
    for ep, (step, e, t, h) in sorted(last.items()):
        # CAUSE = le drive le PLUS BAS a la derniere mesure (le log echantillonne tous les 10 pas →
        # le vrai zero est manque ; l'argmin est le proxy honnete). Sante = seule chose que la zone
        # nocive abaisse → sante minimale = mort par danger. Seuil 15 : au-dessus = vie tronquee.
        levels = {"faim": e, "soif": t, "danger": h}
        killer = min(levels, key=levels.get)
        cause = killer if levels[killer] <= 15.0 else "tronque"
        lives.append({"ep": ep, "energy": e, "thirst": t, "health": h, "cause": cause,
                      "steps": step, "meals": meals.get(ep, 0), "drinks": drinks.get(ep, 0)})
    return lives


def parse_hazard(path: str) -> dict[int, dict]:
    out = {}
    for line in open(path, errors="ignore"):
        m = HAZ.search(line)
        if m:
            # signal ROBUSTE = pas_dans_zone > 0 (le flag texte est "true"/"false" en GDScript,
            # minuscule → une comparaison "==True" ratait ; on lit directement les pas comptés).
            steps_in = int(m.group(3))
            out[int(m.group(1))] = {"entered": steps_in > 0,
                                    "steps_in": steps_in, "damage": float(m.group(4))}
    return out


def selfcheck() -> None:
    """Invariants du parseur BUT : survie = dernier step ; repas/boisson = REMONTEE de niveau."""
    import tempfile

    lines = [
        # vie 0 : energie 70->69.5 (drain) ->95 (REPAS +40 cape) ->94.5 ; soif 70->69.5->90 (BOISSON)
        "[Godot] Episode 0 | Step 10 | Energy: 70.0 | Thirst: 70.0 | Health: 100.0 |",
        "[Godot] Episode 0 | Step 20 | Energy: 69.5 | Thirst: 69.5 | Health: 100.0 |",
        "[Godot] Episode 0 | Step 30 | Energy: 95.0 | Thirst: 69.0 | Health: 100.0 |",
        "[Godot] Episode 0 | Step 40 | Energy: 94.5 | Thirst: 90.0 | Health: 100.0 |",
        "[Godot] Episode 0 | Step 200 | Energy: 5.0 | Thirst: 60.0 | Health: 100.0 |",
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        f.write("\n".join(lines))
        path = f.name
    lives = parse_lives(path)
    assert len(lives) == 1
    lv = lives[0]
    assert lv["steps"] == 200, lv                      # survie = dernier step vu
    assert lv["meals"] == 1 and lv["drinks"] == 1, lv  # 1 remontee de chaque = 1 repas + 1 boisson
    assert lv["cause"] == "faim", lv                   # energie 5 = min <= 15
    print("[selfcheck] OK : survie=dernier step ; remontees comptees (1 repas, 1 boisson) ; cause=faim")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", default="/tmp/gate_godot_off.log")
    ap.add_argument("--on", default="/tmp/gate_godot_on.log")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return

    off, on = parse_lives(args.off), parse_lives(args.on)
    haz = parse_hazard(args.on)
    if not off or not on:
        print(f"logs vides ? off={len(off)} vies, on={len(on)} vies")
        return

    def health_med(lives):
        return st.median([l["health"] for l in lives])

    def deaths(lives, cause):
        return sum(1 for l in lives if l["cause"] == cause)

    # NB : le manager logue la vie N au DEBUT de la vie N+1 → la DERNIERE vie n'a pas de ligne
    # [hazard]. On divise donc par le nombre de vies AVEC donnee hazard, pas par len(on).
    entered = [h for h in haz.values() if h["entered"]]
    entry_rate = len(entered) / len(haz) if haz else 0.0
    dmg_med = st.median([h["damage"] for h in haz.values()]) if haz else 0.0
    haz_deaths = deaths(on, "danger")

    def med(lives, key):
        return st.median([l[key] for l in lives])

    def tot(lives, key):
        return sum(l[key] for l in lives)

    print(f"\n=== GATE ZONE NOCIVE — {len(off)} vies OFF vs {len(on)} vies ON (memes graines) ===\n")
    print(f"{'':22}{'OFF':>10}{'ON':>10}")
    print("-" * 42)
    print(f"{'sante med a la mort':22}{health_med(off):>10.0f}{health_med(on):>10.0f}")
    print(f"{'morts par DANGER':22}{deaths(off, 'danger'):>10d}{haz_deaths:>10d}")
    print(f"{'morts par faim':22}{deaths(off, 'faim'):>10d}{deaths(on, 'faim'):>10d}")
    print(f"{'morts par soif':22}{deaths(off, 'soif'):>10d}{deaths(on, 'soif'):>10d}")
    print("-" * 42)
    # LE BUT (pas le proxy cause-de-mort) : survie + forage. En monde marginal, 'eviter' peut
    # seulement deplacer la cause de mort -- ces trois lignes disent si ca ACHETE quelque chose.
    print(f"{'survie med (pas)':22}{med(off, 'steps'):>10.0f}{med(on, 'steps'):>10.0f}")
    print(f"{'repas (total)':22}{tot(off, 'meals'):>10d}{tot(on, 'meals'):>10d}")
    print(f"{'boissons (total)':22}{tot(off, 'drinks'):>10d}{tot(on, 'drinks'):>10d}")
    print("-" * 42)
    print(f"\nZone nocive (bras ON) :")
    print(f"  entree dans la zone : {len(entered)}/{len(on)} vies ({entry_rate * 100:.0f}%)")
    print(f"  degats-danger par vie (mediane) : {dmg_med:.1f}")
    print(f"  pas passes dans la zone (mediane) : "
          f"{st.median([h['steps_in'] for h in haz.values()]) if haz else 0:.0f}")

    # CRITERES PRE-ENREGISTRES
    blind = entry_rate >= 0.50
    cost = (health_med(on) < health_med(off)) or (haz_deaths >= 1) or (dmg_med >= 20.0)
    print("\n--- VERDICT (criteres ecrits AVANT le run) ---")
    print(f"  (a) AVEUGLEMENT  entree {entry_rate*100:.0f}% >= 50% : {'OUI' if blind else 'NON'}")
    print(f"  (b) COUT REEL    (sante ON<OFF) ou (>=1 mort danger) ou (degats>=20) : {'OUI' if cost else 'NON'}")
    if blind and cost:
        print("\n  ✅ PLACE PROUVEE. L'entite aveugle SUBIT un cout qu'elle ne peut pas eviter (pas de")
        print("     perception du danger, pas de terme dans le cout inne). BASELINE AVEUGLE ci-dessus =")
        print("     le chiffre a battre par une entite qui PERCOIT et DECIDE. Prochain etage justifie :")
        print("     donner au WM le sens 'danger' (re-collecte + retrain), puis mesurer qu'elle l'evite.")
    elif not blind:
        print(f"\n  ⚠️ REGLAGE (pas concept) : entree {entry_rate*100:.0f}% trop basse → zone mal placee.")
        print("     Monter le rayon / ajuster frac (placement sur le trajet). Re-run.")
    else:
        print("\n  ⚠️ REGLAGE (pas concept) : ON ~ OFF → danger trop faible pour constituer une 'place'.")
        print("     Monter SYLVAN_HAZARD_DAMAGE. Re-run.")


if __name__ == "__main__":
    main()
