"""diag_critic_landscape, GRATUIT (offline : corpus + checkpoints deja sur disque, ZERO entrainement).

QUESTION FALSIFIABLE (2026-07-08). Le critique appris pilote maintenant 100% des replans (gate corrige),
mais il FORAGE MOINS que la formule codee-main qu'il remplace (-25% de consommations, KILL pre-enregistre
que l'owner a choisi d'outrepasser pour garder la purete). POURQUOI arbitre-t-il moins bien ?

Trois causes possibles, qui appellent des fixes COMPLETEMENT DIFFERENTS :
  (1) PAYSAGE PLAT : V varie a peine entre les 33 candidats -> l'argmax pique AU BRUIT.
      (Mecanisme deja rencontre dans ce projet : "score PLAT std=0.000" du cout survie.)
  (2) MAUVAIS CLASSEMENT : V discrimine bien, mais classe mal -> il prefere des candidats qui
      n'approchent PAS la ressource urgente. Le probleme est l'OBJECTIF d'entrainement (MSE sur des
      retours Monte-Carlo n'optimise PAS le rangement de candidats quasi-identiques).
  (3) VALEUR COMPRIMEE : monde marginal (67% des vies condamnees, cf diag_metabolic_ceiling) ->
      V apprend "tout est mauvais" -> plage utile ecrasee -> gradient faible.

MESURE (sur de VRAIS etats de replan rejoues depuis le corpus, avec le VRAI WM et le VRAI critique) :
  A. DISCRIMINABILITE : dispersion des scores sur les 33 candidats, pour le CRITIQUE et pour le
     coût DESIGNED, sur LES MEMES etats. Metrique robuste a l'echelle (les 2 couts ont des unites
     differentes) : la MARGE RELATIVE = (best - 2e) / (max - min). Si elle est ~0, les meilleurs
     candidats sont ex-aequo -> l'argmax est du bruit.
  B. LE BUT (pas le proxy) : l'argmax FERME-T-IL sur la ressource URGENTE ? Pour la pulsion la plus
     basse, on compare la distance minimale atteinte dans le reve par le candidat CHOISI a la
     MEILLEURE atteignable parmi les 33. "regret" = choisi - meilleur (en metres). Un critique qui
     departage bien mais choisit un candidat qui n'approche pas = cause (2). Un critique dont le
     regret est ~celui d'un choix ALEATOIRE = cause (1).

CRITERES PRE-ENREGISTRES (ecrits AVANT de lancer) :
  - CAUSE (1) PAYSAGE PLAT si : marge relative du critique << celle du designed (ex. < 1/3), ET le
    regret du critique ~ celui d'un choix aleatoire -> le fix est la DISCRIMINABILITE (objectif
    contrastif/ranking, ou normalisation des cibles), PAS plus de donnees.
  - CAUSE (2) MAUVAIS CLASSEMENT si : marge relative saine (~ designed) MAIS regret du critique
    >> regret du designed -> il departage franchement mais se trompe de candidat -> le fix est
    l'OBJECTIF d'entrainement (TD/bootstrap, ou apprendre a RANGER, pas a predire un retour).
  - CAUSE (3) VALEUR COMPRIMEE si : l'ecart-type de V sur des etats VARIES est minuscule devant sa
    propre erreur -> le fix est la normalisation des cibles / le reequilibrage du corpus.
  (Les causes ne sont pas exclusives : on rapporte les 3 mesures et on lit ce qui domine.)

Lancer : PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_critic_landscape.py [--selfcheck] [-n 300]
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import random
import statistics as st
from pathlib import Path

import torch


def relative_margin(scores: list[float]) -> float:
    """(best - 2e) / (max - min) : 0 = les 2 meilleurs sont ex-aequo (argmax au bruit),
    1 = le vainqueur ecrase tout. Invariant a l'echelle ET au decalage -> comparable entre
    deux couts qui n'ont PAS les memes unites (c'est tout l'interet)."""
    if len(scores) < 2:
        return float("nan")
    s = sorted(scores, reverse=True)
    spread = s[0] - s[-1]
    if spread <= 1e-12:
        return 0.0                      # tous les candidats strictement ex-aequo
    return (s[0] - s[1]) / spread


def urgent_regret(scores: list[float], min_d_urgent: list[float]) -> tuple[float, float, float]:
    """-> (regret du choisi, regret d'un choix ALEATOIRE, meilleure distance atteignable).

    regret = (distance min atteinte par le candidat retenu) - (meilleure distance atteignable).
    Le regret ALEATOIRE est la reference : si le regret du critique l'egale, son argmax n'apporte
    RIEN par rapport a tirer au hasard -> paysage plat.
    """
    best_d = min(min_d_urgent)
    chosen = int(max(range(len(scores)), key=lambda i: scores[i]))
    regret_chosen = min_d_urgent[chosen] - best_d
    regret_random = st.mean(min_d_urgent) - best_d      # esperance d'un tirage uniforme
    return regret_chosen, regret_random, best_d


def selfcheck() -> None:
    # marge relative : vainqueur net
    assert abs(relative_margin([10.0, 0.0, 0.0]) - 1.0) < 1e-9
    # marge relative : 2 ex-aequo en tete -> 0
    assert abs(relative_margin([10.0, 10.0, 0.0]) - 0.0) < 1e-9
    # tous strictement egaux -> 0 (argmax = bruit pur)
    assert relative_margin([5.0, 5.0, 5.0]) == 0.0
    # invariance d'echelle et de decalage (le point clef pour comparer critic vs designed)
    a = [3.0, 1.0, 0.0]
    assert abs(relative_margin(a) - relative_margin([100.0 * x + 7.0 for x in a])) < 1e-9
    # regret : le score pointe le candidat 1, qui est aussi le plus proche -> regret 0
    rc, rr, bd = urgent_regret([0.0, 9.0, 0.0], [5.0, 1.0, 3.0])
    assert abs(rc) < 1e-9 and abs(bd - 1.0) < 1e-9 and rr > 0.0, (rc, rr, bd)
    # regret : le score pointe le PIRE candidat -> regret = 5 - 1 = 4
    rc2, _, _ = urgent_regret([9.0, 0.0, 0.0], [5.0, 1.0, 3.0])
    assert abs(rc2 - 4.0) < 1e-9, rc2
    print("[selfcheck] OK : marge relative (invariante echelle/decalage), regret choisi vs aleatoire")


def load_states(dirs: list[str], n: int, seed: int) -> list[dict]:
    """Etats de replan REELS : proprio(132) + retina(144) + energie/soif."""
    rows: list[dict] = []
    for d in dirs:
        f = Path(d) / "ep_0000.jsonl"
        if not f.exists():
            continue
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                wm = r.get("wm") or {}
                obs = r.get("obs") or {}
                ret = wm.get("retina0")
                pro = obs.get("proprio")
                if not ret or not pro or len(ret) != 144 or len(pro) != 132:
                    continue
                rows.append({"proprio": pro, "retina": ret,
                             "energy": float(obs["energy"]), "thirst": float(obs["thirst"])})
    rnd = random.Random(seed)
    rnd.shuffle(rows)
    return rows[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/replay_buffer/critic_kin_[ab]")
    ap.add_argument("--wm", default="data/checkpoints/wm_objcentric_kin/wm_best.pt")
    ap.add_argument("--critic", default="data/checkpoints/survival_critic_kin/critic_best.pt")
    ap.add_argument("-n", type=int, default=300, help="nb d'etats de replan rejoues")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.environ["SYLVAN_PLANNER_CRITIC"] = args.critic
    os.environ.setdefault("SYLVAN_PLANNER_DRAIN", "0.0005")
    os.environ.setdefault("SYLVAN_PLANNER_RESTORE", "0.4")
    os.environ["SYLVAN_PLANNER_FAR_ALIGN"] = "0"

    from sylvan.control.planning.command_planner import CommandPlanConfig, CommandPlanner
    from sylvan.models.command_wm import CommandWorldModel

    # Chargement STRICTEMENT identique a serve_planner_command.py:142-156 (source de verite) :
    # une divergence ici mesurerait un WM different de celui qui decide en vrai.
    payload = torch.load(args.wm, map_location="cpu", weights_only=False)
    meta = payload["meta"]
    wm = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                           predictor_arch=meta.get("predictor_arch", "shallow"),
                           with_slot=meta.get("with_slot", False),
                           slot_resources=meta.get("slot_resources", 1))
    wm.load_state_dict(payload["model"])
    wm.eval()
    wm.food_idx = meta.get("food_idx", 0)
    wm.water_idx = meta.get("water_idx")
    print(f"[wm] obs_dim={meta['obs_dim']} slots={meta.get('slot_resources', 1)} "
          f"(food_idx={wm.food_idx}, water_idx={wm.water_idx})")

    states = load_states(sorted(globmod.glob(args.glob)), args.n, args.seed)
    if not states:
        print(f"AUCUN etat exploitable dans {args.glob}")
        return

    # Deux planners sur LE MEME WM : l'un score au CRITIQUE, l'autre au coût DESIGNED.
    os.environ["SYLVAN_PLANNER_COST"] = "critic"
    p_critic = CommandPlanner(wm, CommandPlanConfig(horizon=80))
    os.environ["SYLVAN_PLANNER_COST"] = "designed"
    p_design = CommandPlanner(wm, CommandPlanConfig(horizon=80))

    marg_c, marg_d = [], []
    reg_c, reg_d, reg_rand = [], [], []
    v_means = []
    n_flat = 0

    for s in states:
        obs = torch.tensor(s["proprio"] + s["retina"] + [s["energy"] / 100.0], dtype=torch.float32)
        e, t = s["energy"] / 100.0, s["thirst"] / 100.0
        # food/water non-None a l'ENTREE = ce que fait le serveur (radar EMA) -> route vers la branche
        # MULTI-RESSOURCE ; le bloc slot-2 les ECRASE ensuite avec les slots du WM + gate de visibilite.
        kw = dict(radar=[0.0] * 12, energy=e, thirst=t, override_pos=True,
                  food_override=(0.0, 3.0), water_override=(0.0, 3.0), debug_scores=True)
        with torch.no_grad():
            rc = p_critic.plan(obs, **kw)
            rd = p_design.plan(obs, **kw)
        if "scores" not in rc or "scores" not in rd:
            continue
        sc, sd = rc["scores"], rd["scores"]
        # La pulsion URGENTE = la plus basse ; sa ressource = celle qu'il FAUT approcher.
        urgent_food = e <= t
        key = "min_df" if urgent_food else "min_dw"
        if key not in rc or rc[key] is None:
            continue
        md = rc[key]
        if any(x != x or x == float("inf") for x in md):   # NaN/inf -> ressource inconnue, on saute
            continue

        marg_c.append(relative_margin(sc))
        marg_d.append(relative_margin(sd))
        a, r, _ = urgent_regret(sc, md)
        b, _, _ = urgent_regret(sd, md)
        reg_c.append(a)
        reg_d.append(b)
        reg_rand.append(r)
        v_means.append(st.mean(sc))
        if (max(sc) - min(sc)) < 1e-6:
            n_flat += 1

    if not marg_c:
        print("Aucun etat exploitable (ressource urgente inconnue partout ?)")
        return

    n = len(marg_c)
    print("=" * 78)
    print(f"ETATS DE REPLAN REJOUES : {n} (vrai WM + vrai critique, 33 candidats chacun)")
    print("=" * 78)
    print()
    print("A. DISCRIMINABILITE -- marge relative (best - 2e) / (max - min), invariante a l'echelle")
    print(f"   CRITIQUE  : mediane {st.median(marg_c):.4f}   (moyenne {st.mean(marg_c):.4f})")
    print(f"   DESIGNED  : mediane {st.median(marg_d):.4f}   (moyenne {st.mean(marg_d):.4f})")
    print(f"   candidats strictement ex-aequo (spread < 1e-6) : {n_flat}/{n} chez le critique")
    ratio = st.median(marg_c) / max(st.median(marg_d), 1e-9)
    print(f"   -> ratio critique/designed = {ratio:.2f}")
    print()
    print("B. LE BUT -- regret sur la ressource URGENTE (metres de plus que le meilleur candidat)")
    print(f"   CRITIQUE  : regret median {st.median(reg_c):.3f} m   (moyenne {st.mean(reg_c):.3f})")
    print(f"   DESIGNED  : regret median {st.median(reg_d):.3f} m   (moyenne {st.mean(reg_d):.3f})")
    print(f"   ALEATOIRE : regret median {st.median(reg_rand):.3f} m  <- la reference 'aucune info'")
    print()
    frac_c = st.mean(reg_c) / max(st.mean(reg_rand), 1e-9)
    frac_d = st.mean(reg_d) / max(st.mean(reg_rand), 1e-9)
    print(f"   regret critique / regret aleatoire = {frac_c:.2f}   (1.0 = n'apporte RIEN vs le hasard)")
    print(f"   regret designed / regret aleatoire = {frac_d:.2f}")
    print()
    print("C. VALEUR -- plage utile de V sur des etats varies")
    print(f"   ecart-type de V (moyenne sur candidats) entre etats : {st.pstdev(v_means):.4f}")
    print(f"   V median : {st.median(v_means):.4f}")

    print()
    print("=" * 78)
    print("VERDICT (criteres pre-enregistres)")
    print("=" * 78)
    flat = ratio < 0.34 or frac_c > 0.8
    misrank = (not flat) and (st.mean(reg_c) > 1.5 * st.mean(reg_d))
    if flat:
        print("  >>> CAUSE (1) = PAYSAGE PLAT <<<")
        print(f"  Le critique ne departage PAS ses candidats (marge {ratio:.2f}x celle du designed ;")
        print(f"  regret = {frac_c:.2f}x celui du HASARD). Son argmax est du bruit.")
        print("  => Le fix n'est PAS 'plus de donnees' : c'est la DISCRIMINABILITE.")
        print("     Pistes : objectif contrastif/ranking (apprendre a RANGER des candidats, pas a")
        print("     predire un retour), normalisation des cibles, ou enrichir le token du critique.")
    elif misrank:
        print("  >>> CAUSE (2) = MAUVAIS CLASSEMENT <<<")
        print(f"  Le critique departage franchement (marge {ratio:.2f}x designed) MAIS choisit mal :")
        print(f"  regret {st.mean(reg_c):.2f} m contre {st.mean(reg_d):.2f} m pour le designed.")
        print("  => Le fix est l'OBJECTIF d'entrainement : des retours Monte-Carlo n'enseignent pas")
        print("     le RANGEMENT de candidats quasi-identiques. Pistes : labels TD/bootstrap,")
        print("     ou entrainer directement sur des paires de candidats (preference learning).")
    else:
        print("  >>> NI (1) NI (2) de facon nette <<<")
        print(f"  marge {ratio:.2f}x designed, regret {st.mean(reg_c):.2f} m vs {st.mean(reg_d):.2f} m.")
        print("  Le critique classe ~aussi bien que le designed sur la ressource urgente -> la perte")
        print("  de forage vient d'AILLEURS (arbitrage ENTRE pulsions ? timing ? engagement ?).")
        print("  -> Regarder l'equilibre repas/boissons et les poursuites avortees.")


if __name__ == "__main__":
    main()
