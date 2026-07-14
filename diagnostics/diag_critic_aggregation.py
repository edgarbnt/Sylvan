"""SONDE GRATUITE — comment le planner CONSOMME la valeur (2026-07-14).

QUESTION FALSIFIABLE. Le planner note chaque candidat par `score = moyenne_t V(etat_reve_t)`
(command_planner.py:725). La litterature (TD-MPC, MuZero, Dreamer) ne fait JAMAIS ca : elle
somme les RECOMPENSES le long du rollout et BOOTSTRAPPE avec la valeur TERMINALE.

Pourquoi la moyenne serait nuisible : a t=0 les 33 candidats partent du MEME etat -> V est
IDENTIQUE pour tous. Ils ne divergent qu'en fin de reve. Moyenner sur l'horizon melange donc
le signal (la fin, qui discrimine) avec un socle CONSTANT (le debut, qui ne discrimine rien)
-> l'ecart entre candidats est DILUE, alors que l'erreur du reseau, elle, ne l'est pas
(elle est correlee le long de la trajectoire : meme reseau, etats voisins).

=> La moyenne ne casse pas l'ORDRE quand la valeur est PARFAITE (un retrecissement monotone
   preserve l'argmax) -- c'est pourquoi la sonde ORACLE ne pouvait PAS l'incriminer. Elle ne
   nuit QUE lorsque la valeur est BRUITEE. C'est exactement notre cas.

CE QUE LA SONDE MESURE (aucun entrainement, aucun episode Godot) : sur des etats de replan
REELS, on recupere la valeur PAR PAS [33 candidats x 80 pas] -- pour le critique APPRIS et
pour l'ORACLE analytique (meme reve, meme code) -- puis on rejoue plusieurs AGREGATIONS :

  mean      : moyenne sur l'horizon              <- le vivant (ligne 725)
  tail25    : moyenne du dernier quart
  terminal  : V(etat final) seul                 <- la forme canonique (bootstrap)
  discount  : somme escomptee gamma^t V_t

Pour chacune, 3 chiffres qui decident :
  SIGNAL  = ecart median (meilleur - pire) entre les 33 candidats, chez l'ORACLE
  BRUIT   = erreur mediane |critique - oracle| sur les scores des candidats
  SNR     = SIGNAL / BRUIT       <- si SNR <= 1, l'argmax est du bruit, POINT.
  ACCORD  = % de fois ou l'argmax du critique == l'argmax de l'oracle (33 candidats)
  REGRET  = metres de plus que le meilleur candidat, sur la ressource URGENTE
            (compare au regret d'un choix ALEATOIRE = la reference du hasard)

CRITERES ECRITS AVANT (falsifiables) :
  - VALIDE le changement d'agregation si `terminal` (ou tail25) ameliore le SNR d'un facteur
    >= 1.5 ET l'ACCORD de >= +10 points vs `mean`, chez le MEME critique (zero retrain).
  - TUE le changement si le SNR et l'ACCORD sont plats (< 1.2x et < +5 pts) : alors la facon
    de consommer la valeur n'est PAS le goulot, et le probleme est bien la valeur elle-meme
    (=> aller vers l'ecart-d'action / la perte de classement / l'acteur amorti).

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_critic_aggregation.py --selfcheck
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_critic_aggregation.py -n 120
"""

from __future__ import annotations

import argparse
import glob as globmod
import os
import random
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_critic_landscape import load_states, urgent_regret  # noqa: E402

# Agregations rejouees sur la MEME valeur par pas. vmap: [n_cand, horizon] -> [n_cand]
AGGS: dict[str, object] = {
    "mean": lambda v: v.mean(dim=1),
    "tail25": lambda v: v[:, -max(1, v.shape[1] // 4):].mean(dim=1),
    "terminal": lambda v: v[:, -1],
    "discount": lambda v: (v * (0.97 ** torch.arange(v.shape[1], dtype=v.dtype))).sum(dim=1),
}


def spread(x: torch.Tensor) -> float:
    """Ecart meilleur-pire entre candidats = le SIGNAL a resoudre."""
    return float(x.max() - x.min())


def action_gap(x: torch.Tensor) -> float:
    """ECART D'ACTION (Farahmand, NeurIPS 2011) = meilleur - DEUXIEME meilleur.

    C'est LUI que l'argmax doit resoudre, pas l'ecart meilleur-pire. Un critique dont l'erreur
    est petite devant le spread mais GRANDE devant l'ecart d'action inversera quand meme le
    classement en tete -- et c'est le classement en tete, seul, qui decide."""
    top2 = torch.topk(x, 2).values
    return float(top2[0] - top2[1])


def value_regret(sc: torch.Tensor, so: torch.Tensor) -> float:
    """LE VRAI COUT d'une erreur d'argmax : valeur ORACLE perdue en suivant le choix du critique.

    Se tromper de candidat est SANS CONSEQUENCE si le candidat retenu vaut presque autant.
    -> si ce regret est ~0, l'accord a 22% est un FAUX PROBLEME et la cause est ailleurs."""
    return float(so.max() - so[int(sc.argmax())])


def selfcheck() -> None:
    """Verifie les agregations et la metrique de dilution sur un cas construit a la main."""
    # 3 candidats, 4 pas. Ils partent du MEME V a t=0 (socle commun) et divergent a la fin :
    # c'est la structure exacte du probleme (tous les candidats partagent l'etat initial).
    v = torch.tensor([[0.5, 0.5, 0.5, 0.5],     # cand 0 : stagne
                      [0.5, 0.6, 0.7, 0.9],     # cand 1 : progresse (le MEILLEUR)
                      [0.5, 0.5, 0.5, 0.6]])    # cand 2 : progresse peu
    term = AGGS["terminal"](v)
    mean = AGGS["mean"](v)
    assert int(term.argmax()) == 1 and int(mean.argmax()) == 1, "meme argmax sans bruit"
    # LE POINT : la moyenne DILUE l'ecart (le socle commun t=0 le tire vers 0), la terminale non.
    assert spread(term) > spread(mean), "la terminale doit mieux separer les candidats"
    assert abs(spread(term) - 0.4) < 1e-6 and abs(spread(mean) - 0.175) < 1e-6
    # -> avec un bruit de 0.2, la moyenne (ecart 0.175) est NOYEE, la terminale (0.40) survit.
    assert AGGS["tail25"](v).shape == (3,) and AGGS["discount"](v).shape == (3,)
    # regret : le score pointe le candidat le plus proche -> regret nul
    rc, _, _ = urgent_regret([0.0, 9.0, 0.0], [5.0, 1.0, 3.0])
    assert abs(rc) < 1e-9
    print("[selfcheck] OK : agregations + la moyenne dilue bien l'ecart entre candidats (0.15 vs 0.40)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/replay_buffer/critic_kin_[ab]")
    ap.add_argument("--wm", default="data/checkpoints/wm_objcentric_kin/wm_best.pt")
    ap.add_argument("--critic", default="data/checkpoints/survival_critic_kin/critic_best.pt")
    ap.add_argument("-n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=80,
                    help="longueur du reve. LE TEST : l'ecart d'action grandit-il quand le reve "
                         "atteint enfin la ressource (80 pas x 0.02 m = 1.6 m vs cible a 2-8 m) ?")
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
    os.environ["SYLVAN_CRITIC_ALWAYS"] = "1"
    os.environ["SYLVAN_PLANNER_COST"] = "critic"

    from sylvan.control.planning.command_planner import CommandPlanConfig, CommandPlanner
    from sylvan.models.command_wm import CommandWorldModel

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

    planner = CommandPlanner(wm, CommandPlanConfig(horizon=args.horizon))
    states = load_states(sorted(globmod.glob(args.glob)), args.n, args.seed)
    if not states:
        print(f"AUCUN etat exploitable dans {args.glob}")
        return

    # accumulateurs par agregation
    sig: dict[str, list[float]] = {k: [] for k in AGGS}     # ecart entre candidats (oracle)
    noi: dict[str, list[float]] = {k: [] for k in AGGS}     # |critique - oracle|
    agree: dict[str, list[float]] = {k: [] for k in AGGS}   # argmax critique == argmax oracle
    reg_c: dict[str, list[float]] = {k: [] for k in AGGS}   # regret du CRITIQUE (metres)
    reg_o: dict[str, list[float]] = {k: [] for k in AGGS}   # regret de l'ORACLE (metres)
    # "tourne du bon cote" : la convention de signe d'omega est CALIBREE PAR L'ORACLE lui-meme
    # (s'il sort 15%, ma convention est inversee et le vrai chiffre est 85%). Ce qui compte est
    # l'ECART critique-vs-oracle, invariant a la convention.
    side: dict[tuple[str, str], list[float]] = {(k, w): [] for k in AGGS for w in ("crit", "orac")}
    gap: dict[str, list[float]] = {k: [] for k in AGGS}     # ecart d'action (best - 2e), oracle
    vreg: dict[str, list[float]] = {k: [] for k in AGGS}    # valeur oracle PERDUE par le critique
    reg_rand: list[float] = []
    base_frac: list[float] = []                             # part du score = socle commun a t=0
    blind_c: list[float] = []                               # spread du score, ressource urgente HORS-VUE
    blind_o: list[float] = []

    for s in states:
        obs = torch.tensor(s["proprio"] + s["retina"] + [s["energy"] / 100.0], dtype=torch.float32)
        e, t = s["energy"] / 100.0, s["thirst"] / 100.0
        kw = dict(radar=[0.0] * 12, energy=e, thirst=t, override_pos=True,
                  food_override=(0.0, 3.0), water_override=(0.0, 3.0), debug_scores=True)
        with torch.no_grad():
            os.environ["SYLVAN_CRITIC_ORACLE"] = "0"
            rc = planner.plan(obs, **kw)
            os.environ["SYLVAN_CRITIC_ORACLE"] = "1"
            ro = planner.plan(obs, **kw)
        if "vmap" not in rc or "vmap" not in ro:
            continue
        vc = torch.tensor(rc["vmap"])                       # [n_cand, h] critique appris
        vo = torch.tensor(ro["vmap"])                       # [n_cand, h] oracle analytique
        fall = torch.tensor(rc["fall"])                     # facteur de chute (commun aux deux)

        urgent_food = e <= t
        key = "min_df" if urgent_food else "min_dw"
        md = rc.get(key)
        if not md or any(x != x or x == float("inf") for x in md):
            # RESSOURCE URGENTE INCONNUE (hors-vue). La sonde precedente SAUTAIT ces etats -- elle
            # ne jugeait donc le critique QUE la ou c'est facile. Or c'est ~40% des replans en epars,
            # et en deploiement le critique DOIT quand meme y produire une commande. Que vaut son
            # paysage quand le token dit "connu=0" ? S'il est PLAT, l'argmax est arbitraire : pas de
            # recherche, pas d'exploration -- l'entite est paralysee la ou il faudrait CHERCHER.
            sc = AGGS["mean"](vc) * fall
            so = AGGS["mean"](vo) * fall
            blind_c.append(spread(sc))
            blind_o.append(spread(so))
            continue

        # LE BUT, PAS LE PROXY : le candidat retenu tourne-t-il DU BON COTE ?
        # La ressource urgente est en (px, pz) ego -> son bearing a pour signe celui de px.
        # Le forage reussi est l'INTEGRALE de centaines de ces micro-decisions : si l'argmax
        # tourne du bon cote a peine mieux qu'a pile ou face, le beeline ne s'accumule jamais.
        tgt = rc.get("food") if urgent_food else rc.get("water")
        cmds = rc.get("cand_cmd0")
        if tgt and cmds and abs(tgt[0]) > 1e-3:
            want_left = tgt[0] < 0.0                       # px<0 -> il faut tourner a gauche
            for name, fn in AGGS.items():
                for who, v in (("crit", vc), ("orac", vo)):
                    om = cmds[int((fn(v) * fall).argmax())][1]   # omega du candidat retenu
                    if abs(om) > 1e-6:
                        side[(name, who)].append(1.0 if ((om < 0.0) == want_left) else 0.0)

        # Part du score qui est un SOCLE COMMUN : V a t=0 est le meme etat pour les 33 candidats.
        base_frac.append(float(vo[:, 0].mean() / (vo.mean() + 1e-9)))

        for name, fn in AGGS.items():
            sc = fn(vc) * fall
            so = fn(vo) * fall
            # les deux echelles different (l'oracle est en pas-vecus/horizon) -> on aligne
            # affinement le critique sur l'oracle AVANT de mesurer le bruit, sinon on mesurerait
            # un decalage d'echelle et non une erreur de CLASSEMENT.
            a = torch.stack([sc, torch.ones_like(sc)], dim=1)
            coef = torch.linalg.lstsq(a, so.unsqueeze(1)).solution.squeeze(1)
            sc_al = sc * coef[0] + coef[1]

            sig[name].append(spread(so))
            noi[name].append(float((sc_al - so).abs().median()))
            agree[name].append(1.0 if int(sc.argmax()) == int(so.argmax()) else 0.0)
            gap[name].append(action_gap(so))
            vreg[name].append(value_regret(sc, so))
            rc_m, rr, _ = urgent_regret(sc.tolist(), md)
            ro_m, _, _ = urgent_regret(so.tolist(), md)
            reg_c[name].append(rc_m)
            reg_o[name].append(ro_m)
        reg_rand.append(rr)

    n = len(reg_rand)
    if n == 0:
        print("AUCUN etat n'a produit de valeur par pas (le critique n'a pas note ?)")
        return

    if blind_c:
        tied = sum(1 for s in blind_c if s < 1e-6)
        print(f"\n=== RESSOURCE URGENTE HORS-VUE — {len(blind_c)} etats "
              f"({len(blind_c) / (len(blind_c) + n) * 100:.0f}% des replans) ===")
        print(f"  spread du score CRITIQUE : {st.median(blind_c):.6f}   "
              f"strictement ex-aequo (<1e-6) : {tied}/{len(blind_c)}")
        print(f"  spread du score ORACLE   : {st.median(blind_o):.6f}")
        print("  -> si ~0 : le critique ne DEPARTAGE RIEN quand il ne voit pas la ressource ;")
        print("     l'argmax y est arbitraire. CHERCHER n'est pas une capacite qu'il possede.")

    print(f"\n=== AGREGATION DE LA VALEUR — {n} etats x 33 candidats x {args.horizon} pas ===")
    print(f"Socle commun (V a t=0 / V moyen) : {st.mean(base_frac):.2f}  "
          "<- part du score qui ne discrimine AUCUN candidat\n")
    print(f"{'agregation':<10} {'SPREAD':>8} {'ECART-ACT':>10} {'BRUIT':>8} {'bruit/ecart':>12} "
          f"{'ACCORD':>7} {'VAL PERDUE':>11}")
    print("-" * 74)
    for name in AGGS:
        s_, g_, b_ = st.median(sig[name]), st.median(gap[name]), st.median(noi[name])
        ratio = b_ / g_ if g_ > 1e-12 else float("inf")
        # valeur perdue, en % du spread : "combien du signal disponible l'argmax gaspille-t-il"
        lost = st.median([v / s if s > 1e-12 else 0.0 for v, s in zip(vreg[name], sig[name])])
        print(f"{name:<10} {s_:>8.4f} {g_:>10.5f} {b_:>8.5f} {ratio:>12.2f} "
              f"{st.mean(agree[name]) * 100:>6.0f}% {lost * 100:>10.0f}%")
    print("-" * 74)
    print("\nLE BUT — le candidat retenu tourne-t-il DU BON COTE (vers la ressource urgente) ?")
    for name in AGGS:
        so_ = side[(name, "orac")]
        sc_ = side[(name, "crit")]
        if not so_:
            continue
        o, c = st.mean(so_) * 100, st.mean(sc_) * 100
        if o < 50.0:                       # convention de signe inversee -> l'oracle la calibre
            o, c = 100 - o, 100 - c
        print(f"  {name:<10} oracle {o:>3.0f}%   critique {c:>3.0f}%   (pile ou face = 50%)  n={len(so_)}")
    print("\nECART-ACT = meilleur - 2e meilleur (ce que l'argmax doit VRAIMENT resoudre)")
    print("bruit/ecart > 1  =>  l'argmax est du bruit, quelle que soit l'agregation")
    print("VAL PERDUE = valeur oracle sacrifiee par le choix du critique, en % du spread")
    print(f"\nregret metrique (m de plus que le meilleur candidat, ressource urgente) : "
          f"critique {st.median(reg_c['mean']):.2f} | oracle {st.median(reg_o['mean']):.2f} | "
          f"hasard {st.median(reg_rand):.2f}")

    m_snr = st.median(sig["mean"]) / max(st.median(noi["mean"]), 1e-12)
    t_snr = st.median(sig["terminal"]) / max(st.median(noi["terminal"]), 1e-12)
    d_ag = (st.mean(agree["terminal"]) - st.mean(agree["mean"])) * 100
    print(f"\nVERDICT vs criteres pre-enregistres (SNR x>=1.5 ET accord >=+10 pts) :")
    print(f"  terminal vs mean : SNR x{t_snr / max(m_snr, 1e-9):.2f}, accord {d_ag:+.0f} pts -> "
          f"{'VALIDE' if (t_snr >= 1.5 * m_snr and d_ag >= 10) else 'NON VALIDE'}")


if __name__ == "__main__":
    main()
