"""diag_critic_corpus_coverage, GRATUIT (offline, buffers deja sur disque).

QUESTION FALSIFIABLE (2026-07-06). Le critique appris a echoue 3x en boucle fermee par PLAFOND
D'IMITATION : diagnostic = son corpus (142 vies, toutes vecues par le planner analytique) manque
de reussites de POURSUITE LOINTAINE, donc il a appris le fatalisme ("affame + bouffe loin =
condamne"). AVANT de payer des heures de collecte, on MESURE si le corpus manque vraiment de ces
exemplaires positifs.

Un exemplaire "poursuite-lointaine reussie" = un etat ou (a) la pulsion urgente est BASSE (< 0.35)
et (b) sa ressource est LOIN (> 4 m, coords crues du planner), et (c) cette pulsion est ENSUITE
REMPLIE dans le meme episode (saut de drive = consommation). C'est exactement ce que le critique
doit voir pour apprendre "va la chercher, ca passe" au lieu de "condamne".

Corpus = meme glob que train_survival_critic.py (hesit_probe_*_surv). Monde lu du nom de dossier
(55 = dense 5+5, 11 = epars 1+1). Le 1+1 est le monde qui echoue ; ce sont SES exemplaires qui
comptent.

CRITERES PRE-ENREGISTRES :
- CONFIRME "collecter plus" si, en EPARS : peu d'episodes contiennent un exemplaire (< ~8) ET le
  taux "urgent+loin -> rempli" est bas (< 25 %) -> le critique n'avait pas les positifs, la
  collecte de vies eparses reussies est le levier ; le compte donne le volume a viser.
- REFUTE si : exemplaires epars deja abondants (bien peuples) -> le corpus n'est pas le goulot,
  creuser ailleurs (horizon 80 trop court pour l'epars).

Lancer : PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_critic_corpus_coverage.py [--selfcheck]
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import math
import statistics as st
from pathlib import Path

URGENT_LOW = 0.35     # pulsion "basse" (normalisee 0-1)
FAR_M = 4.0           # ressource "loin" (metres)
JUMP_LO, JUMP_HI = 0.05, 0.50    # saut de drive = consommation (respawn = > 0.50)
DEATH_LVL = 0.03      # drive quasi nul en fin d'episode = mort
MIN_EP_LEN = 15


def _dist(pos) -> float | None:
    if pos is None:
        return None
    return math.hypot(float(pos[0]), float(pos[1]))


def load_episodes(d: str) -> list[list[dict]]:
    """ep_0000.jsonl -> episodes (coupes aux respawns), chaque etat = {e,t,fd,wd}."""
    f = Path(d) / "ep_0000.jsonl"
    if not f.exists():
        return []
    rows = []
    for line in open(f):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        p = r.get("plan")
        if p is None:
            continue
        rows.append({"e": float(r["obs"]["energy"]) / 100.0,
                     "t": float(r["obs"]["thirst"]) / 100.0,
                     "fd": _dist(p.get("food")), "wd": _dist(p.get("water"))})
    segs, cur = [], []
    for row in rows:
        if cur and (row["e"] - cur[-1]["e"] > JUMP_HI or row["t"] - cur[-1]["t"] > JUMP_HI):
            segs.append(cur)
            cur = []
        cur.append(row)
    if cur:
        segs.append(cur)
    return [s for s in segs if len(s) >= MIN_EP_LEN]


def analyze_episode(ep: list[dict]) -> dict:
    """Compte les etats "urgent+loin" (denominateur) et ceux SUIVIS d'un remplissage de CETTE
    pulsion (numerateur = poursuite-lointaine reussie)."""
    urgent_far = 0
    far_success = 0
    for i, s in enumerate(ep):
        # pulsion urgente = la plus basse ; sa ressource = rouge(bouffe) si e<=t sinon bleu(eau)
        food_urgent = s["e"] <= s["t"]
        lvl = s["e"] if food_urgent else s["t"]
        dist = s["fd"] if food_urgent else s["wd"]
        if lvl >= URGENT_LOW or dist is None or dist <= FAR_M:
            continue
        urgent_far += 1
        key = "e" if food_urgent else "t"
        # cette pulsion est-elle REMPLIE plus tard dans l'episode ? (saut de consommation)
        if any(JUMP_LO < ep[j][key] - ep[j - 1][key] < JUMP_HI for j in range(i + 1, len(ep))):
            far_success += 1
    last = ep[-1]
    died = min(last["e"], last["t"]) < DEATH_LVL
    return {"urgent_far": urgent_far, "far_success": far_success, "died": died,
            "surv_lvl": min(last["e"], last["t"])}


def selfcheck() -> None:
    # episode ou l'agent est affame (e bas), bouffe loin, puis MANGE (e saute) = 1 far_success
    ep = ([{"e": 0.20, "t": 0.80, "fd": 6.0, "wd": 3.0}] * 20
          + [{"e": 0.70, "t": 0.75, "fd": 0.5, "wd": 3.0}])   # e saute 0.20->0.70 au dernier
    m = analyze_episode(ep)
    assert m["urgent_far"] == 20 and m["far_success"] == 20, m
    # episode ou affame + bouffe loin mais MEURT sans manger = 0 success, died
    ep2 = [{"e": max(0.0, 0.20 - i * 0.012), "t": 0.80, "fd": 6.0, "wd": 3.0} for i in range(18)]
    m2 = analyze_episode(ep2)
    assert m2["urgent_far"] == 18 and m2["far_success"] == 0 and m2["died"], m2
    print("[selfcheck] OK, exemplaire reussi vs mort-sans-manger classes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/replay_buffer/hesit_probe_*_surv")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return

    worlds = {"dense (5+5)": "_55_", "epars (1+1)": "_11_"}
    print(f"corpus glob = {args.glob}\n")
    header = f"{'monde':<14}{'ep':>5}{'morts':>7}{'survie med':>12}{'etats urgent+loin':>19}{'-> remplie (reussi)':>21}{'ep avec exemplaire':>20}"
    print(header)
    for wname, tag in worlds.items():
        dirs = [d for d in sorted(globmod.glob(args.glob)) if tag in d + "_"]
        eps = []
        for d in dirs:
            eps += load_episodes(d)
        if not eps:
            print(f"{wname:<14} (aucun episode)")
            continue
        ms = [analyze_episode(e) for e in eps]
        n_ep = len(ms)
        deaths = sum(m["died"] for m in ms)
        surv_med = st.median([len(e) for e in eps])
        uf = sum(m["urgent_far"] for m in ms)
        fs = sum(m["far_success"] for m in ms)
        ep_with = sum(1 for m in ms if m["far_success"] > 0)
        rate = f"{fs} ({100 * fs / uf:.0f}%)" if uf else "0"
        print(f"{wname:<14}{n_ep:>5}{deaths:>7}{surv_med:>12.0f}{uf:>19}{rate:>21}{ep_with:>20}")

    print("\n--- VERDICT (criteres pre-enregistres) ---")
    dirs11 = [d for d in sorted(globmod.glob(args.glob)) if "_11_" in d + "_"]
    eps11 = [e for d in dirs11 for e in load_episodes(d)]
    ms11 = [analyze_episode(e) for e in eps11]
    uf11 = sum(m["urgent_far"] for m in ms11)
    fs11 = sum(m["far_success"] for m in ms11)
    ep_with11 = sum(1 for m in ms11 if m["far_success"] > 0)
    rate11 = fs11 / uf11 if uf11 else 0.0
    if ep_with11 < 8 and rate11 < 0.25:
        print(f"CONFIRME 'collecter plus' : EPARS n'a que {ep_with11} episodes avec un exemplaire de")
        print(f"  poursuite-lointaine reussie, taux urgent+loin->remplie = {100*rate11:.0f}% ({fs11}/{uf11}).")
        print("  Le critique ne pouvait PAS apprendre 'va la chercher' : ses donnees eparses disent 'mort'.")
        print("  -> collecter des vies EPARSES REUSSIES avec le planner analytique (il y arrive : 1760-1800),")
        print("     puis re-entrainer le critique dessus, puis re-juger. Le compte donne le volume a viser.")
    elif ep_with11 >= 8 and rate11 >= 0.25:
        print(f"REFUTE : EPARS a deja {ep_with11} episodes avec exemplaires (taux {100*rate11:.0f}%).")
        print("  Le corpus n'est pas le goulot -> creuser ailleurs (horizon 80 trop court pour l'epars ?).")
    else:
        print(f"MITIGE : EPARS ep-avec-exemplaire={ep_with11}, taux={100*rate11:.0f}% -> lire les sous-scores.")


if __name__ == "__main__":
    main()
