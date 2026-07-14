"""SONDE GRATUITE — le vecu de l'entite contient-il quelque chose que son INNE ignore ? (2026-07-15)

LA QUESTION (owner) : « si le critique n'a pas lieu d'etre, notre entite pourra-t-elle quand meme
PROGRESSER a partir de ses experiences ? »

CE QU'UN CRITIQUE PEUT APPRENDRE, EXACTEMENT. Le cout analytique (l'INNE, cable a la conception :
distance / drain / vitesse / temps de virage) donne deja une prediction de « combien de pas vais-je
encore vivre depuis ici ». Un critique appris ne peut apporter que le RESIDU :

        residu  =  ce qui est REELLEMENT arrive  -  ce que l'inne PREDISAIT

Si ce residu est du bruit pur, alors l'experience vecue ne contient RIEN que l'entite ne sache deja
de naissance -> aucun critique, aussi bien entraine soit-il, n'a quoi que ce soit a apporter, et le
monde est trop pauvre pour que le vecu vaille quelque chose.
Si ce residu est PREDICTIBLE depuis l'etat, alors il y a bien une lecon dans le vecu -- et on sait
exactement laquelle, et combien elle vaut.

DEUX MESURES :
  A. FIDELITE DE L'INNE  : l'inne explique-t-il deja ce qui arrive ? (R^2, correlation de rang)
  B. LE VECU EST-IL INSTRUCTIF ? : on entraine un petit reseau a predire le RESIDU depuis l'etat,
     et on mesure son R^2 sur des EPISODES JAMAIS VUS. C'est LA reponse a la question de l'owner.
       R^2 hors-echantillon ~ 0     -> le residu est du bruit : RIEN a apprendre dans ce monde.
       R^2 hors-echantillon > 0.15  -> il y a une lecon structuree, et le critique doit apprendre
                                       CA (le residu), pas la valeur absolue.

HONNETETE (§2) : le corpus est collecte sous UNE politique (deterministe). Le residu mesure donc ce
qui est apprenable SOUS CETTE POLITIQUE. Un R^2 nul ne prouve pas que le monde est vide dans l'absolu
-- il prouve que CE vecu-la n'enseigne rien. (Boucle auto-confirmante deja mesuree : 18.6% d'approches
alignees.) On le dit, on ne le cache pas.

CENSURE : un episode coupe par la fin du run (pas par la mort) a une survie TRONQUEE (« au moins X »).
Ces etats sont EXCLUS -- les garder biaiserait le residu vers le negatif.

Usage :
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_experience_residual.py --selfcheck
  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_experience_residual.py
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import os
from pathlib import Path

import torch
from torch import nn

STEPS_PER_REPLAN = 10           # 1 replan planner = 10 pas Godot (cf H_SURV dans train_survival_critic)


def load_raw(dirs: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> (X [N,2,5] tokens, ACTUAL [N] pas REELLEMENT vecus ensuite, EID [N] id d'episode).

    Ne garde QUE les segments finis par une MORT (survie non tronquee = label honnete)."""
    from scripts.train_survival_critic import token

    X: list[list[list[float]]] = []
    actual: list[float] = []
    eids: list[int] = []
    eid = 0
    for d in dirs:
        f = Path(d) / "ep_0000.jsonl"
        if not f.exists():
            continue
        rows = []
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = r.get("plan")
            if p is None:
                continue
            rows.append((float(r["obs"]["energy"]) / 100.0, float(r["obs"]["thirst"]) / 100.0,
                         p.get("food"), p.get("water")))
        segs: list[list] = []
        cur: list = []
        for row in rows:                       # coupe aux respawns (les 2 drives remontent d'un coup)
            if cur and (row[0] - cur[-1][0] > 0.5 or row[1] - cur[-1][1] > 0.5):
                segs.append(cur)
                cur = []
            cur.append(row)
        if cur:
            segs.append(cur)
        for seg in segs:
            L = len(seg)
            if L < 15:
                continue
            if min(seg[-1][0], seg[-1][1]) >= 0.03:        # CENSURE : pas mort -> survie tronquee
                continue
            for t, (e, th, fp, wp) in enumerate(seg):
                X.append([token(e, fp), token(th, wp)])
                actual.append((L - t) * STEPS_PER_REPLAN)  # pas reellement vecus apres cet instant
                eids.append(eid)
            eid += 1
    return torch.tensor(X), torch.tensor(actual, dtype=torch.float32), torch.tensor(eids)


def r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Part de la variance de `target` expliquee par `pred`. 0 = ne fait pas mieux que la moyenne."""
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot.clamp(min=1e-9))


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Correlation de RANG : est-ce que l'inne ORDONNE bien les situations (meme si l'echelle est fausse) ?"""
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    # std de POPULATION (unbiased=False) : avec le std non-biaise (n-1) la correlation d'une
    # variable avec elle-meme vaudrait (n-1)/n, pas 1.
    ra = (ra - ra.mean()) / ra.std(unbiased=False).clamp(min=1e-9)
    rb = (rb - rb.mean()) / rb.std(unbiased=False).clamp(min=1e-9)
    return float((ra * rb).mean())


def selfcheck() -> None:
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert abs(r2(t, t) - 1.0) < 1e-6                            # prediction parfaite -> R2 = 1
    assert abs(r2(torch.full_like(t, t.mean()), t)) < 1e-6       # predire la moyenne -> R2 = 0
    assert r2(t.flip(0), t) < 0.0                                # prediction inversee -> R2 negatif
    assert abs(spearman(t, t * 3.0 + 7.0) - 1.0) < 1e-5          # rang invariant a l'echelle
    assert abs(spearman(t, -t) + 1.0) < 1e-5
    print("[selfcheck] OK : R2 (1 / 0 / negatif) et correlation de rang (+1 / -1)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/replay_buffer/critic_kin_[ab]")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return

    torch.manual_seed(args.seed)
    os.environ.setdefault("SYLVAN_PLANNER_DRAIN", "0.0005")
    os.environ.setdefault("SYLVAN_PLANNER_RESTORE", "0.4")
    from scripts.train_survival_critic import analytic_labels
    from sylvan.control.planning.command_planner import CommandPlanConfig

    X, actual, eid = load_raw(sorted(globmod.glob(args.glob)))
    if X.numel() == 0:
        print(f"AUCUN episode mort (non tronque) dans {args.glob}")
        return
    n_ep = int(eid.max()) + 1
    cfg = CommandPlanConfig()
    innate = analytic_labels(X) * cfg.surv_horizon        # prediction de l'INNE, en pas

    print(f"\n=== LE VECU CONTIENT-IL UNE LECON ? — {len(X)} instants, {n_ep} vies "
          f"(mortes, non tronquees) ===")
    print(f"survie reelle : mediane {actual.median():.0f} pas, moyenne {actual.mean():.0f}, "
          f"max {actual.max():.0f}")
    print(f"prediction de l'inne : mediane {innate.median():.0f} pas, moyenne {innate.mean():.0f}")

    # ── A. FIDELITE DE L'INNE ────────────────────────────────────────────────────────────────
    # On lui accorde le meilleur recalage affine possible (son echelle est arbitraire) : on teste
    # sa capacite a EXPLIQUER, pas son calibrage.
    a = torch.stack([innate, torch.ones_like(innate)], dim=1)
    coef = torch.linalg.lstsq(a, actual.unsqueeze(1)).solution.squeeze(1)
    fitted = innate * coef[0] + coef[1]
    print(f"\nA. L'INNE explique-t-il deja ce qui arrive ?")
    print(f"   R2 (apres recalage affine) : {r2(fitted, actual):+.3f}   "
          f"correlation de rang : {spearman(innate, actual):+.3f}")
    print(f"   erreur typique : {(fitted - actual).abs().median():.0f} pas "
          f"(pour une vie mediane de {actual.median():.0f} pas)")

    # ── B. LE RESIDU EST-IL APPRENABLE ? (LA question) ───────────────────────────────────────
    # ⚠️ VALIDATION CROISEE 4 PLIS, PAS UN SPLIT UNIQUE (corrige le 2026-07-15). La 1re version de
    # cette sonde tirait UN decoupage aleatoire et annoncait R2 +0.21 : c'etait un pli CHANCEUX.
    # Avec 57 vies, un pli unique n'en teste que ~14 -> l'estimation oscille de -0.13 a +0.10 selon
    # le tirage. Un seul chiffre etait donc une AUTODECEPTION (CLAUDE.md §2). On moyenne sur 4 plis.
    resid = actual - fitted
    r2s: list[float] = []
    for k in range(4):
        te = torch.tensor([int(e) % 4 == k for e in eid])          # split par EPISODE (jamais par
        tr = ~te                                                   # instant : ce serait une fuite)
        if int(te.sum()) < 20 or int(tr.sum()) < 20:
            continue
        torch.manual_seed(args.seed)
        mu, sd = resid[tr].mean(), resid[tr].std().clamp(min=1e-6)
        net = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                            nn.Linear(64, 1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        xtr, xte = X[tr].reshape(-1, 10), X[te].reshape(-1, 10)
        ytr = ((resid[tr] - mu) / sd).unsqueeze(1)
        for _ in range(args.epochs):
            nn.functional.mse_loss(net(xtr), ytr).backward()
            opt.step()
            opt.zero_grad()
        with torch.no_grad():
            pred_te = net(xte).squeeze(1) * sd + mu
        r2s.append(r2(pred_te, resid[te]))
        print(f"   pli {k} : R2 hors-echantillon {r2s[-1]:+.3f}")

    r2_te = sum(r2s) / len(r2s)
    print(f"\nB. LE RESIDU (ce que l'inne RATE) est-il APPRENABLE depuis l'etat ?")
    print(f"   R2 MOYEN sur les vies JAMAIS VUES : {r2_te:+.3f}   <<< LA REPONSE")
    print(f"   dispersion entre plis : {min(r2s):+.3f} a {max(r2s):+.3f}")
    print(f"   dispersion du residu : {resid.std():.0f} pas "
          f"({resid.std() / actual.mean() * 100:.0f}% de la vie moyenne)")

    print("\n--- VERDICT (critere ecrit AVANT) ---")
    if r2_te < 0.05:
        print(f"  R2 hors-echantillon {r2_te:+.3f} < 0.05 -> CE VECU-LA N'ENSEIGNE RIEN DE FIABLE.")
        print("  Ce que l'inne rate n'est pas retrouvable sur des vies jamais vues -> aucun critique")
        print("  ne peut le rattraper A PARTIR DE CE CORPUS.")
        print("  => la cause la plus probable n'est pas le cerveau mais l'EXPERIENCE elle-meme :")
        print("     57 vies, UNE politique deterministe (boucle auto-confirmante deja mesuree).")
        print("     Il faut de l'EXPLORATION et des vies VARIEES avant d'esperer apprendre quoi que")
        print("     ce soit -- ou un monde plus riche (obstacles, ressources qui s'epuisent, danger).")
    elif r2_te < 0.15:
        print(f"  R2 hors-echantillon {r2_te:+.3f} : lecon FAIBLE, non robuste entre plis. Marge etroite.")
    else:
        print(f"  R2 hors-echantillon {r2_te:+.3f} >= 0.15 -> IL Y A UNE LECON DANS LE VECU.")
        print("  => le critique doit apprendre CE RESIDU (et non la valeur absolue, dont 98% est un")
        print("     socle commun) : score = cout inne + critique-du-residu.")


if __name__ == "__main__":
    main()
