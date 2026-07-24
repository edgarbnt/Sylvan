"""TEST GRATUIT — un critique APPRIS bat-il la meilleure FORMULE, ou est-il redondant avec elle ?

POURQUOI CE TEST EXISTE. Le levier « proie mobile » crée bien une marge (test d'interception : 67,5 %
de capture contre 56,2 % pour la poursuite). Mais l'interception a une FORME CLOSE : « vise la
position prédite » s'écrit à la main. Si on s'arrête là, un critique appris ne ferait que ré-apprendre
une formule — exactement l'anomalie A3 de l'audit, et le piège dans lequel on est tombé cinq fois
(corrélation de rang mesurée +0,93 entre notre meilleur critique et `-min_dist`).

L'hypothèse à tester est donc : ce qui rend un critique NÉCESSAIRE n'est pas UNE feature
non-géométrique, c'est le CHOIX entre options HÉTÉROGÈNES dont l'arbitrage n'a pas de forme close.

CE QU'ON MESURE. Des scénarios avec K proies hétérogènes (distance, direction, VITESSE et VALEUR
toutes différentes). L'agent doit CHOISIR laquelle poursuivre, puis la poursuit en interception.
Gain = valeur de la proie × actualisation du temps de capture (0 si ratée). On compare :

  NEAREST   : la plus proche                     <- ce que fait `-min_dist` aujourd'hui
  FORMULE   : meilleure règle paramétrique AJUSTÉE sur les données (on est GÉNÉREUX avec elle)
  LINÉAIRE  : score linéaire appris sur les features  <- un critique « simple »
  MLP       : score non-linéaire appris               <- un vrai critique
  ORACLE    : simule les K choix et prend le meilleur <- borne supérieure atteignable

LECTURE :
  MLP ≈ FORMULE            -> le critique est REDONDANT, une formule suffit. Piste morte.
  MLP nettement > FORMULE  -> il existe une marge qu'AUCUNE formule simple n'atteint : c'est là que
                              le critique devient nécessaire et pas seulement utile.
L'écart ORACLE − MLP dit ce qui reste hors de portée même d'un bon appris.

Aucun Godot, aucun WM : constantes MESURÉES de notre corps.

Usage : PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_choice_headroom.py
"""
from __future__ import annotations

import argparse
import math

import torch
from torch import nn

SPEED = 0.011          # m/tick (mesuré)
MAX_TURN = 1.5 * 0.05  # rad/tick (mesuré)
CAPTURE = 1.0          # m
BUDGET = 1500          # ticks accordés à une poursuite
GAMMA = 0.999          # le temps coûte (drain d'énergie)


def chase(pos0, head0, prey0, pdir, vp, budget=BUDGET):
    """Poursuite en INTERCEPTION d'une proie. → (capturée, tick de capture)."""
    n = pos0.shape[0]
    pos, head, prey = pos0.clone(), head0.clone(), prey0.clone()
    caught = torch.zeros(n, dtype=torch.bool)
    tc = torch.full((n,), float(budget))
    for t in range(budget):
        dist = (prey - pos).norm(dim=1)
        new = (~caught) & (dist <= CAPTURE)
        tc[new] = float(t)
        caught |= new
        if bool(caught.all()):
            break
        tau = dist / SPEED
        for _ in range(3):
            tau = (prey + pdir * vp.unsqueeze(1) * tau.unsqueeze(1) - pos).norm(dim=1) / SPEED
        tgt = prey + pdir * vp.unsqueeze(1) * tau.unsqueeze(1)
        want = torch.atan2(tgt[:, 1] - pos[:, 1], tgt[:, 0] - pos[:, 0])
        err = torch.remainder(want - head + math.pi, 2 * math.pi) - math.pi
        head = head + err.clamp(-MAX_TURN, MAX_TURN)
        pos = torch.where(caught.unsqueeze(1), pos,
                          pos + torch.stack([torch.cos(head), torch.sin(head)], 1) * SPEED)
        prey = torch.where(caught.unsqueeze(1), prey, prey + pdir * vp.unsqueeze(1))
    return caught, tc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--k", type=int, default=3)      # proies proposées par scénario
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    n, K = args.n, args.k

    # POPULATION HÉTÉROGÈNE : distance, direction, VITESSE et VALEUR varient indépendamment.
    # C'est l'hétérogénéité qui crée l'arbitrage sans forme close (loin+lente+riche vs proche+rapide+pauvre).
    dist0 = 3.0 + 8.0 * torch.rand(n, K, generator=g)
    azim = 2 * math.pi * torch.rand(n, K, generator=g)
    pdir_a = 2 * math.pi * torch.rand(n, K, generator=g)
    vp = SPEED * (0.2 + 1.1 * torch.rand(n, K, generator=g))     # 0,2x à 1,3x la vitesse agent
    val = 10.0 + 30.0 * torch.rand(n, K, generator=g)            # énergie rendue : 10 à 40
    head0 = 2 * math.pi * torch.rand(n, generator=g)

    # Issue RÉELLE de chaque choix possible (c'est ça qui définit l'oracle).
    outcome = torch.zeros(n, K)
    for k in range(K):
        prey = torch.stack([dist0[:, k] * torch.cos(azim[:, k]),
                            dist0[:, k] * torch.sin(azim[:, k])], 1)
        pd = torch.stack([torch.cos(pdir_a[:, k]), torch.sin(pdir_a[:, k])], 1)
        c, tc = chase(torch.zeros(n, 2), head0, prey, pd, vp[:, k])
        outcome[:, k] = torch.where(c, val[:, k] * (GAMMA ** tc), torch.zeros(n))

    # features disponibles au moment du CHOIX (rien d'oraculaire : tout est perceptible)
    rel_ang = torch.remainder(azim - head0.unsqueeze(1) + math.pi, 2 * math.pi) - math.pi
    feats = torch.stack([dist0, vp / SPEED, val, rel_ang.abs(), torch.cos(rel_ang)], -1)  # [n,K,5]

    ntr = int(0.7 * n)
    def score_gain(score: torch.Tensor) -> float:
        pick = score[ntr:].argmax(dim=1)
        return float(outcome[ntr:].gather(1, pick.unsqueeze(1)).mean())

    res = {}
    res["NEAREST (-min_dist)"] = score_gain(-dist0)

    # FORMULE : famille valeur^a / (distance^b · (1+vitesse)^c), exposants AJUSTÉS sur le train.
    # On est GÉNÉREUX : on lui donne le meilleur membre de sa famille, pas un réglage arbitraire.
    best, bp = -1.0, None
    for a in torch.linspace(0.0, 2.0, 9):
        for bx in torch.linspace(0.0, 2.0, 9):
            for c in torch.linspace(0.0, 3.0, 10):
                s = val ** a / (dist0 ** bx * (1 + vp / SPEED) ** c)
                pick = s[:ntr].argmax(dim=1)
                v = float(outcome[:ntr].gather(1, pick.unsqueeze(1)).mean())
                if v > best:
                    best, bp = v, (float(a), float(bx), float(c))
    s = val ** bp[0] / (dist0 ** bp[1] * (1 + vp / SPEED) ** bp[2])
    res[f"FORMULE ajustée a={bp[0]:.2f} b={bp[1]:.2f} c={bp[2]:.2f}"] = score_gain(s)

    # APPRIS : linéaire puis MLP, entraînés à prédire l'issue (régression), choix = argmax
    mu, sd = feats[:ntr].reshape(-1, 5).mean(0), feats[:ntr].reshape(-1, 5).std(0).clamp_min(1e-6)
    x = (feats - mu) / sd
    for name, net in (("LINÉAIRE appris", nn.Linear(5, 1)),
                      ("MLP appris", nn.Sequential(nn.Linear(5, 128), nn.SiLU(),
                                                   nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 1)))):
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(4000):
            i = torch.randperm(ntr)[:512]
            opt.zero_grad()
            ((net(x[:ntr][i]).squeeze(-1) - outcome[:ntr][i]) ** 2).mean().backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            res[name] = score_gain(net(x).squeeze(-1))

    res["ORACLE (borne)"] = float(outcome[ntr:].max(dim=1).values.mean())
    res["hasard"] = float(outcome[ntr:].mean())

    print(f"  {n} scénarios × {K} proies hétérogènes (vitesse 0,2-1,3× agent, valeur 10-40)\n")
    base, top = res["hasard"], res["ORACLE (borne)"]
    for k2, v in res.items():
        pct = 100 * (v - base) / (top - base) if top > base else float("nan")
        print(f"  {k2:38s} gain moyen {v:6.2f}   ({pct:5.1f} % de la marge oracle)")
    gap = res["MLP appris"] - max(v for k2, v in res.items() if k2.startswith("FORMULE"))
    print(f"\n  ÉCART MLP − meilleure FORMULE : {gap:+.3f}")
    print("  -> le critique est REDONDANT avec une formule" if gap < 0.15 else
          "  -> il existe une marge qu'AUCUNE formule simple n'atteint : le critique devient NÉCESSAIRE")


if __name__ == "__main__":
    main()
