"""TEST GRATUIT — l'ARBITRAIRE rend-il enfin le critique nécessaire ?

L'HISTORIQUE QUI MOTIVE CE TEST. Quatre leviers ont été mesurés et aucun ne rend un critique
nécessaire : conséquence (périssable, 33 %, sans effet), feature non-géométrique (maturité, lisible
dans le latent à 0,65, sans effet), prédiction (interception : FORME CLOSE), hétérogénéité (formule
ajustée 64,3 % vs MLP 64,0 % — écart −0,022). Point commun : la relation entre ce qu'on PERÇOIT et ce
que ça VAUT était toujours DÉRIVABLE, donc une formule la capture — et une formule bat un appris,
puisqu'elle ne paie ni bruit d'estimation ni données.

L'HYPOTHÈSE À TESTER : ce qui manque est l'ARBITRAIRE — une relation apparence→valeur qu'on ne peut
pas CALCULER, seulement avoir VÉCUE. Le type rouge nourrit, le bleu rend malade, le vert s'échappe :
rien dans la physique perceptible ne le dit.

DÉFINITION HONNÊTE de « formule » ici : une règle dérivable de grandeurs physiques perceptibles
(distance, vitesse, angle) SANS expérience. Une table de correspondance type→valeur n'est PAS une
formule : la remplir exige d'avoir goûté. C'est exactement la frontière calculer / apprendre.

ON EST GÉNÉREUX AVEC LA FORMULE : elle reçoit les exposants AJUSTÉS sur les données, et on lui donne
même le type comme SCALAIRE (ce qu'un monde réel ne fournit pas — une teinte n'est pas un rang).

  formule ≈ appris  -> l'arbitraire ne suffit pas non plus, il faut le dire.
  appris >> formule -> première condition mesurée où le critique est NÉCESSAIRE, pas juste utile.

Usage : PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_arbitrary_headroom.py
"""
from __future__ import annotations

import argparse
import math

import torch
from torch import nn

SPEED = 0.011
MAX_TURN = 1.5 * 0.05
CAPTURE = 1.0
BUDGET = 1500
GAMMA = 0.999
N_TYPES = 6


def chase(head0, prey0, pdir, vp):
    n = prey0.shape[0]
    pos = torch.zeros(n, 2)
    head, prey = head0.clone(), prey0.clone()
    caught = torch.zeros(n, dtype=torch.bool)
    tc = torch.full((n,), float(BUDGET))
    for t in range(BUDGET):
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
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    n, K = args.n, args.k

    # TABLE ARBITRAIRE type -> valeur, tirée une fois pour ce monde. Rien dans la géométrie ne la
    # prédit ; l'agent ne peut la connaître qu'en ayant mangé de chaque type.
    type_value = 5.0 + 35.0 * torch.rand(N_TYPES, generator=g)
    print("  table ARBITRAIRE type -> valeur : " +
          "  ".join(f"t{i}={float(v):.1f}" for i, v in enumerate(type_value)))

    dist0 = 3.0 + 8.0 * torch.rand(n, K, generator=g)
    azim = 2 * math.pi * torch.rand(n, K, generator=g)
    pdir_a = 2 * math.pi * torch.rand(n, K, generator=g)
    vp = SPEED * (0.2 + 1.1 * torch.rand(n, K, generator=g))
    typ = torch.randint(0, N_TYPES, (n, K), generator=g)
    val = type_value[typ]                                    # valeur RÉELLE, jamais perçue
    head0 = 2 * math.pi * torch.rand(n, generator=g)

    outcome = torch.zeros(n, K)
    for k in range(K):
        prey = torch.stack([dist0[:, k] * torch.cos(azim[:, k]),
                            dist0[:, k] * torch.sin(azim[:, k])], 1)
        pd = torch.stack([torch.cos(pdir_a[:, k]), torch.sin(pdir_a[:, k])], 1)
        c, tc = chase(head0, prey, pd, vp[:, k])
        outcome[:, k] = torch.where(c, val[:, k] * (GAMMA ** tc), torch.zeros(n))

    rel = torch.remainder(azim - head0.unsqueeze(1) + math.pi, 2 * math.pi) - math.pi
    ntr = int(0.7 * n)

    def gain(score: torch.Tensor) -> float:
        pick = score[ntr:].argmax(dim=1)
        return float(outcome[ntr:].gather(1, pick.unsqueeze(1)).mean())

    res = {}
    res["NEAREST (-min_dist)"] = gain(-dist0)

    # FORMULE sur la GÉOMÉTRIE seule (elle ne peut RIEN savoir de la valeur)
    best, bp = -1.0, None
    for b in torch.linspace(0.0, 2.0, 11):
        for c in torch.linspace(0.0, 3.0, 13):
            s = 1.0 / (dist0 ** b * (1 + vp / SPEED) ** c)
            v = float(outcome[:ntr].gather(1, s[:ntr].argmax(1).unsqueeze(1)).mean())
            if v > best:
                best, bp = v, (float(b), float(c))
    res["FORMULE géométrie ajustée"] = gain(1.0 / (dist0 ** bp[0] * (1 + vp / SPEED) ** bp[1]))

    # FORMULE + le type comme SCALAIRE — on lui offre une information qu'un monde réel ne donne pas
    # (une teinte n'est pas un rang), et pourtant elle ne peut pas exprimer une table arbitraire.
    best2, bp2 = -1.0, None
    for a in torch.linspace(-1.0, 1.0, 9):
        for b in torch.linspace(0.0, 2.0, 9):
            for c in torch.linspace(0.0, 3.0, 10):
                s = (1.0 + typ.float()) ** a / (dist0 ** b * (1 + vp / SPEED) ** c)
                v = float(outcome[:ntr].gather(1, s[:ntr].argmax(1).unsqueeze(1)).mean())
                if v > best2:
                    best2, bp2 = v, (float(a), float(b), float(c))
    res["FORMULE + type-scalaire"] = gain((1.0 + typ.float()) ** bp2[0]
                                          / (dist0 ** bp2[1] * (1 + vp / SPEED) ** bp2[2]))

    # APPRIS : le type en ONE-HOT (l'apparence, sans ordre) + la géométrie
    oh = torch.nn.functional.one_hot(typ, N_TYPES).float()
    feats = torch.cat([dist0.unsqueeze(-1), (vp / SPEED).unsqueeze(-1),
                       rel.abs().unsqueeze(-1), torch.cos(rel).unsqueeze(-1), oh], -1)
    mu = feats[:ntr].reshape(-1, feats.shape[-1]).mean(0)
    sd = feats[:ntr].reshape(-1, feats.shape[-1]).std(0).clamp_min(1e-6)
    x = (feats - mu) / sd
    d = feats.shape[-1]
    for name, net in (("LINÉAIRE appris (type one-hot)", nn.Linear(d, 1)),
                      ("MLP appris (type one-hot)",
                       nn.Sequential(nn.Linear(d, 128), nn.SiLU(),
                                     nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 1)))):
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(4000):
            i = torch.randperm(ntr)[:512]
            opt.zero_grad()
            ((net(x[:ntr][i]).squeeze(-1) - outcome[:ntr][i]) ** 2).mean().backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            res[name] = gain(net(x).squeeze(-1))

    res["ORACLE (borne)"] = float(outcome[ntr:].max(dim=1).values.mean())
    res["hasard"] = float(outcome[ntr:].mean())

    print(f"\n  {n} scénarios × {K} proies | {N_TYPES} types, valeur ARBITRAIRE par type\n")
    base, top = res["hasard"], res["ORACLE (borne)"]
    for k2, v in res.items():
        pct = 100 * (v - base) / (top - base) if top > base else float("nan")
        print(f"  {k2:34s} gain {v:6.2f}   ({pct:5.1f} % de la marge oracle)")
    gap = res["MLP appris (type one-hot)"] - max(v for k2, v in res.items() if k2.startswith("FORMULE"))
    print(f"\n  ÉCART MLP − meilleure FORMULE : {gap:+.3f}")
    print("  -> encore REDONDANT : l'arbitraire ne suffit pas non plus" if gap < 0.15 else
          "  -> le critique est NÉCESSAIRE : aucune formule ne peut contenir une table arbitraire")


if __name__ == "__main__":
    main()
