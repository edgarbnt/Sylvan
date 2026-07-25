"""A1 — L'ENCODEUR NE PEUT PAS, OU N'EST PAS SOLLICITÉ ? (test GRATUIT, sans Godot, sans WM)

POURQUOI CETTE SONDE. Mesuré (2026-07-25) : l'encodeur du WM ne porte pas le type de proie (33,3 %
contre 27,3 % de majorité) alors que la teinte est 100 % séparable dans la RÉTINE. Une pression
explicite de décodage ne le répare pas — 10x le poids n'achète que +1,6 pt (35,7 → 37,3 %). Trois
explications ont déjà été écartées par la mesure : le gradient atteint bien l'encodeur, ce n'est pas
un problème de SÉLECTION (une sonde « type présent ? », sans argmin, est aussi au hasard), et ce
n'est pas un effondrement (rang effectif 38/128, mieux que l'historique 34/128).

RESTE DEUX CAUSES, ET ELLES APPELLENT DES FIX OPPOSÉS :
  (A) COMPÉTITION — l'architecture pourrait, mais la tâche perd face aux pertes WM (latent, énergie,
      déplacement, rollout, VICReg). Fix : rééquilibrer l'objectif.
  (B) ARCHITECTURE — un MLP à UNE couche cachée sur 278 dims concaténées ne peut pas extraire une
      couleur PAR RAYON. Fix : un encodeur à ATTENTION sur les rayons (ce que fait le slot, qui y
      arrive sans effort sur la même rétine).

COMMENT ON TRANCHE, SANS RIEN PAYER. On entraîne les deux architectures sur la SEULE tâche de teinte,
sans aucune perte WM. Si le MLP y arrive quand c'est sa seule tâche, l'architecture est hors de cause
et le coupable est (A). S'il échoue même seul, c'est (B) — et l'attention doit alors le battre nettement.

Les deux bras partagent tout ce qui pourrait fausser la comparaison : même corpus, même découpe
train/held-out, même largeur de sortie (128), même optimiseur, même nombre de pas. Seule la façon de
LIRE la rétine change.

CRITÈRES PRÉ-ENREGISTRÉS :
  T1 le MLP-seul .......... précision type > 60 % ⇒ l'architecture SUFFIT ⇒ cause (A) compétition.
                            ≤ 60 % ⇒ l'architecture est en cause.
  T2 l'attention-seule .... précision > 60 % ET nettement au-dessus du MLP (≥ +15 pts) ⇒ cause (B)
                            confirmée, et le levier est identifié.
  T3 contrôle « l'info est dans la rétine » ... ⚠️ CRITÈRE CORRIGÉ APRÈS COUP, et il faut le dire :
                            j'avais posé « une régression LINÉAIRE sur la rétine brute doit être
                            quasi parfaite ». C'est mal posé — un modèle linéaire sur une rétine
                            APLATIE ne peut pas non plus faire la sélection (trouver le rayon
                            nourriture le plus proche PUIS lire sa couleur) ; son échec ne dit rien
                            du corpus. Le bon contrôle est qu'un modèle CAPABLE de sélectionner y
                            arrive : le bras attention à 99,0 % l'établit. Mesuré par ailleurs et
                            indépendamment : teinte 100 % séparable dans la rétine (cos 1,0000).

CLI :
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_a1_architecture.py
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_a1_architecture.py --selfcheck
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from sylvan.critic_corpus import load_bc_corpora  # noqa: E402

N_RAYS, RAY_CH = 36, 4
LATENT = 128            # même largeur de sortie que l'encodeur du WM
HIDDEN = 256            # même largeur cachée que ProprioEncoder
PAL = torch.tensor([[0.90, 0.12, 0.10], [0.90, 0.55, 0.08],
                    [0.85, 0.10, 0.45], [0.80, 0.42, 0.42]])
T1_T2_BAR = 0.60        # « l'architecture y arrive » = 60 % pour 4 classes (hasard ~27 %)
GAP_BAR = 0.15          # l'attention doit battre le MLP d'au moins 15 points pour trancher (B)


class DenseEncoder(nn.Module):
    """LA COPIE EXACTE de l'encodeur du WM (ProprioEncoder) : un MLP à UNE couche cachée sur l'obs
    ENTIÈRE concaténée. C'est le bras à réfuter ou à disculper."""

    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, LATENT))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class RayAttentionEncoder(nn.Module):
    """Lit la rétine RAYON PAR RAYON, puis agrège par ATTENTION — la structure que le slot exploite.

    Chaque rayon est un token de 4 nombres (profondeur, R, G, B) ; un petit MLP partagé les plonge,
    un score d'attention (appris) pondère les rayons, et la somme pondérée forme la sortie avec la
    proprioception. La différence essentielle avec le bras dense n'est PAS la capacité — c'est que
    la couleur d'un rayon reste une entité manipulable au lieu d'être noyée dans 278 dims aplaties.
    """

    def __init__(self, obs_dim: int, d: int = 64) -> None:
        super().__init__()
        self.n_other = obs_dim - N_RAYS * RAY_CH
        self.token = nn.Sequential(nn.Linear(RAY_CH, d), nn.SiLU(), nn.Linear(d, d))
        self.score = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
        self.out = nn.Sequential(nn.Linear(d + self.n_other, HIDDEN), nn.SiLU(),
                                 nn.Linear(HIDDEN, LATENT))

    def forward(self, obs: torch.Tensor, retina_at: int) -> torch.Tensor:
        ret = obs[:, retina_at:retina_at + N_RAYS * RAY_CH].reshape(-1, N_RAYS, RAY_CH)
        other = torch.cat([obs[:, :retina_at], obs[:, retina_at + N_RAYS * RAY_CH:]], dim=1)
        tok = self.token(ret)                                  # [B, 36, d]
        w = torch.softmax(self.score(tok).squeeze(-1), dim=1)  # [B, 36] — quels rayons comptent
        pooled = (tok * w.unsqueeze(-1)).sum(dim=1)            # [B, d]
        return self.out(torch.cat([pooled, other], dim=1))


def type_labels(retina: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Type de la proie VISÉE (rayon nourriture le plus proche) — vérité lue dans la rétine."""
    r = retina.reshape(len(retina), N_RAYS, RAY_CH)
    depth, rgb = r[..., 0], r[..., 1:4]
    norm = rgb.norm(dim=-1)
    unit = rgb / (norm.unsqueeze(-1) + 1e-6)
    food = (unit[..., 0] > 0.55) & (norm > 1e-3)
    d = torch.where(food, depth, torch.full_like(depth, 9e9))
    nearest = d.argmin(dim=1)
    picked = unit[torch.arange(len(unit)), nearest]
    ref = PAL / PAL.norm(dim=-1, keepdim=True)
    return (picked @ ref.T).argmax(dim=1), food.any(dim=1)


def train_arm(enc: nn.Module, obs: torch.Tensor, typ: torch.Tensor, retina_at: int,
              steps: int, attn: bool, seed: int = 0) -> tuple[float, int]:
    """Entraîne encodeur + tête sur la SEULE tâche de type. Renvoie (précision held-out, nb params).

    La graine est REPOSÉE ici, pas seulement au début du script : sans ça le second bras hérite de
    l'état du générateur consommé par le premier (init des poids ET ordre des batches différents), et
    on attribuerait à l'architecture un écart qui vient du tirage. Un A/B non reproductible est un
    A/B inutile.
    """
    torch.manual_seed(seed)
    head = nn.Sequential(nn.Linear(LATENT, 128), nn.SiLU(), nn.Linear(128, 4))
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    ntr = int(0.7 * len(obs))
    for _ in range(steps):
        i = torch.randperm(ntr)[:256]
        opt.zero_grad()
        z = enc(obs[i], retina_at) if attn else enc(obs[i])
        lossf(head(z), typ[i]).backward()
        opt.step()
    n_par = sum(q.numel() for q in enc.parameters())
    enc.eval(); head.eval()
    with torch.no_grad():
        acc = []
        for i in range(ntr, len(obs), 512):
            z = enc(obs[i:i + 512], retina_at) if attn else enc(obs[i:i + 512])
            acc.append((head(z).argmax(1) == typ[i:i + 512]).float())
        return float(torch.cat(acc).mean()), n_par


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+",
                    default=["data/replay_buffer/foret_v1_planner",
                             "data/replay_buffer/foret_v1b_planner"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    torch.manual_seed(0)
    torch.set_num_threads(4)

    obs, energy, _cmd, _b = load_bc_corpora(a.corpus)
    retina_at = obs.shape[1] - N_RAYS * RAY_CH - 1
    idx = torch.arange(0, len(energy), a.stride)
    typ, valid = type_labels(obs[idx, retina_at:retina_at + N_RAYS * RAY_CH])
    idx, typ = idx[valid], typ[valid]
    X = obs[idx]
    counts = torch.bincount(typ, minlength=4).float()
    majority = float(counts.max() / counts.sum())
    print("Découpe held-out = les 30 % DERNIERS ticks du corpus concaténé, donc des ÉPISODES "
          "distincts\n(pas un tirage aléatoire, qui ferait fuir des ticks voisins quasi identiques).")
    print(f"MÊME corpus, MÊME découpe, MÊME largeur (128), MÊME budget de pas, MÊME graine — seule la LECTURE "
          f"de la rétine change.\n{len(X)} états | majorité {majority:.1%} | obs {X.shape[1]} "
          f"(rétine à {retina_at})\n")

    # T3 — contrôle : l'information est-elle dans la rétine ? (sinon rien d'autre n'a de sens)
    ret = X[:, retina_at:retina_at + N_RAYS * RAY_CH]
    lin = nn.Linear(ret.shape[1], 4)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-3)
    ntr = int(0.7 * len(X))
    for _ in range(a.steps):
        i = torch.randperm(ntr)[:256]
        opt.zero_grad()
        nn.functional.cross_entropy(lin(ret[i]), typ[i]).backward()
        opt.step()
    lin.eval()
    with torch.no_grad():
        acc_ret = float((lin(ret[ntr:]).argmax(1) == typ[ntr:]).float().mean())

    torch.manual_seed(0)
    acc_dense, p_dense = train_arm(DenseEncoder(X.shape[1]), X, typ, retina_at, a.steps, attn=False)
    torch.manual_seed(0)
    acc_attn, p_attn = train_arm(RayAttentionEncoder(X.shape[1]), X, typ, retina_at, a.steps, attn=True)

    print(f"  rétine BRUTE (linéaire)          : {acc_ret:.1%}")
    print(f"  MLP dense (copie de l'encodeur)  : {acc_dense:.1%}   ({p_dense:,} paramètres)")
    print(f"  ATTENTION par rayon              : {acc_attn:.1%}   ({p_attn:,} paramètres)")
    print("  ⚠️  l'attention a plus de paramètres — on le DIT. La question posée n'est pas « qui est")
    print("     meilleur toutes choses égales », c'est « l'encodeur RÉELLEMENT SERVI par le WM")
    print("     peut-il lire une couleur par rayon ? ». T1 y répond seul, sans comparaison.")
    print("=" * 78)

    # T3 est porté par le bras attention, pas par la régression linéaire (cf. docstring) : un
    # linéaire sur rétine aplatie ne peut pas sélectionner, son score ne prouve rien.
    t3 = acc_attn > 0.80
    print(f"{'✅' if t3 else '❌'} T3 INFO DANS RÉTINE  attention {acc_attn:.1%} — l'information "
          f"{'EST bien là (un lecteur capable la trouve)' if t3 else 'MANQUE vraiment'} "
          f"| linéaire-aplati {acc_ret:.1%}, non concluant PAR CONSTRUCTION")
    t1 = acc_dense > T1_T2_BAR
    print(f"{'✅' if t1 else '❌'} T1 MLP SEUL          {acc_dense:.1%} — "
          f"{'l architecture SUFFIT → la cause est la COMPÉTITION avec les pertes WM' if t1 else 'l architecture NE SUFFIT PAS, même sans concurrence'}")
    t2 = acc_attn > T1_T2_BAR and (acc_attn - acc_dense) >= GAP_BAR
    print(f"{'✅' if t2 else '❌'} T2 ATTENTION         {acc_attn:.1%} "
          f"({acc_attn - acc_dense:+.1%} vs dense) — "
          f"{'lecture PAR RAYON = le levier' if t2 else 'l attention ne tranche pas'}")
    print("=" * 78)
    if t1:
        print("⇒ CAUSE (A) : rééquilibrer l'objectif du WM, pas son architecture.")
    elif t2:
        print("⇒ CAUSE (B) : le MLP dense ne PEUT pas lire une couleur par rayon ; l'attention si.")
        print("  Le levier est architectural, et il est déjà validé ailleurs (le slot lit la même")
        print("  rétine par attention géométrique).")
    else:
        print("⇒ NI L'UN NI L'AUTRE : aucune des deux architectures n'y arrive seule. La difficulté")
        print("  est dans la TÂCHE ou la DONNÉE — re-diagnostiquer avant toute construction.")
    return 0


def selfcheck() -> int:
    torch.manual_seed(0)
    # les labels : une rétine synthétique où le rayon 5 porte la teinte du type 2, plus proche
    ret = torch.zeros(3, N_RAYS * RAY_CH)
    r = ret.reshape(3, N_RAYS, RAY_CH)
    r[:, 5, 0] = 0.2                              # proche
    r[:, 5, 1:4] = PAL[2]
    r[:, 9, 0] = 0.8                              # plus loin
    r[:, 9, 1:4] = PAL[0]
    typ, valid = type_labels(r.reshape(3, -1))
    assert bool(valid.all()) and int(typ[0]) == 2, (typ, valid)
    print("  [ok] l'étiquette suit le rayon nourriture le PLUS PROCHE (type 2, pas le type 0 lointain)")

    # les deux encodeurs rendent bien la même largeur, et l'attention voit les 36 rayons
    obs = torch.randn(4, 133 + N_RAYS * RAY_CH + 1)
    d, at = DenseEncoder(obs.shape[1]), RayAttentionEncoder(obs.shape[1])
    assert d(obs).shape == (4, LATENT) and at(obs, 133).shape == (4, LATENT)
    print(f"  [ok] les deux bras rendent {LATENT} dims — comparaison à largeur ÉGALE")

    # l'attention est bien une pondération normalisée sur les rayons (pas un raccourci)
    ret2 = obs[:, 133:133 + N_RAYS * RAY_CH].reshape(-1, N_RAYS, RAY_CH)
    w = torch.softmax(at.score(at.token(ret2)).squeeze(-1), dim=1)
    assert w.shape == (4, N_RAYS) and torch.allclose(w.sum(1), torch.ones(4), atol=1e-5)
    print(f"  [ok] l'attention pondère les {N_RAYS} rayons et somme à 1")
    print("SELFCHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
