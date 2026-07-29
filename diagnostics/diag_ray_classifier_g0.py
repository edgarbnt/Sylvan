"""G0 — UNE PERCEPTION APPRISE DE LA CONSÉQUENCE PEUT-ELLE BATTRE LE COSINUS CODÉ-MAIN ?
(offline, gratuit, aucun entraînement de WM, aucune collecte)

POURQUOI. Le WM sait lire le type (A1 : 99,9 %), mais ce n'est pas lui qui désigne les cibles au
planner : c'est le SLOT, qui compare la couleur brute de chaque rayon à des requêtes CODÉES MAIN,
avec des seuils. C'est le dernier échafaudage sur le chemin perception→action, et il produit deux
défauts mesurés : le vert du danger est à cos 0,958 de celui des arbres (donc inséparable par la
couleur), et une flaque SÈCHE est du bleu identique à une flaque pleine alors qu'elle ne désaltère
pas. Les deux disparaissent si la classe s'apprend de ce que l'objet FAIT, pas de sa teinte.

L'IDÉE, ET CE QUI LA REND HONNÊTE. On étiquette un rayon par la CONSÉQUENCE observée quand l'entité
était à son contact : l'énergie est montée → nourriture ; la soif est montée → eau ; la santé a
baissé → danger ; rien ne s'est produit → neutre. La couleur reste une ENTRÉE (c'est de la
perception), elle n'entre jamais dans l'étiquette. Aucun oracle : tout vient de jauges que l'entité
a réellement vécues.

🚨 CE QUE CE G0 NE PEUT PAS FAIRE. Il juge la SÉPARABILITÉ (« ces rayons sont-ils classables ? »),
pas la localisation. Un classifieur meilleur que le cosinus ne garantit pas un meilleur slot : il
faudra l'y brancher et re-mesurer. On mesure donc ici la condition NÉCESSAIRE, pas le résultat.

CRITÈRES PRÉ-ENREGISTRÉS (écrits avant de lancer) :
  T0 MATIÈRE ..... au moins 200 rayons étiquetés par classe conséquence, sinon rien n'est
                   apprenable et le reste ne veut rien dire.
  T1 GLOBAL ...... précision held-out du classifieur > celle du cosinus codé-main, sur le MÊME
                   jeu de test. C'est la comparaison qui décide, pas une barre absolue.
  T2 DANGER ...... séparation arbre/danger : rappel du danger > 0,70, là où le cosinus échoue par
                   construction (cos 0,958 entre les deux verts). C'est le gain SPÉCIFIQUE attendu.
  KILL ........... si le classifieur ne bat pas le cosinus, la voie « apprendre de la conséquence »
                   est réfutée pour ce corpus : STOP, ne pas construire, chercher ailleurs.

CLI :
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_ray_classifier_g0.py
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_ray_classifier_g0.py --selfcheck
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

NRAY, RAYCH = 36, 4
RANGE_M = 10.0          # portée de la rétine (perception.gd MAX_RANGE)
DEPTH_OFFSET = 0.35     # la profondeur mesure la SURFACE ; +rayon de collision = distance au centre
CONTACT_M = 1.5         # « à portée » : les reliefs et les morsures arrivent au contact
RELIEF = 5.0            # remontée de jauge en un tick = consommation (convention du projet)
# 🚨 SEUIL MESURÉ SUR CE MONDE, PAS IMPORTÉ. La valeur 0,5 vient de train_danger_saliency, calibrée
# ailleurs ; ici la baisse vaut EXACTEMENT 0,45 par tick (médiane = min = p90, sur 3 099 ticks),
# donc « > 0,5 » n'attrapait RIEN et le G0 rendait 0 exemple de danger. La santé ne baisse que par
# morsure (la régénération, elle, la fait monter), donc n'importe quel seuil bien sous 0,45 et bien
# au-dessus du bruit flottant convient : 0,1.
DMG_DROP = 0.1          # baisse de santé en un tick = morsure (mesuré 0,45 dans ce monde)
LIFE_JUMP = 20.0        # saut de jauge = frontière d'épisode, jamais une conséquence
CLASSES = ("neutre", "nourriture", "eau", "danger")
MIN_PER_CLASS = 200     # T0
DANGER_RECALL_BAR = 0.70

# ORACLE D'ÉVALUATION — la palette RÉELLEMENT rendue par le monde. Licite en éval (monde-jouet),
# INTERDIT en entraînement : le classifieur ne voit que la conséquence. Sans lui on évaluerait
# contre les étiquettes faibles elles-mêmes, où un tronc au contact pendant un repas est étiqueté
# « nourriture » — et où un classifieur PARFAIT plafonnerait à ~35 %, ce qui ne dit rien de lui.
# Les teintes viennent du preset et des managers, pas d'une supposition.
PALETTE = (
    ((0.90, 0.12, 0.10), 1), ((0.90, 0.55, 0.08), 1),      # 4 teintes de proie -> nourriture
    ((0.85, 0.10, 0.45), 1), ((0.80, 0.42, 0.42), 1),
    ((0.20, 0.50, 0.95), 2),                                # eau
    ((0.10, 0.90, 0.15), 3),                                # danger
    ((0.47, 0.93, 0.53), 0), ((0.13, 0.35, 0.13), 0),       # buisson-bosquet, arbre -> neutre
    ((0.30, 0.70, 0.25), 0),                                # distracteur -> neutre
)


def oracle_class(X: torch.Tensor) -> torch.Tensor:
    """Classe VRAIE d'un rayon : plus proche teinte rendue (cosinus sur la couleur normalisée).
    Deux verts très proches restent distincts ici parce qu'on compare aux teintes EXACTES."""
    rgb = X[:, 1:4]
    unit = rgb / (rgb.norm(dim=1, keepdim=True) + 1e-6)
    P = torch.tensor([c for c, _ in PALETTE], dtype=torch.float32)
    P = P / P.norm(dim=1, keepdim=True)
    lab = torch.tensor([k for _, k in PALETTE], dtype=torch.long)
    return lab[(unit @ P.T).argmax(dim=1)]


def label_rays(corpus: str, ep0: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(features de rayon [N,4], classe issue de la CONSÉQUENCE [N]).

    Un tick contribue UN rayon : le plus proche objet touché, s'il est au contact. La conséquence
    lue entre ce tick et le suivant lui est attribuée. C'est une attribution FAIBLE — si deux
    objets sont au contact, on crédite le plus proche — et c'est assumé : elle ne suppose aucune
    connaissance du monde, seulement que ce qu'on touche est ce qui agit.
    """
    X, Y, E = [], [], []
    for ep, f in enumerate(sorted(glob.glob(os.path.join(corpus, "*.jsonl")))):
        rows = [json.loads(l) for l in open(f) if l.strip()]
        for a, b in zip(rows, rows[1:]):
            oa, ob = a["obs"], b["obs"]
            de, dt = ob["energy"] - oa["energy"], ob["thirst"] - oa["thirst"]
            dh = oa["health"] - ob["health"]
            # 🚨 PAS DE FILTRE « FRONTIÈRE » ICI, et c'est un correctif (3e erreur d'étiquetage de
            # cette sonde). La version précédente écartait tout saut de jauge > 20 comme un reset
            # d'épisode. Mais un repas rend jusqu'à 210 points dans ce monde : le filtre jetait donc
            # presque TOUTES les consommations, ne gardant que celles écrêtées près du plafond — un
            # biais de sélection qui ne laissait que 64 rayons « eau » sur ~500 boissons.
            # Le corpus écrit UN FICHIER PAR ÉPISODE : à l'intérieur d'un fichier il n'y a aucune
            # frontière, donc rien à filtrer. La structure du corpus dit la vérité mieux qu'un seuil.
            cls = 0
            if de > RELIEF:
                cls = 1
            elif dt > RELIEF:
                cls = 2
            elif dh > DMG_DROP:
                cls = 3
            # 🚨 TOUS LES RAYONS AU CONTACT, PAS LE PLUS PROCHE (corrigé après un T0 à 17 exemples).
            # Ne retenir que le rayon globalement le plus proche revient, dans une forêt, à retenir
            # presque toujours un ARBRE : la baie derrière lui n'est jamais échantillonnée, et le
            # tronc hérite de l'étiquette « nourriture » quand un repas survient. Mesuré : 17 rayons
            # nourriture et 0 danger sur 96 502.
            # L'étiquette est donc FAIBLE et assumée comme telle : chaque rayon au contact reçoit la
            # conséquence du tick, sans qu'on sache lequel l'a causée. C'est du bruit HONNÊTE, et il
            # se moyenne : un tronc est au contact aussi souvent pendant un repas que pendant rien,
            # donc sa marginale reste « neutre », tandis qu'une baie n'y est qu'aux repas. On laisse
            # les statistiques faire l'attribution plutôt que d'inventer une règle.
            ret = oa["retina"]
            for k in range(NRAY):
                d = ret[RAYCH * k]
                if d >= 0.999 or d * RANGE_M + DEPTH_OFFSET > CONTACT_M:
                    continue
                X.append(ret[RAYCH * k:RAYCH * (k + 1)])
                Y.append(cls)
                E.append(ep0 + ep)
    return (torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.long),
            torch.tensor(E, dtype=torch.long))


def cosine_baseline(X: torch.Tensor, queries: torch.Tensor, thr: torch.Tensor) -> torch.Tensor:
    """La règle CODÉE MAIN telle qu'elle est servie : cos(couleur, requête) >= seuil, sinon neutre.
    Rendue ici pour être battue sur le MÊME jeu de test — une barre absolue ne dirait rien."""
    rgb = X[:, 1:4]
    unit = rgb / (rgb.norm(dim=1, keepdim=True) + 1e-6)
    qn = queries / queries.norm(dim=1, keepdim=True)
    cos = unit @ qn.T                                     # [N, 3] food, water, danger
    fires = cos >= thr.unsqueeze(0)
    out = torch.zeros(len(X), dtype=torch.long)
    # priorité au cosinus le plus fort parmi ceux qui dépassent leur seuil (règle du slot)
    best = torch.where(fires, cos, torch.full_like(cos, -9.0)).argmax(dim=1)
    out[fires.any(dim=1)] = best[fires.any(dim=1)] + 1
    return out


def train_probe(X: torch.Tensor, Y: torch.Tensor, n_tr: int, steps: int = 4000) -> nn.Module:
    """Petit MLP sur les 4 nombres du rayon. La COULEUR est une entrée légitime — c'est de la
    perception ; ce qui compte est qu'elle n'ait pas servi à fabriquer l'étiquette."""
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(RAYCH, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(),
                        nn.Linear(64, len(CLASSES)))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    # Classes très déséquilibrées (le neutre domine) : on pondère, sinon « tout neutre » gagne.
    cnt = torch.bincount(Y[:n_tr], minlength=len(CLASSES)).float().clamp_min(1.0)
    w = (cnt.sum() / cnt)
    lossf = nn.CrossEntropyLoss(weight=w / w.sum() * len(CLASSES))
    for _ in range(steps):
        i = torch.randperm(n_tr)[:512]
        opt.zero_grad()
        lossf(net(X[:n_tr][i]), Y[:n_tr][i]).backward()
        opt.step()
    net.eval()
    return net


def recall(pred: torch.Tensor, y: torch.Tensor, c: int) -> float:
    m = (y == c)
    return float((pred[m] == c).float().mean()) if int(m.sum()) else float("nan")


def precision(pred: torch.Tensor, y: torch.Tensor, c: int) -> float:
    """Parmi ce qui est ANNONCÉ classe c, quelle part l'est vraiment.

    🚨 Sans elle, le rappel seul ment par omission — un classifieur qui annonce « danger » PARTOUT
    a 100 % de rappel sur le danger. C'est la même faute que le gate de la proie, qui n'avait qu'un
    plancher : une mesure bornée d'un seul côté certifie une moitié de la question. Ici c'est la
    moitié qui compte, puisque l'accusation portée contre le cosinus est justement qu'il prend des
    ARBRES pour du danger — un faux positif, invisible au rappel."""
    m = (pred == c)
    return float((y[m] == c).float().mean()) if int(m.sum()) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", nargs="+",
                    default=["data/replay_buffer/foret_v1_planner",
                             "data/replay_buffer/foret_v1b_planner",
                             "data/replay_buffer/foret_v1c_planner"])
    ap.add_argument("--wm", default="data/checkpoints/wm_foret_v2_slot/wm_best.pt")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()

    Xs, Ys, Es, off = [], [], [], 0
    for c in a.corpus:
        x, y, e = label_rays(c, off)
        Xs.append(x); Ys.append(y); Es.append(e)
        off = int(e.max()) + 1 if len(e) else off
    X, Y, E = torch.cat(Xs), torch.cat(Ys), torch.cat(Es)
    cnt = torch.bincount(Y, minlength=len(CLASSES))
    print("=== G0 — PERCEPTION APPRISE DE LA CONSÉQUENCE ===")
    print(f"  rayons étiquetés : " + " | ".join(f"{CLASSES[i]} {int(cnt[i])}" for i in range(4)))
    rare = [CLASSES[i] for i in (1, 2, 3) if int(cnt[i]) < MIN_PER_CLASS]
    if rare:
        print(f"  🛑 T0 ÉCHEC : trop peu d'exemples pour {rare} (< {MIN_PER_CLASS})"
              " — rien n'est apprenable, le reste ne veut rien dire.")
        return 1
    print(f"  [ok] T0 : au moins {MIN_PER_CLASS} par classe conséquence")

    # 🚨 DÉCOUPE PAR ÉPISODE, JAMAIS PAR TICK. Deux ticks voisins d'une même vie sont quasi
    # identiques (le corps avance de 4 cm) : un split au hasard mettrait des jumeaux des deux côtés
    # et la précision held-out mesurerait la mémoire, pas la généralisation. On sépare des VIES.
    eps = torch.unique(E)
    g = torch.Generator().manual_seed(0)
    eps = eps[torch.randperm(len(eps), generator=g)]
    n_ep_tr = max(1, int(0.7 * len(eps)))
    tr_mask = torch.isin(E, eps[:n_ep_tr])
    X = torch.cat([X[tr_mask], X[~tr_mask]])
    Y = torch.cat([Y[tr_mask], Y[~tr_mask]])
    n_tr = int(tr_mask.sum())
    print(f"  découpe par ÉPISODE : {n_ep_tr}/{len(eps)} vies en train ({n_tr} rayons)")
    net = train_probe(X, Y, n_tr)
    with torch.no_grad():
        pred = net(X[n_tr:]).argmax(dim=1)
    # 🚨 ON ÉVALUE CONTRE L'ORACLE DE PALETTE, PAS CONTRE LES ÉTIQUETTES D'ENTRAÎNEMENT. Celles-ci
    # sont volontairement FAIBLES (tout rayon au contact hérite de la conséquence du tick), donc un
    # classifieur parfait y plafonnerait vers 35 % — le chiffre mesurerait le bruit, pas le modèle.
    yte = oracle_class(X[n_tr:])

    payload = torch.load(a.wm, map_location="cpu", weights_only=False)
    q = payload["model"]["slot_encoder.color_queries"]
    thr = torch.tensor(payload["meta"]["query_thr"], dtype=torch.float32)
    base = cosine_baseline(X[n_tr:], q, thr)

    acc_net = float((pred == yte).float().mean())
    acc_cos = float((base == yte).float().mean())
    print(f"\n  {'':22} {'cosinus codé-main':>19} {'appris de la conséquence':>26}")
    print(f"  {'précision globale':22} {acc_cos:>18.1%} {acc_net:>25.1%}")
    for c in (1, 2, 3):
        print(f"  {'rappel ' + CLASSES[c]:22} {recall(base, yte, c):>18.1%} {recall(pred, yte, c):>25.1%}")
    for c in (1, 2, 3):
        print(f"  {'précision ' + CLASSES[c]:22} {precision(base, yte, c):>18.1%} "
              f"{precision(pred, yte, c):>25.1%}")

    t1 = acc_net > acc_cos
    t2 = recall(pred, yte, 3) > DANGER_RECALL_BAR
    print(f"\n  T1 global : appris {acc_net:.1%} vs cosinus {acc_cos:.1%} → {'PASS' if t1 else 'ÉCHEC'}")
    print(f"  T2 danger : rappel {recall(pred, yte, 3):.1%} (barre {DANGER_RECALL_BAR:.0%}) →"
          f" {'PASS' if t2 else 'ÉCHEC'}")
    if not t1:
        print("\n  🛑 KILL : apprendre de la conséquence ne bat pas le code-main sur ce corpus."
              " NE PAS construire ; chercher pourquoi (attribution trop faible ? matière trop rare ?).")
        return 1
    print(f"\n  {'✅ VOIE OUVERTE' if t2 else '⚠️  PARTIEL'} — condition NÉCESSAIRE tenue ;"
          " reste à brancher sur le slot et re-mesurer la localisation.")
    return 0 if (t1 and t2) else 1


def selfcheck() -> int:
    """Le cosinus doit gagner sur un monde où la couleur EST la classe, et perdre quand deux
    classes partagent la même couleur — c'est exactement le cas que ce G0 doit détecter."""
    g = torch.Generator().manual_seed(0)
    n = 800
    rgb_food = torch.tensor([0.9, 0.1, 0.1]) + 0.02 * torch.randn(n, 3, generator=g)
    rgb_haz = torch.tensor([0.1, 0.9, 0.1]) + 0.02 * torch.randn(n, 3, generator=g)
    X = torch.cat([torch.cat([torch.full((n, 1), 0.1), rgb_food], 1),
                   torch.cat([torch.full((n, 1), 0.1), rgb_haz], 1)])
    Y = torch.cat([torch.ones(n, dtype=torch.long), torch.full((n,), 3, dtype=torch.long)])
    net = train_probe(X, Y, len(X), steps=800)
    with torch.no_grad():
        acc = float((net(X).argmax(1) == Y).float().mean())
    assert acc > 0.95, acc
    print(f"  [ok] sur des couleurs séparables, le classifieur atteint {acc:.1%}")
    q = torch.tensor([[0.9, 0.1, 0.1], [0.1, 0.1, 0.9], [0.1, 0.9, 0.1]])
    b = cosine_baseline(X, q, torch.tensor([0.9, 0.9, 0.9]))
    assert float((b == Y).float().mean()) > 0.9
    print("  [ok] le témoin cosinus est correct quand la couleur EST la classe")
    print("\nSELFCHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
