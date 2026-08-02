"""G0 GRATUIT — le corpus de la FAIM porte-t-il assez de conséquence pour apprendre à voir ?

Jumeau de `diagnostics/diag_saliency_corpus.py` (qui a gaté la tête DANGER avant de
l'entraîner). Aucun entraînement long, aucun Godot : on lit le corpus déjà collecté et on
mesure si la recette MIL du danger peut se transposer à la faim.

    P(je consomme bientôt | rétine) = σ( b + max_k  s_faim(rgb_k) · g_faim(d_k) )

Le risque est PRÉ-INSCRIT (docs/design_perception_pure_faim.md §6) : le danger disposait de
9 372 ticks de dégâts, la faim n'a que ~534 repas sur 271 731 ticks — 17× moins de signal.
Ce diagnostic dit AVANT de payer si le signal suffit, et si non, LEQUEL des deux murs bloque :

  MUR-SIGNAL     — pas assez de repas / repas non observables au tick de consommation
  MUR-APPARENCE  — la couleur SEULE ne sépare pas la nourriture des arbres

Le second est le plus important et n'a jamais été mesuré ainsi. §5 du design a réfuté un
classifieur par rayon sur (depth, RGB) : 39,9 % des rayons d'arbres partagent ce volume.
Mais `s(rgb)` ne voit PAS la profondeur — c'est justement la propriété qui fait que ce
qu'elle apprend de près vaut à toute distance. La question « RGB SEULE sépare-t-elle ? »
est donc distincte, et c'est le PLAFOND de tout le chantier.

⚠️ La règle-couleur servie (slot_head.color_queries) est utilisée ici comme ORACLE D'ÉVAL
   UNIQUEMENT — jamais comme cible d'entraînement du chantier. `wm.food_rel0` idem.
⚠️ Découpe PAR ÉPISODE, jamais aléatoire : à 0,05 m/tick les ticks voisins sont quasi
   identiques (mesuré 2026-08-02 : 0,42 m en split aléatoire contre 2,58 m par épisode).

Usage :
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_drive_corpus.py
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_drive_corpus.py \
        --runs data/replay_buffer/foret_v1_planner --max-eps 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

N_RAY = 36
RETINA_DIM = 144
RETINA_RANGE_M = 10.0
DEPTH_OFFSET = 0.35  # slot_head.DEPTH_OFFSET : depth=surface → distance au CENTRE
TOUCH_MAX = 0.999  # d >= 0.999 = le rayon ne touche rien
MATCH_TOL_M = 0.6  # un rayon « porte » la nourriture s'il tombe à moins de ça du centre

# Palette SERVIE par le monde (sylvan.world.FORET_V1.food_type_hues) — 4 teintes rougeâtres,
# aucune verte. ORACLE D'ÉVAL uniquement : sert à contrôler la vérité géométrique, jamais à
# entraîner quoi que ce soit.
FOOD_HUES = ((0.9, 0.12, 0.1), (0.9, 0.55, 0.08), (0.85, 0.1, 0.45), (0.8, 0.42, 0.42))

DEFAULT_RUNS = [
    f"data/replay_buffer/foret_v1{s}_{k}"
    for s in ("", "b", "c")
    for k in ("planner", "babble", "explore")
]
LIVE_WM = "data/checkpoints/wm_foret_v2_slot/wm_best.pt"

# Barres pré-enregistrées, transposées de diag_saliency_corpus (danger).
BAR_ONSETS = 150  # repas observables minimum
BAR_VISIBLE = 0.90  # part des repas où un rayon porte vraiment la nourriture
BAR_NEG_NEAR = 500  # ticks proche-sans-repas (ils enseignent la portée à g)
BAR_SECTORS = 2  # secteurs angulaires distincts couverts
BAR_SEP_AUC = 0.90  # plafond apparence : RGB seule sépare-t-elle nourriture/reste ?


def load_food_rule(wm_ckpt: str) -> tuple[torch.Tensor, float]:
    """Requête-couleur SERVIE + son seuil. Oracle d'ÉVAL uniquement."""
    payload = torch.load(wm_ckpt, map_location="cpu", weights_only=False)
    meta = payload["meta"]
    q = payload["model"]["slot_encoder.color_queries"]
    idx = int(meta.get("food_idx", 0))
    thr = float(meta["query_thr"][idx])
    return q[idx].clone(), thr


def food_mask(rgb: torch.Tensor, query: torch.Tensor, thr: float) -> torch.Tensor:
    """[..., 3] -> bool. Exactement la règle de slot_head._attend (cos > seuil)."""
    n = rgb / (rgb.norm(dim=-1, keepdim=True) + 1e-6)
    return (n @ query) > thr


def ray_angles() -> torch.Tensor:
    """Angles des 36 rayons — copie EXACTE de slot_head (FOV via SYLVAN_RETINA_FOV_DEG)."""
    fov = math.radians(float(os.environ.get("SYLVAN_RETINA_FOV_DEG", "360")))
    return torch.tensor(
        [(k if k <= N_RAY // 2 else k - N_RAY) * fov / N_RAY for k in range(N_RAY)]
    )


def true_food_rays(ret: torch.Tensor, food: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """VÉRITÉ GÉOMÉTRIQUE : quels rayons tombent vraiment sur la nourriture ?

    ret  [B, 36, 4] rétine · food [B, 3] = (x, z, visible) — `wm.food_rel0`, ORACLE D'ÉVAL.
    Renvoie (label [B,36] bool, valide [B] bool).

    On ne compare PAS des couleurs : on projette chaque rayon dans le repère ego avec ses
    angles CONNUS et on regarde s'il atterrit sur l'objet. C'est la seule étiquette qui ne
    présuppose pas la règle qu'on veut justement remplacer.

    ⚠️ `food_rel0[2]` n'est PAS la visibilité du CÔNE : mesuré le 2026-08-02, 49 % des ticks
       qu'il déclare visibles ont la cible au-delà de ±60°. Il faut filtrer sur le champ,
       exactement comme `diag_bilan.load_perception_pairs`. Sans ce filtre, la moitié des
       ticks « visibles » sont hors-vue et l'étiquette géométrique est du bruit.

    VALIDATION (2026-08-02) : avec le filtre, cette projection reproduit indépendamment les
    chiffres pré-inscrits du design §4 — 40 % de ticks avec contact réel (annoncé 39 %) et
    67/45/20 % par bande de distance (annoncé 70/38/16 %). La convention est donc la bonne.
    """
    th = ray_angles()
    d = ret[..., 0]
    touch = d < TOUCH_MAX
    dist = d * RETINA_RANGE_M + DEPTH_OFFSET
    x = dist * torch.sin(th)
    z = dist * torch.cos(th)
    fx, fz = food[:, 0:1], food[:, 1:2]
    fov = float(os.environ.get("SYLVAN_RETINA_FOV_DEG", "360"))
    in_cone = torch.atan2(food[:, 0], food[:, 1]).abs() <= math.radians(fov / 2.0)
    vis = (food[:, 2] > 0.5) & in_cone
    err = torch.sqrt((x - fx) ** 2 + (z - fz) ** 2)
    lab = touch & (err < MATCH_TOL_M)
    return lab, vis


def scan(runs: list[str], max_eps: int | None, keep_neg: int) -> dict:
    """Un passage streaming sur le corpus. Ne garde en mémoire que l'utile."""
    pos_ret: list[list[float]] = []  # rétines aux ticks de repas
    neg_ret: list[list[float]] = []  # rétines sous-échantillonnées sans repas
    pos_ep: list[int] = []
    neg_ep: list[int] = []
    pos_food: list[list[float]] = []  # food_rel0 apparié — ORACLE D'ÉVAL
    neg_food: list[list[float]] = []
    n_ticks = 0
    n_meals = 0
    n_meals_noretina = 0
    ep = 0
    per_run: dict[str, tuple[int, int, int]] = {}

    for run in runs:
        d = Path(run)
        if not d.is_dir():
            continue
        files = sorted(d.glob("*.jsonl"))
        if max_eps is not None:
            files = files[:max_eps]
        r_ticks = r_meals = 0
        for f in files:
            got = False
            i = 0
            with f.open() as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    wm = rec.get("wm")
                    if wm is None:
                        continue
                    ret = wm.get("retina0") or (rec.get("obs") or {}).get("retina")
                    n_ticks += 1
                    r_ticks += 1
                    ate = float(wm.get("ate", 0.0)) > 0.5
                    if ate:
                        n_meals += 1
                        r_meals += 1
                    if ret is None or len(ret) != RETINA_DIM:
                        if ate:
                            n_meals_noretina += 1
                        i += 1
                        continue
                    got = True
                    fr = wm.get("food_rel0") or [0.0, 0.0, 0.0]
                    if ate:
                        pos_ret.append(ret)
                        pos_ep.append(ep)
                        pos_food.append(fr)
                    elif i % keep_neg == 0:
                        neg_ret.append(ret)
                        neg_ep.append(ep)
                        neg_food.append(fr)
                    i += 1
            if got:
                ep += 1
        per_run[d.name] = (len(files), r_ticks, r_meals)

    return {
        "pos": torch.tensor(pos_ret, dtype=torch.float32).view(-1, N_RAY, 4),
        "neg": torch.tensor(neg_ret, dtype=torch.float32).view(-1, N_RAY, 4),
        "pos_ep": torch.tensor(pos_ep, dtype=torch.long),
        "neg_ep": torch.tensor(neg_ep, dtype=torch.long),
        "pos_food": torch.tensor(pos_food, dtype=torch.float32).view(-1, 3),
        "neg_food": torch.tensor(neg_food, dtype=torch.float32).view(-1, 3),
        "n_ticks": n_ticks,
        "n_meals": n_meals,
        "n_meals_noretina": n_meals_noretina,
        "n_ep": ep,
        "per_run": per_run,
    }


def _auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Mann-Whitney. Renvoie nan si une classe est vide."""
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    allv = torch.cat([pos, neg])
    rank = allv.argsort().argsort().float() + 1.0
    rp = rank[: pos.numel()].sum()
    n1, n2 = float(pos.numel()), float(neg.numel())
    return float((rp - n1 * (n1 + 1) / 2) / (n1 * n2))


def section_signal(data: dict) -> dict:
    print("=" * 78)
    print("1. SIGNAL — combien de conséquence le corpus rend-il ?")
    print("=" * 78)
    print(f"  {data['n_ep']} épisodes exploitables, {data['n_ticks']} ticks")
    for name, (nf, nt, nm) in data["per_run"].items():
        print(f"      {name:28s} {nf:3d} fich.  {nt:7d} ticks  {nm:4d} repas")
    n_obs = data["pos"].shape[0]
    print(f"\n  repas TOTAUX          : {data['n_meals']}")
    print(f"  repas AVEC rétine     : {n_obs}   (sans rétine : {data['n_meals_noretina']})")
    rate = n_obs / max(1, data["n_ticks"])
    print(f"  taux de positifs      : {rate * 100:.3f} %  (le danger avait 9 372 positifs)")
    ok = n_obs >= BAR_ONSETS
    print(f"  {'✅' if ok else '❌'} G0-sig : {n_obs} repas observables (barre {BAR_ONSETS})")
    return {"n_obs": n_obs, "pass": ok}


def section_visibilite(data: dict, query: torch.Tensor, thr: float) -> dict:
    print()
    print("=" * 78)
    print("2. VISIBILITÉ AU REPAS — la nourriture est-elle DANS la rétine quand on mange ?")
    print("=" * 78)
    pos = data["pos"]
    if pos.shape[0] == 0:
        print("  (aucun repas observable)")
        return {"pass": False, "vis": 0.0, "rho": float("nan")}
    d = pos[..., 0]
    touch = d < TOUCH_MAX
    # VÉRITÉ GÉOMÉTRIQUE, pas la règle-couleur : demander à la règle si elle voit la
    # nourriture serait circulaire (c'est elle qu'on veut remplacer).
    is_food, _ = true_food_rays(pos, data["pos_food"])
    has = is_food.any(dim=-1)
    vis = float(has.float().mean())
    print(f"  repas où un rayon porte vraiment la nourriture : {vis * 100:.1f} %  "
          f"(barre {BAR_VISIBLE * 100:.0f} %)")

    # ρ̂ candidat : à quelle distance est le rayon-nourriture le plus proche au repas ?
    dm = torch.where(is_food, d, torch.ones_like(d)) * RETINA_RANGE_M
    dmin = dm.min(dim=-1).values[has]
    if dmin.numel():
        q = torch.quantile(dmin, torch.tensor([0.5, 0.95]))
        print(f"  distance au repas     : méd={float(q[0]):.2f} m   q95={float(q[1]):.2f} m")
        print(f"      → ρ̂ attendu ≈ {float(q[0]):.2f} m (eat_radius = 1.0 m)")
        rho = float(q[0])
    else:
        rho = float("nan")

    # Couverture angulaire : la conséquence doit venir de plusieurs directions.
    ang = torch.arange(N_RAY, dtype=torch.float32) * (2 * math.pi / N_RAY)
    sect = ((ang / (math.pi / 2)).long() % 4)
    hit = torch.zeros(4, dtype=torch.bool)
    for s in range(4):
        hit[s] = bool(is_food[:, sect == s].any())
    n_sect = int(hit.sum())
    print(f"  secteurs angulaires   : {n_sect}/4 couverts (barre {BAR_SECTORS})")

    ok = vis >= BAR_VISIBLE and n_sect >= BAR_SECTORS
    print(f"  {'✅' if ok else '❌'} G0-vis")
    return {"pass": ok, "vis": vis, "rho": rho, "sectors": n_sect}


def section_negatifs(data: dict, query: torch.Tensor, thr: float) -> dict:
    print()
    print("=" * 78)
    print("3. NÉGATIFS PROCHES — y a-t-il de quoi apprendre la FRONTIÈRE ?")
    print("=" * 78)
    neg = data["neg"]
    if neg.shape[0] == 0:
        print("  (aucun négatif)")
        return {"pass": False}
    d, rgb = neg[..., 0], neg[..., 1:]
    touch = d < TOUCH_MAX
    near = touch & (d * RETINA_RANGE_M < 2.0)
    n_near = int(near.any(dim=-1).sum())
    is_food = food_mask(rgb, query, thr) & touch
    near_food = int((near & is_food).any(dim=-1).sum())
    near_other = int((near & ~is_food).any(dim=-1).sum())
    print(f"  ticks sans repas avec un objet à moins de 2 m : {n_near}  (barre {BAR_NEG_NEAR})")
    print(f"      dont un rayon NOURRITURE proche : {near_food}   (→ enseignent la portée à g)")
    print(f"      dont un rayon AUTRE proche      : {near_other}   (→ enseignent l'apparence à s)")
    ok = n_near >= BAR_NEG_NEAR and near_other >= 100
    print(f"  {'✅' if ok else '❌'} G0-neg")
    return {"pass": ok, "n_near": n_near}


def section_apparence(data: dict, query: torch.Tensor, thr: float, seed: int) -> dict:
    """LE test décisif : la couleur SEULE sépare-t-elle la nourriture du reste ?

    Plafond du chantier, mesuré contre la VÉRITÉ GÉOMÉTRIQUE (`food_rel0`), pas contre la
    règle-couleur — sinon on demanderait à un MLP sur RGB de reproduire une fonction de RGB,
    ce qui est vrai par construction et ne mesure rien.

    Deux chiffres, la même vérité :
      (a) le TENANT — ce que la règle-couleur codée-main obtient. C'est la barre à ne pas
          casser (doctrine 2026-07-22 : on accepte un peu moins bon, jamais nuisible).
      (b) le PLAFOND — un s(rgb) (MLP 3→16→1, la forme exacte de la recette) entraîné avec
          des étiquettes PARFAITES. Si ce plafond est bas, aucune étiquette de conséquence
          ne peut faire mieux : le mur serait l'APPARENCE, pas le signal.
    """
    print()
    print("=" * 78)
    print("4. PLAFOND D'APPARENCE — mesuré contre la vérité géométrique (food_rel0)")
    print("=" * 78)
    ret = torch.cat([data["pos"], data["neg"]], dim=0)
    eps = torch.cat([data["pos_ep"], data["neg_ep"]], dim=0)
    food = torch.cat([data["pos_food"], data["neg_food"]], dim=0)
    lab, vis = true_food_rays(ret, food)
    d, rgb = ret[..., 0], ret[..., 1:]
    touch = d < TOUCH_MAX

    # On ne juge que les ticks où l'oracle déclare la nourriture visible ET où au moins un
    # rayon la porte vraiment : ailleurs, « rayon-nourriture » n'est pas défini.
    usable = vis & lab.any(dim=-1)
    print(f"  ticks retenus : {int(usable.sum())} / {len(usable)} "
          f"(oracle visible ET au moins un rayon dessus)")
    if int(usable.sum()) < 50:
        print("  ❌ trop peu de ticks exploitables")
        return {"pass": False}

    ret, eps, lab, touch, rgb = ret[usable], eps[usable], lab[usable], touch[usable], rgb[usable]

    # (a) le TENANT : la règle-couleur servie, sur la même vérité.
    rule = food_mask(rgb, query, thr) & touch
    r_rec = float(rule[lab].float().mean())
    r_fpr = float(rule[touch & ~lab].float().mean())
    print(f"\n  (a) TENANT — règle-couleur codée-main :")
    print(f"      rappel nourriture {r_rec * 100:.1f} %   faux-flags autres {r_fpr * 100:.1f} %")

    # ⚠️ CONTRÔLE DE LA VÉRITÉ ELLE-MÊME, contre la palette SERVIE par le monde
    # (`sylvan.world.FORET_V1.food_type_hues` — 4 teintes rougeâtres). Elle est utilisée en
    # ORACLE D'ÉVAL, jamais à l'entraînement. `food_rel0` ne suit qu'UNE cible dans un monde
    # qui en sert plusieurs, et la forêt occulte : l'étiquette géométrique est donc faussée
    # DANS LES DEUX SENS. Sans ce contrôle on conclurait « la couleur ne sépare pas » alors
    # qu'on mesure sa propre étiquette — le piège n°2 du design §5.
    palette = torch.tensor(FOOD_HUES)
    palette = palette / palette.norm(dim=-1, keepdim=True)
    rgbn = rgb / (rgb.norm(dim=-1, keepdim=True) + 1e-6)
    is_hue = (rgbn @ palette.T).amax(dim=-1) > 0.98
    neg_contam = float(is_hue[touch & ~lab].float().mean())  # autres fruits pris pour du décor
    pos_contam = float((~is_hue)[lab & touch].float().mean())  # occulteurs pris pour du fruit
    print(f"      NÉGATIFS à couleur de fruit (= un AUTRE fruit) : {neg_contam * 100:.1f} %")
    print(f"      POSITIFS sans couleur de fruit (= occultés)    : {pos_contam * 100:.1f} %")
    contamination = max(neg_contam, pos_contam)
    if contamination > 0.15:
        print("      ⚠️  l'étiquette géométrique est faussée dans les deux sens : (a) et (b)")
        print("          sont des PLANCHERS, pas des verdicts. Trancher de bout en bout.")

    # (b) le PLAFOND. Découpe PAR ÉPISODE (jamais aléatoire).
    uniq = eps.unique()
    if len(uniq) < 4:
        print("  ❌ trop peu d'épisodes pour une découpe honnête")
        return {"pass": False}
    cut = uniq[int(0.8 * len(uniq))]
    tr_t, te_t = eps < cut, eps >= cut

    def flat(m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sel = touch & m.unsqueeze(-1)
        return rgb[sel], lab[sel]

    xtr, ytr = flat(tr_t)
    xte, yte = flat(te_t)
    print(f"\n  (b) PLAFOND — s(rgb) avec étiquettes PARFAITES :")
    print(f"      rayons touchants train {len(xtr)}  test {len(xte)}")
    if len(xte) == 0 or int(yte.sum()) == 0 or len(xtr) == 0 or int(ytr.sum()) == 0:
        print("      ❌ une classe est vide dans un des plis")
        return {"pass": False}
    print(f"      part nourriture train {float(ytr.float().mean()) * 100:.1f} %  "
          f"test {float(yte.float().mean()) * 100:.1f} %")

    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(3, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), 1e-2)
    yf = ytr.float()
    for _ in range(1500):
        bi = torch.randint(0, len(xtr), (4096,))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            net(xtr[bi]).squeeze(-1), yf[bi]
        )
        loss.backward()
        opt.step()
        opt.zero_grad()
    with torch.no_grad():
        s = torch.sigmoid(net(xte).squeeze(-1))
    auc = _auc(s[yte], s[~yte])
    rec = float((s[yte] > 0.5).float().mean())
    fpr = float((s[~yte] > 0.5).float().mean())
    print(f"      AUC {auc:.3f}   rappel {rec * 100:.1f} %   faux-flags {fpr * 100:.1f} %  (seuil 0,5)")
    ok = auc >= BAR_SEP_AUC
    print(f"  {'✅' if ok else '❌'} G0-sep (barre AUC {BAR_SEP_AUC})")
    if ok and r_rec > 0 and rec < r_rec - 0.10:
        print("      ⚠️  le plafond appris reste sous le tenant au seuil 0,5 — calibration à revoir")
    return {"pass": ok, "auc": auc, "recall": rec, "fpr": fpr,
            "rule_recall": r_rec, "rule_fpr": r_fpr, "contamination": contamination}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--wm", default=LIVE_WM)
    ap.add_argument("--max-eps", type=int, default=None)
    ap.add_argument("--keep-neg", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    query, thr = load_food_rule(args.wm)
    print(f"règle-couleur SERVIE (oracle d'ÉVAL) : requête={tuple(round(float(v), 3) for v in query)}"
          f"  seuil={thr:.3f}\n")

    data = scan(args.runs, args.max_eps, args.keep_neg)
    r_sig = section_signal(data)
    r_vis = section_visibilite(data, query, thr)
    r_neg = section_negatifs(data, query, thr)
    r_sep = section_apparence(data, query, thr, args.seed)

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    contam = r_sep.get("contamination", float("nan"))
    contaminated = contam == contam and contam > 0.15
    if not r_sep.get("pass") and contaminated:
        print("  ⚠️  G0-sep NON CONCLUANT — c'est l'ÉTIQUETTE qui est fausse, pas la couleur.")
        print(f"     Contamination mesurée {contam * 100:.0f} %, dans les DEUX sens : des rayons")
        print("     comptés « non-nourriture » tombent sur un AUTRE fruit (food_rel0 ne suit")
        print("     qu'une cible), et des rayons comptés « nourriture » tombent sur l'arbre qui")
        print("     l'occulte. Le per-rayon mesuré est un PLANCHER, pas un verdict : conclure")
        print("     « la couleur ne sépare pas » ici serait juger sa propre étiquette — le")
        print("     piège n°2 du design §5.")
        print("     ⇒ ENTRAÎNEMENT LICENCIÉ, mais on tranche sur le GISEMENT de bout en bout")
        print("       (G-gis, vérité non ambiguë = la position de la cible), pas sur ce chiffre.")
    elif not r_sep.get("pass"):
        print("  🛑 MUR-APPARENCE : la couleur SEULE ne sépare pas la nourriture du reste,")
        print("     même avec des étiquettes PARFAITES. Aucune étiquette de conséquence ne")
        print("     peut faire mieux — la substitution s_faim(rgb) > 0.5 est impossible ICI.")
        print("     Ce n'est pas « l'appris ne marche pas » : c'est ce MONDE dont les teintes")
        print("     de nourriture ne sont pas séparables en RGB des troncs et du feuillage.")
    elif not (r_sig.get("pass") and r_vis.get("pass")):
        print("  🛑 MUR-SIGNAL : l'apparence EST séparable, mais le corpus ne rend pas assez")
        print("     de conséquence pour l'apprendre. Le levier est le MONDE (plus de repas,")
        print("     ou un drive qui les rend plus fréquents), pas la forme du modèle.")
    elif not r_neg.get("pass"):
        print("  ⚠️  Signal et apparence OK, mais peu de négatifs proches : g (la portée)")
        print("     sera mal contrainte. Entraînable, à surveiller sur ρ̂.")
    else:
        print("  ✅ LICENCIÉ : le corpus porte assez de conséquence ET l'apparence est")
        print("     séparable en RGB seule. La recette MIL du danger est transposable.")
    print(f"\n  G0-sig {'✅' if r_sig.get('pass') else '❌'} · "
          f"G0-vis {'✅' if r_vis.get('pass') else '❌'} · "
          f"G0-neg {'✅' if r_neg.get('pass') else '❌'} · "
          f"G0-sep {'✅' if r_sep.get('pass') else '❌'}")


if __name__ == "__main__":
    main()
