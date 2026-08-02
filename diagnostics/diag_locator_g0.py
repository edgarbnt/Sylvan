"""G0 — une perception PURE de la position peut-elle battre le cosinus codé-main ?

LA QUESTION. Le slot servi lit la rétine rayon par rayon avec des requêtes-couleur écrites en dur
(« rouge = nourriture »). C'est la dernière clé-apparence structurelle du projet : ajouter une
ressource d'une nouvelle couleur demande de toucher au code, pas de vivre. On veut la remplacer par
une tête qui lit le LATENT et n'apprend QUE de ce que l'entité vit.

CE QUI REND CE GATE POSSIBLE MAINTENANT (mesuré le 2026-08-02) :
  * le latent PORTE la position — sonde supervisée 0,45 m près, 0,89 m au-delà de 6 m, là où le
    slot rend 0,60 m et 4,10 m. L'information est là, c'est le readout qui manque.
  * la proximité PRÉDIT le repas à 95 % — le lien perception→conséquence est quasi déterministe.
  * la distance au moment du repas est 1,08 m, très serrée (q25 1,06 / q75 1,11) = `eat_radius`,
    une constante du CORPS. Le gisement, lui, est étalé (médiane 27°) : la conséquence donne la
    DISTANCE, pas la DIRECTION.

LES TROIS SIGNAUX, TOUS PURS. Aucun n'emploie de couleur ni `food_rel0` :

  1. TRANSPORT  ‖transport(p_t, ego) − sg(p_{t+1})‖²  — dense (chaque tick).
     Dit « ta sortie est une POSITION dans l'espace ego », par équivariance à l'ego-motion.
     L'ego-motion vient de la proprioception : le corps sait comment il a bougé.
     ⚠️ Ce terme SEUL est dégénéré : son optimum est de suivre un TRONC (statique, résidu nul),
     alors que la proie bouge (résidu = prey_speed × gap, mesuré exactement). C'est pourquoi il
     ne peut pas être le seul — le négatif est banké dans le commit f85ced1.

  2. PORTÉE     (‖p_t‖ − reach)²  aux ticks suivis d'une consommation — épars (~0,2 %).
     Dit « la chose que tu suis, c'est CELLE qui m'a nourri ». `reach` = `eat_radius`, connu du
     corps (c'est sa bouche), pas du monde.

  3. HORS-PORTÉE  relu(reach − ‖p_t‖)²  aux ticks SANS consommation à venir — dense.
     Dit « si tu avais été si près, j'aurais mangé ». C'est ce terme qui casse la dégénérescence
     du transport : un tronc qu'on frôle ne nourrit pas.

Le gisement n'est ancré par AUCUN des trois directement — c'est la translation qui le pinne :
un biais de gisement θ produit un écart de transport (Rot(θ)−I)·Rot(−Δyaw)·t, non nul dès que
l'entité avance. Calculé : 20° → 0,017 m/tick, 45° → 0,038 m/tick. Les gros biais sont corrigés,
un flou résiduel d'environ 20° est attendu (soit ~1 m à 3 m).

CRITÈRES PRÉ-ENREGISTRÉS (posés AVANT de lancer) :
  ✅ PASS  erreur médiane < 1,00 m — le rayon de bouche, donc le seuil où viser sert à quelque chose
  🎯 CIBLE erreur au-delà de 2 m < 1,50 m — c'est LÀ que le slot s'effondre (1,50 → 4,10 m) et
           donc là que se joue le comportement (nourriture à 3,16 m médians)
  🛑 KILL  erreur médiane > 1,43 m — n'apporte rien sur le slot servi, la voie est réfutée

═══════════════════════════════════════════════════════════════════════════════════════════
RÉSULTAT : 🛑 KILL SUR LES DEUX MODES (2026-08-02). Les deux échouent, mais PAS pour la même
raison — et c'est la conjonction des deux qui ferme la voie.

  mode         médiane   < 2 m    > 2 m
  implicite     3,43 m   1,05 m   4,00 m
  retro         3,47 m   1,13 m   4,01 m
  slot servi    1,43 m   0,60 m   2,30 m
  sonde SUPERVISÉE (référence de ce que le latent CONTIENT)   0,58 m / 0,45 m / 0,68 m

MODE `implicite` — l'échelle n'est ancrée nulle part. La tête satisfait les trois pertes SANS
RIEN SUIVRE : norme 1,06 m aux ticks de repas (cible 1,0 ✅), 1,56 m ailleurs (vérité 3,72 ✗),
corrélation avec la vérité r = −0,09 / +0,06. Elle a appris « c'est toujours à ~1,5 m dans une
direction quelconque ». Le calcul l'explique : une erreur d'échelle α produit un résidu de
transport |α−1|×0,05 m/tick, alors que le plancher de bruit de la proie mobile est
0,023 m/tick — pour l'α mesuré (0,42) le SNR vaut 1,26, insuffisant. La conséquence, elle,
n'ancre l'échelle qu'en UN point : à la bouche.

MODE `retro` — les étiquettes sont bonnes mais ne couvrent QUE le court terme. Reconstruire par
ego-motion intégrée donne 0,67 m à k=1, 0,87 m à k=10, 1,09 m à k=15 (5 030 cibles). Mais 15
ticks de remontée à 0,05 m/tick ne couvrent que 0,75 m de trajet : on n'étiquette JAMAIS au-delà
de ~2 m. La tête tient son régime d'entraînement (1,13 m sous 2 m) et s'effondre ailleurs
(4,01 m), faute d'un seul exemple.

⇒ LIMITE STRUCTURELLE, pas un défaut de réglage : **l'entité ne peut étiqueter par la conséquence
que ce qu'elle a ATTEINT.** La nourriture lointaine qu'elle n'a jamais rejointe ne laisse aucune
trace exploitable. Et c'est précisément le régime (> 2 m) où le slot codé-main échoue et où le
comportement se joue.

⇒ LA CAUSE EST ISOLÉE, ET C'EST UN RÉGLAGE DU MONDE. En rétro-propageant depuis une ancre
PARFAITE (la vraie position au repas), on sépare l'erreur d'ANCRE de la dérive de PROIE :

  k        ancre corps   ancre parfaite   prédiction proie (0,046·k)
  10          0,88 m         0,46 m              0,46 m
  20          1,24 m         0,92 m              0,92 m
  40          2,11 m         1,86 m              1,84 m
  60          3,11 m         2,81 m              2,76 m

La colonne « ancre parfaite » suit EXACTEMENT le déplacement relatif de la proie, à trois
décimales sur quatre profondeurs. L'erreur d'ancre, elle, reste CONSTANTE (~0,3 m) : elle ne
limite pas la profondeur.

⇒ Avec une proie IMMOBILE, la rétro-propagation rendrait des étiquettes à ~0,3 m jusqu'à k=60,
soit 3 m de trajet — donc une couverture jusqu'à ~4 m, exactement le régime où le slot
codé-main échoue. **Ce n'est pas l'apprentissage qui manque, c'est que le monde efface sa
propre trace.** `prey_speed` est d'ailleurs déjà noté dans `world.py` comme « UNE SONDE, pas le
correctif final », et l'audit a montré que le planner fait de la poursuite pure et ne peut de
toute façon pas intercepter. Le gate à poser ensuite est un A/B de monde, pas un entraînement.
═══════════════════════════════════════════════════════════════════════════════════════════

L'ÉVALUATION seule lit `food_rel0` (oracle de MESURE, jamais d'entraînement) — comme
`diag_slot_localise`. Une perception qu'on ne confronte jamais au monde est une perception qu'on
croit sur parole.

CLI :
    PYTHONPATH=python SYLVAN_RETINA_FOV_DEG=120 env_pytorch_3.12/bin/python \
        diagnostics/diag_locator_g0.py
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import warnings

import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from sylvan.models.command_wm import CommandWorldModel  # noqa: E402

PASS_BAR_M = 1.00        # rayon de bouche
RANGE_BAR_M = 1.50       # cible au-delà de 2 m
KILL_BAR_M = 1.43        # le slot servi


# ── géométrie du transport : la MÊME convention que command_wm.transport_slot ────────────────

def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def ego_step(t0: list[float], t1: list[float]) -> tuple[float, float, float]:
    """(d_fwd, d_lat, Δyaw) entre deux poses — de la PROPRIOCEPTION, pas du monde.

    ⚠️ ORDRE IMPOSÉ par `CommandWorldModel.transport_slot`, qui lit disp_real[0]=d_fwd,
    [1]=d_lat, [2]=d_yaw. On rend le tuple dans CET ordre pour pouvoir le lui passer tel quel."""
    x0, z0, y0 = t0
    x1, z1, y1 = t1
    dx, dz = x1 - x0, z1 - z0
    return (dx * math.sin(y0) + dz * math.cos(y0),      # d_fwd
            dx * math.cos(y0) - dz * math.sin(y0),      # d_lat
            wrap(y1 - y0))                              # d_yaw


def transport(p: torch.Tensor, ego: torch.Tensor, wm: CommandWorldModel) -> torch.Tensor:
    """p [N,2] transporté d'un pas par l'ego-motion [N,3] = (d_fwd, d_lat, d_yaw).

    ⚠️ ON APPELLE LE TRANSPORT DE PRODUCTION, on ne le ré-implémente pas. La première version de
    ce diagnostic le recodait avec Rot(−Δyaw) et d_lat soustrait, ce qui a rendu un KILL FAUX :
    balayage des 8 conventions de signe sur 21 869 transitions réelles, la convention servie
    (`slot_calib`, Rot(+Δyaw) et d_lat ajouté) rend un résidu de 0,046 m — le plancher de la proie
    mobile — et se classe 1re sur 8, tandis que celle du diagnostic rendait 0,241 m (5e sur 8).
    Un diagnostic qui recode la géométrie du modèle qu'il juge finit par juger sa propre erreur."""
    return wm.transport_slot(p, ego)


def transport_inv(q: tuple[float, float], e: tuple[float, float, float]) -> tuple[float, float]:
    """Inverse EXACT du transport de production (aller-retour vérifié à 2e-15).

    production : p_next = Rot(+d_yaw)·(p_x + d_lat, p_z − d_fwd)
    inverse    : p = Rot(−d_yaw)·p_next puis on défait la translation."""
    dfwd, dlat, dyaw = e
    c, s = math.cos(-dyaw), math.sin(-dyaw)
    qx, qz = c * q[0] - s * q[1], s * q[0] + c * q[1]
    return (qx - dlat, qz + dfwd)


# ── la tête ─────────────────────────────────────────────────────────────────────────────────

class DriveLocator(nn.Module):
    """latent → position ego de ce qui satisfait UNE pulsion.

    Une tête par pulsion : ajouter la soif ou le danger = instancier une seconde tête sur le MÊME
    latent, avec le signal de conséquence de CETTE pulsion. Rien d'autre ne bouge — c'est la
    modularité que les requêtes-couleur codées-main empêchaient."""

    def __init__(self, latent_dim: int = 128, hidden: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 2),
        )
        self.register_buffer("mu", torch.zeros(latent_dim))
        self.register_buffer("sd", torch.ones(latent_dim))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net((latent - self.mu) / self.sd)


# ── corpus ──────────────────────────────────────────────────────────────────────────────────

def load(corpora: list[str], wm: CommandWorldModel, horizon_no_eat: int = 30,
         reach: float = 1.0, retro_k: int = 15):
    """Latents + signaux PURS + vérité d'évaluation, découpés par épisode.

    Rend, par tick t (et t+1 valide dans le MÊME épisode) :
      latent[t], ego[t→t+1], eat_next[t] (consommation à t+1), no_eat[t] (aucune dans K ticks),
      truth[t] (food_rel0 — ÉVALUATION SEULEMENT), episode_id[t]
    """
    lat_l, ego_l, eat_l, noeat_l, truth_l, epi_l, retro_l = [], [], [], [], [], [], []
    eid = 0
    for corpus in corpora:
        for f in sorted(glob.glob(os.path.join(corpus, "episode_*.jsonl"))):
            rows = [json.loads(l) for l in open(f) if l.strip()]
            if len(rows) < 3:
                continue
            ate = [r.get("wm", {}).get("ate", 0) > 0.5 for r in rows]
            obs_ep, ego_ep, keep = [], [], []
            for i in range(len(rows) - 1):
                w0, w1 = rows[i].get("wm", {}), rows[i + 1].get("wm", {})
                t0, t1 = w0.get("torso0"), w1.get("torso0")
                ret = w0.get("retina0")
                if not (t0 and t1 and ret and len(ret) == 144):
                    continue
                d = math.hypot(t1[0] - t0[0], t1[1] - t0[1])
                if d > 1.0:                       # respawn : pas un déplacement
                    continue
                o = rows[i]["obs"]
                obs_ep.append(o["proprio"] + ret + [o["energy"] / 100.0])
                ego_ep.append(list(ego_step(t0, t1)))
                keep.append(i)
            if len(obs_ep) < 8:
                continue
            with torch.no_grad():
                O = torch.tensor(obs_ep, dtype=torch.float32)
                lat = torch.cat([wm.encoder(O[j:j + 2048]) for j in range(0, len(O), 2048)])

            # ── CIBLES RÉTRO-PROPAGÉES (mode `retro`) ────────────────────────────────────────
            # « La nourriture était à ma bouche quand j'ai mangé » + l'ego-motion INTÉGRÉE
            # reconstruit où elle était avant. Trois ingrédients, tous purs : l'événement vécu,
            # le rayon de bouche (constante du CORPS) et la proprioception. Zéro couleur.
            # Qualité mesurée contre food_rel0 : 0,67 m à k=1, 0,87 m à k=10, 1,09 m à k=15.
            slot_of_row = {i: j for j, i in enumerate(keep)}
            retro = [None] * len(keep)
            for t in range(len(rows)):
                if not ate[t] or t - 1 < 0:
                    continue
                p = (0.0, reach)                       # ancre : devant, à portée de bouche
                r = t - 1
                while r >= 0 and (t - 1 - r) <= retro_k:
                    j = slot_of_row.get(r)
                    if j is None:
                        break
                    if retro[j] is None or (t - 1 - r) < retro[j][1]:
                        retro[j] = (p, t - 1 - r)      # on garde le repas le PLUS PROCHE
                    p = transport_inv(p, ego_ep[j])
                    r -= 1

            for j, i in enumerate(keep):
                v = rows[i].get("wm", {}).get("food_rel0")
                lat_l.append(lat[j])
                ego_l.append(ego_ep[j])
                eat_l.append(ate[i + 1])
                noeat_l.append(not any(ate[i + 1:i + 1 + horizon_no_eat]))
                truth_l.append([v[0], v[1]] if (v and len(v) >= 3 and v[2] > 0.5) else [float("nan")] * 2)
                epi_l.append(eid)
                retro_l.append(list(retro[j][0]) if retro[j] else [float("nan")] * 2)
            eid += 1
    return (torch.stack(lat_l),
            torch.tensor(ego_l, dtype=torch.float32),
            torch.tensor(eat_l),
            torch.tensor(noeat_l),
            torch.tensor(truth_l, dtype=torch.float32),
            torch.tensor(epi_l),
            torch.tensor(retro_l, dtype=torch.float32))


# ── entraînement ────────────────────────────────────────────────────────────────────────────

def train(head: DriveLocator, lat, ego, eat, noeat, tr_mask, wm, *,
          reach: float, steps: int, lr: float, w_reach: float, w_far: float) -> None:
    """Aucun des trois termes n'emploie de couleur ni `food_rel0`."""
    # Les paires (t, t+1) doivent rester dans le TRAIN et être contiguës.
    idx = torch.nonzero(tr_mask[:-1] & tr_mask[1:]).squeeze(-1)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    n = len(idx)
    bs = min(4096, n)
    for step in range(steps):
        b = idx[torch.randint(0, n, (bs,))]
        p = head(lat[b])
        p_next = head(lat[b + 1]).detach()                     # stop-grad : cible, pas prédiction
        # 1. TRANSPORT — équivariance à l'ego-motion (« c'est une position »)
        l_tr = ((transport(p, ego[b], wm) - p_next) ** 2).sum(1).mean()
        norm = p.norm(dim=1)
        # 2. PORTÉE — « ce que tu suis est ce qui m'a nourri »
        m_eat = eat[b]
        l_reach = (((norm - reach) ** 2)[m_eat]).mean() if int(m_eat.sum()) > 0 else norm.sum() * 0
        # 3. HORS-PORTÉE — « si tu étais si près, j'aurais mangé »
        m_far = noeat[b]
        l_far = ((torch.relu(reach - norm) ** 2)[m_far]).mean() if int(m_far.sum()) > 0 else norm.sum() * 0
        opt.zero_grad()
        (l_tr + w_reach * l_reach + w_far * l_far).backward()
        opt.step()


def train_retro(head: DriveLocator, lat, retro, tr_mask, *, steps: int, lr: float) -> int:
    """Régression sur les cibles RÉTRO-PROPAGÉES — même information que le mode `implicite`,
    mais rendue EXPLICITE au lieu d'espérer que le terme de transport la propage.

    C'est ce qui change tout : l'intégration de l'ego-motion est EXACTE (arithmétique, pas
    apprise), donc l'échelle est préservée sur les ~15 ticks de remontée. Le mode `implicite`
    échouait précisément là — son signal d'échelle (|α−1|×0,05 m/tick) passait sous le plancher
    de bruit de la proie (0,023 m/tick)."""
    ok = tr_mask & ~torch.isnan(retro[:, 0])
    idx = torch.nonzero(ok).squeeze(-1)
    if len(idx) < 100:
        return len(idx)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    bs = min(1024, len(idx))
    for _ in range(steps):
        b = idx[torch.randint(0, len(idx), (bs,))]
        opt.zero_grad()
        ((head(lat[b]) - retro[b]) ** 2).sum(1).mean().backward()
        opt.step()
    return len(idx)


@torch.no_grad()
def evaluate(head: DriveLocator, lat, truth, mask) -> dict:
    """ÉVALUATION SEULEMENT : `food_rel0` sert d'oracle de mesure, jamais d'entraînement."""
    ok = mask & ~torch.isnan(truth[:, 0])
    p = head(lat[ok])
    err = (p - truth[ok]).norm(dim=1)
    d = truth[ok].norm(dim=1)
    out = {"n": int(ok.sum()), "med": float(err.median()),
           "lt1": float((err < 1.0).float().mean())}
    for lo, hi, k in [(0, 2, "proche"), (2, 99, "loin")]:
        m = (d >= lo) & (d < hi)
        out[k] = float(err[m].median()) if int(m.sum()) > 10 else float("nan")
        out[k + "_n"] = int(m.sum())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wm", default="data/checkpoints/wm_foret_v2_slot/wm_best.pt")
    ap.add_argument("--corpus", nargs="+", default=[
        "data/replay_buffer/bootstrap_poshead_multi",
        "data/replay_buffer/foret_v1_planner",
        "data/replay_buffer/foret_v1b_planner",
        "data/replay_buffer/foret_v1c_planner"])
    ap.add_argument("--reach", type=float, default=1.0, help="eat_radius — constante du CORPS")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w-reach", type=float, default=1.0)
    ap.add_argument("--w-far", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["implicite", "retro"], default="retro",
                    help="implicite = transport + portée (RÉFUTÉ, 3,43 m) | "
                         "retro = régression sur les cibles rétro-propagées")
    ap.add_argument("--retro-k", type=int, default=15,
                    help="profondeur de remontée (15 → étiquettes à ~1,09 m)")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    print("=== G0 LOCALISATEUR PUR — la conséquence bat-elle le cosinus codé-main ? ===")
    print(f"    mode={a.mode}  reach={a.reach} m (constante du corps)  "
          f"steps={a.steps}  seed={a.seed}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wm = CommandWorldModel.from_checkpoint(
            torch.load(a.wm, map_location="cpu", weights_only=False))
    wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)

    print(f"\n1. Corpus : {len(a.corpus)} sources")
    lat, ego, eat, noeat, truth, epi, retro = load(a.corpus, wm, reach=a.reach, retro_k=a.retro_k)
    n_ep = int(epi.max()) + 1
    n_retro = int((~torch.isnan(retro[:, 0])).sum())
    print(f"   {len(lat)} ticks · {n_ep} épisodes · {int(eat.sum())} consommations "
          f"({100 * float(eat.float().mean()):.2f} %)")
    print(f"   cibles rétro-propagées : {n_retro} ({100 * n_retro / len(lat):.1f} % des ticks)")

    # Split PAR ÉPISODE — deux ticks du même épisode ne peuvent pas tomber de part et d'autre.
    cut = int(0.75 * n_ep)
    tr_mask, te_mask = epi < cut, epi >= cut
    print(f"   train {int(tr_mask.sum())} ticks ({cut} ép.) · "
          f"test {int(te_mask.sum())} ticks ({n_ep - cut} ép.)")

    head = DriveLocator(latent_dim=lat.shape[1])
    head.mu.copy_(lat[tr_mask].mean(0))
    head.sd.copy_(lat[tr_mask].std(0).clamp(min=1e-2))

    if a.mode == "implicite":
        print(f"\n2. Entraînement IMPLICITE — transport + portée + hors-portée "
              f"(zéro couleur, zéro food_rel0)")
        train(head, lat, ego, eat, noeat, tr_mask, wm,
              reach=a.reach, steps=a.steps, lr=a.lr, w_reach=a.w_reach, w_far=a.w_far)
    else:
        n = train_retro(head, lat, retro, tr_mask, steps=a.steps, lr=a.lr)
        print(f"\n2. Entraînement RÉTRO — régression sur {n} cibles reconstruites "
              f"(zéro couleur, zéro food_rel0)")
    head.eval()

    # Référence : le slot SERVI, mesuré sur les mêmes ticks de test.
    print(f"\n3. Résultats (held-out, {int(te_mask.sum())} ticks)")
    r = evaluate(head, lat, truth, te_mask)
    print(f"   {'':22} {'médiane':>9} {'<1m':>6} {'proche <2m':>12} {'loin >2m':>10}")
    print(f"   {'-' * 62}")
    print(f"   {'LOCALISATEUR PUR':22} {r['med']:8.2f}m {100*r['lt1']:5.0f}% "
          f"{r['proche']:11.2f}m {r['loin']:9.2f}m")
    print(f"   {'slot cosinus (servi)':22} {'1.43m':>9} {'42%':>6} {'0.60m':>12} {'2.30m':>10}")
    print(f"   {'sonde SUPERVISÉE':22} {'0.58m':>9} {'—':>6} {'0.45m':>12} {'0.68m':>10}")

    print(f"\n{'=' * 66}")
    if r["med"] > KILL_BAR_M:
        print(f"🛑 KILL : {r['med']:.2f} m > {KILL_BAR_M:.2f} m — n'apporte rien sur le slot servi.")
        print("   La conséquence seule ne suffit pas à ancrer la position. Voie réfutée en l'état.")
        return 1
    if r["med"] < PASS_BAR_M:
        print(f"✅ PASS : {r['med']:.2f} m < {PASS_BAR_M:.2f} m (rayon de bouche).")
        print("   Une perception SANS aucune couleur codée bat le cosinus codé-main.")
        if r["loin"] < RANGE_BAR_M:
            print(f"🎯 CIBLE ATTEINTE : {r['loin']:.2f} m au-delà de 2 m < {RANGE_BAR_M:.2f} m — "
                  "c'est là que le comportement se joue.")
        return 0
    print(f"🟡 ZONE GRISE : {r['med']:.2f} m — bat le slot ({KILL_BAR_M:.2f} m) mais reste au-dessus "
          f"du rayon de bouche ({PASS_BAR_M:.2f} m).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
