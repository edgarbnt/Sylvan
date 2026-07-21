"""G0 GRATUIT du chantier FORÊT SOLIDE — deux questions, aucun entraînement, aucun Godot.

CONTEXTE. Le monde actuel est un plan vide : `forest_manager.gd` est marqué « VISUAL-ONLY forest
decor [...] NO collision, NO physics » et n'est instancié qu'en mode visuel. On veut rendre les
arbres SOLIDES et OCCULTANTS. Le precedent obstacle a montre qu'un obstacle solide ne demande PAS de
re-entrainer le WM (G0 obstacle : l'info est DEJA dans le latent, Δlatent 0.0118 vs 0.0, Δslot
6.58 m ; la voie « absorber dans le WM » a ete explicitement ecartee, SIGNAL D'ALERTE §3). Restent
deux inconnues que ce precedent ne couvre pas, parce qu'il testait UN MUR :

  TEST 1 — OCCLUSION. Avec des arbres, une ressource peut DISPARAITRE de la retine. Que fait le slot
  du WM GELE quand l'objet est masque : tient-il (permanence) ou s'effondre-t-il ? Un effondrement
  n'est PAS un argument pour re-entrainer -- c'est exactement le role de MultiSlotMemory, qui existe.
  Mais il faut le SAVOIR avant de construire, pour brancher la memoire du bon cote.

  TEST 2 — TRANSFERT D'APPARENCE. Le predicteur d'affordance (voie B) a ete entraine sur du CYAN.
  Repond-il a l'apparence d'un ARBRE (brun/vert) ? Dette OOD deja notee : il tirait aussi sur bleu et
  vert, faute d'avoir vu ces couleurs en monde food-only. Si le transfert echoue, le cout est un
  re-entrainement de CETTE PETITE TETE, jamais du WM.

METHODE. Occlusion SIMULEE hors-ligne sur des observations REELLES : on remplace les rayons pointant
vers la ressource par ce que produirait un arbre interpose (distance plus proche + couleur d'ecorce),
puis on compare le slot du WM avant/apres. C'est le meme procede que diag_cone_g0.py (masque
hors-ligne sur corpus 360).

CLI :
    PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_foret_g0.py [--selfcheck]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

WM_CKPT = "data/checkpoints/wm_objcentric_kin/wm_best.pt"
AFF_CKPT = "data/checkpoints/obstacle_affordance"
N_RAY, RETINA_RANGE_M = 36, 12.0

# Apparences, en RGB rendu [0,1]. Les trois premieres sont les references DEJA mesurees par le
# gate d'affordance ; les suivantes sont celles qu'on veut introduire (ecorce, feuillage, rocher).
COLORS = {
    "cyan (obstacle entraine)": (0.05, 0.70, 0.95),
    "rouge (bouffe, passable)": (1.00, 0.00, 0.00),
    "bleu (eau, passable)":     (0.00, 0.20, 1.00),
    "ECORCE brun fonce":        (0.36, 0.25, 0.15),
    "ECORCE brun clair":        (0.55, 0.40, 0.25),
    "FEUILLAGE vert fonce":     (0.13, 0.35, 0.13),
    "FEUILLAGE vert moyen":     (0.25, 0.55, 0.22),
    "ROCHER gris":              (0.50, 0.50, 0.52),
}


def _load_wm():
    from sylvan.models.command_wm import CommandWorldModel
    payload = torch.load(WM_CKPT, map_location="cpu", weights_only=False)
    meta = payload["meta"]
    wm = CommandWorldModel(obs_dim=meta["obs_dim"], proprio_dim=meta["proprio_dim"],
                           predictor_arch=meta.get("predictor_arch", "shallow"),
                           with_slot=meta.get("with_slot", False),
                           slot_resources=meta.get("slot_resources", 1))
    wm.load_state_dict(payload["model"])
    wm.eval()
    wm.food_idx = meta.get("food_idx", 0)
    wm.water_idx = meta.get("water_idx")
    return wm, meta


def _ticks_with_retina(run: str, limit: int) -> list[dict]:
    import gzip
    out = []
    for fp in sorted(glob.glob(os.path.join(run, "ep_*.jsonl*"))):
        op = gzip.open if fp.endswith(".gz") else open
        with op(fp, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("wm", {}).get("retina0") and r.get("plan", {}).get("food"):
                    out.append(r)
                    if len(out) >= limit:
                        return out
    return out


def occlude(retina: list[float], bearing_rad: float, half_width_rad: float,
            tree_rgb: tuple[float, float, float], tree_dist_m: float) -> list[float]:
    """Interpose un ARBRE : les rayons dans le secteur prennent la distance et la couleur de l'ecorce.

    C'est ce que la retine VERRAIT reellement -- pas un simple effacement : un arbre est un objet
    PLUS PROCHE qui remplace l'objet lointain dans ces rayons.
    """
    r = list(retina)
    for k in range(N_RAY):
        b = 2.0 * math.pi * k / N_RAY
        db = abs(math.atan2(math.sin(b - bearing_rad), math.cos(b - bearing_rad)))
        if db <= half_width_rad:
            r[4 * k] = min(r[4 * k], tree_dist_m / RETINA_RANGE_M)
            r[4 * k + 1], r[4 * k + 2], r[4 * k + 3] = tree_rgb
    return r


def test1_occlusion(runs: list[str], limit: int = 60) -> None:
    print("=" * 78)
    print("TEST 1 — le slot du WM GELÉ tient-il quand la ressource est MASQUÉE par un arbre ?")
    print("=" * 78)
    wm, _ = _load_wm()
    ticks = []
    for r in runs:
        ticks += _ticks_with_retina(r, limit - len(ticks))
        if len(ticks) >= limit:
            break
    if not ticks:
        raise SystemExit("aucun tick avec retina0 + plan.food")

    kept, moved, lost = [], [], []
    for t in ticks:
        ret = t["wm"]["retina0"]
        food = t["plan"]["food"]
        d0 = math.hypot(food[0], food[1])
        brg = math.atan2(food[0], food[1])          # convention planner : indice 1 = AVANT
        occ = occlude(ret, brg, math.radians(20.0), COLORS["ECORCE brun fonce"], max(0.8, d0 - 1.0))
        # obs du WM = proprio(132) ++ retine(144) ++ energie(1) = 277 (serve_planner_command.py:154)
        pro = t["obs"]["proprio"]
        e = [float(t["obs"]["energy"]) / 100.0]
        # On interroge la PERCEPTION (`encode_slot` = « ou est l'objet »), pas le rollout : le
        # transport du slot est GEOMETRIQUE et n'est pas concerne par l'occlusion.
        with torch.no_grad():
            sa = wm.encode_slot(torch.tensor([pro + ret + e], dtype=torch.float32))[0]
            sb = wm.encode_slot(torch.tensor([pro + occ + e], dtype=torch.float32))[0]
        da, db = float(sa.norm()), float(sb.norm())
        kept.append(db > 0.05)
        moved.append(float((sa - sb).norm()))
        lost.append(db < 0.05 <= da)

    n = len(ticks)
    print(f"  {n} instants RÉELS, ressource masquée par un arbre (secteur ±20°)")
    print(f"  slot NON nul après occlusion : {100 * sum(kept) / n:5.1f} %")
    print(f"  slot ÉTEINT par l'occlusion  : {100 * sum(lost) / n:5.1f} %")
    print(f"  déplacement médian du slot   : {sorted(moved)[n // 2]:.2f} m")
    print("\n  Lecture : slot éteint = pas de permanence sous occlusion → la MÉMOIRE devient")
    print("  load-bearing (c'est le rôle prévu de MultiSlotMemory, PAS un motif de retrain WM).")
    print("  Slot conservé = le WM porte déjà une permanence → la mémoire aurait moins de place.")


def test2_transfert(thr: float = 0.5) -> None:
    print("\n" + "=" * 78)
    print("TEST 2 — le prédicteur d'affordance (entraîné sur CYAN) répond-il à un ARBRE ?")
    print("=" * 78)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python", "scripts"))
    from train_obstacle_affordance import ObstacleAffordance
    ck = os.path.join(AFF_CKPT, "obstacle_best.pt")
    if not os.path.exists(ck):
        cands = glob.glob(os.path.join(AFF_CKPT, "*.pt"))
        if not cands:
            raise SystemExit(f"aucun checkpoint dans {AFF_CKPT}")
        ck = cands[0]
    payload = torch.load(ck, map_location="cpu", weights_only=False)
    model = ObstacleAffordance()
    model.load_state_dict(payload["state_dict"] if "state_dict" in payload else payload)
    model.eval()
    print(f"  checkpoint : {ck}")
    with torch.no_grad():
        for name, rgb in COLORS.items():
            s = float(model.s(torch.tensor([list(rgb)], dtype=torch.float32)).item())
            tag = "BLOQUANT" if s > thr else "passable"
            print(f"    s({name:26s}) = {s:.3f}   {tag}")
    print("\n  Lecture : pour que la forêt soit utilisable SANS ré-entraîner, il faut que l'écorce et")
    print("  le feuillage sortent BLOQUANTS et que rouge/bleu restent passables. Sinon le coût est un")
    print("  ré-entraînement de CETTE PETITE TÊTE (minutes), jamais du WM.")


def forest_retina(ret: list[float], dens: float, hide_brg: float | None = None,
                  half: float = math.radians(20.0), seed: int = 1234) -> list[float]:
    """Simule une retine REELLEMENT forestiere : une fraction `dens` des rayons est occupee par des
    troncs/feuillages a distances variees, et optionnellement la cible est masquee.

    POURQUOI : tester l'occlusion sur une retine quasi vide (2 rayons touches) n'est PAS le regime
    de la foret. C'est la densite qui revele la vraie degradation.
    """
    import random
    r, rng = list(ret), random.Random(seed)
    for k in range(N_RAY):
        b = 2.0 * math.pi * k / N_RAY
        occ = hide_brg is not None and abs(math.atan2(math.sin(b - hide_brg),
                                                     math.cos(b - hide_brg))) <= half
        if occ or rng.random() < dens:
            d = 2.5 if occ else rng.uniform(1.5, 8.0)
            if occ or d / RETINA_RANGE_M < r[4 * k]:
                r[4 * k] = d / RETINA_RANGE_M
                rgb = COLORS["ECORCE brun fonce"] if rng.random() < 0.5 else COLORS["FEUILLAGE vert fonce"]
                r[4 * k + 1], r[4 * k + 2], r[4 * k + 3] = rgb
    return r


def test3_densite(runs: list[str], limit: int = 80) -> None:
    """Le slot survit-il a une retine forestiere -- et SAIT-IL quand il a perdu la cible ?"""
    import statistics as st
    print("\n" + "=" * 78)
    print("TEST 3 — RÉTINE FORESTIÈRE : dégradation vs densité, et signal de perte")
    print("=" * 78)
    wm, _ = _load_wm()
    ticks = []
    for r in runs:
        ticks += _ticks_with_retina(r, limit - len(ticks))
        if len(ticks) >= limit:
            break
    print(f"{'densité arbres':>14} {'cible cachée':>13} | {'slot PERDU':>11} {'erreur médiane':>15}")
    print("-" * 60)
    for dens in (0.0, 0.3, 0.6):
        for hide in (False, True):
            lost, err = [], []
            for t in ticks:
                ret, f = t["wm"]["retina0"], t["plan"]["food"]
                d0, brg = math.hypot(*f), math.atan2(f[0], f[1])
                occ = forest_retina(ret, dens, brg if hide else None)
                pro = t["obs"]["proprio"]
                e = [float(t["obs"]["energy"]) / 100.0]
                with torch.no_grad():
                    n = float(wm.encode_slot(torch.tensor([pro + occ + e],
                                                          dtype=torch.float32))[0].norm())
                lost.append(n < 0.05)
                err.append(abs(n - d0))
            print(f"{dens:13.0%} {'OUI' if hide else 'non':>13} | "
                  f"{100 * sum(lost) / len(ticks):10.1f}% {st.median(err):14.2f}m")
    print("\n  Lecture : `slot PERDU` = 0 % PARTOUT signifie que la perception ne se déclare JAMAIS")
    print("  aveugle — elle rapporte une position FAUSSE. L'entité poursuivrait un fantôme, et la")
    print("  mémoire n'aurait aucun signal pour prendre le relais. C'est la pièce manquante.")


def _selfcheck() -> None:
    ret = [1.0, 0.0, 0.0, 0.0] * N_RAY
    occ = occlude(ret, 0.0, math.radians(20.0), (0.36, 0.25, 0.15), 3.0)
    changed = [k for k in range(N_RAY) if occ[4 * k:4 * k + 4] != ret[4 * k:4 * k + 4]]
    assert changed, "l'occlusion ne modifie aucun rayon"
    # secteur ±20° sur 36 rayons de 10° => 5 rayons (0, ±10, ±20)
    assert len(changed) == 5, f"secteur attendu 5 rayons, obtenu {len(changed)}"
    assert abs(occ[0] - 3.0 / RETINA_RANGE_M) < 1e-6, "distance de l'arbre non appliquee"
    # un secteur a 180 deg ne doit PAS toucher le rayon 0
    occ2 = occlude(ret, math.pi, math.radians(20.0), (0.36, 0.25, 0.15), 3.0)
    assert occ2[0] == ret[0], "l'occlusion arriere a touche le rayon avant"
    print("selfcheck OK — occlusion ciblée (5 rayons), distance appliquée, secteur correct")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*",
                    default=["data/replay_buffer/arbgrad_graded_s1_r40_fa0"])
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck()
        return
    test1_occlusion(a.runs, a.limit)
    test3_densite(a.runs, max(a.limit, 80))
    test2_transfert()


if __name__ == "__main__":
    main()
