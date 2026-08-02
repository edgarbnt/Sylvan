"""G0 GRATUIT — le signal qui distingue une bonne décision d'une mauvaise est-il DANS le rêve ?

Gate pré-inscrit dans `docs/design_critique_rang.md`. Zéro entraînement, zéro Godot : on rejoue le
WM GELÉ sur des candidats dont la conséquence RÉELLE est déjà mesurée.

CE QUI REND CE G0 POSSIBLE, et ce qui manquait en juillet : `scripts/cf_fork_probe.sh` a produit
des CONTREFACTUELS RÉELS — 3 forks x 21 commandes x conséquence mesurée en rejeu déterministe.
Le chantier de juillet n'avait que des données off-policy (l'action prise, jamais les autres), donc
aucune façon de savoir si une meilleure action existait, ni si elle était identifiable.

LA QUESTION : si aucune lecture du rêve ne corrèle avec le résultat réel, alors aucune tête posée
sur ce rêve ne peut trouver la bonne action — et on l'apprend pour zéro, sans payer un entraînement.

CE QUE LA LITTÉRATURE AJOUTE (recherche du 2026-08-02) :
  · Bellemare et al. 2016, « Increasing the Action Gap » — quand les valeurs de plusieurs actions
    sont proches, l'erreur d'approximation domine le choix glouton. C'est le diagnostic EXACT de
    l'échec de juillet (« erreur réseau 19-47x l'écart à trancher entre 33 candidats quasi
    ex-aequo »). La parade n'est pas de mieux régresser, c'est d'ÉLARGIR l'écart. On mesure donc
    l'écart réel entre actions — en DESCRIPTIF, pas comme critère : le transformer en critère après
    avoir vu les données serait déplacer les poteaux.
  · « Value-Guided Action Planning with JEPA World Models » (arXiv 2601.00844) — planifier en
    minimisant la distance à l'objectif dans l'espace de représentation « a de nombreux minima
    locaux ». C'est littéralement notre `-min_dist`. C'est une raison de tester d'AUTRES lectures
    que celle qui est servie.

BARRES PRÉ-ENREGISTRÉES (design §G0, écrites AVANT de regarder les données) :
  G0-signal   au moins une lecture atteint |rho| >= 0,40 avec le résultat réel
  G0-marge    la lecture la mieux classée désigne un candidat qui BAT `ref` sur >= 2 forks sur 3
  G0-contrôle le coût SERVI, lui, ne les bat PAS (sinon rien à apprendre : il suffit de le lire)
  🛑 STOP     aucune lecture ne dépasse |rho| = 0,20 ⇒ le rêve ne porte pas le signal ; le levier
              redevient le WM (cécité au mouvement des objets), pas le critique.
  Zone grise 0,20-0,40 ⇒ règle PRÉ-DÉCLARÉE : 3 forks de plus, re-jugés UNE fois.

Usage :
    PYTHONPATH=python SYLVAN_RETINA_FOV_DEG=120 env_pytorch_3.12/bin/python \
        diagnostics/diag_critique_rang_g0.py
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path

import torch

from sylvan.models.command_wm import CommandWorldModel

FORK_K = 60
HORIZON = 80
BAR_RHO = 0.40
STOP_RHO = 0.20


def real_outcomes(path: str) -> tuple[int | None, list[tuple[float, float, int]]]:
    """Lit une sortie de cf_fork_probe : (ref, [(vx, om, repas), ...])."""
    txt = Path(path).read_text()
    m = re.search(r"A=(\d+)\s+B=(\d+)", txt)
    ref = int(m.group(1)) if m else None
    rows = []
    for line in txt.splitlines():
        mm = re.match(r"\s+(0\.\d+)\s+([+-]?\d\.\d+)\s+(\d+)\s*$", line)
        if mm:
            rows.append((float(mm.group(1)), float(mm.group(2)), int(mm.group(3))))
    return ref, rows


def obs_at_fork(run_dir: str, k: int, proprio_dim: int) -> torch.Tensor | None:
    """Observation servie au tick du fork : proprio ++ retina0 ++ energie/100."""
    files = sorted(glob.glob(f"{run_dir}/*.jsonl"))
    if not files:
        return None
    rows = []
    for line in open(files[0]):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        ob, wm = r.get("obs"), r.get("wm")
        if not ob or not wm:
            continue
        ret = wm.get("retina0")
        if not ret or len(ret) != 144:
            continue
        rows.append((ob["proprio"], ret, float(ob["energy"])))
    if len(rows) <= k:
        return None
    p, ret, e = rows[k]
    if len(p) != proprio_dim:
        return None
    return torch.tensor(p + ret + [e / 100.0], dtype=torch.float32)


def spearman(a: list[float], b: list[float]) -> float:
    """Corrélation de rang, sans scipy. nan si une série est constante."""
    n = len(a)
    if n < 4:
        return float("nan")

    def rank(v: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return float("nan") if da * db == 0 else num / (da * db)


def readings(wm: CommandWorldModel, obs0: torch.Tensor,
             cmds: list[tuple[float, float]]) -> dict[str, list[float]]:
    """Fait rêver le WM GELÉ sur chaque commande TENUE, et extrait plusieurs lectures."""
    n = len(cmds)
    seqs = torch.zeros(n, HORIZON, 2)
    for i, (vx, om) in enumerate(cmds):
        seqs[i, :, 0] = vx
        seqs[i, :, 1] = om
    with torch.no_grad():
        out = wm.rollout_open_loop(obs0.unsqueeze(0).expand(n, -1), seqs)
    slot = out["slot"]                                   # [n, h, 2]
    dist = torch.linalg.vector_norm(slot, dim=-1)        # [n, h]
    energy = out["predicted_next_obs"][..., -1]
    done = torch.sigmoid(out["predicted_done_logits"])
    cos_brg = slot[..., 1] / (dist + 1e-6)
    r = {
        "cout SERVI (-min_dist + energie)": (-dist.min(dim=1).values
                                             + 1.0 * energy[:, -1]).tolist(),
        "-min_dist seul": (-dist.min(dim=1).values).tolist(),
        "-distance FINALE": (-dist[:, -1]).tolist(),
        "energie predite": energy[:, -1].tolist(),
        "alignement moyen": cos_brg.mean(dim=1).tolist(),
        "-risque (done)": (-done.sum(dim=1)).tolist(),
        "rapprochement (d0 - dmin)": (dist[:, 0] - dist.min(dim=1).values).tolist(),
    }
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", default="data/checkpoints/wm_foret_v2_slot/wm_best.pt")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = ap.parse_args()

    payload = torch.load(args.wm, map_location="cpu", weights_only=False)
    wm = CommandWorldModel.from_checkpoint(payload)
    wm.eval()
    wm.requires_grad_(False)
    pdim = int(payload["meta"]["proprio_dim"])

    outs = {1: "/tmp/cf_foret2.out", 2: "/tmp/cf_foret_s2.out", 3: "/tmp/cf_foret_s3.out"}
    pooled: dict[str, list[float]] = {}
    pooled_real: list[float] = []
    per_fork = []

    for s in args.seeds:
        ref, rows = real_outcomes(outs[s])
        obs0 = obs_at_fork(f"/tmp/fork_obs_s{s}", FORK_K, pdim)
        if obs0 is None or not rows:
            print(f"  ⚠️ graine {s} : données manquantes (obs={obs0 is not None}, "
                  f"candidats={len(rows)}) — écartée")
            continue
        cmds = [(vx, om) for vx, om, _ in rows]
        real = [float(m) for _, _, m in rows]
        rd = readings(wm, obs0, cmds)
        per_fork.append((s, ref, real, rd))
        for k, v in rd.items():
            pooled.setdefault(k, []).extend(v)
        pooled_real.extend(real)
        gap = max(real) - min(real)
        print(f"  graine {s} : ref={ref}  max={int(max(real))}  ecart entre actions={int(gap)}  "
              f"(candidats a zero : {sum(1 for x in real if x == 0)}/{len(real)})")

    if not per_fork:
        print("\n❌ aucune graine exploitable — le G0 ne peut pas être rendu.")
        return

    print(f"\n{len(pooled_real)} triplets (etat, action, consequence REELLE) poolés\n")
    print("CORRÉLATION DE RANG entre chaque lecture du rêve et le résultat RÉEL :")
    ranked = []
    for k, v in pooled.items():
        rho = spearman(v, pooled_real)
        ranked.append((abs(rho) if rho == rho else -1, rho, k))
    ranked.sort(reverse=True)
    for _, rho, k in ranked:
        flag = "✅" if abs(rho) >= BAR_RHO else ("~" if abs(rho) >= STOP_RHO else " ")
        print(f"  {flag} {k:34s} rho = {rho:+.3f}")

    best_abs, best_rho, best_name = ranked[0]
    served = next(r for _, r, k in ranked if k.startswith("cout SERVI"))

    # G0-marge : la meilleure lecture désigne-t-elle un candidat qui bat ref ?
    def picks_better(name: str) -> int:
        wins = 0
        for s, ref, real, rd in per_fork:
            v = rd[name]
            i = max(range(len(v)), key=lambda j: v[j])
            if ref is not None and real[i] > ref:
                wins += 1
        return wins

    w_best = picks_better(best_name)
    w_served = picks_better("cout SERVI (-min_dist + energie)")
    print(f"\nG0-marge   : « {best_name} » désigne un candidat battant ref sur "
          f"{w_best}/{len(per_fork)} forks")
    print(f"G0-contrôle: le coût SERVI, lui, y arrive sur {w_served}/{len(per_fork)} forks")

    print("\n" + "=" * 74)
    if best_abs < STOP_RHO:
        print("🛑 STOP pré-enregistré — aucune lecture ne dépasse |rho| = 0,20.")
        print("   Le rêve du WM NE PORTE PAS le signal : aucune tête posée dessus ne peut")
        print("   trouver la bonne action. Le levier redevient le WM lui-même (cécité au")
        print("   mouvement des objets), pas le critique. Ne pas payer G1.")
    elif best_abs < BAR_RHO:
        print(f"~ ZONE GRISE ({best_abs:.2f}) — règle PRÉ-DÉCLARÉE : collecter 3 forks de plus")
        print("  (graines 4-6, mêmes paramètres) et re-juger UNE fois. Pas de collecte répétée.")
    elif w_best >= 2 and w_served < w_best:
        print("✅ G0 PASSÉ — le signal est dans le rêve, une lecture le lit mieux que le coût")
        print("   servi, et elle désigne de meilleures actions. G1 (tête de RANG) est licencié.")
    else:
        print("⚠️ G0 PARTIEL — corrélation suffisante mais la lecture ne bat pas le coût servi")
        print("   là où il faut. Re-scoper AVANT de payer un entraînement.")


if __name__ == "__main__":
    main()
