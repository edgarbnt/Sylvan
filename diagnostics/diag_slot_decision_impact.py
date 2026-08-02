"""Le slot à 1,43 m casse-t-il la DÉCISION du planner ? (offline, gratuit, zéro Godot)

POURQUOI. Le slot en forêt est ~4× moins précis qu'en monde typé (1,43 m vs 0,35 m), mais
l'erreur est-elle suffisante pour FAIRE DIVERGER la commande choisie par le planner ? C'est
la seule question qui décide si on touche à la rétine ou non.

Le test est conceptuellement simple : sur chaque tick du corpus où la nourriture est visible,
on roule le MÊME rêve (mêmes candidats, même WM) avec deux positions initiales différentes —
celle du slot (encodée depuis la rétine) et la position VRAIE (food_rel0, l'oracle). On compare
les commandes choisies par argmax. Si les commandes sont identiques (ou quasi) dans la majorité
des cas, 1,43 m est un bruit que le planner absorbe — le problème est ailleurs. Si la commande
change souvent, l'erreur du slot CASSE la décision → il faut corriger la rétine.

CRITÈRES PRÉ-ENREGISTRÉS :
  T1 STABILITÉ .... ≥ 70 % des ticks où la commande choisie est IDENTIQUE (même indice de
                     candidat). Le planner est robuste au bruit de position → ne pas toucher
                     à la rétine, chercher ailleurs.
  T2 PROXIMITÉ ..... score du candidat-slot ≥ 80 % du score du meilleur candidat-oracle
                     sur ≥ 80 % des ticks. Même si l'argmax flippe, le slot pointe vers un
                     candidat quasi aussi bon.
  KILL ............. < 50 % de commandes identiques → le slot CASSE la décision → la rétine
                     est le bottleneck, il faut la corriger AVANT tout closed-loop.

Le plan_wm_slot (single-resource food-only) est le chemin le plus simple et le plus direct :
score = -min_dist (heading_weight=0 en config vivante). On compare les vecteurs de score
pour tous les candidats et on mesure l'accord d'argmax.

CLI :
    PYTHONPATH=python SYLVAN_RETINA_FOV_DEG=120 env_pytorch_3.12/bin/python \
        diagnostics/diag_slot_decision_impact.py \
        --wm data/checkpoints/wm_foret_v2_slot/wm_best.pt \
        --corpus data/replay_buffer/gate_foret_cl
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from sylvan.models.command_wm import CommandWorldModel, DISPLACEMENT_SCALE  # noqa: E402

STABILITY_BAR = 0.70       # ≥ 70 % de commandes identiques
PROXIMITY_BAR = 0.80       # ≥ 80 % des ticks où le score slot ≥ 80 % du score oracle
KILL_BAR = 0.50            # < 50 % de commandes identiques → le slot casse la décision
_HALF_FOV = float(os.environ.get("SYLVAN_RETINA_FOV_DEG", "360")) / 2.0
HEADING_WEIGHT = float(os.environ.get("SYLVAN_PLANNER_HEADING_W", "0.0"))


def load_pairs(corpus: str, key: str) -> tuple[torch.Tensor, torch.Tensor, list]:
    """(obs, positions vraies, raw_lines) sur les ticks où la ressource est visible."""
    obs_list, tgt, lines = [], [], []
    for f in sorted(glob.glob(os.path.join(corpus, "*.jsonl"))):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            v = r.get("wm", {}).get(key)
            if not v or len(v) < 3 or v[2] <= 0.5:
                continue
            bearing = math.degrees(math.atan2(v[0], v[1]))
            if abs(bearing) > _HALF_FOV:
                continue
            obs = r["obs"]
            obs_vec = obs["proprio"] + obs["retina"] + [obs["energy"] / 100.0]
            obs_list.append(obs_vec)
            tgt.append([v[0], v[1]])
            lines.append(r)
    if not obs_list:
        raise SystemExit(f"aucun tick avec {key} visible dans {corpus}")
    return torch.tensor(obs_list, dtype=torch.float32), torch.tensor(tgt, dtype=torch.float32), lines


def prepare_wm(wm_path: str) -> CommandWorldModel:
    payload = torch.load(wm_path, map_location="cpu", weights_only=False)
    meta = payload["meta"]
    if not meta.get("with_slot"):
        raise SystemExit(f"{wm_path} n'a pas de canal-slot")
    wm = CommandWorldModel.from_checkpoint(payload)
    wm.load_state_dict(payload["model"])
    qthr = meta.get("query_thr")
    if qthr is not None and getattr(wm.slot_encoder, "query_thr", None) is not None:
        wm.slot_encoder.query_thr.copy_(torch.tensor(qthr, dtype=torch.float32))
    fov = float(os.environ.get("SYLVAN_RETINA_FOV_DEG", "360"))
    if abs(fov - 360.0) > 1e-6 and getattr(wm.slot_encoder, "sin", None) is not None:
        n = wm.slot_encoder.sin.shape[0]
        th = torch.tensor(
            [(k if k <= n // 2 else k - n) * math.radians(fov) / n for k in range(n)],
            dtype=torch.float32,
        )
        wm.slot_encoder.sin.copy_(torch.sin(th))
        wm.slot_encoder.cos.copy_(torch.cos(th))
    wm.eval()
    return wm


def build_candidates(wm: CommandWorldModel) -> torch.Tensor:
    """Réplique de _build_candidates() du CommandPlanner, version minimale.
    On garde la grille constante + les 2-segments pour couvrir virages + droites."""
    h = 80  # horizon WM
    seqs: list[list[list[float]]] = []
    # (a) CONSTANT-command candidates
    vx_grid = (0.55, 0.65, 0.75)
    omega_grid = (-0.6, -0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6)
    for vx in vx_grid:
        for om in omega_grid:
            seqs.append([[vx, om]] * h)
    # (b) 2-SEGMENT candidates
    for split in (40,):
        for vx1 in (0.6, 0.75):
            for om1 in (-0.6, -0.3, 0.0, 0.3, 0.6):
                for vx2 in (0.6, 0.75):
                    for om2 in (-0.6, -0.3, 0.0, 0.3, 0.6):
                        if (vx1, om1) == (vx2, om2):
                            continue
                        seqs.append([[vx1, om1]] * split + [[vx2, om2]] * (h - split))
    # (c) PIVOT candidates
    for plen in (30,):
        for om1 in (-0.6, -0.45, 0.45, 0.6):
            seqs.append([[0.55, om1]] * plen + [[0.7, 0.0]] * (h - plen))
    return torch.tensor(seqs, dtype=torch.float32)


@torch.no_grad()
def score_candidates(
    wm: CommandWorldModel,
    obs0: torch.Tensor,
    cmd_seqs: torch.Tensor,
    slot_override: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score de chaque candidat (plan_wm_slot, single-resource food-only).
    slot_override [B, 2] = position initiale alternative (None → encode depuis l'obs).
    Retourne le vecteur de score [n_candidates], PLUS HAUT = meilleur."""
    n = cmd_seqs.shape[0]
    h = cmd_seqs.shape[1]
    obs_batch = obs0.reshape(1, -1).expand(n, -1).contiguous()
    if slot_override is not None:
        out = wm.rollout_open_loop(obs_batch, cmd_seqs, slot0=slot_override)
    else:
        out = wm.rollout_open_loop(obs_batch, cmd_seqs)
    slot = out["slot"]                                              # [n, h, 2]
    dist = torch.linalg.vector_norm(slot, dim=-1)                   # [n, h]
    min_dist = dist.min(dim=1).values                               # [n]
    done_prob = torch.sigmoid(out["predicted_done_logits"])
    energy_pred = out["predicted_next_obs"][..., -1].clamp(0.0, 1.0)
    cos_brg = slot[..., 1] / (dist + 1e-6)
    far_gate = (dist / 2.0).clamp(max=1.0)                          # heading_far_gate = 2.0
    mean_align = (cos_brg * far_gate).mean(dim=1)
    alive = torch.ones(n)
    survival_pen = torch.zeros(n)
    for t in range(h):
        survival_pen = survival_pen + alive * done_prob[:, t]
        alive = alive * (1.0 - done_prob[:, t])
    return -min_dist + HEADING_WEIGHT * mean_align \
        + 2.0 * energy_pred[:, -1] - 3.0 * survival_pen           # energy_weight=2, done_penalty=3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wm", default="data/checkpoints/wm_foret_v2_slot/wm_best.pt")
    ap.add_argument("--corpus", default="data/replay_buffer/gate_foret_cl")
    ap.add_argument("--max-ticks", type=int, default=0,
                    help="limite les ticks traités (0 = tous)")
    a = ap.parse_args()

    print(f"=== LE SLOT CASSE-T-IL LA DÉCISION ? | {a.wm} | {a.corpus} ===")
    wm = prepare_wm(a.wm)
    cmd_seqs = build_candidates(wm)
    n_cand = cmd_seqs.shape[0]
    print(f"  candidats : {n_cand}, horizon : {cmd_seqs.shape[1]}, heading_weight : {HEADING_WEIGHT}")

    obs, truth, _ = load_pairs(a.corpus, "food_rel0")
    n_total = obs.shape[0]
    if a.max_ticks > 0:
        obs, truth = obs[:a.max_ticks], truth[:a.max_ticks]
    n_ticks = obs.shape[0]
    print(f"  ticks : {n_ticks} (sur {n_total} total)")

    # Pré-encoder les slots (batch) — le slot est encodé en interne par rollout_open_loop
    # quand on ne passe PAS de slot_override. On veut le slot t0 séparément pour info.
    with torch.no_grad():
        slots_enc = wm.encode_slot(obs)                             # [B, 2] depuis la rétine

    # Erreur de position par tick (info seulement)
    slot_err = (slots_enc - truth).norm(dim=1)
    print(f"\n  Erreur slot→vérité : méd={slot_err.median():.2f} m, "
          f"moy={slot_err.mean():.2f} m, p90={slot_err.quantile(0.9):.2f} m")

    # ── Comparaison des décisions ──
    same_cmd = 0
    slot_better = 0        # le slot trouve un MEILLEUR candidat que la vérité (bruit de l'argmax)
    truth_wins = 0          # la vérité trouve un MEILLEUR candidat que le slot
    score_ratios: list[float] = []           # score(slot_choice) / score(truth_choice) — les deux
                                            # scores sont ceux du scoreur-VÉRITÉ (le juge impartial)

    for i in range(n_ticks):
        # Score avec position VRAIE (oracle)
        truth0 = truth[i:i+1]  # [1, 2]
        score_truth = score_candidates(wm, obs[i], cmd_seqs, slot_override=truth0)
        best_truth = int(torch.argmax(score_truth).item())

        # Score avec position SLOT (encodée depuis la rétine)
        slot0 = slots_enc[i:i+1]  # [1, 2]
        score_slot = score_candidates(wm, obs[i], cmd_seqs, slot_override=slot0)
        best_slot = int(torch.argmax(score_slot).item())

        if best_slot == best_truth:
            same_cmd += 1
        else:
            # Qui a raison ? On juge avec le scoreur-VÉRITÉ (impartial) :
            # le score du candidat-slot évalué par la VÉRITÉ vs le score du candidat-vérité
            # évalué par la VÉRITÉ.
            slot_score_by_truth = float(score_truth[best_slot])
            truth_score_by_truth = float(score_truth[best_truth])
            if slot_score_by_truth >= truth_score_by_truth:
                slot_better += 1    # l'argmax du slot a trouvé un meilleur candidat (jugé par le vrai score)
            else:
                truth_wins += 1

        # Ratio de score : le candidat choisi par le slot, jugé par le scoreur-vérité,
        # comparé au meilleur score possible (le candidat-vérité, jugé par le scoreur-vérité)
        ratio = float(score_truth[best_slot] / score_truth[best_truth]) if float(score_truth[best_truth]) > 0 else 1.0
        if not math.isnan(ratio):
            score_ratios.append(ratio)

    # ── Verdicts ──
    same_pct = same_cmd / n_ticks
    proximity = sum(1.0 for r in score_ratios if r >= 0.80) / max(len(score_ratios), 1)
    mean_ratio = sum(score_ratios) / max(len(score_ratios), 1)

    print(f"\n  {'─' * 50}")
    print(f"  Commandes IDENTIQUES  : {same_cmd}/{n_ticks} = {100*same_pct:.1f}%  (barre ≥ {100*STABILITY_BAR:.0f}%)")
    print(f"  Slot meilleur (jugé) : {slot_better}/{n_ticks}")
    print(f"  Vérité gagne  (jugé) : {truth_wins}/{n_ticks}")
    print(f"  Ratio de score médian : {sorted(score_ratios)[len(score_ratios)//2]:.3f}"
          f"  (1.0 = slot trouve le meilleur candidat)")
    print(f"  Proximité ≥ 80%      : {100*proximity:.1f}%  (barre ≥ {100*PROXIMITY_BAR:.0f}%)")

    # Décomposition par bande d'erreur
    print(f"\n  ── Par bande d'erreur slot ──")
    err_bins = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 5.0), (5.0, 100.0)]
    for lo, hi in err_bins:
        mask = (slot_err >= lo) & (slot_err < hi)
        n_bin = mask.sum().item()
        if n_bin < 5:
            continue
        same_bin = sum(1 for j in range(n_ticks) if mask[j] and
                       int(torch.argmax(score_candidates(wm, obs[j], cmd_seqs,
                                                         slot_override=slots_enc[j:j+1])).item())
                       == int(torch.argmax(score_candidates(wm, obs[j], cmd_seqs,
                                                            slot_override=truth[j:j+1])).item()))
        print(f"  [{lo:.1f}, {hi:.1f}) m : {n_bin:4d} ticks, "
              f"identiques={same_bin}/{n_bin} = {100*same_bin/n_bin:.0f}%")

    print()
    if same_pct >= STABILITY_BAR:
        print(f"  ✅ T1 PASSÉ : le planner absorbe l'erreur du slot → ne pas toucher à la rétine,"
              f" chercher ailleurs.")
    elif same_pct < KILL_BAR:
        print(f"  🛑 KILL : le slot CASSE la décision → la rétine est le bottleneck, il faut la"
              f" corriger AVANT tout closed-loop.")
    else:
        print(f"  🟡 ZONE GRISE : {same_pct:.1%} < {STABILITY_BAR:.0%} mais ≥ {KILL_BAR:.0%}."
              f" Proximité={proximity:.1%}.")
        if proximity >= PROXIMITY_BAR:
            print(f"     → la commande flippe mais le candidat-slot est quasi aussi bon que"
                  f" l'oracle → l'erreur est un BRUIT TOLÉRABLE, pas un bloqueur.")
        else:
            print(f"     → la commande flippe ET le score dégrade → l'erreur PÈSE sur la décision.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
