"""CRITIQUE-SPRINT de l'étage waypoint — forme IC+TC (docs/design_critique_sprint.md).

Forme D1 (tranchée owner 2026-07-16) : au déploiement, le scoreur ANALYTIQUE reste le socle et
    score(c) = leg1 + leg2 + (W − g(s,c)) · intrusion(c),   g = W · p(s,c) ∈ [0, W]
p(s,c) = P(la traversée PAIE | état, candidat). La correction ne touche que les candidats qui
croisent le vert et ne peut qu'ADOUCIR la pénalité (jamais l'aggraver) : g=0 ⇒ bit-identique à
l'analytique — le plancher de perf est le bras géométrie. Elle n'apprend QUE la licence de sprint,
ce que la géométrie ignore (drives, santé, douleur prédite).

Label PINNÉ (Phase 0, diag_sprint_corpus) : y = 1[U > 0], U = gain_repas_observé/drain −
κ_data·dégâts_de_poursuite (LINÉAIRE — plancher-mort non retenu, 3 % < 10 %) ; κ_data = 9.5,
drain mesuré 0.05, valeur repas ≈ 799 pas.

CE FICHIER porte le CONTRAT train=déploiement : `SprintCritic` + `sprint_inputs` (14-d).
Le trainer et les gates pré-enregistrés (G-rank AUC>0.70, G-res +10 pts, G-consist ≤1.2×)
arrivent avec la Phase C — APRÈS la collecte ε monde-v2 (le corpus g24 n'a aucun contrefactuel
blessé : l'oracle ne sprintait jamais sous santé 60).

Usage (Phase A) :
  PYTHONPATH=python env_pytorch_3.12/bin/python -m scripts.train_sprint_critic --selfcheck
"""

from __future__ import annotations

import argparse

import torch
from torch import nn

from sylvan.control.waypoint_layer import WP_FEAT_DIM

SPRINT_IN_DIM = WP_FEAT_DIM + 4   # + énergie/100, soif/100, santé/100, douleur prédite (/100)


class SprintCritic(nn.Module):
    """entrées [B, 14] → p = P(la traversée paie) ∈ (0, 1). Déploiement : g = W·p."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(SPRINT_IN_DIM, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def p(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x).squeeze(-1))


def sprint_inputs(feats: list[list[float]], drives: tuple[float, float, float],
                  pain: list[float]) -> torch.Tensor:
    """Assemble les entrées 14-d PAR CANDIDAT — LE point de parité train/déploiement.

    feats = `candidate_features` (déjà miroir-canoniques — la symétrie s'impose, ne se fitte pas) ;
    drives = (énergie, soif, santé) 0-100 (échelle payload/BC) ; pain = sortie BRUTE de
    PainCritic.pain (unités /100) sur les MÊMES feats — le savoir douleur bankée devient une entrée."""
    e, t, h = drives
    rows = [list(f) + [e / 100.0, t / 100.0, h / 100.0, float(p)] for f, p in zip(feats, pain)]
    x = torch.tensor(rows, dtype=torch.float32)
    assert x.shape[-1] == SPRINT_IN_DIM, x.shape
    return x


def make_checkpoint(critic: SprintCritic, pain_ckpt: str, **meta) -> dict:
    """Format de checkpoint unique (déploiement : waypoint_layer recharge pain_ckpt d'ici —
    la parité de la feature douleur est portée par le chemin bankée, pas par une convention)."""
    return {"state_dict": critic.state_dict(), "in_dim": SPRINT_IN_DIM,
            "pain_ckpt": pain_ckpt, **meta}


def selfcheck() -> None:
    """Contrat 14-d + intégration waypoint_layer : g≡0 ⇒ bit-identique à l'analytique ;
    g≡W ⇒ la licence ouvre le direct bloqué ; exclusivité des modes de scoring."""
    import math
    import os
    import tempfile
    from pathlib import Path

    from sylvan.control.waypoint_layer import WaypointLayer

    x = sprint_inputs([[0.1] * WP_FEAT_DIM] * 3, (30.0, 70.0, 100.0), [0.1, 0.2, 0.3])
    assert tuple(x.shape) == (3, SPRINT_IN_DIM) and abs(float(x[1, -1]) - 0.2) < 1e-6

    # scène synthétique : cible à 4 m droit devant, vert SUR la ligne à 2 m → direct bloqué.
    retina = [1.0, 0.0, 0.0, 0.0] * 36
    retina[0:4] = [0.2, 0.0, 1.0, 0.0]           # rayon k=0 (droit devant), d=0.2 → vert à 2 m
    target = (0.0, 4.0)
    pain_ckpt = "data/checkpoints/waypoint_pain_v3/pain_best.pt"
    if not Path(pain_ckpt).exists():
        print("[selfcheck] ⚠️ pain_v3 absent — intégration sautée (contrat 14-d seul vérifié)")
        return
    base = WaypointLayer()
    rec0 = base.decide("food", target, retina)
    assert rec0["intr_direct"] > 0.5, rec0        # la scène bloque bien le direct

    with tempfile.TemporaryDirectory() as td:
        for bias, name in ((-20.0, "g0"), (+20.0, "gW")):
            c = SprintCritic()
            with torch.no_grad():
                c.net[-1].weight.zero_()
                c.net[-1].bias.fill_(bias)        # σ(∓20) → p ≈ 0 / 1 quelle que soit l'entrée
            torch.save(make_checkpoint(c, pain_ckpt), Path(td) / f"{name}.pt")
        try:
            os.environ["SYLVAN_WP_SPRINT_CRITIC"] = str(Path(td) / "g0.pt")
            lay = WaypointLayer()
            lay.maybe_decide("food", target, retina, drives=(30.0, 70.0, 100.0))
            rec = lay.decide("food", target, retina)
            assert (rec["choice"], rec["cost_direct"], rec["cost_best_wp"]) == \
                (rec0["choice"], rec0["cost_direct"], rec0["cost_best_wp"]), (rec, rec0)

            os.environ["SYLVAN_WP_SPRINT_CRITIC"] = str(Path(td) / "gW.pt")
            lay = WaypointLayer()
            lay.maybe_decide("food", target, retina, drives=(30.0, 70.0, 100.0))
            rec = lay.decide("food", target, retina)
            assert rec["choice"] == "direct", rec  # pénalité verte licenciée → direct ≈ 4 m gagne
            assert abs(rec["cost_direct"] - math.hypot(*target)) < 0.2, rec

            os.environ["SYLVAN_WP_ORACLE_SPRINT"] = "1"
            try:
                WaypointLayer()
                raise AssertionError("exclusivité sprint-critic/oracle non levée")
            except ValueError:
                pass
        finally:
            os.environ.pop("SYLVAN_WP_SPRINT_CRITIC", None)
            os.environ.pop("SYLVAN_WP_ORACLE_SPRINT", None)
    print("[selfcheck] OK — 14-d, g≡0 bit-identique, g≡W licencie le direct, exclusivité")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    raise SystemExit("[sprint] trainer = Phase C (après collecte ε) — voir docs/design_critique_sprint.md")


if __name__ == "__main__":
    main()
