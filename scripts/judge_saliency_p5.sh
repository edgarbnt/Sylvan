#!/bin/zsh
# JUGE closed-loop du chantier P5 (docs/design_purete_hjepa.md §P5, PRÉ-ENREGISTRÉ) :
# config vivante + LUNETTE SAILLANCE (SYLVAN_WP_SALIENCY) + têtes DÉ-CONTAMINÉES
# (sprint_critic_decont, pain_ckpt→waypoint_pain_decont). 2×24 vies monde v2, seeds 1+2.
# Réf vivante MESURÉE (pas de re-run) : 45 repas / 8 morts-danger poolés.
# PASS = repas poolés ≥ 40 ET morts-danger ≤ 10 ; KILL précoce = seed 1 < 14 repas.
#
# Usage : bash scripts/judge_saliency_p5.sh [seed=1] [tag=sal${seed}]
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
SEED=${1:-1}; TAG=${2:-sal${SEED}}

export SYLVAN_WP_SALIENCY=data/checkpoints/danger_saliency/saliency_best.pt
[[ -f "$SYLVAN_WP_SALIENCY" ]] || { echo "[judge-p5] saillance introuvable"; exit 1; }

# le harnais du juge sprint (env monde v2 + WM kin_haz + waypoint, parité collecte) fait le reste
bash scripts/judge_sprint_critic_v2.sh "$SEED" "$TAG" \
    data/checkpoints/sprint_critic_decont/sprint_best.pt
