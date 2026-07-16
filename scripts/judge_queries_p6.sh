#!/bin/zsh
# JUGE closed-loop du volet P6 (docs/design_purete_hjepa.md §P6, PRÉ-ENREGISTRÉ) :
# config vivante (lunette saillance + sprint decont) avec WM À REQUÊTES APPRISES
# (wm_objcentric_kin_lq — les requêtes-couleur main sont sorties du buffer). 2×24 vies seeds 1+2.
# Réf vivante MESURÉE : bras saillance 41 repas / 9 morts-danger poolés (2026-07-17).
# PASS = repas poolés ≥ 36 ET morts-danger ≤ 11 ; KILL précoce = seed 1 < 14 repas.
#
# Usage : bash scripts/judge_queries_p6.sh [seed=1] [tag=lq${seed}]
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
SEED=${1:-1}; TAG=${2:-lq${SEED}}

export WM_CKPT=data/checkpoints/wm_objcentric_kin_lq/wm_best.pt
[[ -f "$WM_CKPT" ]] || { echo "[judge-p6] WM requêtes-apprises introuvable (build_learned_queries d'abord)"; exit 1; }

# la lunette saillance + le critique decont = le vivant jugé 41/9 ; seul le WM change
bash scripts/judge_saliency_p5.sh "$SEED" "$TAG"
