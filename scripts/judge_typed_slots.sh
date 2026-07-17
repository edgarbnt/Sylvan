#!/bin/zsh
# JUGE closed-loop du WM TYPÉ (P6-reopen, docs/design_perception_types.md §gates, PRÉ-ENREGISTRÉ) :
# config vivante (lunette saillance + sprint decont) + WM à SLOTS TYPÉS APPRIS
# (wm_objcentric_kin_typed : requêtes mesurées du rendu, marges par-type, lien découvert)
# + MONDE À APPARENCES VARIÉES (le monde du chantier). 2×24 vies seeds 1+2.
# Réf vivante MESURÉE : bras saillance 41 repas / 9 morts-danger poolés (monde plat, 2026-07-17).
# PASS-parité = repas poolés ≥ 36 ET morts-danger ≤ 11 ; KILL précoce = seed 1 < 14 repas.
#
# Usage : bash scripts/judge_typed_slots.sh [seed=1] [tag=typj${seed}]
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
SEED=${1:-1}; TAG=${2:-typj${SEED}}

export WM_CKPT=data/checkpoints/wm_objcentric_kin_typed/wm_best.pt
[[ -f "$WM_CKPT" ]] || { echo "[judge-typed] WM typé introuvable (build_typed_slots d'abord)"; exit 1; }
export SYLVAN_FOOD_APPEARANCE_VAR=0.15
export SYLVAN_WATER_APPEARANCE_VAR=0.15

# la lunette saillance + le critique decont = le vivant jugé 41/9 ; changent : WM + monde varié
bash scripts/judge_saliency_p5.sh "$SEED" "$TAG"
