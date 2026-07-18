#!/bin/zsh
# Collecte ε-CIBLE du chantier CRITIQUE-ARBITRAGE — G1 (docs/design_critique_arbitrage.md §G1).
# Base = CONFIG VIVANTE de l'étage waypoint (train=déploiement : lunette saillance + sprint-critic
# dé-contaminé ACTIFS — le critique d'arbitrage se déploiera PAR-DESSUS eux). L'ε vit au niveau
# CIBLE (choix food/eau du planner, plan_multi_surv), TENU K replans : la politique de cible
# explorée = le coût survie designé lui-même (pas d'auto-confirmation). ε waypoint OFF (pas le
# sujet). Décisions forcées FLAGGÉES `explore_target` dans le BC_LOG (corpus honnête).
#
# Calibrage déclaré : fraction forcée en régime permanent ≈ K/(K + 1/ε) ; à ε=0.05, K=15 →
# ~43 % des replans MULTI forcés, ~2-3 bascules TENUES par vie (~150 pas ≈ 3 m de poursuite).
# La masse on-policy vient des 240 vies déjà instrumentées (corpus G0) ; ce corpus apporte les
# CONTREFACTUELS de bascule qui n'existent pas dans le vécu designé (leçon auto-confirmante).
#
# Seeds 3+4 (les seeds 1+2 restent la propriété du juge).
# Usage : bash scripts/collect_arb_corpus.sh [ep=24] [seed=3] [tag=arb<seed>]
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
NEP=${1:-24}; SEED=${2:-3}; TAG=${3:-arb${SEED}}
OUT="data/replay_buffer/critic_kin_${TAG}"

export WM_CKPT=data/checkpoints/wm_objcentric_kin_haz/wm_best.pt
export SYLVAN_HAZARD_COUNT=1
export SYLVAN_HAZARD_ENGULF_P=0.5
export SYLVAN_HEALTH_REGEN=0.05
export SYLVAN_WAYPOINT=1
export SYLVAN_WAYPOINT_DEBUG=1
export SYLVAN_WP_LOG="$OUT"
export SYLVAN_WP_SALIENCY=data/checkpoints/danger_saliency/saliency_best.pt
export SYLVAN_WP_SPRINT_CRITIC=data/checkpoints/sprint_critic_decont/sprint_best.pt
export SYLVAN_TARGET_EXPLORE_EPS=${EPS:-0.05}
export SYLVAN_TARGET_EXPLORE_PERSIST=${PERSIST:-15}
export SYLVAN_TARGET_EXPLORE_SEED=$SEED
# garde-fous : pas d'ε waypoint ni d'oracle pendant la collecte ε-cible
unset SYLVAN_WP_EXPLORE_EPS SYLVAN_WP_ORACLE_SPRINT SYLVAN_WP_PAIN_CRITIC

bash scripts/collect_critic_corpus_kin.sh "$NEP" "$SEED" "$TAG"

# le trainer/diag attendent godot.log DANS le run (leçon g24)
cp "/tmp/critic_free_${TAG}.log" "$OUT/godot.log" 2>/dev/null
NREP=$(grep -c '"plan"' "$OUT/ep_0000.jsonl" 2>/dev/null)
NEXP=$(grep -c '"explore_target"' "$OUT/ep_0000.jsonl" 2>/dev/null)
gzip -f "$OUT/ep_0000.jsonl" 2>/dev/null
echo "[arb-collect] $TAG : $NREP replans, forcés ε-cible=$NEXP"
echo "ALL_DONE_ARB_${TAG}"
