#!/bin/zsh
# Juge G3 du CRITIQUE-ARBITRAGE (docs/design_critique_arbitrage.md — G-mono-v2 tranché owner
# 2026-07-20). 2 bras × 2 seeds × 24 vies, monde v2, config VIVANTE waypoint (lunette saillance +
# sprint-critic décontaminé), ε OFF. Bras `ref` = choix de cible DESIGNÉ (coût survie) ;
# bras `arb` = + SYLVAN_ARB_CRITIC (forme pinnée D1). La réf est RE-MESURÉE AVANT le bras appris
# et le PASS chiffré à partir d'elle (formule pré-enregistrée au doc). Les métriques (conso
# poolées, morts totales/danger, morts-par-arbitrage) se parsent du BC_LOG des runs avec
# diagnostics/diag_arbitrage_g0.py --runs (MÊME parseur pour les deux bras).
# Usage : bash scripts/judge_arb_critic.sh <ref|arb> <seed 1|2>
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
ARM=${1:-ref}; SEED=${2:-1}; TAG="arbj_${ARM}_s${SEED}"
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
# juge = déploiement déterministe : aucune exploration, aucun oracle
unset SYLVAN_WP_EXPLORE_EPS SYLVAN_TARGET_EXPLORE_EPS SYLVAN_WP_ORACLE_SPRINT SYLVAN_WP_PAIN_CRITIC
if [[ "$ARM" == "arb" ]]; then
  export SYLVAN_ARB_CRITIC=data/checkpoints/arb_critic/arb_best.pt
else
  unset SYLVAN_ARB_CRITIC
fi

bash scripts/collect_critic_corpus_kin.sh 24 "$SEED" "$TAG"

cp "/tmp/critic_free_${TAG}.log" "$OUT/godot.log" 2>/dev/null
NARB=$(grep -c '"arb"' "$OUT/ep_0000.jsonl" 2>/dev/null)
gzip -f "$OUT/ep_0000.jsonl" 2>/dev/null
echo "[judge-arb] $TAG : replans arb-décidés=$NARB"
echo "ALL_DONE_JUDGE_${TAG}"
