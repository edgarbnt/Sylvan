#!/bin/zsh
# Collecte ε du CRITIQUE-SPRINT — monde v2 (bouffe au cœur du danger), étage waypoint + exploration.
# docs/design_critique_sprint.md Phase B : politique de base = GÉOMÉTRIE (oracle OFF, sprint-critic
# OFF — le déploiement y retombe à g=0) + ε uniforme au MANAGER (contrefactuels : sprint blessé,
# traversées profondes… que le corpus g24 n'a pas — 0% ε mesuré). Seeds 3+4 (tranché owner :
# les seeds 1+2 restent la propriété du juge).
#
# Env RÉPLIQUÉ des runs g24 (bannières /tmp/critic_srv_g24*.log, 2026-07-16) : harnais
# collect_critic_corpus_kin.sh tel quel + WM 3-slots (wm_objcentric_kin_haz, SLOT-2 vient du meta
# du ckpt) + monde v2 de référence (HAZARD_COUNT=1, ENGULF_P=0.5, HEALTH_REGEN=0.05 ;
# dégâts/rayon/frac = défauts verrouillés) ; margin_w=200 = défaut du planner, rien à poser.
#
# Usage : bash scripts/collect_sprint_corpus_v2.sh [ep=24] [seed=3] [tag=spx3]
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
NEP=${1:-24}; SEED=${2:-3}; TAG=${3:-spx${SEED}}
OUT="data/replay_buffer/critic_kin_${TAG}"

export WM_CKPT=data/checkpoints/wm_objcentric_kin_haz/wm_best.pt
export SYLVAN_HAZARD_COUNT=1
export SYLVAN_HAZARD_ENGULF_P=0.5
export SYLVAN_HEALTH_REGEN=0.05
export SYLVAN_WAYPOINT=1
export SYLVAN_WAYPOINT_DEBUG=1
export SYLVAN_WP_LOG="$OUT"
export SYLVAN_WP_EXPLORE_EPS=${EPS:-0.15}
export SYLVAN_WP_EXPLORE_SEED=$SEED
# garde-fous : jamais d'oracle ni de critique pendant la collecte ε (base = géométrie pure)
unset SYLVAN_WP_ORACLE_SPRINT SYLVAN_WP_SPRINT_CRITIC SYLVAN_WP_PAIN_CRITIC

bash scripts/collect_critic_corpus_kin.sh "$NEP" "$SEED" "$TAG"

# le trainer/diag attendent godot.log DANS le run (leçon g24 : logs restés dans gate_logs)
cp "/tmp/critic_free_${TAG}.log" "$OUT/godot.log" 2>/dev/null
gzip -f "$OUT/ep_0000.jsonl" 2>/dev/null
NDEC=$(wc -l < "$OUT/decisions.jsonl" 2>/dev/null)
NEXP=$(grep -c '"explore": true' "$OUT/decisions.jsonl" 2>/dev/null)
NHAZ=$(grep -c '\[hazard\]' "$OUT/godot.log" 2>/dev/null)
echo "[sprint-collect] $TAG : $NDEC décisions, ε=$NEXP ($((100 * ${NEXP:-0} / ${NDEC:-1}))%), lignes hazard=$NHAZ"
echo "ALL_DONE_SPRINT_${TAG}"
