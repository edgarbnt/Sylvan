#!/bin/zsh
# JUGE closed-loop du CRITIQUE-SPRINT (docs/design_critique_sprint.md §gates, PRÉ-ENREGISTRÉ) :
# 2×24 vies monde v2, seeds 1+2 (propriété du juge), bras apprenant SYLVAN_WP_SPRINT_CRITIC,
# oracle OFF, ε=0 (déploiement déterministe). Réfs MESURÉES (pas de re-run) : géométrie 34 repas/
# 11 morts poolés ; plafond oracle 47/9. PASS = repas poolés ≥ 42 ET morts poolées ≤ 13.
# KILL précoce = premier seed < géométrie−5 repas. Même harnais/env que la collecte (parité).
#
# Usage : bash scripts/judge_sprint_critic_v2.sh [seed=1] [tag=judge1] [ckpt=data/checkpoints/sprint_critic/sprint_best.pt]
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
SEED=${1:-1}; TAG=${2:-judge${SEED}}; CKPT=${3:-data/checkpoints/sprint_critic/sprint_best.pt}
OUT="data/replay_buffer/critic_kin_${TAG}"

[[ -f "$CKPT" ]] || { echo "[judge] ckpt introuvable : $CKPT"; exit 1; }

export WM_CKPT=data/checkpoints/wm_objcentric_kin_haz/wm_best.pt
export SYLVAN_HAZARD_COUNT=1
export SYLVAN_HAZARD_ENGULF_P=0.5
export SYLVAN_HEALTH_REGEN=0.05
export SYLVAN_WAYPOINT=1
export SYLVAN_WAYPOINT_DEBUG=1
export SYLVAN_WP_LOG="$OUT"
export SYLVAN_WP_SPRINT_CRITIC="$CKPT"
export SYLVAN_WP_EXPLORE_EPS=0
# garde-fous : le juge mesure l'APPRENANT seul (ni oracle, ni douleur-scoreur)
unset SYLVAN_WP_ORACLE_SPRINT SYLVAN_WP_PAIN_CRITIC

bash scripts/collect_critic_corpus_kin.sh 24 "$SEED" "$TAG"

cp "/tmp/critic_free_${TAG}.log" "$OUT/godot.log" 2>/dev/null
gzip -f "$OUT/ep_0000.jsonl" 2>/dev/null
echo "[judge] bras apprenant seed=$SEED → $OUT (parse : diag_hazard_gate.parse_lives sur $OUT/godot.log)"
echo "ALL_DONE_JUDGE_${TAG}"
