#!/bin/zsh
# VISUEL 🅑 — VOIR l'agent forager en planifiant DANS LE LATENT (coût-valeur, coordonnées débranchées,
# symétrisation à l'inférence). Comme run_forage_latent.sh MAIS Godot en FENÊTRE (pas --headless).
# Usage: bash voir_forage_latent.sh [eat_radius=1.0] [horizon=300] [episodes=3]
set +e
ER=${1:-1.0}; HZ=${2:-220}; NEP=${3:-3}   # horizon 220 (vs 300 headless) = compromis FPS/portée pour le VISUEL
# Config gagnante 🅑 (2026-06-21) : WM symétrisé + value-rêve multi-pas + agrégat MEAN (cf run_forage_latent.sh).
WM=${WM_CKPT:-data/checkpoints/wm_rich_fidele_sym/wm_best.pt}
VALUE=${VALUE_CKPT:-data/checkpoints/value_head_food_dream/value_best.pt}
export SYLVAN_VALUE_AGG=${SYLVAN_VALUE_AGG:-mean}
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
echo "🅑 VISUEL WM=$WM value=$VALUE eat_radius=$ER horizon=$HZ episodes=$NEP"

SYLVAN_PLANNER_THREADS=${SYLVAN_PLANNER_THREADS:-12} \
PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
  --wm "$WM" \
  --residual data/checkpoints/hexapod_v2/policy_best.pt \
  --value-head "$VALUE" \
  --host 127.0.0.1 --port 6052 --horizon $HZ --replan-every 12 > /tmp/planner_latent.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ':6052' && break; sleep 1; done

# PAS de --headless → fenêtre Godot visible. Moins d'épisodes, mêmes réglages moteur/perception que le headless.
SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 SYLVAN_EAT_RADIUS=$ER \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=2000 SYLVAN_SEED=1 SYLVAN_FOOD_COUNT=${FC:-6} \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=6052 \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 \
SYLVAN_RUN_DIR=data/replay_buffer/forage_latent_vis \
./tools/godot/godot --path godot > /tmp/forage_latent_vis.log 2>&1
kill -9 $SRV 2>/dev/null
echo "fenêtre fermée. planner log: /tmp/planner_latent.log"
