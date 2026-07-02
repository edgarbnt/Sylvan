#!/bin/zsh
# Voir la SALAMANDRE ENTRAÎNÉE (cerveau = policy .pt sur le CPG). Démarre son PROPRE serveur.
# Usage :  bash scripts/voir_salamandre_cerveau.sh [omega] [checkpoint]
#   bash scripts/voir_salamandre_cerveau.sh 0                         # marche droite, dernier best
#   bash scripts/voir_salamandre_cerveau.sh 0.6                       # virage
#   bash scripts/voir_salamandre_cerveau.sh 0 data/checkpoints/sal_fwd/policy_best.pt
# Une fenêtre Godot s'ouvre. Ferme-la pour arrêter (le serveur est tué automatiquement).
cd /home/edgarbrunet/Documents/PERSO/SylvanV1
OM=${1:-0.0}
CKPT=${2:-data/checkpoints/hexapod_v2/policy_latest.pt}
[ -f "$CKPT" ] || CKPT=data/checkpoints/hexapod_v1/policy_latest.pt
PORT=6063
PF=$(mktemp); AF=$(mktemp)
echo "[voir] checkpoint = $CKPT | omega = $OM"
( cd python && ../env_pytorch_3.12/bin/python -m scripts.serve_ppo_collect \
    --checkpoint "../$CKPT" --host 127.0.0.1 --port $PORT --seed 1 \
    --port-file "$PF" --ack-file "$AF" > /tmp/voir_sal_srv.log 2>&1 ) &
SRV=$!
for i in $(seq 1 40); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done
trap "kill -9 $SRV 2>/dev/null" EXIT INT TERM
# HEXAPOD v2 base (2026-06-17): 6-leg tripod + speed-coupled cadence + grip 7 + symmetry-trained,
# residual gain 0.4, turn-fade OFF. CMD_VX 0.7 → ~0.49 m/s, cap droit (~3× the old quadruped).
SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 \
SYLVAN_CPG_SPEEDCAD=0.6 SYLVAN_FOOT_FRICTION=7.0 \
SYLVAN_CMD_VX=0.7 SYLVAN_CMD_OMEGA=$OM SYLVAN_CPG_PERIOD=0.5 \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=20 SYLVAN_MAX_EPISODE_STEPS=600 SYLVAN_SEED=1 SYLVAN_DISABLE_HOMEOSTASIS=1 \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 \
SYLVAN_RUN_DIR=data/replay_buffer/view \
./tools/godot/godot --path godot
