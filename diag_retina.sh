#!/bin/zsh
# SANITY RÉTINE — étage 0 (GRATUIT, zéro entraînement). Vérifie que le raycast couleur fonctionne :
# une SEULE pastille rouge épinglée à distance connue → un rayon doit la toucher, lire (R haut, G/B bas)
# et une depth ≈ dist/10. Si aucun rayon ne touche / mauvaise couleur → bug géométrie/collider à corriger
# AVANT de payer quoi que ce soit (étage 1 tête, étage 2 WM). Le gait marche via hexapod_v2 (serveur policy).
# Usage: bash diag_retina.sh [food_radius=3]
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1
cd "$ROOT"
FR=${1:-3}
CKPT=data/checkpoints/hexapod_v2/policy_best.pt
[ -f "$CKPT" ] || CKPT=data/checkpoints/hexapod_v2/policy_latest.pt
PORT=6071
pkill -9 -f serve_ppo_collect 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1

( cd python && ../env_pytorch_3.12/bin/python -m scripts.serve_ppo_collect \
    --checkpoint "../$CKPT" --host 127.0.0.1 --port $PORT --seed 1 > /tmp/retina_srv.log 2>&1 ) &
SRV=$!
for i in $(seq 1 40); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done

SYLVAN_RETINA_DEBUG=1 \
SYLVAN_FOOD_COUNT=1 SYLVAN_FOOD_ANGLE_DEG=0 SYLVAN_FOOD_MIN_RADIUS=$FR SYLVAN_FOOD_SPAWN_RADIUS=$FR \
SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
SYLVAN_CPG_PERIOD=0.5 SYLVAN_CMD_VX=0.0 SYLVAN_CMD_OMEGA=0.0 \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=1 SYLVAN_MAX_EPISODE_STEPS=40 SYLVAN_SEED=1 SYLVAN_DISABLE_HOMEOSTASIS=1 \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 \
SYLVAN_RUN_DIR=data/replay_buffer/retina_diag \
./tools/godot/godot --path godot --headless > /tmp/retina_diag.log 2>&1
kill -9 $SRV 2>/dev/null
echo "=== [Retina] lines ==="
grep -a "\[Retina\]" /tmp/retina_diag.log
echo "=== errors (if any) ==="
grep -a -iE "error|push_error|SCRIPT ERROR|Invalid" /tmp/retina_diag.log | head -20
echo "done -> /tmp/retina_diag.log (food pinned at radius $FR, color expected RED ~0.9,0.3,0.2)"
