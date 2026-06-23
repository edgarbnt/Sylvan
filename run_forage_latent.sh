#!/bin/zsh
# 🅑-PUR CLOSED-LOOP — forager en planifiant DANS LE LATENT : coût-valeur (tête apprise V sur les latents
# RÊVÉS du WM-rétine), COORDONNÉES DÉBRANCHÉES (pas de food_xz/radar/min_dist). La bouffe n'existe que dans
# ce que le WM perçoit (rétine→latent) + ce que la tête de valeur en lit. C'est le jalon 🅑 (JEPA-pur).
# Compare à la baseline COORDONNÉES = run_forage_retina.sh (même corps, même WM-rétine famille).
#
# Usage: bash run_forage_latent.sh [eat_radius=1.0] [horizon=80] [episodes=8]
set +e
ER=${1:-1.0}; HZ=${2:-300}; NEP=${3:-8}   # horizon LONG (300) : le rêve fidèle atteint la bouffe (~1.5 m) →
                                          # Vmax devient discriminant (sinon V plat → 'tourne dans le vide')
# CONFIG GAGNANTE 🅑 (2026-06-21, hypothèse TRANSFERT validée par A/B 0→3 repas/4 ép) : WM symétrisé +
# value entraînée sur les latents RÊVÉS multi-pas (value_head_food_dream) + agrégat MEAN (calibrée à
# toutes profondeurs → la moyenne débruite ; .max sur-récompense un seul pic). L'ancien teacher-forced+max
# ne closait JAMAIS (0 repas, min food_d 1.5+). Voir docs/BUG_OUVERT_close_latent.md (close RÉSOLU).
WM=${WM_CKPT:-data/checkpoints/wm_rich_fidele_sym/wm_best.pt}
VALUE=${VALUE_CKPT:-data/checkpoints/value_head_food_dream/value_best.pt}
export SYLVAN_VALUE_AGG=${SYLVAN_VALUE_AGG:-mean}   # agrégat du score-valeur sur l'horizon (mean = bon pour value-rêve)
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
echo "🅑 LATENT WM=$WM value=$VALUE eat_radius=$ER horizon=$HZ episodes=$NEP"

PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
  --wm "$WM" \
  --residual data/checkpoints/hexapod_v2/policy_best.pt \
  --value-head "$VALUE" \
  --host 127.0.0.1 --port 6052 --horizon $HZ --replan-every 10 > /tmp/planner_latent.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ':6052' && break; sleep 1; done

SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 SYLVAN_EAT_RADIUS=$ER \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=1500 SYLVAN_SEED=1 SYLVAN_FOOD_COUNT=${FC:-6} \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=6052 \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 \
SYLVAN_RUN_DIR=data/replay_buffer/forage_latent \
./tools/godot/godot --path godot --headless > /tmp/forage_latent.log 2>&1
kill -9 $SRV 2>/dev/null
echo "=== survie par épisode (dernier Step de chaque épisode) ==="
grep -a "\[Godot\] Episode" /tmp/forage_latent.log | awk '{for(i=1;i<=NF;i++){if($i=="Episode")e=$(i+1);if($i=="Step")s=$(i+1)}; key=e; last[key]=s} END{for(k in last) print "ep"k" steps="last[k]}' | sort -t= -k2 -n
echo "=== planner errors? ==="; grep -a -iE "error|traceback|plan_latent" /tmp/planner_latent.log | head
echo "done -> /tmp/forage_latent.log + /tmp/planner_latent.log"
