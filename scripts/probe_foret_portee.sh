#!/bin/bash
# SONDE DE PORTÉE (G11) — le trajet par repas SOUS UNE POLITIQUE, forêt vs monde-ancien.
#
# POURQUOI (dry-run 2026-07-25). Le babillage ne mange jamais → le trajet RÉEL par repas, le terme
# décisif de la calibration de densité, était non mesurable. Une POLITIQUE forage : on sert le
# planner du WM ACTUEL et on LIT ce qu'il parcourt entre deux repas. C'est cheap (bien moins qu'une
# vraie collecte) et ça donne le chiffre que G2 devinait.
#
# TROIS HONNÊTETÉS CÂBLÉES DANS LE PROTOCOLE (retours du pair) :
#  1. BORNE HAUTE. Le WM est entraîné sur l'ancien monde : en forêt il est OOD (occlusion → slot vidé
#     → errance), donc il GONFLE le trajet. Le WM re-entraîné fera mieux.
#  2. ANCRE. On mesure le MÊME WM, la MÊME nourriture, la MÊME politique dans DEUX mondes (forêt ON /
#     OFF). Le ratio isole la forêt+terrain ; l'OOD commun (types, proies) se simplifie.
#  3. CORPS DU WM, PAS L'ÉVENTAIL. kin_speed 0,8 + grille 0,55-0,75 (ce que le WM connaît), pas kin
#     2,83 ni l'éventail : sinon on empile un second OOD. Le trajet est géométrique (mètres), il
#     transfère ; la conversion mètres→énergie se fait in vivo après le retrain.
#
# Usage : bash scripts/probe_foret_portee.sh [episodes=4] [max_steps=3000]
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

NEP="${1:-4}"
MS="${2:-3000}"
WM="${WM_CKPT:-data/checkpoints/wm_objcentric_kin/wm_best.pt}"
PORT="${PORT:-6061}"
FOREST_DIR="$ROOT/data/replay_buffer/probe_portee_foret"
ANCHOR_DIR="$ROOT/data/replay_buffer/probe_portee_ancre"

export PYTHONPATH=python
export GODOT_BIN="$ROOT/tools/godot/godot"

# LE MONDE, depuis sa source unique — mais on RETIRE ce qui empilerait un second OOD (nuance 3) et
# ce qui confond le trajet-FOOD (l'eau). Le corps redevient celui du WM ; la nourriture reste celle
# de foret_v1 (12 bosquets, 4 types, proies), IDENTIQUE dans les deux conditions.
eval "$(env_pytorch_3.12/bin/python -m sylvan.world --preset foret_v1 --env)"
unset SYLVAN_SPEED_COST SYLVAN_WM_VX_MIN SYLVAN_WM_VX_MAX SYLVAN_PLANNER_VX_GRID   # pas l'éventail
unset SYLVAN_GAZE                                                                  # WM à 132
unset SYLVAN_THIRST_DRAIN SYLVAN_WATER_COUNT SYLVAN_WATER_PATCHES \
      SYLVAN_WATER_REGROW SYLVAN_DRINK_RADIUS SYLVAN_INIT_THIRST SYLVAN_WATER_PUDDLE_PERIOD  # food-only
export SYLVAN_KIN_SPEED=0.8 SYLVAN_KIN_TURN=1.5                                    # le corps du WM
# Métabolisme DOUX + réservoir plein : on veut BEAUCOUP de repas pour un trajet/repas robuste. Le
# drain ne biaise pas la grandeur géométrique (mètres entre deux repas) ; il ne fait que régler le
# nombre d'échantillons. IDENTIQUE dans les deux conditions.
export SYLVAN_ENERGY_DRAIN=0.05 SYLVAN_INIT_ENERGY=100.0

run_condition() {
  local tag="$1" run_dir="$2" forest_on="$3"
  echo "=== CONDITION $tag | WM=$WM | corps kin_speed=0.8 | $NEP vies x $MS ticks ==="
  pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
  rm -rf "$run_dir"; mkdir -p "$run_dir"

  PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
    --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt \
    --host 127.0.0.1 --port "$PORT" --horizon 80 --replan-every 10 \
    > "/tmp/probe_${tag}_srv.log" 2>&1 &
  local srv=$!
  for _ in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done

  # Les massifs, le terrain et les distracteurs ne sont servis QUE si forest_on=1. La nourriture, le
  # corps et le métabolisme sont déjà dans l'environnement exporté, IDENTIQUES des deux côtés.
  local forest_env=""
  if [[ "$forest_on" == "1" ]]; then
    forest_env="SYLVAN_FOREST_COUNT=$SYLVAN_FOREST_COUNT SYLVAN_FOREST_STANDS=$SYLVAN_FOREST_STANDS \
SYLVAN_FOREST_CLEARINGS=$SYLVAN_FOREST_CLEARINGS SYLVAN_FOREST_APPEARANCE_VAR=$SYLVAN_FOREST_APPEARANCE_VAR \
SYLVAN_TERRAIN_SLOW=$SYLVAN_TERRAIN_SLOW SYLVAN_TERRAIN_RADIUS=$SYLVAN_TERRAIN_RADIUS \
SYLVAN_TERRAIN_FLOOR=$SYLVAN_TERRAIN_FLOOR SYLVAN_DISTRACTOR_COUNT=$SYLVAN_DISTRACTOR_COUNT"
  fi
  # Sans forêt, on efface les clés forêt de l'environnement pour qu'aucune ne fuie dans l'ANCRE.
  env -u SYLVAN_FOREST_COUNT -u SYLVAN_FOREST_STANDS -u SYLVAN_FOREST_CLEARINGS \
      -u SYLVAN_FOREST_APPEARANCE_VAR -u SYLVAN_TERRAIN_SLOW -u SYLVAN_TERRAIN_RADIUS \
      -u SYLVAN_TERRAIN_FLOOR -u SYLVAN_DISTRACTOR_COUNT \
      $forest_env \
      SYLVAN_KINEMATIC=1 SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 \
      SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 SYLVAN_EAT_RADIUS=1.0 \
      SYLVAN_COLLECT=1 SYLVAN_WM_COLLECT=1 SYLVAN_NUM_EPISODES="$NEP" SYLVAN_MAX_EPISODE_STEPS="$MS" SYLVAN_SEED=1 \
      SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT="$PORT" \
      SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
      SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR="$run_dir" \
      ./tools/godot/godot --path godot --headless > "/tmp/probe_${tag}_free.log" 2>&1
  kill -9 "$srv" 2>/dev/null
  echo "    corpus -> $run_dir  ($(ls "$run_dir"/*.jsonl 2>/dev/null | wc -l) épisodes)"
}

run_condition "foret" "$FOREST_DIR" 1
run_condition "ancre" "$ANCHOR_DIR" 0

echo
echo "=== VERDICT G11 ==="
PYTHONPATH=python ./env_pytorch_3.12/bin/python diagnostics/diag_foret_g11_portee.py \
  "$FOREST_DIR" "$ANCHOR_DIR"
