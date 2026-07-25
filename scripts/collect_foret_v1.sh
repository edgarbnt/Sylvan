#!/bin/bash
# COLLECTE MIXTE DU MONDE-FORÊT (foret_v1) — le corpus qui alimentera le retrain du WM.
#
# POURQUOI MIXTE, ET PAS DU BABILLAGE (§6sexies, désormais MESURÉ). Le dry-run du 2026-07-25 a
# collecté 818 ticks de babillage pur sur ce monde : ZÉRO repas, et le garde-fou de la matrice a
# REFUSÉ le corpus (« zéro consommation réelle — comportement dégénéré »). Une politique purement
# aléatoire n'attrape jamais rien, donc ne montre JAMAIS au WM ce qui se passe AU CONTACT — le seul
# moment intéressant. La couverture sans pertinence produit un WM qui connaît le vide.
# Inversement le planner seul ne visite qu'un couloir. D'où les TROIS parts :
#   * PLANNER  (pertinence) — visite les états qui comptent, dont le CONTACT. Sert le WM actuel comme
#     ÉCHAFAUDAGE de collecte : il est OOD en forêt, il forage imparfaitement, et ce n'est pas grave —
#     on lui demande d'amener le corps près des ressources, pas d'être bon.
#   * BABILLAGE (couverture) — l'éventail de vitesse complet + le regard, sur son flux dédié.
#   * EXPLORATION (couverture large) — virage élargi, pour que le WM voie aussi les régimes que ni le
#     planner ni le babillage nominal ne visitent.
#
# Le corpus sort en TROIS dossiers : `train_wm_command --runs a b c` les consomme ensemble. Un seul
# dossier serait écrasé (les épisodes sont numérotés à partir de 0 à chaque run).
#
# §6quinquies E : toute dimension d'ACTION nouvelle doit être EXPLORÉE, sinon le WM n'apprend pas sa
# dynamique. Le regard a son exploration propre (cadence + flux de hasard dédiés, G3) et tourne dans
# les TROIS parts. L'éventail de vitesse est babillé dans SYLVAN_WM_VX_MIN/MAX, posés par le preset.
#
# Usage : bash scripts/collect_foret_v1.sh <lives-total> <seed> [tag]
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

LIVES="${1:-60}"
SEED="${2:-11}"
TAG="${3:-foret_v1}"
STEPS="${STEPS:-3000}"
WM="${WM_CKPT:-data/checkpoints/wm_objcentric_kin/wm_best.pt}"
PORT="${PORT:-6067}"

# Répartition : la pertinence pèse le plus (c'est elle qui produit les CONTACTS, ce que le babillage
# ne produit jamais), la couverture complète le reste.
N_PLAN=$(( LIVES * 50 / 100 ))
N_BABL=$(( LIVES * 30 / 100 ))
N_EXPL=$(( LIVES - N_PLAN - N_BABL ))

export PYTHONPATH=python
export GODOT_BIN="$ROOT/tools/godot/godot"

# LE MONDE, depuis sa source de vérité unique (guillemets : la palette contient des « ; »).
eval "$(env_pytorch_3.12/bin/python -m sylvan.world --preset foret_v1 --env)"

echo "=== COLLECTE MIXTE foret_v1 | $LIVES vies x $STEPS ticks | graine $SEED ==="
echo "    planner $N_PLAN | babillage $N_BABL | exploration $N_EXPL"
echo "    corps kin_speed=$SYLVAN_KIN_SPEED éventail $SYLVAN_WM_VX_MIN..$SYLVAN_WM_VX_MAX"\
" coût $SYLVAN_SPEED_COST | drain $SYLVAN_ENERGY_DRAIN+$SYLVAN_THIRST_DRAIN | $SYLVAN_FOOD_PATCHES sites"

# Réglages communs aux trois parts. SYLVAN_WM_COLLECT écrit le bloc wm (cmd, torso, rétine) dont tout
# l'aval dépend ; sans lui le corpus est illisible pour le contrat comme pour l'entraînement.
COMMON="SYLVAN_COLLECT=1 SYLVAN_WM_COLLECT=1 SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.0 SYLVAN_TURN_FADE=0 \
SYLVAN_MAX_EPISODE_STEPS=$STEPS SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 \
SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0"

collect_babble() {   # $1=nom  $2=vies  $3=graine  $4=omega max
  local dir="$ROOT/data/replay_buffer/${TAG}_$1"
  [[ "$2" -le 0 ]] && { echo "  ($1 : 0 vie, ignoré)"; return 0; }
  echo "--- part $1 : $2 vies, omega max $4 -> $dir"
  rm -rf "$dir"; mkdir -p "$dir"
  env $COMMON SYLVAN_COLLECTOR_MODE=babbling SYLVAN_WM_WMAX="$4" \
      SYLVAN_NUM_EPISODES="$2" SYLVAN_SEED="$3" SYLVAN_RUN_DIR="$dir" \
      ./tools/godot/godot --path godot --headless > "/tmp/collect_${TAG}_$1.log" 2>&1
  echo "    -> $(ls "$dir"/*.jsonl 2>/dev/null | wc -l) épisodes"
}

collect_planner() {  # $1=vies  $2=graine
  local dir="$ROOT/data/replay_buffer/${TAG}_planner"
  [[ "$1" -le 0 ]] && { echo "  (planner : 0 vie, ignoré)"; return 0; }
  echo "--- part planner : $1 vies (échafaudage : WM actuel, OOD en forêt) -> $dir"
  rm -rf "$dir"; mkdir -p "$dir"
  pkill -9 -f serve_planner_command 2>/dev/null; sleep 1
  PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
    --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt \
    --host 127.0.0.1 --port "$PORT" --horizon 80 --replan-every 10 \
    > "/tmp/collect_${TAG}_planner_srv.log" 2>&1 &
  local srv=$!
  for _ in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done
  env $COMMON SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 \
      SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT="$PORT" \
      SYLVAN_NUM_EPISODES="$1" SYLVAN_SEED="$2" SYLVAN_RUN_DIR="$dir" \
      ./tools/godot/godot --path godot --headless > "/tmp/collect_${TAG}_planner.log" 2>&1
  kill -9 "$srv" 2>/dev/null
  echo "    -> $(ls "$dir"/*.jsonl 2>/dev/null | wc -l) épisodes"
}

collect_planner "$N_PLAN" "$SEED"
collect_babble  "babble" "$N_BABL" "$((SEED + 101))" 0.6
collect_babble  "explore" "$N_EXPL" "$((SEED + 202))" 1.0

echo
echo "=== CORPUS ==="
for part in planner babble explore; do
  d="data/replay_buffer/${TAG}_${part}"
  [[ -d "$d" ]] && echo "  $d : $(ls "$d"/*.jsonl 2>/dev/null | wc -l) épisodes, $(cat "$d"/*.jsonl 2>/dev/null | wc -l) ticks"
done
echo
echo "Vérifier AVANT d'entraîner :"
echo "  PYTHONPATH=python env_pytorch_3.12/bin/python -m sylvan.world --preset foret_v1 --env | sed 's/^export //; s/\"//g' > /tmp/foret_v1.env"
echo "  PYTHONPATH=python env_pytorch_3.12/bin/python diagnostics/diag_world_contract.py data/replay_buffer/${TAG}_planner --preset-file /tmp/foret_v1.env"
