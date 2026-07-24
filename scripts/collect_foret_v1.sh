#!/bin/bash
# COLLECTE DU MONDE-FORÊT (foret_v1) — le monde complet de docs/design_foret_complete.md.
#
# POURQUOI CE SCRIPT EXISTE. Le monde ne se décrit plus par ~44 variables recopiées à la main dans
# chaque harnais : il vient du PRESET GELÉ `sylvan.world.FORET_V1`, qui les émet toutes et les fait
# atteindre les DEUX consommateurs (Godot ET le serveur planner). Un réglage recopié est un réglage
# qui divergera ; celui-ci ne peut pas.
#
# POURQUOI GODOT EN DIRECT, ET PAS `scripts.collect_wm_data`. Cette dernière lève un SERVEUR DE
# POLITIQUE dont l'entrée est dimensionnée sur la proprioception d'avant (132). Avec le regard, la
# proprioception fait 133 : le serveur recevrait une observation qu'il ne sait pas lire. En
# babillage la politique ne pilote rien de toute façon (les commandes viennent du tirage de main.gd),
# donc on lance Godot directement — la route que les sondes G4 à G9 ont déjà éprouvée avec le regard.
#
# ⚠️ Ce script COLLECTE. Ne l'utiliser pour de vrai qu'après un dry-run VERT
# (bash scripts/dryrun_foret_v1.sh) : la combinaison des 9 briques n'a jamais tourné ensemble.
#
# Usage : bash scripts/collect_foret_v1.sh <run-dir> <episodes> <seed> [max-steps]
set -euo pipefail
cd "$(dirname "$0")/.."

# CHEMIN ABSOLU OBLIGATOIRE. Godot tourne avec --path godot : un SYLVAN_RUN_DIR RELATIF se résout
# depuis le dossier du projet Godot, pas depuis la racine du dépôt. Mesuré : le corpus atterrissait
# dans godot/data/replay_buffer/ pendant que tout l'aval le cherchait dans data/replay_buffer/ —
# aucune erreur, juste un corpus introuvable et un dossier vide à l'endroit attendu.
RUN_DIR="$(realpath -m "${1:-data/replay_buffer/foret_v1_smoke}")"
EPS="${2:-4}"
SEED="${3:-7}"
STEPS="${4:-3000}"

export PYTHONPATH=python

# LE MONDE, depuis sa source de vérité unique (guillemets : la palette contient des « ; »).
eval "$(env_pytorch_3.12/bin/python -m sylvan.world --preset foret_v1 --env)"

# LE RÉGIME DE COLLECTE (≠ le monde) : babillage de commandes sur l'ÉVENTAIL COMPLET, que le preset
# a déjà posé dans SYLVAN_WM_VX_MIN/MAX. §6quinquies E : toute dimension d'ACTION nouvelle doit être
# EXPLORÉE pendant la collecte, sinon le WM n'apprend jamais sa dynamique et la capacité reste inerte.
# Le regard a sa propre exploration (cadence et flux de hasard dédiés, cf. G3).
export SYLVAN_COLLECT=1 SYLVAN_WM_COLLECT=1 SYLVAN_COLLECTOR_MODE=babbling
export SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.0 SYLVAN_TURN_FADE=0
export SYLVAN_WM_WMAX=0.6
export SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0
export SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0
export SYLVAN_NUM_EPISODES="$EPS" SYLVAN_MAX_EPISODE_STEPS="$STEPS"
export SYLVAN_SEED="$SEED" SYLVAN_RUN_DIR="$RUN_DIR"

echo "=== MONDE foret_v1 | $EPS vies x $STEPS ticks | graine $SEED | -> $RUN_DIR ==="
echo "    corps kin_speed=$SYLVAN_KIN_SPEED | éventail vx $SYLVAN_WM_VX_MIN..$SYLVAN_WM_VX_MAX"\
" coût $SYLVAN_SPEED_COST | drain $SYLVAN_ENERGY_DRAIN + soif $SYLVAN_THIRST_DRAIN"

mkdir -p "$RUN_DIR"
./tools/godot/godot --path godot --headless
