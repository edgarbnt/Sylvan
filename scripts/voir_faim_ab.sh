#!/bin/bash
# VOIR le forage du monde-forêt, dans la CONFIG EXACTEMENT TESTÉE par gate_foret_closedloop.sh.
#
# POURQUOI CE SCRIPT plutôt que voir_foret.sh : `voir_foret.sh` sert en plus `--slot-memory` et
# `--egomotion-head`, et un autre WM par défaut. Il montrerait donc une AUTRE entité que celle dont
# on vient de publier les chiffres — un visuel qui sert à juger ne doit pas montrer un cousin.
# Ici : mêmes flags, même WM, même coût, même horizon que le gate. Ce que tu vois EST ce qu'on a mesuré.
#
# LES DEUX BRAS (2026-08-02, chantier « perception de la faim apprise ») :
#   ARM=a  la nourriture est reconnue par une RÈGLE DE COULEUR écrite en dur (« ce qui est rouge »)
#   ARM=b  elle est reconnue par un petit réseau qui l'a APPRISE en observant ce qu'elle a mangé
# Les deux se valent sur mesure directe ; le comportement, lui, n'a pas pu être départagé
# (36 vies par bras, aucune différence significative). D'où ce visuel : regarder ce que les
# chiffres ne tranchent pas.
#
# CE QU'IL FAUT REGARDER (le goulot est le MÊME dans les deux bras) :
#   1. VISE-T-ELLE À CÔTÉ ? l'erreur de visée mesurée est ~23°, soit 1,2 m d'écart à 3 m alors que
#      sa bouche fait 1 m. Si elle frôle la nourriture sans la prendre, c'est ÇA qu'on voit.
#   2. POURSUIT-ELLE UN FANTÔME ? 6 ticks sur 10, aucun rayon ne touche vraiment la cible : elle
#      devine une position. Si elle fonce vers une zone vide, c'est ça.
#   3. HÉSITE-T-ELLE entre manger et boire ? la plupart des morts sont des morts de SOIF (18/36).
#   4. EST-ELLE BLOQUÉE par les arbres ? (elle les traverse pas : ils sont solides)
#
# Usage :  bash scripts/voir_faim_ab.sh          # bras a (règle de couleur)
#          ARM=b bash scripts/voir_faim_ab.sh    # bras b (perception apprise)
#          VIES=3 bash scripts/voir_faim_ab.sh
# Fermer la fenêtre pour arrêter (le serveur est tué automatiquement).
set +e
cd "$(dirname "$0")/.." || exit 1; ROOT="$(pwd)"

ARM=${ARM:-a}
PORT=${PORT:-6391}
VIES=${VIES:-6}
PAS=${PAS:-3000}
WM=${WM_CKPT:-data/checkpoints/wm_foret_v2_slot/wm_best.pt}
SAL=data/checkpoints/drive_saliency_food/saliency_best.pt

# On ne tue QUE nos propres processus : une autre session peut travailler à côté.
pkill -9 -f "serve_planner_command.*$PORT" 2>/dev/null
pkill -9 -f "$ROOT/tools/godot" 2>/dev/null
sleep 1
export GODOT_BIN="$ROOT/tools/godot/godot"

# Le monde vient du PRESET GELÉ (une seule source de vérité), exactement comme le gate.
eval "$(PYTHONPATH=python ./env_pytorch_3.12/bin/python -m sylvan.world --preset foret_v1 --env)" \
  || { echo "[voir-faim] preset foret_v1 introuvable"; exit 1; }

if [ "$ARM" = "b" ]; then
  if [ ! -f "$SAL" ]; then
    echo "[voir-faim] ❌ tête apprise absente : $SAL"; exit 1
  fi
  export SYLVAN_SLOT_DRIVE_SALIENCY="0:$SAL"
  echo "[voir-faim] BRAS B — la nourriture est reconnue par le réseau APPRIS (règle de couleur court-circuitée)"
else
  unset SYLVAN_SLOT_DRIVE_SALIENCY
  echo "[voir-faim] BRAS A — la nourriture est reconnue par la RÈGLE DE COULEUR écrite en dur"
fi

# Coût survie : sans lui le planner ne connaît que la faim et meurt de soif par construction.
# Flags STRICTEMENT ceux du gate (pas de slot-memory, pas d'egomotion-head).
env SYLVAN_PLANNER_COST=survival SYLVAN_PLANNER_DRAIN=0.0005 SYLVAN_PLANNER_RESTORE=0.4 \
    SYLVAN_PLANNER_HEADING_W=0.0 \
    PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
    --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt \
    --host 127.0.0.1 --port $PORT --horizon 80 --replan-every 10 \
    > "/tmp/voir_faim_${ARM}_srv.log" 2>&1 &
SRV=$!
trap "kill -9 $SRV 2>/dev/null; pkill -9 -f \"$ROOT/tools/godot\" 2>/dev/null" EXIT INT TERM
for _ in $(seq 1 90); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done
if ! ss -ltn 2>/dev/null | grep -q ":$PORT"; then
  echo "[voir-faim] le serveur n'a pas démarré — voir /tmp/voir_faim_${ARM}_srv.log"
  tail -20 "/tmp/voir_faim_${ARM}_srv.log"; exit 1
fi
# PREUVE À L'ÉCRAN de ce qui est réellement servi : sans ça, un bras B silencieusement inactif
# ressemblerait à « aucune différence » — exactement le faux verdict qu'on veut éviter.
grep -m1 "SAILLANCE APPRISE" "/tmp/voir_faim_${ARM}_srv.log" \
  || [ "$ARM" = "a" ] || { echo "[voir-faim] ❌ bras B demandé mais la tête n'est PAS chargée"; exit 1; }
echo "[voir-faim] serveur prêt (WM=$WM) — $VIES vies x $PAS pas. Ferme la fenêtre pour arrêter."

# Fenêtré (PAS de --headless) => temps réel, regardable.
# ⚠️ CE BLOC EST CELUI DE voir_foret.sh, À L'IDENTIQUE — ne pas le « simplifier ». Deux pièges
# vérifiés le 2026-08-02 en le réécrivant de mémoire :
#   · SYLVAN_COLLECT=0 rend l'entité IMMOBILE : c'est ce flag qui arme la boucle qui interroge le
#     planner. À 0, elle ne demande jamais quoi faire et reste plantée.
#   · et comme le maillage du loup est chargé DANS le pas cinématique (sylvan_agent.gd:793), une
#     entité immobile n'a pas de loup non plus — un seul défaut, deux symptômes trompeurs.
env SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 \
  SYLVAN_CPG_SPEEDCAD=0.6 SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 \
  SYLVAN_KINEMATIC=1 \
  SYLVAN_WOLF=1 SYLVAN_WOLF_SCALE=${WOLF_SCALE:-0.4} SYLVAN_WOLF_Y=${WOLF_Y:--0.30} \
  SYLVAN_WOLF_LIGHTEN=${WOLF_LIGHTEN:-0.35} \
  SYLVAN_FOREST_DECOR=0 \
  SYLVAN_FOREST_MESH=${SYLVAN_FOREST_MESH:-1} \
  SYLVAN_FOREST_UNDERGROWTH=${SYLVAN_FOREST_UNDERGROWTH:-75} \
  SYLVAN_FOREST_CLEARING_R=${CLEARING_R:-4.0} SYLVAN_FOREST_STAND_SIGMA=${STAND_SIGMA:-3.0} \
  SYLVAN_FOOD_BUSH=${SYLVAN_FOOD_BUSH:-0} \
  SYLVAN_INIT_ENERGY=${INIT_ENERGY:-70} \
  SYLVAN_COLLECT=1 SYLVAN_WM_COLLECT=1 SYLVAN_RUN_DIR=/tmp/voir_faim_run \
  SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
  SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
  SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 \
  SYLVAN_NUM_EPISODES="$VIES" SYLVAN_MAX_EPISODE_STEPS="$PAS" SYLVAN_SEED="${SEED:-1}" \
  ./tools/godot/godot --path godot
