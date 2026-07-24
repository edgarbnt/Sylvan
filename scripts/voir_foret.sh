#!/bin/bash
# VOIR LA FORÊT — la configuration EXACTE qui a passé le gate d'arrangement (2026-07-24).
#
# Ce qui est à l'écran = ce que l'entité PERÇOIT, et rien d'autre :
#   * 45 massifs SOLIDES (bit 2) et PERCEPTIBLES (bit 7 + retina_color), vert foncé — teinte choisie
#     par la MESURE (fuite 0,0000 sur la requête rouge ; un tronc brun fuirait à 0,2271, donc un
#     tronc brun EST perceptuellement de la nourriture pour le slot) ;
#   * arrangés en 6 PEUPLEMENTS (processus de Neyman-Scott/Thomas) avec 3 CLAIRIÈRES.
#     Mesuré sur 24 tirages : Clark-Evans 0,794 ± 0,044 contre 1,054 ± 0,031 en tirage uniforme.
#     C'est ce groupement qui crée une occlusion NON uniforme — couloirs, écrans, ouvertures — donc
#     des positions qui ne se valent pas. Un semis uniforme n'est qu'un brouillard homogène.
#
# 🚨 DÉCOR VISUAL-ONLY ÉTEINT (SYLVAN_FOREST_DECOR=0), et c'est le point le plus important de ce
# script. forest_manager.gd dessine des arbres SANS collision et SANS couleur-rétine : l'entité ne
# les voit pas et les traverse. Les laisser à côté de la forêt solide rendrait l'image INDÉCHIFFRABLE
# — impossible de distinguer un arbre réel d'un arbre qui n'existe que pour l'observateur. Or ce
# visuel sert à JUGER (§2.1 : le visuel ne doit pas mentir). Mettre SYLVAN_FOREST_DECOR=1 pour
# comparer les deux, en sachant ce qu'on regarde.
#
# ⛔ LE REGARD N'EST PAS ACTIVÉ ICI, et ce n'est pas un oubli : il porte la proprioception à 133,
# alors que le WM servi en attend 132. Le montrer piloté par le planner exigerait le retrain. Le
# mécanisme du regard se vérifie séparément, sans fenêtre : diagnostics/diag_foret_g3_regard.py.
#
# Usage : bash scripts/voir_foret.sh            (fermer la fenêtre pour arrêter)
#         TREES=28 STANDS=4 bash scripts/voir_foret.sh
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1-foret; cd "$ROOT" || exit 1

TREES=${TREES:-45}          # 45 = plafond de navigabilité MESURÉ (54 → immobile 85 % du temps)
STANDS=${STANDS:-6}
CLEARINGS=${CLEARINGS:-3}
PORT=${PORT:-6291}
WM=${WM_CKPT:-data/checkpoints/wm_objcentric_kin/wm_best.pt}

# On ne tue QUE nos propres processus : une autre session peut travailler dans un worktree voisin.
pkill -9 -f "serve_planner_command.*$PORT" 2>/dev/null
pkill -9 -f "$ROOT/tools/godot" 2>/dev/null
sleep 1
export GODOT_BIN="$ROOT/tools/godot/godot"

env SYLVAN_PLANNER_COST=survival SYLVAN_PLANNER_DRAIN=0.0005 SYLVAN_PLANNER_RESTORE=0.4 \
    SYLVAN_PLANNER_HEADING_W=0.0 \
    PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
    --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt \
    --host 127.0.0.1 --port $PORT --horizon 80 --replan-every 10 \
    --egomotion-head data/checkpoints/egomotion_head/best.pt --slot-memory \
    > /tmp/voir_foret_srv.log 2>&1 &
SRV=$!
trap "kill -9 $SRV 2>/dev/null; pkill -9 -f \"$ROOT/tools/godot\" 2>/dev/null" EXIT INT TERM
for _ in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done
if ! ss -ltn 2>/dev/null | grep -q ":$PORT"; then
  echo "[voir-foret] le serveur planner n'a pas démarré — voir /tmp/voir_foret_srv.log"; tail -20 /tmp/voir_foret_srv.log; exit 1
fi
echo "[voir-foret] serveur prêt (WM=$WM). Fenêtre Godot : $TREES massifs, $STANDS peuplements, $CLEARINGS clairières."
echo "[voir-foret] décor visual-only ÉTEINT — tout ce qui est dessiné est perçu par l'entité."

# Fenêtré (PAS de --headless) => temps réel, regardable.
env SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 \
  SYLVAN_CPG_SPEEDCAD=0.6 SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 \
  SYLVAN_KINEMATIC=1 SYLVAN_KIN_SPEED=${KIN_SPEED:-0.8} SYLVAN_KIN_TURN=${KIN_TURN:-1.5} \
  SYLVAN_WOLF=1 SYLVAN_WOLF_SCALE=${WOLF_SCALE:-0.4} SYLVAN_WOLF_Y=${WOLF_Y:--0.30} \
  SYLVAN_WOLF_LIGHTEN=${WOLF_LIGHTEN:-0.35} \
  SYLVAN_FOREST_DECOR=${SYLVAN_FOREST_DECOR:-0} \
  SYLVAN_FOREST_COUNT=$TREES SYLVAN_FOREST_STANDS=$STANDS SYLVAN_FOREST_CLEARINGS=$CLEARINGS \
  SYLVAN_FOREST_CLEARING_R=${CLEARING_R:-4.0} SYLVAN_FOREST_STAND_SIGMA=${STAND_SIGMA:-3.0} \
  SYLVAN_EAT_RADIUS=1.0 SYLVAN_DRINK_RADIUS=1.0 \
  SYLVAN_FOOD_COUNT=3 SYLVAN_WATER_COUNT=3 SYLVAN_ENERGY_DRAIN=0.05 SYLVAN_THIRST_DRAIN=0.05 \
  SYLVAN_INIT_ENERGY=70 SYLVAN_INIT_THIRST=70 \
  SYLVAN_FOOD_MIN_RADIUS=2.0 SYLVAN_FOOD_SPAWN_RADIUS=6.0 \
  SYLVAN_WATER_MIN_RADIUS=2.0 SYLVAN_WATER_SPAWN_RADIUS=6.0 \
  SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=10 SYLVAN_MAX_EPISODE_STEPS=3000 SYLVAN_SEED=${SEED:-1} \
  SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
  SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
  SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR=/tmp/voir_foret_run \
  ./tools/godot/godot --path godot
