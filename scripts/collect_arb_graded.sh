#!/bin/zsh
# RÉOUVERTURE DU CHANTIER ARBITRAGE sur hypothèse NOUVELLE (2026-07-21) : les 3 échecs passés
# (critique G3, commitment, remise-capée) portaient peut-être sur une TÂCHE DÉGÉNÉRÉE — avec des
# drains identiques, l'écart d'urgence ne prend que 2 valeurs (0 ou ~40 : 49,9 % / 0,3 % / 49,8 %),
# donc aucun régime ne récompense un arbitrage subtil.
# Ce harnais collecte les DEUX bras à comparer :
#   sym    = drains 0.05/0.05, planner symétrique (le monde dégénéré = statu quo)
#   graded = drains 0.05/0.035 + planner drain_t=0.00035 (zone grise 41,9 % + modèle interne HONNÊTE)
#
# ⚠️ MÉTRIQUES VALIDES = TAUX seulement. Le bras gradué a MOINS de drain total (soif 30 % plus lente)
# donc un monde plus FACILE → comparer survie/consommations entre bras serait CONFONDU (§2).
# On compare : part des morts imputable à l'ARBITRAGE, flottement PAR consommation, distribution des
# écarts d'urgence. QUESTION PRÉ-INSCRITE : le déficit d'arbitrage PERSISTE-t-il en tâche graduée ?
#   - il S'EFFONDRE  → la tâche dégénérée était le problème ; le coût designé suffit une fois graduée ;
#                       rien à apprendre → les 3 échecs sont EXPLIQUÉS, chantier clos pour de bon ;
#   - il PERSISTE    → place réelle dans une tâche enfin non-dégénérée → réouverture G1/G2/G3 licenciée.
# Monde SANS hazard (isoler l'arbitrage), épars 1+1 distances 2-8 m, corps cinématique, coût survival.
# Parallèle-safe : port + run-dir uniques, tue SEULEMENT son serveur.
# Usage : PORT=62xx bash scripts/collect_arb_graded.sh <sym|graded> [seed=1] [ep=24]
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
ARM=${1:-sym}; SEED=${2:-1}; NEP=${3:-24}
# CALIBRAGE DE VIE (2026-07-21) : restore par repas. 40 = reglage de COLLECTE (defaut historique) ;
# 60 = "vie" (equilibre a 6 m > mediane de spawn 5 m => derive positive, mais 6-8 m punit encore).
EPF=${EPF:-40}
# A/B ÉCHAFAUDAGE far-target (2026-07-21) : FA=1 = bequille codee-main ACTIVE (statu quo de TOUTES
# nos mesures) ; FA=0 = boucle PURE. Juge = courbe atteinte-vs-distance (bandes LOINTAINES, la ou
# l echafaudage agit) + ratio d errance ; JAMAIS la survie. Enjeu : la competence mesuree
# aujourd hui (trajets 1.08, choix 98%) est-elle celle de l ENTITE ou celle de la bequille ?
FA=${FA:-1}
PORT=${PORT:-6270}; TAG="arbgrad_${ARM}_s${SEED}_r${EPF}_fa${FA}"
OUT="data/replay_buffer/${TAG}"; RUNDIR="data/replay_buffer/${TAG}_run"
export GODOT_BIN="$(pwd)/tools/godot/godot"
if [[ "$ARM" == "graded" ]]; then
  EDRAIN=0.05; TDRAIN=0.035; PDRAIN_T=0.00035     # monde gradué + planner qui le SAIT
else
  EDRAIN=0.05; TDRAIN=0.05;  PDRAIN_T=0.0005      # statu quo dégénéré (bit-identique à avant)
fi
rm -rf "$OUT" "$RUNDIR"
echo "=== ARB-GRADED $ARM : seed=$SEED port=$PORT | monde e=$EDRAIN t=$TDRAIN | planner drain_t=$PDRAIN_T ==="

env SYLVAN_PLANNER_HEADING_W=2.0 SYLVAN_PLANNER_URGENCY_W=6.0 \
    SYLVAN_PLANNER_COST=survival SYLVAN_PLANNER_DRAIN=0.0005 SYLVAN_PLANNER_DRAIN_T=$PDRAIN_T \
    SYLVAN_PLANNER_RESTORE=$(env_pytorch_3.12/bin/python -c "print($EPF/100)") SYLVAN_PLANNER_FAR_ALIGN=$FA SYLVAN_PLANNER_ALIGN_GAIN=60 \
    SYLVAN_CMD_EXPLORE_STD=0 SYLVAN_BC_LOG="$OUT" \
    PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
    --wm data/checkpoints/wm_objcentric_kin/wm_best.pt --residual data/checkpoints/hexapod_v2/policy_best.pt \
    --host 127.0.0.1 --port $PORT --horizon 80 --replan-every 10 > /tmp/arbgrad_srv_${TAG}.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done

env SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
SYLVAN_KINEMATIC=1 SYLVAN_KIN_SPEED=0.8 SYLVAN_KIN_TURN=1.5 \
SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 SYLVAN_EAT_RADIUS=1.0 SYLVAN_DRINK_RADIUS=1.0 \
SYLVAN_FOOD_COUNT=1 SYLVAN_WATER_COUNT=1 SYLVAN_ENERGY_DRAIN=$EDRAIN SYLVAN_THIRST_DRAIN=$TDRAIN \
 SYLVAN_FOOD_ENERGY_PER=$EPF SYLVAN_WATER_ENERGY_PER=$EPF \
SYLVAN_INIT_ENERGY=70 SYLVAN_INIT_THIRST=70 \
SYLVAN_FOOD_MIN_RADIUS=2.0 SYLVAN_FOOD_SPAWN_RADIUS=8.0 SYLVAN_FOOD_RESPAWN_MIN=2.0 SYLVAN_FOOD_RESPAWN_MAX=8.0 \
SYLVAN_WATER_MIN_RADIUS=2.0 SYLVAN_WATER_SPAWN_RADIUS=8.0 SYLVAN_WATER_RESPAWN_MIN=2.0 SYLVAN_WATER_RESPAWN_MAX=8.0 \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=3000 SYLVAN_SEED=$SEED \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR="$RUNDIR" \
./tools/godot/godot --path godot --headless > /tmp/arbgrad_godot_${TAG}.log 2>&1
kill -9 $SRV 2>/dev/null
rm -rf "$RUNDIR"
grep -m1 'COÛT SURVIE actif' /tmp/arbgrad_srv_${TAG}.log
echo "corpus -> $OUT ($(wc -l < "$OUT/ep_0000.jsonl" 2>/dev/null) ticks)"
echo "ALL_DONE_ARBGRAD_${ARM}_s${SEED}"
