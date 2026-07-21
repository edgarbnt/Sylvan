#!/bin/zsh
# Sonde n°1 (debug arbitrage vs moteur) : collecte APPARIÉE mono-drive vs multi-drive, tout IDENTIQUE
# sauf l'eau, pour mesurer P(atteindre une ressource EN VUE) à distance égale. mono = bouffe SEULE,
# soif GELÉE (THIRST_DRAIN=0) → ZÉRO arbitrage possible → plancher de nav/commitment PUR. multi =
# bouffe+eau, soif qui draine → arbitrage réel. Écart à 2 m = coût du FLOTTEMENT d'arbitrage.
# Monde SANS hazard (isoler l'arbitrage). Parallèle-safe : port + run-dir uniques, tue SEULEMENT son
# serveur (jamais de pkill global — deux instances coexistent).
#
# ⭐ TEST kin_speed (PRÉ-INSCRIT 2026-07-21) — env SPEED : le corps plus rapide REPOUSSE-T-IL le mur
# de portée ? CRITÈRE HONNÊTE (§2) = la COURBE atteinte-vs-distance, JAMAIS la survie totale (qui
# confond gain-de-portée légitime et masquage du déficit de près) :
#   PASS  = l'atteinte aux bandes LOINTAINES (5-6 m, 6-8 m) monte MATÉRIELLEMENT (> bruit ~5 pts)
#           => le mur de portée est bien une enveloppe physique, la vitesse est le vrai levier ;
#   NUL   = les bandes lointaines ne bougent pas => la vitesse n'est PAS le levier (ou le WM décalé
#           annule le gain — voir caveat) ;
#   GARDE anti-masquage = le près (0-2 m) est RAPPORTÉ mais ne sert JAMAIS de preuve de succès.
# CAVEAT PRÉ-INSCRIT : le WM (wm_objcentric_kin) a été collecté à kin_speed=0.5 et tourne à 0.8 ;
# monter à 1.2 AUGMENTE le décalage de régime → un résultat NUL serait AMBIGU (vitesse vs décalage
# WM) et la version propre exigerait une re-collecte WM. Bump modéré (1.5×) pour limiter ce biais.
#
#
# ⭐ TEST DRAINS ASYMÉTRIQUES (PRÉ-INSCRIT 2026-07-21) — env EDRAIN/TDRAIN. MESURÉ : avec des drains
# IDENTIQUES, l'écart |énergie−soif| ne change QU'aux consommations (+40) et reste FIGÉ entre deux →
# il ne prend que 2 valeurs (0 ou ~40) : 49,9 % du temps égalité EXACTE, 49,8 % écrasement, 0,3 %
# entre les deux. La tâche d'arbitrage n'a donc AUCUNE ZONE GRISE (elle est triviale des 2 côtés) —
# explication candidate de l'échec du chantier critique-arbitrage (rien à apprendre).
# QUESTION DE CE TEST : des drains asymétriques créent-ils un CONTINUUM d'urgences (les jauges
# dérivent l'une par rapport à l'autre entre les repas) ?
#   MESURES : (1) distribution de |e−t| — la bande intermédiaire (5-30) se peuple-t-elle ?
#             (2) GARDE anti-dégénérescence : une jauge devient-elle systématiquement l'urgente
#                 (→ on aurait troqué une dégénérescence pour une autre = quasi-mono-drive) ;
#             (3) CAS NON TRIVIAUX : fréquence des situations où la ressource la PLUS URGENTE est la
#                 PLUS LOINTAINE (c'est là, et seulement là, qu'un bon arbitrage paie).
#   ⚠️ CAVEAT PRÉ-INSCRIT : le planner suppose un drain SYMÉTRIQUE en interne (cfg.resource_drain,
#   une seule valeur) → son modèle d'urgence sera légèrement faux en asymétrique. Ce test mesure la
#   STRUCTURE DU MONDE, pas la qualité de réponse de l'entité ; toute suite exigerait un drain
#   interne PAR-JAUGE. Asymétrie DÉCLARÉE une fois (0.05/0.035 = −30 %), non ajustée après coup (§2).
#
# Usage : PORT=62xx [SPEED=1.2] [EDRAIN=0.05 TDRAIN=0.035] bash scripts/collect_reachprobe.sh <mono|multi> [seed] [ep]
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
MODE=${1:-mono}; SEED=${2:-5}; NEP=${3:-16}
PORT=${PORT:-6250}; DELTA=${DELTA:-0}; SPEED=${SPEED:-0.8}
EDRAIN=${EDRAIN:-0.05}; TDRAIN=${TDRAIN:-0.05}
TAG="reach_${MODE}_s${SEED}_d${DELTA}_v${SPEED}_e${EDRAIN}t${TDRAIN}"
OUT="data/replay_buffer/critic_kin_${TAG}"
export GODOT_BIN="$(pwd)/tools/godot/godot"
if [[ "$MODE" == "mono" ]]; then WC=0; TD=0; else WC=1; TD=0.05; fi
rm -rf "$OUT"
echo "=== REACHPROBE $MODE : ep=$NEP seed=$SEED port=$PORT (WC=$WC thirst_drain=$TD, no hazard) ==="

env SYLVAN_PLANNER_HEADING_W=2.0 SYLVAN_PLANNER_URGENCY_W=6.0 \
    SYLVAN_PLANNER_COST=survival SYLVAN_PLANNER_DRAIN=0.0005 SYLVAN_PLANNER_RESTORE=0.4 \
    SYLVAN_PLANNER_FAR_ALIGN=1 SYLVAN_PLANNER_ALIGN_GAIN=60 \
    SYLVAN_PLANNER_COMMIT_DELTA=$DELTA \
    SYLVAN_PLANNER_CRITIC=data/checkpoints/survival_critic_kin/critic_best.pt \
    SYLVAN_BC_LOG="$OUT" \
    PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
    --wm data/checkpoints/wm_objcentric_kin/wm_best.pt --residual data/checkpoints/hexapod_v2/policy_best.pt \
    --host 127.0.0.1 --port $PORT --horizon 80 --replan-every 10 > /tmp/reach_srv_${TAG}.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done

env SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
SYLVAN_KINEMATIC=1 SYLVAN_KIN_SPEED=$SPEED SYLVAN_KIN_TURN=1.5 \
SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 SYLVAN_EAT_RADIUS=1.0 SYLVAN_DRINK_RADIUS=1.0 \
SYLVAN_FOOD_COUNT=1 SYLVAN_WATER_COUNT=$WC SYLVAN_ENERGY_DRAIN=$EDRAIN SYLVAN_THIRST_DRAIN=$([[ "$MODE" == "mono" ]] && echo 0 || echo $TDRAIN) \
SYLVAN_INIT_ENERGY=70 SYLVAN_INIT_THIRST=70 \
SYLVAN_FOOD_MIN_RADIUS=2.0 SYLVAN_FOOD_SPAWN_RADIUS=8.0 SYLVAN_FOOD_RESPAWN_MIN=2.0 SYLVAN_FOOD_RESPAWN_MAX=8.0 \
SYLVAN_WATER_MIN_RADIUS=2.0 SYLVAN_WATER_SPAWN_RADIUS=8.0 SYLVAN_WATER_RESPAWN_MIN=2.0 SYLVAN_WATER_RESPAWN_MAX=8.0 \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=3000 SYLVAN_SEED=$SEED \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR=data/replay_buffer/reachtmp_${TAG} \
./tools/godot/godot --path godot --headless > /tmp/reach_godot_${TAG}.log 2>&1
kill -9 $SRV 2>/dev/null
rm -rf "data/replay_buffer/reachtmp_${TAG}"
echo "reach -> $OUT ($(wc -l < "$OUT/ep_0000.jsonl" 2>/dev/null) ticks)"
echo "ALL_DONE_REACH_${MODE}"
