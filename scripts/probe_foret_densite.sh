#!/bin/bash
# SONDE DENSITÉ — éclaircir le monde fait-il qu'un repas vaut enfin quelque chose ?
#
# POURQUOI. Mesuré le 2026-07-28 sur le visuel : l'entité mange à énergie 74 et n'encaisse que 18
# points sur les 84 servis (21 % de rendement), parce que la jauge plafonne à 100 et qu'il y a de la
# nourriture tous les 2 m. Il lui faudrait ~53 repas par vie pour un budget qui n'en porte que ~12 :
# ce monde n'est pas survivable, et l'hypothèse est que c'est la DENSITÉ qui le rend tel.
# Les critères de succès ET de kill sont pré-enregistrés dans diagnostics/diag_foret_densite.py.
#
# UNE SEULE VARIABLE CHANGE. On fait varier le nombre de BOSQUETS (l'étalement spatial) et on garde
# SYLVAN_*_COUNT constant : la masse de nourriture du monde est identique dans les trois conditions,
# seule sa concentration change. Sans ça on confondrait « plus épars » et « moins de nourriture »,
# et un rendement qui monte ne voudrait rien dire. Même graine, même WM, même corps partout.
#
# ⚠️ RÉSERVE À CITER AVEC TOUT RÉSULTAT : le WM servi a été entraîné sur l'ANCIENNE dynamique. Il est
# hors-distribution dans les trois conditions. Les ABSOLUS sont donc des bornes, pas des chiffres
# finaux ; ce qui transfère, c'est la COMPARAISON entre densités, qui partage cet OOD.
#
# Usage : bash scripts/probe_foret_densite.sh [vies=6] [max-steps=3000] [densités="180 60 25"]
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

NEP="${1:-6}"
MS="${2:-3000}"
DENS="${3:-180 60 25}"
WM="${WM_CKPT:-data/checkpoints/wm_foret_attn_slot/wm_best.pt}"
PORT="${PORT:-6071}"
SEED="${SEED:-1}"

export PYTHONPATH=python
eval "$(env_pytorch_3.12/bin/python -m sylvan.world --preset foret_v1 --env)"

echo "=== SONDE DENSITÉ | WM=$WM | $NEP vies x $MS ticks par condition | graine $SEED ==="
echo "    densités testées : $DENS bosquets/ressource (masse de nourriture CONSTANTE)"

pkill -9 -f serve_planner_command 2>/dev/null; sleep 1
PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
  --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt \
  --host 127.0.0.1 --port "$PORT" --horizon 80 --replan-every 10 \
  > /tmp/probe_dens_srv.log 2>&1 &
SRV=$!
for _ in $(seq 1 90); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done
if ! ss -ltn 2>/dev/null | grep -q ":$PORT"; then
  echo "❌ le serveur planner n'a pas démarré — voir /tmp/probe_dens_srv.log"
  tail -20 /tmp/probe_dens_srv.log; exit 1
fi

RUNS=""
for d in $DENS; do
  RUN_DIR="$ROOT/data/replay_buffer/dens_$d"          # le diag lit la densité dans CE nom
  rm -rf "$RUN_DIR"; mkdir -p "$RUN_DIR"
  # L'ÉCART DOIT SUIVRE LA DENSITÉ, sinon « moins de bosquets » ne veut PAS dire « plus épars ».
  # Mesuré : à écart max figé à 6 m, le placeur exige que chaque bosquet touche la chaîne existante
  # -> 25 bosquets se tassent dans un coin à 3,10 m d'écart, exactement comme 180. On dériverait
  # alors le rendement d'un monde « tout au même endroit », pas d'un monde épars. On impose donc
  # l'écart typique d'un semis uniforme, sqrt(aire/n), la même règle dans les trois conditions.
  eval "$(env_pytorch_3.12/bin/python - "$d" <<'PY'
import math, sys
sys.path.insert(0, "python")
from sylvan.world import FORET_V1 as f
n = int(sys.argv[1])
r_in, r_out = f.spawn_annulus_m
typical = math.sqrt(math.pi * (r_out**2 - r_in**2) / n)
print(f"export SYLVAN_FOOD_PATCH_SPACING={0.70 * typical:.2f}")
print(f"export SYLVAN_FOOD_PATCH_SPACING_MAX={1.50 * typical:.2f}")
print(f"export SYLVAN_WATER_PATCH_SPACING={0.70 * typical:.2f}")
print(f"export SYLVAN_WATER_PATCH_SPACING_MAX={1.50 * typical:.2f}")
PY
)"
  echo "--- $d bosquets/ressource (écart ${SYLVAN_FOOD_PATCH_SPACING}-${SYLVAN_FOOD_PATCH_SPACING_MAX} m) ---"
  env SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.0 SYLVAN_TURN_FADE=0 \
      SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 \
      SYLVAN_COLLECT=1 SYLVAN_WM_COLLECT=1 \
      SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT="$PORT" \
      SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
      SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 \
      SYLVAN_FOOD_PATCHES="$d" SYLVAN_WATER_PATCHES="$d" \
      SYLVAN_NUM_EPISODES="$NEP" SYLVAN_MAX_EPISODE_STEPS="$MS" SYLVAN_SEED="$SEED" \
      SYLVAN_RUN_DIR="$RUN_DIR" \
      ./tools/godot/godot --path godot --headless > "/tmp/probe_dens_${d}.log" 2>&1
  grep -E "^\[patch\] (FOOD|WATER) :" "/tmp/probe_dens_${d}.log" | head -2 | sed 's/^/    /'
  RUNS="$RUNS $RUN_DIR"
done
kill -9 "$SRV" 2>/dev/null

PYTHONPATH=python ./env_pytorch_3.12/bin/python diagnostics/diag_foret_densite.py $RUNS
echo "ALL_DONE_PROBE_DENSITE"
