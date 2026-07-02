#!/bin/zsh
# Lance un run foraging vivant AVEC l'écriture live.json (pour regarder l'Archi-HUD s'animer).
# Workflow : ouvre la carte dans un AUTRE terminal (`bash scripts/voir_archi.sh`), puis lance ce script ici.
# Le run est headless ; sa progression est streamée dans CE terminal. Ctrl-C nettoie TOUT (anti-orphelin).
# Usage: bash scripts/run_forage_hud.sh [eat_radius=1.0] [horizon=160] [episodes=6]
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
mkdir -p data/hud
export SYLVAN_HUD=1
LOG=/tmp/forage_purslot.log

cleanup() {
  pkill -9 -f 'scripts.serve_planner_command' 2>/dev/null
  pkill -9 -f 'godot --path godot' 2>/dev/null
  [[ -n "$TAILPID" ]] && kill "$TAILPID" 2>/dev/null
  pkill -f "tail -n +1 -F $LOG" 2>/dev/null  # filet : tout tail de streaming résiduel
}
# EXIT couvre aussi les sorties non-signalées (ex. SIGPIPE) → jamais d'orphelin de streaming.
trap 'echo; echo "[archi-hud] arrêt demandé — nettoyage du serveur planner + godot…"; cleanup; exit 0' INT TERM
trap cleanup EXIT

echo "════════════════════════════════════════════════════════════════════"
echo " Archi-HUD LIVE"
echo "  1) Ouvre la carte dans un AUTRE terminal :  bash scripts/voir_archi.sh"
echo "     puis  http://127.0.0.1:8765/tools/archi_hud/index.html"
echo "  2) Le run tourne en headless ci-dessous (progression en direct)."
echo "     Les modules s'illuminent dans la carte. Ctrl-C = arrêt propre."
echo "════════════════════════════════════════════════════════════════════"

: > "$LOG"
zsh run_forage_purslot.sh "${1:-1.0}" "${2:-160}" "${3:-6}" &
RUNPID=$!
# Attend que le log se remplisse puis le streame (sinon le terminal paraît figé).
for i in $(seq 1 30); do [[ -s "$LOG" ]] && break; sleep 1; done
tail -n +1 -F "$LOG" 2>/dev/null &
TAILPID=$!
wait "$RUNPID"
sleep 1
kill "$TAILPID" 2>/dev/null
cleanup
echo "[archi-hud] run terminé."
