#!/bin/zsh
# ÉTAPE A.1 (vers 🅑) — collecter des données FORAGING EAT-RICHES pour que le WM apprenne l'EAT-DYNAMICS
# (le diag a montré : il ne capte que 4% de la bosse-repas car ~23 repas/18k lignes). Pilotage par le
# PLANNER-COORDONNÉES validé (qui forage bien) ; SYLVAN_WM_COLLECT=1 logge retina0+food_rel0+ate.
# RECETTE eat-riche : faim FORTE (drain) + énergie initiale BASSE + densité MODÉRÉE → l'agent voyage entre
# repas (draine → marge) PUIS mange → beaucoup d'événements (approche→repas→bosse d'énergie) AVEC marge.
# eat_radius reste 1.0 (HONNÊTE §2, on ne gonfle pas la bouche). Régime locomoteur PROPRE (CLAUDE.md).
# Usage: bash scripts/collect_wm_eatrich.sh [episodes=3] [seed=41] [food_count=14] [drain=0.15] [init_energy=50] [outdir=retina_eatrich]
set +e
NEP=${1:-3}; SEED=${2:-41}; FC=${3:-14}; DRAIN=${4:-0.15}; IE=${5:-50}; OUT=${6:-retina_eatrich}; HUNGER=${7:-1.0}
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
rm -rf "godot/data/replay_buffer/$OUT" "data/replay_buffer/$OUT"
echo "EAT-RICH collect: ep=$NEP seed=$SEED food=$FC drain=$DRAIN init_energy=$IE hunger_max=$HUNGER out=$OUT"

SYLVAN_PLANNER_HEADING_W=2.0 \
PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
  --wm data/checkpoints/wm_command_hex_v2/wm_best.pt \
  --residual data/checkpoints/hexapod_v2/policy_best.pt \
  --host 127.0.0.1 --port 6054 --horizon 80 --replan-every 10 > /tmp/planner_eatrich.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ':6054' && break; sleep 1; done

SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 SYLVAN_CPG_SPEEDCAD=0.6 \
SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_WM_COLLECT=1 SYLVAN_EAT_RADIUS=1.0 \
SYLVAN_ENERGY_DRAIN=$DRAIN SYLVAN_INIT_ENERGY=$IE SYLVAN_FOOD_COUNT=$FC SYLVAN_FOOD_HUNGER_MAX=$HUNGER \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=1500 SYLVAN_SEED=$SEED \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=6054 \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 \
SYLVAN_RUN_DIR=data/replay_buffer/$OUT \
./tools/godot/godot --path godot --headless > /tmp/collect_eatrich.log 2>&1
kill -9 $SRV 2>/dev/null

DIR="godot/data/replay_buffer/$OUT"; [ -d "$DIR" ] || DIR="data/replay_buffer/$OUT"
echo "=== EAT-RICHNESS de $DIR ==="
./env_pytorch_3.12/bin/python - "$DIR" <<'PY'
import sys, json, glob, statistics
d = sys.argv[1]
files = sorted(glob.glob(f"{d}/episode_*.jsonl"))
rows = eats = 0
pre_e = []; energies = []
per_ep = {}
for f in files:
    ec = 0
    for ln in open(f):
        r = json.loads(ln); w = r.get("wm", {})
        rows += 1
        e0 = r["obs"]["energy"]; energies.append(e0)
        if w.get("ate"):
            eats += 1; ec += 1; pre_e.append(e0)
    per_ep[f.split('/')[-1]] = ec
print(f"épisodes={len(files)} | lignes={rows} | repas(ate)={eats}  → {eats/max(1,len(files)):.1f}/épisode, {100*eats/max(1,rows):.2f}% des pas")
if pre_e:
    lo = sum(1 for e in pre_e if e < 70)
    pq = statistics.quantiles(pre_e, n=4) if len(pre_e) >= 4 else [min(pre_e)]*3
    print(f"énergie PRÉ-repas: min={min(pre_e):.0f} q25={pq[0]:.0f} méd={statistics.median(pre_e):.0f} max={max(pre_e):.0f}")
    print(f"repas BASSE énergie (pré<70, vraie bosse +marge) = {lo}/{len(pre_e)} ({100*lo/len(pre_e):.0f}%)  ← LE chiffre qui compte pour 🅑")
if energies:
    qs = statistics.quantiles(energies, n=4)
    print(f"énergie GLOBALE: min={min(energies):.0f} q25={qs[0]:.0f} méd={qs[1]:.0f} q75={qs[2]:.0f} max={max(energies):.0f}  (plage large = WM voit varier l'énergie)")
print("repas/épisode:", {k: v for k, v in sorted(per_ep.items())})
PY
echo "errors?"; grep -a -iE "error|traceback" /tmp/planner_eatrich.log | head