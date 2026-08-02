#!/bin/bash
# GATE CLOSED-LOOP DU MONDE-FORÊT — l'entité SURVIT-ELLE avec le WM ré-entraîné ?
#
# POURQUOI. A1 (le type lisible dans le latent, 99,7 %) et l'open-loop (0,132 m à h=50) sont prouvés
# sur `wm_foret_attn_hue`. Le COMPORTEMENT ne l'est pas : aucun foraging n'a jamais été mesuré sur ce
# WM. C'est la seule chose qui manque avant de le promouvoir — et un WM qui rêve juste peut très bien
# mal se conduire (le projet en a l'historique).
#
# CE QU'ON SERT : foret_v1 COMPLET, c'est-à-dire exactement le monde de la collecte (forêt, terrain,
# regard, éventail de vitesse facturé, flaques, distracteurs, danger, palette 4 types). Servir un
# monde plus simple ferait un chiffre flatteur et faux.
#
# ⚠️ RÉSERVE À CITER AVEC TOUT RÉSULTAT : le canal-slot est GREFFÉ, pas rebâti — ses requêtes-couleur
# viennent de l'ANCIEN monde (`wm_objcentric_kin_typed`). C'est un choix assumé pour obtenir un
# premier chiffre en quelques minutes plutôt qu'après 25 min de rebâtissage : si le foraging tient,
# le WM est bon et on rebâtira le canal proprement ; s'il s'effondre, on saura tout de suite que le
# problème est ailleurs. Ce que ce gate NE peut donc PAS trancher : la qualité des requêtes.
#
# Usage : bash scripts/gate_foret_closedloop.sh [vies=12] [max-steps=3000]
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

NEP="${1:-12}"
MS="${2:-3000}"
WM="${WM_CKPT:-data/checkpoints/wm_foret_attn_slot/wm_best.pt}"
PORT="${PORT:-6069}"
# TAG surchargeable : `data/replay_buffer/gate_foret_cl` est le corpus de RÉFÉRENCE de
# `diagnostics/diag_bilan.py` (c'est sur lui qu'est mesurée la baseline 1,42 m / 23,1°). Ce
# script le `rm -rf`, donc tout A/B lancé sous le TAG par défaut REMPLACE silencieusement la
# référence par son propre bras. Poser TAG=... pour comparer sans détruire.
TAG="${TAG:-gate_foret_cl}"
RUN_DIR="$ROOT/data/replay_buffer/$TAG"

export PYTHONPATH=python
eval "$(env_pytorch_3.12/bin/python -m sylvan.world --preset foret_v1 --env)"

echo "=== GATE CLOSED-LOOP | WM=$WM | $NEP vies x $MS ticks | monde foret_v1 COMPLET ==="
echo "    ⚠️  canal-slot GREFFÉ (requêtes de l'ancien monde) — ce gate ne juge PAS les requêtes."

pkill -9 -f serve_planner_command 2>/dev/null; sleep 1
rm -rf "$RUN_DIR"; mkdir -p "$RUN_DIR"

# COÛT SURVIE — ce gate mesure la survie d'une entité à DEUX pulsions ; sans lui le planner ne
# connaît que la faim et meurt de soif par construction, ce qui rendait le verdict ininterprétable
# (mesuré le 2026-07-28 : 6 morts de soif sur 8, que j'avais prises pour un mur d'arbitrage).
PYTHONPATH=python SYLVAN_PLANNER_COST=survival SYLVAN_PLANNER_DRAIN=0.0005 \
  SYLVAN_PLANNER_RESTORE=0.4 SYLVAN_PLANNER_HEADING_W=0.0 \
  ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
  --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt \
  --host 127.0.0.1 --port "$PORT" --horizon 80 --replan-every 10 \
  > "/tmp/${TAG}_srv.log" 2>&1 &
SRV=$!
for _ in $(seq 1 90); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done
if ! ss -ltn 2>/dev/null | grep -q ":$PORT"; then
  echo "❌ le serveur planner n'a pas démarré — voir /tmp/${TAG}_srv.log"; tail -20 "/tmp/${TAG}_srv.log"; exit 1
fi

env SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.0 SYLVAN_TURN_FADE=0 \
    SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 \
    SYLVAN_COLLECT=1 SYLVAN_WM_COLLECT=1 \
    SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT="$PORT" \
    SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
    SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 \
    SYLVAN_NUM_EPISODES="$NEP" SYLVAN_MAX_EPISODE_STEPS="$MS" SYLVAN_SEED="${SEED:-1}" \
    SYLVAN_RUN_DIR="$RUN_DIR" \
    ./tools/godot/godot --path godot --headless > "/tmp/${TAG}_free.log" 2>&1
kill -9 "$SRV" 2>/dev/null

echo
echo "=== SURVIE / REPAS / BOISSONS / CAUSE DE MORT ==="
FREELOG="/tmp/${TAG}_free.log" PYTHONPATH=python ./env_pytorch_3.12/bin/python - <<'PY'
import os, re, statistics as st
eps = {}
pat = re.compile(r'Episode (\d+) \| Step (\d+) .* Energy: ([\d.]+) \| Thirst: ([\d.]+) \| Health: ([\d.]+)')
for line in open(os.environ['FREELOG']):
    m = pat.search(line)
    if not m:
        continue
    ep, s, e, t, h = int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))
    eps.setdefault(ep, []).append((s, e, t, h))
surv, meals, drinks, causes = [], [], [], []
for ep in sorted(eps):
    rows = sorted(eps[ep]); last = rows[-1]
    surv.append(last[0])
    meals.append(sum(1 for i in range(1, len(rows)) if rows[i][1] - rows[i-1][1] > 5))
    drinks.append(sum(1 for i in range(1, len(rows)) if rows[i][2] - rows[i-1][2] > 5))
    # Le log échantillonne tous les 10 ticks : la dernière ligne précède la mort de 0-9 ticks, donc
    # la jauge fatale n'y est jamais pile à 0 (mesuré : E=4 et E=3 sur deux morts de faim évidentes).
    # Un seuil à 1.0 classait donc TOUT en « autre ». Seuil à 10 = la jauge qui va crever, lue à la
    # résolution qu'on a réellement. On prend la plus basse s'il y a ambiguïté.
    low = min((last[1], 'faim'), (last[2], 'soif'), (last[3], 'sante'))
    cause = 'PLEIN' if last[0] >= 2990 else (low[1] if low[0] <= 10.0 else 'autre')
    causes.append(cause)
    print(f"Ep{ep:>2}: survie={last[0]:>5} ({cause:5})  repas={meals[-1]} boissons={drinks[-1]}"
          f"  E={last[1]:.0f} T={last[2]:.0f} H={last[3]:.0f}")
if surv:
    print(f"\nSURVIE méd={st.median(surv):.0f} moy={st.mean(surv):.0f} min={min(surv)} max={max(surv)}")
    print(f"REPAS méd={st.median(meals):.1f} (total {sum(meals)}) | BOISSONS méd={st.median(drinks):.1f} (total {sum(drinks)})")
    jongle = sum(1 for m_, d_ in zip(meals, drinks) if m_ > 0 and d_ > 0)
    print(f"JONGLE faim+soif : {jongle}/{len(surv)} vies | causes : "
          + ", ".join(f"{c}={causes.count(c)}" for c in sorted(set(causes))))
PY
echo "ALL_DONE_GATE_FORET_CL"
