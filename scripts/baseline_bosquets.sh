#!/bin/bash
# BASELINE MONDE BOSQUETS (docs/design_monde_bosquets.md §9) — corps CINÉMATIQUE promu + WM vivant
# `wm_objcentric_kin`, ressources en bosquets FIXES qui s'épuisent et repoussent.
#
# À QUOI ÇA SERT : ÉTAPE 1 de la calibration. Le G0 simulé a montré que l'écart mémoire retombe à
# ZÉRO quand le monde est généreux (57 % de marge alimentaire) — l'information n'a de valeur que si
# se tromper tue. Ce harnais sert donc à trouver le régime où l'entité est AU BORD : elle survit en
# fourrageant bien, elle meurt en errant. Sans mémoire branchée : c'est la ligne de base à battre.
#
# ⚠️ Ce n'est PAS le juge. Le juge est l'A/B mémoire ON/OFF, qui vient après.
#
# Usage: bash scripts/baseline_bosquets.sh [episodes=6] [regrow=2000] [berries=6]
#   SEED=1 PORT=6081 bash scripts/baseline_bosquets.sh 6 2000 6
set +e
NEP=${1:-6}
REGROW=${2:-2000}
BERRIES=${3:-6}
MS=${MS:-3000}
SEED=${SEED:-1}
PORT=${PORT:-6081}
PATCHES=${PATCHES:-2}          # par ressource : 2 bouffe + 2 eau = 4 bosquets
SPACING=${SPACING:-9.0}
DRIVES=${DRIVES:-2}            # 1 = soif COUPEE (mono-pulsion). Voir l ecart mesure ci-dessous.
# POURQUOI L OPTION 1 EXISTE : en bi-pulsion, bouffe et eau sont dans des bosquets SEPARES a ~10 m.
# Chaque bascule de pulsion impose alors une traversee de 909 ticks pour un budget inter-conso de
# 471 -> 1,9x. Alterner faim/soif est ARITHMETIQUEMENT IMPOSSIBLE : mesure, 5/5 episodes morts au
# plancher de famine (2000 pas) avec repas=0 boissons=1. En mono-pulsion le budget passe a 800 ticks
# et la traversee coute 45 pts sur un reservoir de 100 -> payable. Le mur tombe.
PRADIUS=${PRADIUS:-0.95}       # rayon EXTERNE de la couronne de baies ; < eat_radius (1.0)
SPACING_MAX=${SPACING_MAX:-11.0}   # voisin entre 9 et 11 m -> traversee 41-50 pts d energie, comme concu
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT" || exit 1
WM=${WM_CKPT:-data/checkpoints/wm_objcentric_kin/wm_best.pt}
TAG="bosq_d${DRIVES}_r${REGROW}_b${BERRIES}_s${SEED}"

echo "=== BOSQUETS : regrow=$REGROW berries=$BERRIES/ressource patches=$PATCHES espacement=$SPACING-$SPACING_MAX ==="
echo "=== WM=$WM  ep=$NEP  max_steps=$MS  seed=$SEED  port=$PORT ==="

if [ "$DRIVES" = "1" ]; then
  WATER_ENV=""                 # SYLVAN_WATER_COUNT absent => main.gd n active PAS la soif
  echo "=== MONO-PULSION : soif coupee (isole la memoire de l arbitrage) ==="
else
  WATER_ENV="SYLVAN_WATER_COUNT=$BERRIES SYLVAN_WATER_PATCHES=$PATCHES SYLVAN_WATER_PATCH_SPACING=$SPACING SYLVAN_WATER_PATCH_SPACING_MAX=$SPACING_MAX SYLVAN_WATER_PATCH_RADIUS=$PRADIUS SYLVAN_WATER_REGROW=$REGROW SYLVAN_WATER_MIN_RADIUS=3.0 SYLVAN_WATER_SPAWN_RADIUS=11.0 SYLVAN_THIRST_DRAIN=0.05"
fi

SYLVAN_PLANNER_HEADING_W=2.0 SYLVAN_PLANNER_URGENCY_W=6.0 \
SYLVAN_BC_LOG=data/replay_buffer/${TAG} SYLVAN_PLANNER_COST=survival \
SYLVAN_PLANNER_DRAIN=0.0005 SYLVAN_PLANNER_RESTORE=0.4 \
PYTHONPATH=python ./env_pytorch_3.12/bin/python -m scripts.serve_planner_command \
  --wm "$WM" --residual data/checkpoints/hexapod_v2/policy_best.pt \
  --host 127.0.0.1 --port $PORT --horizon 80 --replan-every 10 > /tmp/${TAG}_srv.log 2>&1 &
SRV=$!
for _i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT" && break; sleep 1; done

# $WATER_ENV passe par `env` a la FIN, jamais dans la chaine de prefixes ci-dessous : le shell
# analyse les prefixes AVANT l expansion, donc une variable vide y devient le NOM DE LA COMMANDE.
# Et aucun commentaire ne doit tomber DANS la chaine : il commenterait la commande.
SYLVAN_KINEMATIC=1 SYLVAN_KIN_SPEED=0.8 SYLVAN_KIN_TURN=1.5 \
SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7 \
SYLVAN_CPG_SPEEDCAD=0.6 SYLVAN_CPG_PERIOD=0.5 SYLVAN_CPG_PLANNER=1 SYLVAN_RETINA_PLANNER=1 \
SYLVAN_EAT_RADIUS=1.0 SYLVAN_DRINK_RADIUS=1.0 \
SYLVAN_FOOD_COUNT=$BERRIES SYLVAN_FOOD_PATCHES=$PATCHES \
SYLVAN_FOOD_PATCH_SPACING=$SPACING SYLVAN_FOOD_PATCH_SPACING_MAX=$SPACING_MAX \
SYLVAN_FOOD_PATCH_RADIUS=$PRADIUS SYLVAN_FOOD_REGROW=$REGROW \
SYLVAN_FOOD_MIN_RADIUS=3.0 SYLVAN_FOOD_SPAWN_RADIUS=11.0 \
SYLVAN_ENERGY_DRAIN=0.05 \
SYLVAN_COLLECT=1 SYLVAN_NUM_EPISODES=$NEP SYLVAN_MAX_EPISODE_STEPS=$MS SYLVAN_SEED=$SEED \
SYLVAN_COLLECTOR_MODE=policy_server SYLVAN_POLICY_HOST=127.0.0.1 SYLVAN_POLICY_PORT=$PORT \
SYLVAN_POLICY_EXPLORATION_STD_INITIAL=0 SYLVAN_POLICY_EXPLORATION_STD_FINAL=0 \
SYLVAN_REFLEX_STRENGTH=0 SYLVAN_ASSIST_RATIO=0 SYLVAN_RUN_DIR=data/replay_buffer/${TAG}_run \
env $WATER_ENV ./tools/godot/godot --path godot --headless > /tmp/${TAG}_free.log 2>&1
kill -9 $SRV 2>/dev/null

echo "=== ce qui a VRAIMENT ete servi (le log le prouve) ==="
grep -m2 -E "^\[patch\]" /tmp/${TAG}_free.log

echo "=== SURVIE / REPAS ==="
FREELOG=/tmp/${TAG}_free.log PYTHONPATH=python ./env_pytorch_3.12/bin/python - <<'PY'
import os, re, statistics as st
eps = {}
pat = re.compile(r'Episode (\d+) \| Step (\d+) .* Energy: ([\d.]+) \| Thirst: ([\d.]+)')
for line in open(os.environ['FREELOG']):
    m = pat.search(line)
    if not m: continue
    ep, s, en, th = int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))
    eps.setdefault(ep, []).append((s, en, th))
surv, meals_all, drinks_all = [], [], []
for ep in sorted(eps):
    rows = sorted(eps[ep]); last = rows[-1]
    surv.append(last[0])
    meals = sum(1 for i in range(1, len(rows)) if rows[i][1]-rows[i-1][1] > 5)
    drinks = sum(1 for i in range(1, len(rows)) if rows[i][2]-rows[i-1][2] > 5)
    meals_all.append(meals); drinks_all.append(drinks)
    cause = 'PLEIN' if last[0] >= 2999 else ('faim' if last[1] <= 1.0 else ('soif' if last[2] <= 1.0 else 'autre'))
    print(f"Ep{ep:>2}: survie={last[0]:>5} ({cause:5})  repas={meals} boissons={drinks}")
if surv:
    full = sum(1 for s in surv if s >= 2999)
    print(f"\nSURVIE med={st.median(surv):.0f}  min={min(surv)}  pleins={full}/{len(surv)}")
    print(f"REPAS med={st.median(meals_all):.1f}  BOISSONS med={st.median(drinks_all):.1f}")
    # CRITERES DE CALIBRATION, pre-inscrits (docs/design_monde_bosquets.md)
    floor = 100.0 / 0.05                     # reservoir plein / drain = mort sans jamais manger
    at_floor = sum(1 for s in surv if abs(s - floor) < 60)
    print(f"PLANCHER de famine = {floor:.0f} pas ; episodes AU PLANCHER (= n ont rien mange) : {at_floor}/{len(surv)}")
    if at_floor >= 0.6 * len(surv):
        print("VERDICT: MUR — la majorite meurt sans se nourrir. Ce n est pas une decision.")
    elif full >= 0.8 * len(surv):
        print("VERDICT: TROP FACILE — la survie sature, l'ecart memoire retombera a zero.")
    elif st.median(surv) < 1500:
        print("VERDICT: TROP DUR — on mesurerait un mur, pas une decision.")
    else:
        print("VERDICT: CALIBRE — l'entite est au bord. C'est le regime ou la memoire peut compter.")
PY
echo "ALL_DONE_BOSQUETS"
