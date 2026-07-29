#!/bin/bash
# LES QUATRE GATES POST-RETRAIN, ENCHAÎNÉS ET AUTO-JUGÉS.
#
# POURQUOI CE SCRIPT EXISTE. Les seuils étaient pré-enregistrés dans un document ; les gates étaient
# lancés à la main, un par un, et interprétés à la lecture. Deux fois aujourd'hui, un verdict s'est
# révélé faux non pas parce que le monde allait mal, mais parce que l'INSTRUMENT allait mal (planner
# aveugle à la soif, canal-slot sans index d'eau). Un seuil qu'on relit soi-même à 2 h du matin est
# un seuil qu'on peut arrondir. Ici il est dans le code, il rend PASS ou ÉCHEC, et le script sort en
# erreur si un gate tombe — il n'y a rien à interpréter.
#
# Il enchaîne aussi pour ne pas perdre de temps : le retrain fini, tout part sans intervention.
#
# Usage : bash scripts/gates_foret_v2.sh [nom-du-WM=wm_foret_v2] [vies-closed-loop=12]
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

WM_NAME="${1:-wm_foret_v2}"
NEP="${2:-12}"
BASE="data/checkpoints/${WM_NAME}/wm_best.pt"
SLOT="data/checkpoints/${WM_NAME}_slot/wm_best.pt"
SRC="data/checkpoints/wm_objcentric_kin_typed/wm_best.pt"
export PYTHONPATH=python

# SEUILS PRÉ-ENREGISTRÉS — ne pas les toucher pour faire passer un run (CLAUDE.md §2).
A1_MIN=70.0          # % de lecture du type depuis l'encodeur ; majorité ≈ 27 %
OL_MAX=0.20          # m, erreur de position open-loop à h=50
CL_SURV_MIN=1000     # ticks, survie médiane closed-loop
CL_MEALS_MIN=3       # repas médians par vie

RES="/tmp/gates_${WM_NAME}"
mkdir -p "$RES"
fail=0

echo "############ GATES POST-RETRAIN — $WM_NAME"
if [[ ! -f "$BASE" ]]; then echo "❌ checkpoint absent : $BASE"; exit 1; fi

# --- 1. greffe du canal-slot ------------------------------------------------------------------
echo; echo "=== 1/4 GREFFE DU CANAL-SLOT ==="
env_pytorch_3.12/bin/python scripts/graft_slot_channel.py \
  --dst "$BASE" --src "$SRC" --out "$SLOT" > "$RES/graft.log" 2>&1
if [[ ! -f "$SLOT" ]]; then echo "❌ greffe échouée — voir $RES/graft.log"; tail -5 "$RES/graft.log"; exit 1; fi
# La greffe a éteint une pulsion en silence le 2026-07-28 (water_idx non recopié) : on VÉRIFIE.
env_pytorch_3.12/bin/python - "$SLOT" <<'PY'
import sys, torch
m = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["meta"]
ok = m.get("food_idx") is not None and m.get("water_idx") is not None
print(f"  food_idx={m.get('food_idx')} water_idx={m.get('water_idx')} "
      f"slots={m.get('slot_resources')} attention={m.get('retina_attention')} "
      f"-> {'OK' if ok else '❌ UNE PULSION EST ÉTEINTE'}")
raise SystemExit(0 if ok else 1)
PY
[[ $? -ne 0 ]] && { echo "❌ index de slot manquants — le planner serait borgne"; exit 1; }

# --- 2. A1 : le type survit-il à l'encodeur ? ---------------------------------------------------
echo; echo "=== 2/4 A1 — LE TYPE SURVIT-IL À L'ENCODEUR ? (seuil > ${A1_MIN} %) ==="
env_pytorch_3.12/bin/python diagnostics/diag_latent_carries_type.py \
  --corpus data/replay_buffer/foret_v1_planner data/replay_buffer/foret_v1b_planner \
  --wm "$SLOT" --depth 0 --stride 4 > "$RES/a1.log" 2>&1
grep -E "précision held-out|palette de référence" "$RES/a1.log" | sed 's/^/  /'
A1=$(grep "ENCODEUR" "$RES/a1.log" | grep -oE "[0-9]+\.[0-9]+%" | tr -d '%' | sort -g | tail -1)
if [[ -z "$A1" ]]; then echo "  ⚠️  A1 non mesurable — voir $RES/a1.log"; fail=1
elif (( $(echo "$A1 > $A1_MIN" | bc -l) )); then echo "  ✅ A1 PASS ($A1 % > $A1_MIN %)"
else echo "  ❌ A1 ÉCHEC ($A1 % <= $A1_MIN %)"; fail=1; fi

# --- 3. open-loop : le WM rêve-t-il juste ? -----------------------------------------------------
# ⚠️ LE 0,132 m HISTORIQUE N'EST PAS UNE BASELINE COMPARABLE : il a été mesuré sur l'ANCIEN corpus
# (corps à kin_speed 8, monde dense). Ancre mesurée le 2026-07-29 : le WM d'alors, évalué sur le
# NOUVEAU corpus, rend 1,042 m — il y est hors-distribution. Le nouveau WM doit donc battre 1,042
# largement, et le seuil de 0,20 m reste la vraie barre (« le WM prédit bien SON monde »).
echo; echo "=== 3/4 OPEN-LOOP — POSITION À h=50 (seuil <= ${OL_MAX} m ; ancre OOD 1,042 m) ==="
SYLVAN_WM_USE_RETINA=1 env_pytorch_3.12/bin/python -m scripts.eval_wm_command \
  --checkpoint "$BASE" --horizons 10 50 80 > "$RES/openloop.log" 2>&1
grep -E "^  10 |^  50 |^  80 |JALON" "$RES/openloop.log" | sed 's/^/  /'
OL=$(awk '$1=="50"{gsub("m","",$3); print $3}' "$RES/openloop.log" | head -1)
if [[ -z "$OL" ]]; then echo "  ⚠️  open-loop non mesurable — voir $RES/openloop.log"; fail=1
elif (( $(echo "$OL <= $OL_MAX" | bc -l) )); then echo "  ✅ OPEN-LOOP PASS ($OL m <= $OL_MAX m)"
else echo "  ❌ OPEN-LOOP ÉCHEC ($OL m > $OL_MAX m)"; fail=1; fi

# --- 4. closed-loop : elle SURVIT ? -------------------------------------------------------------
# Le seul gate qui n'a JAMAIS passé. Les deux premières tentatives ont été invalidées par
# l'instrument (planner mono-pulsion, canal-slot borgne) — les deux sont réparés.
echo; echo "=== 4/4 CLOSED-LOOP — SURVIE (seuils : médiane > $CL_SURV_MIN, repas >= $CL_MEALS_MIN) ==="
WM_CKPT="$SLOT" bash scripts/gate_foret_closedloop.sh "$NEP" 3000 > "$RES/closedloop.log" 2>&1
grep -E "^Ep|^SURVIE|^REPAS|^JONGLE" "$RES/closedloop.log" | sed 's/^/  /'
SURV=$(grep "^SURVIE" "$RES/closedloop.log" | grep -oE "méd=[0-9]+" | cut -d= -f2)
MEALS=$(grep "^REPAS" "$RES/closedloop.log" | grep -oE "méd=[0-9.]+" | head -1 | cut -d= -f2)
if [[ -z "$SURV" || -z "$MEALS" ]]; then echo "  ⚠️  closed-loop non mesurable — voir $RES/closedloop.log"; fail=1
else
  ok=1
  (( $(echo "$SURV > $CL_SURV_MIN" | bc -l) )) || { echo "  ❌ survie $SURV <= $CL_SURV_MIN"; ok=0; }
  (( $(echo "$MEALS >= $CL_MEALS_MIN" | bc -l) )) || { echo "  ❌ repas $MEALS < $CL_MEALS_MIN"; ok=0; }
  [[ $ok -eq 1 ]] && echo "  ✅ CLOSED-LOOP PASS (survie $SURV, repas $MEALS)" || fail=1
fi

echo; echo "############ VERDICT"
if [[ $fail -eq 0 ]]; then
  echo "✅ LES 4 GATES PASSENT — $WM_NAME est promouvable (décision owner)."
else
  echo "❌ AU MOINS UN GATE ÉCHOUE. C'est un négatif INFORMATIF : STOP, diagnostiquer"
  echo "   gratuitement, ne PAS enchaîner un tweak (CLAUDE.md §1). Logs : $RES/"
fi
echo "ALL_DONE_GATES_FORET"
exit $fail
