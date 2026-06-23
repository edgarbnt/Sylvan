#!/bin/zsh
# GATE closed-loop de CHERCHER — FC=1 (cas dur : 1 bouffe, respawn possiblement derrière → recherche répétée).
# Compare survie search OFF vs ON, conditions identiques. Métrique falsifiable = survie/épisode + meals.
# SUCCÈS = search ON survit nettement plus + CHERCHER déclenche. KILL = pas de gain / tourne sans fin.
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
ER=1.0; HZ=300; NEP=8; DRAIN=0.05

runit() {  # $1 label  $2 search_enable
  pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
  FC=1 SYLVAN_ENERGY_DRAIN=$DRAIN SYLVAN_SEARCH_ENABLE=$2 SYLVAN_SEARCH_LOG=1 \
    bash run_forage_latent.sh $ER $HZ $NEP > /tmp/gate_$1.out 2>&1
  cp /tmp/forage_latent.log  /tmp/gate_$1_godot.log 2>/dev/null
  cp /tmp/planner_latent.log /tmp/gate_$1_plan.log  2>/dev/null
}

runit base 0
runit search 1

echo "==== SURVIE par épisode (steps) — FC=1, drain=$DRAIN, eat_radius=$ER ===="
echo "--- BASE (search OFF) ---";   grep -a "steps=" /tmp/gate_base.out
echo "--- SEARCH (search ON) ---";  grep -a "steps=" /tmp/gate_search.out
echo "==== CHERCHER a-t-il déclenché ? ===="
echo "base   transitions: $(grep -ac 'CHERCHER' /tmp/gate_base_plan.log)  (attendu 0)"
echo "search transitions: $(grep -ac 'CHERCHER' /tmp/gate_search_plan.log)"
grep -a "CHERCHER" /tmp/gate_search_plan.log | head -4
echo "==== format log [Godot] (pour repérer les meals/energy) ===="
grep -a -oE "\[Godot\] [A-Za-z]+" /tmp/gate_search_godot.log | sort | uniq -c | head
echo "==== erreurs planner ? ===="
grep -a -iE "error|traceback" /tmp/gate_base_plan.log /tmp/gate_search_plan.log | head
echo "GATE done"
