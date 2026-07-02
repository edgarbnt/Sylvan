#!/bin/zsh
# A/B foraging : slot pur à heading_w=0 (pur min_dist, hack retiré) vs heading_w=2 (béquille actuelle). 10 ép chacun.
cd /home/edgarbrunet/Documents/PERSO/SylvanV1
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
echo "########## A) heading_w=0 (pur) ##########"
export SYLVAN_PLANNER_HEADING_W=0
rm -rf data/replay_buffer/forage_purslot 2>/dev/null
zsh run_forage_purslot.sh 1.0 160 10
cp /tmp/forage_purslot.log /tmp/forage_hw0.log
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
echo "########## B) heading_w=2 (béquille) ##########"
export SYLVAN_PLANNER_HEADING_W=2
rm -rf data/replay_buffer/forage_purslot 2>/dev/null
zsh run_forage_purslot.sh 1.0 160 10
cp /tmp/forage_purslot.log /tmp/forage_hw2.log
echo "ALL_DONE"
