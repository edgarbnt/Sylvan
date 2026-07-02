#!/bin/zsh
# A/B foraging : SLOT PUR auto-supervisé vs retina_head supervisé (re-gate Phase 1, survie ≥ baseline). 10 ép chacun.
cd /home/edgarbrunet/Documents/PERSO/SylvanV1
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
rm -rf data/replay_buffer/forage_purslot data/replay_buffer/forage_retina 2>/dev/null
echo "########## A) PUR-SLOT (auto-supervisé, label-free) ##########"
zsh run_forage_purslot.sh 1.0 160 10
cp /tmp/forage_purslot.log /tmp/forage_A_purslot.log
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
echo "########## B) retina_head (supervisé) ##########"
export WM_CKPT=data/checkpoints/wm_rich_fidele_sym/wm_best.pt
zsh run_forage_retina.sh 1.0 160 10
cp /tmp/forage_retina.log /tmp/forage_B_retina.log
echo "ALL_DONE"
