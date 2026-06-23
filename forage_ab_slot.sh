#!/bin/zsh
# A/B foraging : slot (perception apprise, nos changements clé de voûte) vs baseline oracle. 10 ép chacun.
cd /home/edgarbrunet/Documents/PERSO/SylvanV1
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
rm -rf data/replay_buffer/forage_retina data/replay_buffer/forage_hex 2>/dev/null
echo "########## A) SLOT (wm_rich_fidele_sym + retina_head) ##########"
export WM_CKPT=data/checkpoints/wm_rich_fidele_sym/wm_best.pt
zsh run_forage_retina.sh 1.0 160 10
cp /tmp/forage_retina.log /tmp/forage_slot.log
unset WM_CKPT
pkill -9 -f serve_planner_command 2>/dev/null; pkill -9 -f 'godot --path godot' 2>/dev/null; sleep 1
echo "########## B) BASELINE (oracle + wm_command_hex_v2) ##########"
zsh run_forage_hex.sh 1.0 160 10
cp /tmp/forage_hex.log /tmp/forage_base.log
echo "ALL_DONE"
