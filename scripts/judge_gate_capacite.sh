#!/bin/zsh
# JUGE 2-BRAS du Gate-capacite (docs/design_gate_capacite.md Sgates, PRE-ENREGISTRE) : swap
# d'apparence food en cours de vie (T=700 pas/vie, teinte magenta 0.83, DECLARE) + re-mesure
# periodique (embryon jour/nuit, N=150, fenetre glissante 6000, DECLARES -- cf python/sylvan/
# control/remeasure.py). MEME chemin que judge_typed_slots.sh (config vivante WM type + monde
# varie + waypoint/saillance/sprint-critic) pour les DEUX bras -- seul SYLVAN_REMEASURE_EVERY
# differe. Collecte SEQUENTIELLE (4 x 24 vies), verdict via scripts.judge_gate_capacite.
#
# Usage: bash scripts/judge_gate_capacite.sh
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"

export SYLVAN_FOOD_SWAP_TICK=700 SYLVAN_FOOD_SWAP_HUE=0.83

echo "=== [gate-capacite] BRAS CONTROLE (statique) seed=1 ==="
unset SYLVAN_REMEASURE_EVERY
bash scripts/judge_typed_slots.sh 1 gcctl1

echo "=== [gate-capacite] BRAS CONTROLE (statique) seed=2 ==="
bash scripts/judge_typed_slots.sh 2 gcctl2

echo "=== [gate-capacite] BRAS APPRIS (re-mesure N=150) seed=1 ==="
export SYLVAN_REMEASURE_EVERY=150
bash scripts/judge_typed_slots.sh 1 gclrn1

echo "=== [gate-capacite] BRAS APPRIS (re-mesure N=150) seed=2 ==="
bash scripts/judge_typed_slots.sh 2 gclrn2

echo "=== [gate-capacite] VERDICT ==="
PYTHONPATH=python env_pytorch_3.12/bin/python -m scripts.judge_gate_capacite \
    --control data/replay_buffer/critic_kin_gcctl1 data/replay_buffer/critic_kin_gcctl2 \
    --learned data/replay_buffer/critic_kin_gclrn1 data/replay_buffer/critic_kin_gclrn2

echo "ALL_DONE_GATE_CAPACITE"
