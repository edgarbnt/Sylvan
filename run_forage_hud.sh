#!/bin/zsh
# Lance un run foraging vivant AVEC l'écriture live.json (pour regarder l'Archi-HUD s'animer).
# Ouvre d'abord la carte dans un autre terminal : bash voir_archi.sh
# Usage: bash run_forage_hud.sh [eat_radius=1.0] [horizon=160] [episodes=6]
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
mkdir -p data/hud
export SYLVAN_HUD=1
exec zsh run_forage_purslot.sh "${1:-1.0}" "${2:-160}" "${3:-6}"
