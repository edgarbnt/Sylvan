#!/bin/zsh
# PURIFIER LE WM (2026-06-25, principe N°4) — DROP la reconstruction d'entrée (proprio + rétine) → JEPA principe n°1
# (« prédire en REPRÉSENTATION, pas reconstruire l'entrée »). On garde les readouts ABSTRAITS (énergie=coût,
# displacement=ego-motion du slot, done=survie) + la perte latente JEPA + VICReg + w-rollout (fidélité du rêve).
# Warm-start du WM vivant wm_rich_fidele_sym (recette identique, seuls --w-proprio/--w-radar passent à 0).
# Précédent rassurant : v3_jepa2 a droppé la reconstruction → eff_rank tint 21.
#
# CRITÈRES PRÉ-ÉCRITS (gate le retrain, CLAUDE.md §1/§4) — le slot dépend de la displacement-head, donc on VÉRIFIE :
#   GATE GRATUIT (sur le nouveau WM) :
#     - eff_rank ≥ ~13 (baseline ; idéalement ↑ = plus JEPA) ET fidélité rêve↔réel non cassée  [diag_dream_disp]
#     - transport du slot ≥ ~+0.60 (le rêve transporte toujours l'ego-motion)                  [diag_test6_slot_transport]
#   GATE CLOSED-LOOP (si gratuit OK) : engagement diag_nav_ab_purslot + foraging forage_ab → ≥ baseline.
#   PROMOUVOIR seulement si PUR (recon=0) ET ≥ baseline (N°4 : jamais échanger robustesse contre pureté).
#   KILL : eff_rank s'effondre (<6) OU transport slot << 0.60 OU foraging régresse → la reconstruction était
#     porteuse → ne pas promouvoir, garder wm_rich_fidele_sym, rapporter (le drop n'est pas gratuit ici).
#
# Lancer : la commande python SEULE en background (orphelins tués À PART avant).
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
export PYTHONPATH=python SYLVAN_WM_USE_RETINA=1
./env_pytorch_3.12/bin/python -m scripts.train_wm_command \
  --runs data/replay_buffer/retina_wm_a data/replay_buffer/retina_wm_b \
         godot/data/replay_buffer/retina_eat_a godot/data/replay_buffer/retina_eat_b \
  --out data/checkpoints/wm_rich_fidele_sym_jepa \
  --init-from data/checkpoints/wm_rich_fidele_sym/wm_best.pt \
  --latent-loss cosine --vicreg-var 1.0 --vicreg-cov 1.0 --vicreg-gamma 1.0 \
  --w-rollout 3.0 --mirror-augment \
  --w-proprio 0.0 --w-radar 0.0 \
  --lr 1e-4 --epochs 10 --stride 8 --batch-size 48
