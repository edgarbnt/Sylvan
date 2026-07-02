#!/bin/zsh
# TEST DÉCOUPLAGE ARCHI (owner 2026-06-19) — un WM RICHE + IMAGINATION FIDÈLE porte-t-il la bouffe SANS --w-food ?
# Enjeu : --w-food modifie le WM (substrat LENT) → l'utiliser pour chaque ressource = archi qui se retouche à
# chaque pulsion (mauvais). MAIS le chantier fidélité (--w-rollout) rend le rêve fidèle (latent rêvé ≈ réel 0.96)
# → le latent rêvé devrait porter TOUTE la perception rétine (rouge=bouffe, bleu=eau...) NATURELLEMENT, sans patch.
# Si confirmé : WM entraîné UNE fois (perception+dynamique) ; ajouter l'eau = juste une TÊTE DE VALEUR (nuit),
# WM INTACT = le bon découpage "substrat lent / têtes rapides" du cycle de vie.
#
# Setup : warm wm_rollout_fix (riche+fidèle, mais babbling sans bouffe) + eat data, cosine + VICReg(1,1) +
# --w-rollout 3.0, ZÉRO --w-food. Le latent doit porter la bouffe via la recon rétine (radar_loss) + fidélité.
#
# CRITÈRES PRÉ-ÉCRITS (CLAUDE.md §1) :
#   SUCCÈS (archi DÉCOUPLÉE) : food_auc (teacher-forced) ≥ ~0.75 ET fidélité rêve↔réel@40 ≥ ~0.85 ET path ≥ 0.45 m
#     ET eff_rank RICHE (≥~10 en train) → le rêve fidèle porte la bouffe sans forçage → ajouter une pulsion =
#     juste une tête, WM jamais re-touché. (fidélité haute ⟹ latent rêvé ≈ réel ⟹ food_auc rêvé ≈ teacher-forced.)
#   ÉCHEC : food_auc < ~0.70 malgré fidélité haute → le latent ne porte pas assez la bouffe sans --w-food →
#     ne PAS remettre --w-food d'office ; diagnostiquer (poids recon rétine ? autre) AVANT tout patch.
#
# Lancer backgroundé (orphelins tués À PART avant) : nohup zsh run_decoupled_wm.sh > /tmp/decoupled.log 2>&1 &
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
export PYTHONPATH=python
GD=godot/data/replay_buffer
OUT=data/checkpoints/wm_rich_fidele

echo "############### TRAIN : warm rollout_fix + eat data, cosine + VICReg(1,1) + w-rollout 3, SANS w-food ###############"
SYLVAN_WM_USE_RETINA=1 \
./env_pytorch_3.12/bin/python -m scripts.train_wm_command \
  --runs $GD/retina_eat_a $GD/retina_eat_b data/replay_buffer/retina_wm_a data/replay_buffer/retina_wm_b \
  --out $OUT \
  --init-from data/checkpoints/wm_rollout_fix/wm_best.pt \
  --epochs 10 --stride 8 --batch-size 48 --lr 1e-4 \
  --latent-loss cosine --vicreg-var 1.0 --vicreg-cov 1.0 \
  --w-rollout 3.0
echo "train exit=$?"

echo "############### MESURE 1 : fidélité du rêve + path + eff_rank (le rêve reste-t-il fidèle avec eat data ?) ###############"
SYLVAN_WM_USE_RETINA=1 ./env_pytorch_3.12/bin/python diagnostics/diag_dream_disp.py $OUT/wm_best.pt 1 2>&1 | grep -v -i warning
echo "############### MESURE 2 : food_auc (le latent porte-t-il la bouffe SANS w-food ?) ###############"
SYLVAN_WM_USE_RETINA=1 ./env_pytorch_3.12/bin/python diagnostics/diag_eat_value_probe.py $OUT/wm_best.pt 2>&1 | grep -vi warning | sed -n '/frames=/,$p'
echo "DECOUPLED_DONE"
