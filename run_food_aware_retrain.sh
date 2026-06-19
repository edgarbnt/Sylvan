#!/bin/zsh
# 🅑 — RETRAIN WM FOOD-AWARE (le rêve doit transporter la bouffe) + ré-entraîner la tête de valeur + GATE.
# Perte auxiliaire 'repas imminent' sur les latents RÊVÉS (--w-food) → force le dream à garder la nourriture.
# Puis on re-teste le coût-valeur latent (diag_value_direct) : argmax(V) doit ENFIN aller vers la bouffe.
# Lancer backgroundé (tuer orphelins AVANT, à part) : nohup zsh run_food_aware_retrain.sh > /tmp/food_retrain.log 2>&1 &
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
GD=godot/data/replay_buffer
WMOUT=data/checkpoints/wm_command_hex_retina_eat_v2
WFOOD=${WFOOD:-0.5}

echo "=================== 1/4 : RETRAIN WM FOOD-AWARE (w_food=$WFOOD, warm-start + stride8 + batch48) ==================="
# OPTI : warm-start depuis eat_v1 (dynamique déjà bonne → on n'injecte QUE le food) → 10 epochs suffisent ;
# stride 8 (fenêtres quasi-doublons → 2× moins) ; batch 48 (16 cœurs sous-exploités). num_workers INCHANGÉ
# (sur-souscrire fait thrasher la box — leçon connue). ~2h → ~20 min, sans perte de qualité.
export PYTHONPATH=python
SYLVAN_WM_USE_RETINA=1 SYLVAN_EAT_SAMPLE_WEIGHT=60 \
./env_pytorch_3.12/bin/python -m scripts.train_wm_command \
  --runs $GD/retina_eat_a $GD/retina_eat_b data/replay_buffer/retina_wm_a data/replay_buffer/retina_wm_b \
  --out $WMOUT \
  --init-from data/checkpoints/wm_command_hex_retina_eat_v1/wm_best.pt \
  --epochs 10 --stride 8 --batch-size 48 --lr 1e-4 \
  --latent-loss cosine --vicreg-var 1.0 --vicreg-cov 1.0 --vicreg-gamma 1.0 \
  --w-food $WFOOD
echo "retrain exit=$?"

echo "=================== 2/4 : ÉVAL OPEN-LOOP (le déplacement n'a pas régressé ?) ==================="
SYLVAN_WM_USE_RETINA=1 ./env_pytorch_3.12/bin/python -m scripts.eval_wm_command \
  --checkpoint $WMOUT/wm_best.pt --horizons 50 80 100 150 2>&1 | tail -12

echo "=================== 3/4 : RÉ-ENTRAÎNER LA TÊTE DE VALEUR sur le NOUVEAU WM ==================="
SYLVAN_WM_USE_RETINA=1 ./env_pytorch_3.12/bin/python train_value_head.py $WMOUT/wm_best.pt 2>&1 | tail -4

echo "=================== 4/4 : GATE — diag_value_direct (argmax-V → bouffe ?) ==================="
SYLVAN_WM_USE_RETINA=1 ./env_pytorch_3.12/bin/python diag_value_direct.py \
  $WMOUT/wm_best.pt data/checkpoints/value_head_food/value_best.pt 2>&1 | sed -n '/e0=/,$p'
echo "=================== PIPELINE FOOD-AWARE TERMINÉ ==================="