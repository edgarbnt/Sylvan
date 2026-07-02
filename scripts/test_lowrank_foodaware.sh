#!/bin/zsh
# TEST GRATUIT (owner 2026-06-19) — un WM eff_rank-BAS (rêve fidèle) porte-t-il QUAND MÊME la bouffe ?
# = trancher le COMPROMIS « latent riche (VICReg, eff_rank haut) ↔ rêve qui bouge (eff_rank bas) ».
# Recette eff_rank-bas = celle du live v2 (latent-loss mse, VICReg var=0/cov=0) MAIS food-aware via --w-food
# (tête auxiliaire 'repas imminent' sur les latents RÊVÉS, INDÉPENDANTE de VICReg) sur les eat data.
# Warm-start depuis abl_mse (dynamique propre, eff_rank bas) pour garder le rêve mobile.
#
# CRITÈRES PRÉ-ÉCRITS (CLAUDE.md §1) — mesurés après : path (diag_dream_disp) + food_auc (diag_eat_value_probe) :
#   SUCCÈS (PAS de compromis) : food_auc ≥ ~0.75  ET  path ≥ ~0.45 m → un WM dynamique food-aware existe →
#     🅑 redevient jouable (latent appris porte la bouffe + rêve avance). On n'avait pas besoin du VICReg fort.
#   KILL (compromis RÉEL) : food_auc s'effondre vers ~0.67 (niveau énergie-seule) malgré bon path → eff_rank-bas
#     NE porte PAS la bouffe → il FAUT enrichir (VICReg) qui casse le rêve → option (b) dynamique explicite.
#   AMBIGU : path < 0.35 m → le w-food/eat_weight a quand même figé → relancer avec eat_weight plus bas.
#
# Lancer backgroundé (orphelins tués À PART avant) : nohup zsh test_lowrank_foodaware.sh > /tmp/lowrank.log 2>&1 &
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
export PYTHONPATH=python
GD=godot/data/replay_buffer
OUT=data/checkpoints/wm_lowrank_eat

echo "############### TRAIN WM eff_rank-BAS + food-aware (mse, VICReg OFF, --w-food, warm abl_mse) ###############"
SYLVAN_WM_USE_RETINA=1 SYLVAN_EAT_SAMPLE_WEIGHT=20 \
./env_pytorch_3.12/bin/python -m scripts.train_wm_command \
  --runs $GD/retina_eat_a $GD/retina_eat_b data/replay_buffer/retina_wm_a data/replay_buffer/retina_wm_b \
  --out $OUT \
  --init-from data/checkpoints/abl_mse/wm_best.pt \
  --epochs 10 --stride 8 --batch-size 48 --lr 1e-4 \
  --latent-loss mse --vicreg-var 0.0 --vicreg-cov 0.0 \
  --w-food 0.5
echo "train exit=$?"

echo "############### MESURE 1 : path du rêve (le rêve avance-t-il ?) ###############"
SYLVAN_WM_USE_RETINA=1 ./env_pytorch_3.12/bin/python diagnostics/diag_dream_disp.py $OUT/wm_best.pt 1 2>&1 | grep -v -i warning
echo "############### MESURE 2 : food_auc (le latent porte-t-il la bouffe ?) ###############"
SYLVAN_WM_USE_RETINA=1 ./env_pytorch_3.12/bin/python diagnostics/diag_eat_value_probe.py $OUT/wm_best.pt 2>&1 | grep -vi warning | sed -n '/frames=/,$p'
echo "LOWRANK_DONE"
