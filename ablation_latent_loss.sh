#!/bin/zsh
# ABLATION ISOLANTE (owner 2026-06-19) — CONFIRMER que la dérive open-loop du WM-rétine vient du LATENT-LOSS
# (cosine, direction-only) AVANT de payer le full retrain. Deux runs FROM-SCRATCH sur les MÊMES données
# (retina_wm_a+b, babbling propre AVEC rétine), MÊMES epochs, identiques SAUF la perte latente :
#   • abl_mse    : --latent-loss mse                       (recette du live v2, ancre la MAGNITUDE)
#   • abl_cosine : --latent-loss cosine + VICReg(1,1)      (recette rétine actuelle, direction-only)
# Puis diag_dream_disp.py sur chacun (path du rêve droit 120 pas + cohérence latente + eff_rank).
#
# CRITÈRES PRÉ-ÉCRITS (CLAUDE.md §1/§2) :
#   SUCCÈS (cause confirmée = latent-loss) : abl_mse path ≥ ~0.45 m ET ≥ 2× abl_cosine ET eff_rank non
#     re-collapsé (≥ ~8) → le MSE restaure le déplacement open-loop sans tuer le latent → le full retrain
#     food-aware passe en mse (ou hybride mse+cosine). Re-passer diag_cem.py comme gate ENSUITE.
#   KILL-A (pas le latent-loss) : LES DEUX restent ~0.2 m → la dérive vient de l'ENCODEUR/obs rétine
#     (144-dim vision qui noie le proprio), pas de la perte → escalade (skip proprio / rebalance), PAS mse.
#   KILL-B (tension) : abl_mse restaure le path MAIS eff_rank s'effondre (< ~8) alors que abl_cosine le garde
#     → le cosine était porteur du latent food-aware → tester un HYBRIDE mse+cosine, ne pas trancher d'office.
#
# Lancer backgroundé (tuer orphelins AVANT, à part) : nohup zsh ablation_latent_loss.sh > /tmp/abl.log 2>&1 &
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
export PYTHONPATH=python
RUNS="data/replay_buffer/retina_wm_a data/replay_buffer/retina_wm_b"
EP=8

echo "############### 1/2 : ABL MSE (latent-loss mse, vicreg off) — from scratch ###############"
SYLVAN_WM_USE_RETINA=1 ./env_pytorch_3.12/bin/python -m scripts.train_wm_command \
  --runs ${=RUNS} --out data/checkpoints/abl_mse \
  --epochs $EP --stride 8 --batch-size 48 --lr 1e-4 \
  --latent-loss mse
echo "abl_mse exit=$?"

echo "############### 2/2 : ABL COSINE (latent-loss cosine + VICReg) — from scratch ###############"
SYLVAN_WM_USE_RETINA=1 ./env_pytorch_3.12/bin/python -m scripts.train_wm_command \
  --runs ${=RUNS} --out data/checkpoints/abl_cosine \
  --epochs $EP --stride 8 --batch-size 48 --lr 1e-4 \
  --latent-loss cosine --vicreg-var 1.0 --vicreg-cov 1.0 --vicreg-gamma 1.0
echo "abl_cosine exit=$?"

echo "############### DIAG OPEN-LOOP (le verdict) ###############"
echo "=== abl_mse ==="
SYLVAN_WM_USE_RETINA=1 ./env_pytorch_3.12/bin/python diag_dream_disp.py data/checkpoints/abl_mse/wm_best.pt 1 2>&1 | grep -v -i warning
echo "=== abl_cosine ==="
SYLVAN_WM_USE_RETINA=1 ./env_pytorch_3.12/bin/python diag_dream_disp.py data/checkpoints/abl_cosine/wm_best.pt 1 2>&1 | grep -v -i warning
echo "############### ABLATION TERMINÉE — comparer path médian (cible mse ≥0.45m & ≥2× cosine) ###############"
