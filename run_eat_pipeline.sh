#!/bin/zsh
# A2 (vers 🅑) — collecte EAT-RICHE + retrain WM-rétine + GATE (diag). Le WM ne capte que 4% de la bosse-repas
# (23 repas/18k) → on lui donne des centaines de repas BASSE-énergie (marge réelle) puis on re-mesure.
# GATÉ : on ne touchera au planner QUE si diag_latent_foodaware.py remonte (eat-fidelity ≥~40% + test direct).
# Lancer backgroundé : nohup zsh run_eat_pipeline.sh > /tmp/eat_pipeline.log 2>&1 &  (tuer orphelins AVANT, à part)
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"
GD=godot/data/replay_buffer
echo "=================== ÉTAPE 1/4 : COLLECTE EAT-RICHE (gate varié) ==================="
# REPRISE : si la collecte est déjà faite (60+60 ép), on la SAUTE (ne pas re-payer ~30 min). Forcer = RECOLLECT=1.
NA=$(ls $GD/retina_eat_a/episode_*.jsonl 2>/dev/null | wc -l)
NB=$(ls $GD/retina_eat_b/episode_*.jsonl 2>/dev/null | wc -l)
if [ "${RECOLLECT:-0}" != "1" ] && [ "$NA" -ge 60 ] && [ "$NB" -ge 60 ]; then
  echo "collecte déjà présente (A=$NA B=$NB ép) → SAUTÉE (RECOLLECT=1 pour forcer)."
else
  echo "### run A : 60 ép, seed 41, gate 0.65 (grosse marge) ###"
  bash collect_wm_eatrich.sh 60 41 14 0.10 100 retina_eat_a 0.65 2>&1 | grep -E "épisodes=|repas\(ate\)|repas BASSE|énergie GLOB"
  echo "### run B : 60 ép, seed 77, gate 0.82 (plus d'événements) ###"
  bash collect_wm_eatrich.sh 60 77 14 0.10 100 retina_eat_b 0.82 2>&1 | grep -E "épisodes=|repas\(ate\)|repas BASSE|énergie GLOB"
fi

echo "=================== compte total des repas collectés ==================="
./env_pytorch_3.12/bin/python - <<'PY'
import json, glob
tot = lo = rows = 0
for d in ["godot/data/replay_buffer/retina_eat_a", "godot/data/replay_buffer/retina_eat_b"]:
    for f in glob.glob(d + "/episode_*.jsonl"):
        for ln in open(f):
            r = json.loads(ln); rows += 1
            if r.get("wm", {}).get("ate"):
                tot += 1
                if r["obs"]["energy"] < 70: lo += 1
print(f"TOTAL repas={tot} (basse-énergie <70 = {lo}, {100*lo/max(1,tot):.0f}%) sur {rows} lignes  [réf: ancien = 23 repas]")
PY

echo "=================== ÉTAPE 2/4 : RETRAIN WM-RÉTINE (JEPA + eat-weight musclé) ==================="
export PYTHONPATH=python
SYLVAN_WM_USE_RETINA=1 SYLVAN_EAT_SAMPLE_WEIGHT=60 \
./env_pytorch_3.12/bin/python -m scripts.train_wm_command \
  --runs $GD/retina_eat_a $GD/retina_eat_b data/replay_buffer/retina_wm_a data/replay_buffer/retina_wm_b \
  --out data/checkpoints/wm_command_hex_retina_eat_v1 \
  --epochs 20 --lr 1e-4 --latent-loss cosine --vicreg-var 1.0 --vicreg-cov 1.0 --vicreg-gamma 1.0
echo "retrain exit=$?"

echo "=================== ÉTAPE 3/4 : ÉVAL OPEN-LOOP ==================="
SYLVAN_WM_USE_RETINA=1 ./env_pytorch_3.12/bin/python -m scripts.eval_wm_command \
  --checkpoint data/checkpoints/wm_command_hex_retina_eat_v1/wm_best.pt --horizons 50 80 100 150 2>&1 | tail -20

echo "=================== ÉTAPE 4/4 : GATE — diag_latent_foodaware (eat-fidelity + test direct) ==================="
SYLVAN_WM_USE_RETINA=1 ./env_pytorch_3.12/bin/python diag_latent_foodaware.py \
  data/checkpoints/wm_command_hex_retina_eat_v1/wm_best.pt 2>&1
echo "=================== PIPELINE TERMINÉ ==================="