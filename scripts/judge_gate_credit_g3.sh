#!/bin/zsh
# G3 — JUGE closed-loop du chantier ATTRIBUTION DE CRÉDIT (baie-buisson), PRÉ-INSCRIT
# (docs/design_attribution_credit.md §Gates G3). Bras CRÉDIT : WM CRÉDIT-TYPÉ (buisson NEUTRE,
# marges water/danger resserrées pour exclure le teal) dans le monde COMPLET (food/eau/danger) +
# BUISSON co-localisé à la bouffe. 2×24 vies seeds 1+2, config vivante (waypoint+saillance+sprint).
# Réf vivante MESURÉE (pas de re-run) : 42 repas / 10 morts-danger (monde varié SANS buisson).
# PASS-parité = repas poolés ≥ 36 ET morts-danger ≤ 13 (le buisson ne coûte rien) ; « le slot food
# ne se verrouille jamais sur le buisson » est déjà PROUVÉ perceptuellement (cos teal 0.59 < marge
# food 0.761 ; water 0.92 < 0.976, danger 0.86 < 0.936 → buisson exclu de TOUS les slots).
# KILL précoce = seed 1 < 14 repas.
#
# Usage : bash scripts/judge_gate_credit_g3.sh
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1; cd "$ROOT"

export WM_CKPT=data/checkpoints/wm_objcentric_kin_typed_credit/wm_best.pt
[[ -f "$WM_CKPT" ]] || { echo "[g3] WM crédit introuvable (build_typed_slots_credit d'abord)"; exit 1; }
export SYLVAN_FOOD_APPEARANCE_VAR=0.15 SYLVAN_WATER_APPEARANCE_VAR=0.15
# BUISSON NEUTRE co-localisé à la bouffe (mêmes propriétés monde déclarées qu'en G1/G2)
export SYLVAN_FOOD_BUSH=1 SYLVAN_FOOD_BUSH_HUE=0.45 SYLVAN_FOOD_BUSH_P=0.9 SYLVAN_FOOD_BUSH_ALONE=1

echo "=== G3 CRÉDIT seed 1 (WM crédit + monde complet + buisson) ==="
bash scripts/judge_saliency_p5.sh 1 g3cr1
echo "=== G3 CRÉDIT seed 2 ==="
bash scripts/judge_saliency_p5.sh 2 g3cr2

echo "=== G3 VERDICT ==="
PYTHONPATH=python env_pytorch_3.12/bin/python3 -c "
import sys; sys.path.insert(0,'diagnostics')
from diag_hazard_gate import parse_lives
tot_m=tot_d=nlives=0; s1m=0
for tag in ('g3cr1','g3cr2'):
    lv=parse_lives('data/replay_buffer/critic_kin_'+tag+'/godot.log')
    m=sum(x['meals'] for x in lv); d=sum(1 for x in lv if x['cause']=='danger')
    if tag=='g3cr1': s1m=m
    print(f'  {tag}: {len(lv)} vies, {m} repas, {d} morts-danger')
    tot_m+=m; tot_d+=d; nlives+=len(lv)
print(f'POOLÉ: {nlives} vies, {tot_m} repas, {tot_d} morts-danger (réf vivante SANS buisson 42/10)')
kill = s1m < 14
ok = (not kill) and tot_m>=36 and tot_d<=13
print('[g3]', ('KILL précoce (seed1<14)' if kill else ('PASS-parité' if ok else 'ÉCHEC')),
      '- le buisson ne coûte pas le forage' if ok else '- diagnostiquer sur trace')
"
echo "ALL_DONE_G3_CREDIT"
