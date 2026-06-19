#!/bin/zsh
# A/B: real hexapod foraging WITHOUT vs WITH the planner heading-alignment term.
# Same WM/residual/horizon/eat_radius/food_count; only SYLVAN_PLANNER_HEADING_W differs.
# Survival score = max step reached per episode (death = energy critical), median over episodes.
set +e
ROOT=/home/edgarbrunet/Documents/PERSO/SylvanV1
cd "$ROOT"
ER=${1:-1.0}; HZ=${2:-80}; NEP=${3:-12}
for HW in 0.0 2.0; do
  echo "######## foraging heading_w=$HW ########"
  bash run_forage_hex.sh $ER $HZ $NEP $HW > /tmp/forage_ab_run_${HW}.log 2>&1
  cp /tmp/forage_hex.log /tmp/forage_ab_${HW}.log
  echo "saved /tmp/forage_ab_${HW}.log"
done
echo "AB_DONE"
