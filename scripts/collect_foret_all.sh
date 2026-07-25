#!/bin/bash
# Enchaîne les LOTS de la collecte forêt (graines distinctes) puis rapporte le total.
# Trois lots de 150 vies ~ 120 k ticks, le volume des collectes WM historiques.
# Usage : bash scripts/collect_foret_all.sh [lives-par-lot] [tag-base]
set -uo pipefail
cd "$(dirname "$0")/.."
LIVES="${1:-150}"
BASE="${2:-foret_v1}"
for i in 0 1 2; do
  tag="$BASE"; [[ $i -gt 0 ]] && tag="${BASE}$(printf '%b' "$(printf '\\x%x' $((0x61+i)))")"
  seed=$((11 + i))
  echo "########## LOT $((i+1))/3 : tag=$tag graine=$seed"
  bash scripts/collect_foret_v1.sh "$LIVES" "$seed" "$tag"
done
echo
echo "########## TOTAL"
env_pytorch_3.12/bin/python - <<'PYEOF'
import glob, os
t = e = b = 0
for d in sorted(glob.glob("data/replay_buffer/foret_v1*")):
    if not os.path.isdir(d):
        continue
    fs = glob.glob(d + "/*.jsonl")
    t += sum(sum(1 for _ in open(f)) for f in fs)
    e += len(fs)
    b += sum(os.path.getsize(f) for f in fs)
print(f"{e} épisodes | {t} ticks | {b/1e9:.2f} Go")
PYEOF
