#!/usr/bin/env bash
# ─── run_sylvan.sh ────────────────────────────────────────────────────────────
# Wrapper pour lancer le container Sylvan avec les bons flags GPU AMD.
# Usage : ./run_sylvan.sh [options run-sylvan]
# Exemple : ./run_sylvan.sh --num-cycles 30 --steps-per-day 5000 --epochs-per-night 10 --run-name-prefix sylvan_run

set -euo pipefail

# Allocate a TTY only when stdin is an interactive terminal. Without this,
# background / non-TTY launches fail with "cannot attach stdin to a TTY-enabled
# container because stdin is not a terminal".
TTY_FLAGS="-i"
if [ -t 0 ]; then TTY_FLAGS="-it"; fi

docker run --rm ${TTY_FLAGS} \
  --user "$(id -u):$(id -g)" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  -e SYLVAN_TORCH_DEVICE="${SYLVAN_TORCH_DEVICE:-auto}" \
  -v "$(pwd):/workspace" \
  sylvan:latest \
  "$@"
