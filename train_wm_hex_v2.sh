#!/bin/zsh
cd /home/edgarbrunet/Documents/PERSO/SylvanV1
export PYTHONPATH=python
./env_pytorch_3.12/bin/python -m scripts.train_wm_command \
  --runs data/replay_buffer/wm_hex_v2_a data/replay_buffer/wm_hex_v2_b \
  --out data/checkpoints/wm_command_hex_v2 --epochs 20
