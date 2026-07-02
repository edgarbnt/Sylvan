#!/bin/zsh
# Hexapod omni residual retrain: symmetrize turning + keep forward speed while turning.
# Warm-start hexapod_v2, clean regime, omega curriculum 0.15->0.6, command-aware symmetry, omni reward.
cd /home/edgarbrunet/Documents/PERSO/SylvanV1
export SYLVAN_CPG=1 SYLVAN_RESIDUAL_GAIN=0.4 SYLVAN_TURN_FADE=0 SYLVAN_FOOT_FRICTION=7
export SYLVAN_CPG_SPEEDCAD=0.6 SYLVAN_CPG_PERIOD=0.5
export SYLVAN_REWARD_OBJECTIVE=locomotion_omni_v1 SYLVAN_CMD_CURRIC=1 SYLVAN_CMD_VX_MAX=0.6 SYLVAN_MIRROR_COMMAND=1
export PYTHONPATH=python
export GODOT_BIN="$(pwd)/tools/godot/godot"
env_pytorch_3.12/bin/python -m scripts.train_ppo \
  --run-prefix ppo_hexapod_omega --ckpt-name hexapod_omega \
  --init-from data/checkpoints/hexapod_v2/policy_best.pt \
  --iterations 150 --num-workers 8 --lr 1e-4 \
  --cmd-wmax-start 0.15 --cmd-wmax-end 0.6 --cmd-wmax-cycles 120 \
  --sym-coef 1.0 --sym-coef-start 0.0 --sym-coef-cycles 35 --mirror-augment \
  --best-metric stable_fwd
