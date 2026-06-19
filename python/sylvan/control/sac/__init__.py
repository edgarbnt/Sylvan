"""Soft Actor-Critic (off-policy) for sample-efficient locomotion at small scale.

Why SAC here: PPO (on-policy) throws away every transition after one update, so its
sample-efficiency scales with parallelism. We have ~8 Godot/CPU envs (not Isaac's 4096),
and the exhaustive turn-agility investigation proved the ~14 deg/s yaw ceiling is a
sample-efficiency wall, NOT the body or the reward (a spin-specialist PPO with a linear
high-yaw reward still capped at ~15 deg/s). SAC reuses every transition many times from a
persistent replay buffer, so it can extract far more learning per env-step — the last
non-Isaac lever to break the ceiling.

It reuses the WHOLE existing collection pipeline: the SAC actor is served over the same
TCP protocol Godot speaks, and the JSONL it writes already carries (obs, action, reward,
next_obs, done) — exactly the off-policy tuple SAC needs.
"""
