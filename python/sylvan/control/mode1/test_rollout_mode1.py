# Tests Task-8 : build_rollout_batch_mode1 (split épisodes, bootstrap done/truncated, GAE fini).
# Usage: PYTHONPATH=python ./env_pytorch_3.12/bin/python -m pytest \
#            python/sylvan/control/mode1/test_rollout_mode1.py -q
import json

import torch

from sylvan.control.mode1.policy import DriveSymmetricPolicy
from sylvan.control.mode1.rollout_mode1 import (
    build_rollout_batch_mode1,
    _split_episodes,
)


def _line(reward=1.0, steps=10, done=False, truncated=False, energy=50.0, thirst=60.0):
    """Une macro-transition synthétique valide (proprio 132, retina 144, tokens D=2)."""
    return {
        "proprio": [0.01 * (i % 7) for i in range(132)],
        "retina": [0.0] * 144,
        "energy": energy, "thirst": thirst,
        "command_raw": [0.2, -0.1], "command_act": [0.65, -0.06],
        "reward": reward, "steps": steps, "done": done, "truncated": truncated,
    }


def _write_buffer(tmp_path, lines, seed=1):
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / f"part-{seed}.jsonl"
    with open(p, "w") as fh:
        for ln in lines:
            fh.write(json.dumps(ln) + "\n")
    return tmp_path


def test_split_episodes_done_and_truncated():
    lines = [
        _line(), _line(done=True),                 # épisode 1 : 2 transitions, terminé par done
        _line(), _line(), _line(truncated=True),   # épisode 2 : 3 transitions, terminé par truncated
    ]
    eps = _split_episodes(lines)
    assert [len(e) for e in eps] == [2, 3]
    assert eps[0][-1]["done"] is True
    assert eps[1][-1]["truncated"] is True


def test_dangling_episode_kept():
    lines = [_line(done=True), _line(), _line()]  # 2e épisode sans terminal
    eps = _split_episodes(lines)
    assert [len(e) for e in eps] == [1, 2]


def test_build_batch_bootstrap_branches(tmp_path):
    # Épisode DONE (fin réelle → last_value=0) puis TRUNCATED (self-bootstrap).
    lines = [
        _line(reward=1.0), _line(reward=1.0, done=True),
        _line(reward=1.0), _line(reward=1.0, truncated=True),
    ]
    _write_buffer(tmp_path, lines)
    pol = DriveSymmetricPolicy()
    batch, stats = build_rollout_batch_mode1(tmp_path, pol, gamma=0.99, lam=0.95)

    assert batch is not None
    assert stats["n_transitions"] == 4.0
    assert stats["n_episodes"] == 2.0
    assert stats["n_done"] == 1.0
    assert stats["n_truncated"] == 1.0
    # survie = Σ steps par épisode (10+10 et 10+10) → moyenne 20
    assert abs(stats["mean_episode_steps"] - 20.0) < 1e-6
    # tout fini, formes cohérentes
    for t in (batch.obs, batch.actions, batch.old_log_prob, batch.advantages, batch.returns, batch.values):
        assert torch.isfinite(t).all()
    assert batch.obs.shape == (4, 132 + 2 * 38)
    assert batch.actions.shape == (4, 2)
    assert batch.advantages.shape == (4,)
    # avantages normalisés (moyenne ~0)
    assert abs(float(batch.advantages.mean())) < 1e-5


def test_done_zeroes_bootstrap_vs_truncated(tmp_path):
    # Vérifie que la branche done met bien last_value=0 : sur un épisode d'UNE transition done,
    # return == reward (aucun bootstrap), contrairement à un épisode tronqué (return != reward).
    pol = DriveSymmetricPolicy()

    _write_buffer(tmp_path / "d", [_line(reward=2.0, done=True)])
    b_done, _ = build_rollout_batch_mode1(tmp_path / "d", pol, gamma=0.99, lam=0.95)
    assert abs(float(b_done.returns[0]) - 2.0) < 1e-5  # return == reward, pas de futur

    _write_buffer(tmp_path / "t", [_line(reward=2.0, truncated=True)])
    b_trunc, _ = build_rollout_batch_mode1(tmp_path / "t", pol, gamma=0.99, lam=0.95)
    # tronqué : return = reward + gamma * V(obs) → diffère de reward dès que V != 0
    v = float(pol.value(b_trunc.obs[:1])[0])
    assert abs(float(b_trunc.returns[0]) - (2.0 + 0.99 * v)) < 1e-4


def test_empty_buffer_returns_none(tmp_path):
    batch, stats = build_rollout_batch_mode1(tmp_path, DriveSymmetricPolicy(), gamma=0.99, lam=0.95)
    assert batch is None
    assert stats["n_transitions"] == 0.0
