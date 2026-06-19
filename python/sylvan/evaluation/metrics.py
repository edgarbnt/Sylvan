"""Evaluation metrics helpers."""

from __future__ import annotations

from pathlib import Path

from ..buffer.reader import iter_episodes
from ..constants import LOCOMOTION_METRIC_KEYS


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if percentile <= 0.0:
        return float(min(values))
    if percentile >= 100.0:
        return float(max(values))
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def summarize_run(run_dir: Path) -> dict[str, float]:
    episodes = iter_episodes(run_dir)
    if not episodes:
        return {
            "num_episodes": 0,
            "num_transitions": 0,
            "mean_episode_length": 0.0,
            "mean_episode_return": 0.0,
            "mean_reward": 0.0,
            "p10_episode_length": 0.0,
            "p50_episode_length": 0.0,
            "p10_episode_return": 0.0,
            "p50_episode_return": 0.0,
            "done_rate": 0.0,
            "truncated_rate": 0.0,
            "early_termination_rate": 0.0,
            **{f"mean_{key}": 0.0 for key in LOCOMOTION_METRIC_KEYS},
        }
    lengths = [len(episode) for episode in episodes]
    transition_count = sum(lengths)
    metric_sums = {key: 0.0 for key in LOCOMOTION_METRIC_KEYS}
    episode_returns = []
    done_count = 0
    truncated_count = 0
    early_termination_count = 0
    early_termination_step = 80
    for episode in episodes:
        if len(episode) < early_termination_step:
            early_termination_count += 1
        if episode[-1].done:
            done_count += 1
        if episode[-1].truncated:
            truncated_count += 1
        episode_return = 0.0
        for transition in episode:
            episode_return += transition.reward
            for key in LOCOMOTION_METRIC_KEYS:
                metric_sums[key] += transition.obs.metrics.get(key, 0.0)
        episode_returns.append(episode_return)

    if transition_count == 0:
        return {
            "num_episodes": float(len(episodes)),
            "num_transitions": 0.0,
            "mean_episode_length": 0.0,
            "mean_episode_return": 0.0,
            "mean_reward": 0.0,
            "p10_episode_length": 0.0,
            "p50_episode_length": 0.0,
            "p10_episode_return": 0.0,
            "p50_episode_return": 0.0,
            "done_rate": float(done_count / len(episodes)),
            "truncated_rate": float(truncated_count / len(episodes)),
            "early_termination_rate": float(early_termination_count / len(episodes)),
            **{f"mean_{key}": 0.0 for key in LOCOMOTION_METRIC_KEYS},
        }

    return {
        "num_episodes": float(len(episodes)),
        "num_transitions": float(transition_count),
        "mean_episode_length": float(transition_count / len(lengths)),
        "mean_episode_return": float(sum(episode_returns) / len(episode_returns)),
        "mean_reward": float(sum(episode_returns) / transition_count),
        "p10_episode_length": _percentile([float(length) for length in lengths], 10.0),
        "p50_episode_length": _percentile([float(length) for length in lengths], 50.0),
        "p10_episode_return": _percentile(episode_returns, 10.0),
        "p50_episode_return": _percentile(episode_returns, 50.0),
        "done_rate": float(done_count / len(episodes)),
        "truncated_rate": float(truncated_count / len(episodes)),
        "early_termination_rate": float(early_termination_count / len(episodes)),
        **{
            f"mean_{key}": float(metric_sums[key] / transition_count)
            for key in LOCOMOTION_METRIC_KEYS
        },
    }
