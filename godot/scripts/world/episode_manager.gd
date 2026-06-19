extends RefCounted
class_name EpisodeManager

# Episode lifecycle bookkeeping (step counter + truncation at the time-limit). REWRITTEN 2026-06-15
# from main.gd's usage after an accidental deletion; original was never captured in any transcript.
# Interface required by main.gd: max_episode_steps, episode_index, current_episode_id, current_step_id,
# start_episode(), finish_episode(reason), next_step_id()->int, should_truncate()->bool.

var max_episode_steps := 400
var episode_index := 0            # which episode this is (used for per-episode seeding)
var current_episode_id := ""      # STRING id → rollout_writer names the file "%s.jsonl" (episode_0000.jsonl)
var current_step_id := 0          # decision-step counter within the current episode


func start_episode() -> void:
	current_step_id = 0
	current_episode_id = "episode_%04d" % episode_index


func next_step_id() -> int:
	var s := current_step_id
	current_step_id += 1
	return s


func should_truncate() -> bool:
	# Truncate (time-limit, NOT a fall) once we've reached the step budget.
	return current_step_id >= max_episode_steps


func finish_episode(_reason: String) -> void:
	episode_index += 1
