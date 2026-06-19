extends RefCounted
class_name RLBridge

@export var scene_version := "world_v0"
@export var schema_version := "phase1.v1"


func build_transition(
	obs: Dictionary,
	action: Array[float],
	reward: float,
	next_obs: Dictionary,
	done: bool,
	truncated: bool,
	episode_id: String,
	step_id: int,
	seed_value: int,
	agent_version: String
) -> Dictionary:
	return {
		"obs": obs,
		"action": action,
		"reward": reward,
		"next_obs": next_obs,
		"done": done,
		"truncated": truncated,
		"info": {
			"episode_id": episode_id,
			"step_id": step_id,
			"seed": seed_value,
			"scene_version": scene_version,
			"agent_version": agent_version,
			"timestamp": Time.get_datetime_string_from_system(true),
			"schema_version": schema_version,
		},
	}
