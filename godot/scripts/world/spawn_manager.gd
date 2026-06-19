extends RefCounted
class_name SpawnManager

@export var spawn_position := Vector3.ZERO
@export var spawn_radius := 0.35
@export var yaw_range_degrees := 20.0
# No spawn push during the "first steps" phase: a random shove on an inverted
# pendulum guarantees an instant fall. Re-enable (>0) later as a robustness curriculum.
@export var push_impulse_min := 0.0
@export var push_impulse_max := 0.0

var _base_seed := 42
var _rng := RandomNumberGenerator.new()
var _episode_spawn_position := Vector3.ZERO
var _episode_spawn_yaw := 0.0
var _episode_spawn_impulse := Vector3.ZERO
var _episode_ready := false


func set_seed(value: int) -> void:
	_base_seed = value
	_rng.seed = value


func begin_episode(episode_index: int) -> void:
	_rng.seed = int(_base_seed + episode_index * 9973)
	var angle := _rng.randf_range(0.0, TAU)
	var radius := _rng.randf_range(0.0, spawn_radius)
	_episode_spawn_position = spawn_position + Vector3(cos(angle) * radius, 0.0, sin(angle) * radius)
	_episode_spawn_yaw = deg_to_rad(_rng.randf_range(-yaw_range_degrees, yaw_range_degrees))
	var push_angle := _rng.randf_range(0.0, TAU)
	var push_strength := _rng.randf_range(push_impulse_min, push_impulse_max)
	_episode_spawn_impulse = Vector3(cos(push_angle) * push_strength, 0.0, sin(push_angle) * push_strength)
	_episode_ready = true


func get_agent_spawn_position() -> Vector3:
	if _episode_ready:
		return _episode_spawn_position
	return spawn_position


func get_agent_spawn_yaw() -> float:
	if _episode_ready:
		return _episode_spawn_yaw
	return 0.0


func get_agent_spawn_impulse() -> Vector3:
	if _episode_ready:
		return _episode_spawn_impulse
	return Vector3.ZERO
