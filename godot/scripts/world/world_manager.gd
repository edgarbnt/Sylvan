extends RefCounted
class_name WorldManager

# World/terrain reset hook. REWRITTEN 2026-06-15 from main.gd's usage after an accidental deletion
# (original never captured in any transcript). The actual static terrain lives in world_v0.tscn; this
# manager only carries the seed and a per-episode RNG for any future world randomisation. reset_world
# is a safe no-op placeholder — the locomotion/foraging tasks run on the flat world scene + food_manager.
# Interface required by main.gd: set_seed(seed), reset_world(episode_index).

var _base_seed := 42
var _rng := RandomNumberGenerator.new()


func set_seed(value: int) -> void:
	_base_seed = value
	_rng.seed = value


func reset_world(episode_index: int) -> void:
	# Re-seed deterministically per episode so any future world variation is reproducible.
	_rng.seed = int(_base_seed + episode_index * 7919)
