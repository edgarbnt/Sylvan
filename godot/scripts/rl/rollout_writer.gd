extends Node
class_name RolloutWriter

@export var output_directory := "user://replay_buffer"

var _current_episode_id := ""
var _last_step_id := -1
var _file: FileAccess


func configure_output_directory(path: String) -> void:
	output_directory = path


func begin_episode(episode_id: String) -> void:
	end_episode()
	_current_episode_id = episode_id
	_last_step_id = -1
	DirAccess.make_dir_recursive_absolute(output_directory)
	var path := output_directory.path_join("%s.jsonl" % episode_id)
	_file = FileAccess.open(path, FileAccess.WRITE)


func write_transition(transition: Dictionary) -> void:
	if _file == null:
		push_error("Cannot write transition without an open episode file")
		return
	var step_id := int(transition["info"]["step_id"])
	if step_id != _last_step_id + 1:
		push_error(
			"Non contiguous step_id for episode %s: expected %d got %d"
			% [_current_episode_id, _last_step_id + 1, step_id]
		)
		return
	_last_step_id = step_id
	_file.store_line(JSON.stringify(transition))
	# per-step flush() removed: end_episode() flushes+closes, and the trainer only reads completed
	# files after the worker exits (pool waits on p.wait()), so mid-episode durability is pointless —
	# this drops ~16-32k write syscalls/worker/iter off the hot single-core collection loop.


func end_episode() -> void:
	if _file != null:
		_file.flush()
		_file.close()
		_file = null
