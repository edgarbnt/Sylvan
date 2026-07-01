extends RefCounted
class_name PolicyPlayer

var _layers: Array = []
var _socket := StreamPeerTCP.new()
var _response_buffer := ""


func load_policy(path: String) -> void:
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		push_error("Unable to open policy file: %s" % path)
		return
	var payload = JSON.parse_string(file.get_as_text())
	file.close()
	if typeof(payload) != TYPE_DICTIONARY:
		push_error("Invalid policy payload")
		return
	_layers = payload.get("layers", [])


func is_ready() -> bool:
	return not _layers.is_empty() or is_server_ready()


func connect_to_server(host: String, port: int) -> void:
	_response_buffer = ""
	var error := _socket.connect_to_host(host, port)
	if error != OK:
		push_error("Unable to connect to policy server %s:%d" % [host, port])
	_socket.set_no_delay(true)  # TCP_NODELAY: flush the tiny action packet immediately (disable Nagle)


func is_server_ready() -> bool:
	_socket.poll()
	return _socket.get_status() == StreamPeerTCP.STATUS_CONNECTED


func predict(state: Array[float]) -> Array[float]:
	var current := state.duplicate()
	for layer in _layers:
		current = _apply_layer(current, layer)
	return current


func predict_observation(obs: Dictionary) -> Array[float]:
	if not is_server_ready():
		return []
	var payload := JSON.stringify(
		{
			"proprio": obs.get("proprio", []),
			"vision": obs.get("vision", []),
			"energy": obs.get("energy", 0.0),
			"metrics": obs.get("metrics", {}),
		}
	)
	var error = _socket.put_data((payload + "\n").to_utf8_buffer())
	if error != OK:
		push_error("Failed to send data to policy server")
		return []

	var deadline := Time.get_ticks_msec() + 1000
	while Time.get_ticks_msec() < deadline:
		_socket.poll()
		var status = _socket.get_status()
		if status != StreamPeerTCP.STATUS_CONNECTED:
			push_error("Disconnected from policy server")
			return []
			
		var available := _socket.get_available_bytes()
		if available > 0:
			_response_buffer += _socket.get_utf8_string(available)
			var newline_index := _response_buffer.find("\n")
			if newline_index != -1:
				var line := _response_buffer.substr(0, newline_index)
				_response_buffer = _response_buffer.substr(newline_index + 1)
				return _parse_action_response(line)
		OS.delay_usec(1000)  # KEEP 1ms: with N parallel workers sharing cores this SLEEP yields the CPU
		                     # to the other workers + their inference servers. Reducing it to a busy-spin
		                     # (tried 50µs) SATURATED the CPU and slowed collection ~60x — under concurrency
		                     # the poll is hidden, not dead time. Real win is structural (in-Godot inference).
	push_error("Timed out while waiting for policy server response")
	return []


func predict_planner(obs: Dictionary) -> Dictionary:
	# Phase 5: query the combined planner+residual server. Sends {proprio, vision=REAL radar, energy},
	# returns {"action":[12], "command":[vx, omega]}. Same wire protocol as predict_observation, plus
	# the command field the planner chose (Godot feeds it to the CPG so it steers by construction).
	if not is_server_ready():
		return {}
	var payload := JSON.stringify({
		"proprio": obs.get("proprio", []),
		"vision": obs.get("vision", []),
		"vision_fine": obs.get("vision_fine", []),  # A1: finer radar for planner food-localisation only
		"energy": obs.get("energy", 0.0),
		"vision_water": obs.get("vision_water", []), # 2ᵉ pulsion: fine WATER radar (planner-only, hors WM)
		"thirst": obs.get("thirst", 0.0),            # 2ᵉ pulsion: niveau de soif 0..100 (planner-only)
		"retina": obs.get("retina", []),             # RÉTINE étage 1: rayons couleur bruts → localisation apprise
		# Signal EXPLICITE de frontière d'épisode (Mode-1 collecte RL) : additif, ignoré par les autres
		# serveurs. episode_step = index de pas DANS l'épisode (remis à 0 au respawn) ; prev_term = raison
		# de fin de l'épisode PRÉCÉDENT, lisible au 1er tick du nouvel épisode ("death"/"truncated"/"none").
		"episode_step": obs.get("episode_step", -1),
		"prev_term": obs.get("prev_term", "none"),
	})
	var error = _socket.put_data((payload + "\n").to_utf8_buffer())
	if error != OK:
		push_error("Failed to send data to planner server")
		return {}
	var deadline := Time.get_ticks_msec() + 2000
	while Time.get_ticks_msec() < deadline:
		_socket.poll()
		if _socket.get_status() != StreamPeerTCP.STATUS_CONNECTED:
			push_error("Disconnected from planner server")
			return {}
		var available := _socket.get_available_bytes()
		if available > 0:
			_response_buffer += _socket.get_utf8_string(available)
			var newline_index := _response_buffer.find("\n")
			if newline_index != -1:
				var line := _response_buffer.substr(0, newline_index)
				_response_buffer = _response_buffer.substr(newline_index + 1)
				var parsed = JSON.parse_string(line)
				if typeof(parsed) != TYPE_DICTIONARY:
					push_error("Invalid planner server payload")
					return {}
				var action: Array[float] = []
				for v in parsed.get("action", []):
					action.append(float(v))
				var command: Array[float] = []
				for v in parsed.get("command", []):
					command.append(float(v))
				return {"action": action, "command": command}
		OS.delay_usec(1000)
	push_error("Timed out while waiting for planner server response")
	return {}


func _apply_layer(input_vector: Array[float], layer: Dictionary) -> Array[float]:
	var output: Array[float] = []
	var weights: Array = layer.get("weight", [])
	var bias: Array = layer.get("bias", [])
	for row_index in weights.size():
		var row: Array = weights[row_index]
		var value := float(bias[row_index])
		for col_index in row.size():
			value += float(row[col_index]) * input_vector[col_index]
		if layer.get("activation", "") == "tanh":
			value = tanh(value)
		output.append(value)
	return output


func _parse_action_response(line: String) -> Array[float]:
	var payload = JSON.parse_string(line)
	if typeof(payload) != TYPE_DICTIONARY:
		push_error("Invalid policy server payload")
		return []
	var action_payload: Array = payload.get("action", [])
	var action: Array[float] = []
	for value in action_payload:
		action.append(float(value))
	return action
