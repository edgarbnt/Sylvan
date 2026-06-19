extends RefCounted
class_name ActionAdapter

var action_dim := 18  # HEXAPOD: 6 legs × 3. (main.gd also syncs this from the agent.)


func normalize_action(raw_action: Array) -> Array[float]:
	var action: Array[float] = []
	for value in raw_action:
		action.append(clampf(float(value), -1.0, 1.0))
	if action.size() < action_dim:
		for _index in range(action.size(), action_dim):
			action.append(0.0)
	elif action.size() > action_dim:
		action = action.slice(0, action_dim)
	return action
