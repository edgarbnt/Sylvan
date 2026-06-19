extends RefCounted
class_name ObservationBuilder

const PROPRIO_DIM := 132  # HEXAPOD: 7 + 13 bodies×6(=78) + 6 contacts + 3 COM + 18 angles + 18 vels + 2 gait clock

func build_observation(agent, energy: float, health: float, vision: Array = []) -> Dictionary:
	var proprio: Array = agent.get_proprioception() as Array
	if proprio.size() != PROPRIO_DIM:
		push_error("Expected proprio dim %d, got %d" % [PROPRIO_DIM, proprio.size()])
	return {
		"proprio": proprio,
		"vision": vision,   # egocentric food radar (12-d); empty if perception is off
		"energy": energy,
		"health": health,
		"metrics": agent.get_locomotion_metrics(),
	}
