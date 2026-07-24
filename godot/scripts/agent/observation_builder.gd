extends RefCounted
class_name ObservationBuilder

const PROPRIO_DIM := 132  # HEXAPOD: 7 + 13 bodies×6(=78) + 6 contacts + 3 COM + 18 angles + 18 vels + 2 gait clock
const PROPRIO_DIM_GAZE := 133  # + 1 : l'angle de tête, quand SYLVAN_GAZE=1 (docs/design_foret_complete.md §2.4)

func build_observation(agent, energy: float, health: float, vision: Array = []) -> Dictionary:
	var proprio: Array = agent.get_proprioception() as Array
	# La dimension ATTENDUE dépend du regard. On garde une assertion DURE des deux côtés plutôt que
	# de relâcher le contrat : c'est ce garde-fou qui a déjà attrapé des désynchronisations.
	var expected: int = PROPRIO_DIM_GAZE if agent.gaze_enabled else PROPRIO_DIM
	if proprio.size() != expected:
		push_error("Expected proprio dim %d (gaze=%s), got %d" % [expected, agent.gaze_enabled, proprio.size()])
	return {
		"proprio": proprio,
		"vision": vision,   # egocentric food radar (12-d); empty if perception is off
		"energy": energy,
		"health": health,
		"metrics": agent.get_locomotion_metrics(),
	}
