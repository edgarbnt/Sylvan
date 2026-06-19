extends Node3D
class_name FoodManager

# MINIMAL SURVIVAL RESOURCE — food pellets the agent eats to restore energy.
# This is the first perceivable + consumable resource: the substrate for the emergent
# loop "j'ai faim → chercher → manger → survivre". Pellets are scattered on the ground
# around the origin; eating one (agent within eat_radius, horizontally) restores energy
# and respawns the pellet elsewhere so food DENSITY stays constant. Positions are exposed
# (get_positions) for the raycast perception of the next tranche — visual meshes too so the
# owner SEES the food in the viewer. No contract change here (pure Godot/world mechanic);
# perception (vision_shape) and the energy DRIVE get wired in the following steps.

@export var food_count := 10         # NAV-TEST: few DISTINCT pellets so navigation is visible.
                                     # forage4 went too sparse (8) + too much exploration → fall=100%,
                                     # the walk broke. 12 de-saturates the radar enough that it points
                                     # at a couple of clear targets (vs 26 lighting every sector) while
                                     # keeping the reward dense enough to KEEP the gait alive. The turn
                                     # bias is eroded by TASK PRESSURE (sparser, symmetric food punishes
                                     # the right-only habit), NOT by exploration noise. Sparsen further
                                     # (12→8→5) only once the walk holds at this density.
@export var spawn_radius := 7.0      # pellets spawn in an annulus [min_radius, spawn_radius]
@export var min_radius := 1.5        # never right on top of the spawn point
@export var food_y := 0.25           # pellet centre height above the floor top (cosmetic)
@export var eat_radius := 1.0        # realistic head-reach for this body size (was 0.6, smaller than the
                                     # ~1m min turn radius → agent orbited food; 1.0 = a quadruped's mouth
                                     # reach, fixes the terminal-approach orbit. Override: SYLVAN_EAT_RADIUS.
                                     # (precise navigation), so we can see if it truly walks to food.
@export var energy_per_food := 40.0  # WM-DATA: smaller meals → must eat OFTEN → many eat-events for the WM
                                     # 35-energy pellet only bought ~233 steps of life << ~1000 steps to
                                     # reach a 5m pellet → sparse food was unsurvivable → 100% starvation
                                     # → no learning. 100 makes going for a distant, DISTINCT pellet pay
                                     # off → directed foraging becomes both survivable AND steering-driven.

var _rng := RandomNumberGenerator.new()
var _positions: Array[Vector3] = []
var _meshes: Array[MeshInstance3D] = []
var _material: StandardMaterial3D
var consumed_this_episode := 0
# COLLECTE EAT-RICHE (vers 🅑) — leviers de RÉGIME, gated env, défaut = comportement actuel inchangé.
# eat_hunger_max < 1 : ne consommer une pastille QUE si l'énergie (fraction) est sous ce seuil → chaque
# repas a une vraie MARGE (le WM voit la bosse +energy_per_food non écrêtée à 100) → apprend l'eat-dynamics.
# C'est une régime de COLLECTE (comme le babbling overridable), PAS un relâchement du critère d'éval (§2).
var eat_hunger_max := 1.0
var respawn_min := 2.0
var respawn_max := 4.5

# 2ᵉ PULSION (2026-06-18): cette classe sert MAINTENANT n'importe quelle ressource (bouffe OU eau).
# Par défaut = FOOD (comportement identique à avant). `configure()` la repointe sur l'eau : préfixe
# d'env (SYLVAN_<PREFIX>_COUNT/MIN_RADIUS/SPAWN_RADIUS/ANGLE_DEG), nom d'env du rayon de capture,
# et couleur de la pastille. Ainsi main.gd instancie deux managers sans dupliquer le code.
var _prefix := "FOOD"
var _eat_env := "SYLVAN_EAT_RADIUS"
var _albedo := Color(0.9, 0.3, 0.2)        # rouge = bouffe (convention blueprint)
var _emission := Color(0.5, 0.12, 0.05)


func configure(prefix: String, eat_env: String, albedo: Color, emission: Color) -> void:
	_prefix = prefix
	_eat_env = eat_env
	_albedo = albedo
	_emission = emission


func set_seed(value: int) -> void:
	_rng.seed = value


func _ensure_built() -> void:
	var _fc_env := OS.get_environment("SYLVAN_%s_COUNT" % _prefix)  # sparse pellets → one clear target
	if _fc_env != "":
		food_count = maxi(1, int(_fc_env))
	var _er_env := OS.get_environment(_eat_env)  # loosen capture to probe precision-vs-direction
	if _er_env != "":
		eat_radius = maxf(0.1, float(_er_env))
	# Phase 5a nav test: force a controlled spawn annulus so a SINGLE pellet sits at a known distance.
	var _minr_env := OS.get_environment("SYLVAN_%s_MIN_RADIUS" % _prefix)
	if _minr_env != "":
		min_radius = maxf(0.1, float(_minr_env))
	var _maxr_env := OS.get_environment("SYLVAN_%s_SPAWN_RADIUS" % _prefix)
	if _maxr_env != "":
		spawn_radius = maxf(min_radius, float(_maxr_env))
	# Régime EAT-RICHE (collecte WM). Défaut 1.0 = mange toujours (comportement actuel).
	var _hm_env := OS.get_environment("SYLVAN_%s_HUNGER_MAX" % _prefix)
	if _hm_env != "":
		eat_hunger_max = clampf(float(_hm_env), 0.05, 1.0)
	var _rmin_env := OS.get_environment("SYLVAN_%s_RESPAWN_MIN" % _prefix)
	if _rmin_env != "":
		respawn_min = maxf(0.5, float(_rmin_env))
	var _rmax_env := OS.get_environment("SYLVAN_%s_RESPAWN_MAX" % _prefix)
	if _rmax_env != "":
		respawn_max = maxf(respawn_min, float(_rmax_env))
	if _material == null:
		_material = StandardMaterial3D.new()
		_material.albedo_color = _albedo
		_material.emission_enabled = true
		_material.emission = _emission
	if _meshes.is_empty():
		for i in range(food_count):
			var m := MeshInstance3D.new()
			var sphere := SphereMesh.new()
			sphere.radius = 0.18
			sphere.height = 0.36
			m.mesh = sphere
			m.material_override = _material
			add_child(m)
			# RÉTINE (perception apprise) : rendre la pastille PERCEPTIBLE par le raycast couleur, SANS
			# perturber la physique du gait. Area3D (jamais bloquante) sur la couche 8 dédiée (mask 0 :
			# elle ne détecte rien elle-même) ; meta "retina_color" = la couleur que le rayon lira. Sphère
			# de collision un peu > visuel pour tolérer l'écart de hauteur tête↔pastille. La consommation
			# reste par distance (try_consume) — cet Area ne sert QU'À la perception.
			var area := Area3D.new()
			area.collision_layer = 1 << 7   # couche 8 = "perceptible-rétine"
			area.collision_mask = 0
			area.set_meta("retina_color", _albedo)
			var cs := CollisionShape3D.new()
			var col_shape := SphereShape3D.new()
			col_shape.radius = 0.35
			cs.shape = col_shape
			area.add_child(cs)
			m.add_child(area)
			_meshes.append(m)


func reset(_episode_index: int = 0) -> void:
	_ensure_built()
	consumed_this_episode = 0
	_positions.clear()
	for i in range(food_count):
		var p := _random_pos()
		_positions.append(p)
		_meshes[i].global_position = p
		_meshes[i].visible = true


func _random_pos() -> Vector3:
	var angle := _rng.randf_range(0.0, TAU)
	# Phase 5a A→B nav probe: pin the spawn azimuth (world deg) so a SINGLE pellet sits at a known
	# bearing → we can measure WHICH azimuths the planner fails to engage. Ignored when unset.
	var _ang_env := OS.get_environment("SYLVAN_%s_ANGLE_DEG" % _prefix)
	if _ang_env != "":
		angle = deg_to_rad(float(_ang_env))
	var radius := _rng.randf_range(min_radius, spawn_radius)
	return Vector3(cos(angle) * radius, food_y, sin(angle) * radius)


# Eat every pellet within eat_radius (horizontally) of the agent; respawn each eaten one.
# Returns the total energy to restore this step.
func try_consume(agent_pos: Vector3, energy_frac: float = 1.0) -> float:
	# Régime eat-riche : ne pas consommer tant qu'on n'est pas assez affamé (seuil eat_hunger_max).
	# energy_frac = énergie/max. Défaut 1.0 + seuil 1.0 → mange toujours (inchangé).
	if energy_frac > eat_hunger_max:
		return 0.0
	var restored := 0.0
	var ground := Vector3(agent_pos.x, food_y, agent_pos.z)
	for i in range(_positions.size()):
		if ground.distance_to(_positions[i]) <= eat_radius:
			restored += energy_per_food
			consumed_this_episode += 1
			# PERPETUAL FIELD: respawn the eaten pellet in an annulus around the AGENT (not the
			# origin) so food density stays high wherever it roams → survival is limited by
			# falling, not by walking out of a fixed patch. (A later curriculum can make food
			# sparse/clustered to force real directed foraging.)
			_positions[i] = _respawn_near(agent_pos)
			_meshes[i].global_position = _positions[i]
	return restored


func _respawn_near(center: Vector3) -> Vector3:
	# Respawn an eaten pellet FARTHER out (was 2.5-6) so after eating, the next target is a real
	# trek away → the agent keeps STEERING/foraging instead of grazing a local patch.
	var angle := _rng.randf_range(0.0, TAU)
	var radius := _rng.randf_range(respawn_min, respawn_max)   # défaut 2.0-4.5 ; override SYLVAN_FOOD_RESPAWN_MIN/MAX
	return Vector3(center.x + cos(angle) * radius, food_y, center.z + sin(angle) * radius)  # far enough to be a DISTINCT steering target


func get_positions() -> Array:
	return _positions


# Normalised HORIZONTAL direction from the agent to the NEAREST pellet (for the survival reward's
# heading-alignment term). Returns Vector3.ZERO if there is no food.
func nearest_dir(agent_pos: Vector3) -> Vector3:
	var best := 1e9
	var best_dir := Vector3.ZERO
	var ground := Vector3(agent_pos.x, food_y, agent_pos.z)
	for p in _positions:
		var off := Vector3(p.x - ground.x, 0.0, p.z - ground.z)
		var d := off.length()
		if d < best and d > 0.001:
			best = d
			best_dir = off / d
	return best_dir


# Horizontal distance to the NEAREST pellet (for the survival reward's approach term).
# Returns a large value if there is no food.
func nearest_distance(agent_pos: Vector3) -> float:
	var best := 1e9
	var ground := Vector3(agent_pos.x, food_y, agent_pos.z)
	for p in _positions:
		var d := ground.distance_to(p)
		if d < best:
			best = d
	return best
