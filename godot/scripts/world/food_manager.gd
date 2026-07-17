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
var _appearance_var := 0.0    # SYLVAN_<prefix>_APPEARANCE_VAR : jitter d'apparence par instance (0=OFF, bit-identique)
var _areas: Array = []        # Area3D perceptibles — maj de retina_color quand l'apparence varie
var consumed_this_episode := 0
# BASCULE D'APPARENCE EN COURS DE VIE (Gate-capacité, docs/design_gate_capacite.md) — FOOD only
# (jamais lu pour l'eau : _prefix reste "WATER", ces vars restent à leur défaut OFF). Opt-in
# SYLVAN_FOOD_SWAP_TICK/SYLVAN_FOOD_SWAP_HUE, absent = _swap_tick<=0 = jamais appelé, bit-identique.
var _swap_tick := 0           # pas DANS LA VIE où l'apparence bascule (0 = OFF)
var _swap_hue := 0.0          # teinte cible [0,1) (HSV) ; S/V de _albedo conservés
var _swapped := false         # déjà basculé cette vie (une seule fois)
var _life_tick := 0           # compteur de pas depuis le dernier reset() (remis à 0 par reset())
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


func _jitter(base: Color) -> Color:
	# Teinte/saturation/valeur perturbées autour de la couleur de base (même TYPE, apparence variée).
	# Déterministe (via _rng) → corpus reproductible ; magnitude = _appearance_var.
	var h := fposmod(base.h + _rng.randf_range(-_appearance_var, _appearance_var), 1.0)
	var s := clampf(base.s + _rng.randf_range(-_appearance_var, _appearance_var), 0.0, 1.0)
	var v := clampf(base.v + _rng.randf_range(-0.5 * _appearance_var, 0.5 * _appearance_var), 0.0, 1.0)
	return Color.from_hsv(h, s, v)


func _apply_appearance(i: int) -> void:
	# Ré-échantillonne l'apparence de l'item i (couleur du matériau + meta retina_color lue par le
	# raycast). OFF (_appearance_var<=0) : ne touche à rien → couleur unique partagée, bit-identique.
	if _appearance_var <= 0.0 or i >= _areas.size():
		return
	var c := _jitter(_albedo)
	var mat := _meshes[i].material_override as StandardMaterial3D
	if mat != null:
		mat.albedo_color = c
	_areas[i].set_meta("retina_color", c)


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
	var _var_env := OS.get_environment("SYLVAN_%s_APPEARANCE_VAR" % _prefix)
	if _var_env != "":
		_appearance_var = maxf(0.0, float(_var_env))
	# Gate-capacité : bascule d'apparence en cours de vie — FOOD SEULEMENT (le nom n'est pas
	# préfixé par _prefix : l'eau n'y touche jamais, cf déclaration des vars ci-dessus).
	if _prefix == "FOOD":
		var _st_env := OS.get_environment("SYLVAN_FOOD_SWAP_TICK")
		if _st_env != "":
			_swap_tick = maxi(0, int(_st_env))
		var _sh_env := OS.get_environment("SYLVAN_FOOD_SWAP_HUE")
		if _sh_env != "":
			_swap_hue = fposmod(float(_sh_env), 1.0)
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
			# apparence VARIÉE (opt-in) : matériau PAR INSTANCE (couleur ré-échantillonnée à chaque
			# (re)spawn via _apply_appearance) ; OFF = matériau partagé, bit-identique.
			if _appearance_var > 0.0:
				var mat := StandardMaterial3D.new()
				mat.emission_enabled = true
				mat.emission = _emission
				m.material_override = mat
			else:
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
			_areas.append(area)


func reset(_episode_index: int = 0) -> void:
	_ensure_built()
	consumed_this_episode = 0
	_life_tick = 0
	_swapped = false
	_positions.clear()
	for i in range(food_count):
		var p := _random_pos()
		_positions.append(p)
		_meshes[i].global_position = p
		_meshes[i].visible = true
		_apply_appearance(i)


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
	# Gate-capacité : compteur de pas DANS LA VIE + bascule éventuelle — en tête, appelé chaque
	# tick (main.gd) donc le même rythme que le monde. No-op tant que _swap_tick est OFF (0).
	_life_tick += 1
	_maybe_swap_appearance()
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
			_apply_appearance(i)
	return restored


func _maybe_swap_appearance() -> void:
	# Gate-capacité (docs/design_gate_capacite.md) : à _swap_tick pas DANS CETTE VIE, la couleur
	# de base _albedo bascule vers la teinte-cible (HSV, S/V conservés — propriété du MONDE
	# déclarée, jamais ajustée pour faciliter) et se ré-applique à TOUS les items déjà spawnés,
	# une seule fois par vie. OFF (_swap_tick<=0, défaut) : jamais atteint, bit-identique.
	if _swap_tick <= 0 or _swapped or _life_tick < _swap_tick:
		return
	_swapped = true
	_albedo = Color.from_hsv(_swap_hue, _albedo.s, _albedo.v)
	if _material != null:
		_material.albedo_color = _albedo
	for i in range(_meshes.size()):
		if _appearance_var > 0.0:
			_apply_appearance(i)          # ré-échantillonne le jitter autour de la NOUVELLE base
		elif i < _areas.size():
			_areas[i].set_meta("retina_color", _albedo)   # matériau partagé déjà remis à jour ci-dessus


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
