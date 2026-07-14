extends RefCounted
class_name HazardManager

# ZONE NOCIVE (2026-07-15) — premier enrichissement du monde : une région qui ABÎME la santé.
# BUT (docs/etat_critique.md, fork owner) : créer de la PLACE pour que l'expérience compte. Le coût
# codé-main du planner est « rapproche-toi de la ressource » — il n'a AUCUN terme de danger. Donc dès
# qu'une zone nocive est sur le chemin, l'entité fonce dedans en aveugle → coût INÉVITABLE par la
# perception/décision actuelles. Cet échec est PROUVABLE, pas espéré : c'est le gate qui casse la boucle.
#
# ÉTAGE : c'est du MONDE (comme food_manager/water_manager), PAS de la perception. À ce stade l'entité
# ne PERÇOIT pas le danger (le WM ne l'a jamais vu). C'est voulu : on prouve d'abord que la place existe
# (l'entité souffre en aveugle), AVANT de payer le WM-retrain qui lui donnera le sens « danger ».
#
# Tout est OPT-IN (SYLVAN_HAZARD_COUNT=0 par défaut → module inerte, zéro régression). Placement : un
# disque sur le segment spawn→bouffe-la-plus-proche, à HAZARD_FRAC du trajet → « aller tout droit » le
# traverse par construction. Auto-logge ses stats par épisode (pas de dépendance à main.gd pour le log).

var _discs: Array[Vector3] = []          # centres des zones (monde)
var _radius := 1.3
var _damage := 0.1                        # santé/pas tant qu'à l'intérieur
var _frac := 0.55                         # position sur le segment spawn→bouffe (0=spawn, 1=bouffe)
var _count := 0
var _rng := RandomNumberGenerator.new()

# stats de l'épisode courant (le gate les lit dans le log Godot)
var _ep := -1
var _steps_in := 0
var _dmg_done := 0.0
var _entered := false


func _init() -> void:
	_count = int(_env("SYLVAN_HAZARD_COUNT", "0"))
	_radius = _envf("SYLVAN_HAZARD_RADIUS", _radius)
	_damage = _envf("SYLVAN_HAZARD_DAMAGE", _damage)
	_frac = _envf("SYLVAN_HAZARD_FRAC", _frac)


func set_seed(value: int) -> void:
	_rng.seed = value


func active() -> bool:
	return _count > 0


# Placer les zones pour le nouvel épisode ET logger les stats de l'épisode PRÉCÉDENT (self-contained).
func begin_episode(episode_index: int, spawn_pos: Vector3, food_positions: Array) -> void:
	if _ep >= 0 and active():
		print("[hazard] ep %d : entré=%s pas_dans_zone=%d dégâts=%.1f"
			% [_ep, str(_entered), _steps_in, _dmg_done])
	_ep = episode_index
	_steps_in = 0
	_dmg_done = 0.0
	_entered = false
	_discs.clear()
	if not active():
		return
	# Zone sur le segment spawn→bouffe-la-plus-proche (celle que le planner va viser en aveugle).
	var nearest: Vector3 = spawn_pos
	var best := INF
	for p in food_positions:
		var d: float = spawn_pos.distance_to(p)
		if d < best:
			best = d
			nearest = p
	if best < INF:
		_discs.append(spawn_pos.lerp(nearest, _frac))
	# zones supplémentaires (count>1) : dispersées autour, pour peupler l'arène
	for i in range(1, _count):
		var a := _rng.randf_range(0.0, TAU)
		var r := _rng.randf_range(2.0, 6.0)
		_discs.append(spawn_pos + Vector3(cos(a) * r, 0.0, sin(a) * r))


# Dégâts à appliquer à cette position (0 si hors zone). Appelé chaque pas depuis la boucle de main.gd.
func damage_at(agent_pos: Vector3) -> float:
	if not active():
		return 0.0
	var ground := Vector3(agent_pos.x, 0.0, agent_pos.z)
	for c in _discs:
		if ground.distance_to(Vector3(c.x, 0.0, c.z)) <= _radius:
			_steps_in += 1
			_dmg_done += _damage
			_entered = true
			return _damage
	return 0.0


func get_positions() -> Array[Vector3]:
	return _discs


func get_radius() -> float:
	return _radius


func _env(key: String, dflt: String) -> String:
	var v := OS.get_environment(key)
	return v if v != "" else dflt


func _envf(key: String, dflt: float) -> float:
	var v := OS.get_environment(key)
	return float(v) if v != "" else dflt
