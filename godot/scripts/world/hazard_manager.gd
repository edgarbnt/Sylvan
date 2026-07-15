extends Node3D
class_name HazardManager

# ZONE NOCIVE (2026-07-15) — premier enrichissement du monde : une région qui ABÎME la santé.
# BUT (docs/etat_critique.md, fork owner) : créer de la PLACE pour que l'expérience compte. Le coût
# codé-main du planner est « rapproche-toi de la ressource » — il n'a AUCUN terme de danger. Donc dès
# qu'une zone nocive est sur le chemin, l'entité fonce dedans en aveugle → coût INÉVITABLE par la
# perception/décision actuelles. Cet échec est PROUVABLE, pas espéré : c'est le gate qui casse la boucle.
#
# ÉTAGE : c'est du MONDE (comme food_manager/water_manager). La ZONE EST MAINTENANT PERCEPTIBLE (2026-07-15,
# étape 1) : un cylindre coloré (retina_color VIOLET) sur la couche-8 rétine → la rétine la voit GRATIS,
# exactement comme le rouge=bouffe / bleu=eau. MAIS le WM ne SAIT PAS ENCORE lire cette couleur (il ne
# requête que rouge+bleu) → l'entité voit le violet sans le comprendre. Le sens « danger » = étape 2
# (re-collecte + retrain WM, PRINCIPE N°3), gaté derrière la vérif que la rétine capte bien la couleur.
#
# Tout est OPT-IN (SYLVAN_HAZARD_COUNT=0 par défaut → module inerte, zéro visuel, zéro régression). Placement :
# un disque sur le segment spawn→bouffe-la-plus-proche, à HAZARD_FRAC du trajet → « aller tout droit » le
# traverse par construction. Auto-logge ses stats par épisode (pas de dépendance à main.gd pour le log).

const HAZARD_HEIGHT := 1.2                 # hauteur du cylindre (les rayons rétine, à hauteur torse, le touchent)
const HAZARD_COLOR := Color(0.6, 0.12, 0.85)   # VIOLET : distinct du rouge(bouffe) et du bleu(eau) dans les 3 canaux

var _discs: Array[Vector3] = []          # centres des zones (monde)
var _radius := 1.3
var _damage := 0.5                        # santé/pas dedans. 0.5 = niveau LÉTAL choisi au gate (2026-07-15) :
                                          # traverser vide la barre (100 dégâts) → 7/12 vies aveugles TUÉES
                                          # par le danger (vs 0 sans) ; éviter = retour au régime normal. En
                                          # dessous (0.1-0.35) la santé est du slack (la faim tue avant) → non
                                          # conséquent. diagnostics/diag_hazard_gate.py.
var _frac := 0.55                         # position sur le segment spawn→bouffe (0=spawn, 1=bouffe)
var _count := 0
var _rng := RandomNumberGenerator.new()

var _material: StandardMaterial3D
var _visuals: Array[Node3D] = []          # 1 cylindre perceptible par zone (mesh + Area3D couche-8)

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


# Construit les cylindres perceptibles UNE FOIS (mesh violet translucide + Area3D couche-8 rétine).
# Calqué sur food_manager._ensure_built : l'Area3D ne bloque PAS la physique (mask 0), elle sert
# UNIQUEMENT à être lue par le raycast couleur de la rétine (meta "retina_color").
func _ensure_built() -> void:
	if not active() or not _visuals.is_empty():
		return
	if _material == null:
		_material = StandardMaterial3D.new()
		_material.albedo_color = Color(HAZARD_COLOR.r, HAZARD_COLOR.g, HAZARD_COLOR.b, 0.45)
		_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		_material.emission_enabled = true
		_material.emission = HAZARD_COLOR * 0.4
	for i in range(_count):
		var m := MeshInstance3D.new()
		var cyl := CylinderMesh.new()
		cyl.top_radius = _radius
		cyl.bottom_radius = _radius
		cyl.height = HAZARD_HEIGHT
		m.mesh = cyl
		m.material_override = _material
		m.position = Vector3(0.0, HAZARD_HEIGHT * 0.5, 0.0)   # pose sur le sol
		add_child(m)
		# PERCEPTION rétine : Area3D couche 8 (jamais bloquante), meta "retina_color" = ce que le rayon lira.
		var area := Area3D.new()
		area.collision_layer = 1 << 7   # couche 8 = "perceptible-rétine" (idem food_manager)
		area.collision_mask = 0
		area.set_meta("retina_color", HAZARD_COLOR)
		var cs := CollisionShape3D.new()
		var shape := CylinderShape3D.new()
		shape.radius = _radius
		shape.height = HAZARD_HEIGHT
		cs.shape = shape
		area.add_child(cs)
		m.add_child(area)
		m.visible = false
		_visuals.append(m)


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
	_ensure_built()
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
	# positionner/afficher les cylindres perceptibles aux centres des zones
	for i in range(_visuals.size()):
		if i < _discs.size():
			_visuals[i].global_position = Vector3(_discs[i].x, 0.0, _discs[i].z)
			_visuals[i].visible = true
		else:
			_visuals[i].visible = false


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
