extends Node3D
class_name ObstacleManager

# OBSTACLE PHYSIQUE (2026-07-17) — premier canal de conséquence NON-homéostatique (chantier
# docs/design_obstacle_affordance.md). Un MUR SOLIDE qui BLOQUE le mouvement (le corps s'arrête contre
# lui), perceptible par la rétine. La RÉACTION (contourner) n'est PAS ici : elle sera APPRISE par un
# prédicteur d'affordance SÉPARÉ (voie B, tranchée par le diag G0). Ici = uniquement le MONDE + la
# physique (§3 : monde/sens donnés légitimes ; la réaction, jamais codée). Opt-in
# SYLVAN_OBSTACLE_COUNT=0 par défaut → module inerte, zéro nœud créé, bit-identique.
#
# APPARENCE = PROPRIÉTÉ DU MONDE DÉCLARÉE (datée 2026-07-17) : cyan saturé, cos≈0 avec le rouge-bouffe
# → ne fire AUCUN slot pendant une collecte food-only (l'agent l'ignore perceptuellement, fonce vers la
# bouffe, le PERCUTE en chemin → l'événement de blocage est produit sans qu'aucune décision d'évitement
# n'existe encore). La couleur est APPRISE par voie B, pas câblée ici (critère de pureté : la réaction
# survit à un changement d'apparence — cf SYLVAN_OBSTACLE_HUE).
#
# GÉOMÉTRIE (déclarée, viabilité MESURÉE en G1) : un MUR ÉTROIT (demi-largeur ~0.6 m) placé sur le
# segment spawn→bouffe-la-plus-proche à OBSTACLE_FRAC. Étroit EXPRÈS (leçon hazard, header hazard_manager :
# un mur PLEIN occulte la bouffe → la rétine horizontale ne la voit plus → famine → monde NON-viable) :
# il bloque le corps si l'on fonce tout droit, mais n'occulte que ~1-2 rayons rétine → la bouffe reste
# visible autour, le détour reste POSSIBLE + SURVIVABLE.
#
# COUCHES : collision_layer = OBSTACLE_LAYER(bit 2, valeur 4, libre) | RETINA_LAYER(bit 7, 128). Le corps
# cinématique est GELÉ (aucune résolution physique) → le blocage est un raycast MANUEL dans
# sylvan_agent._kinematic_step qui masque le bit 2 (jamais le sol bit 0 ni le corps bit 1). La rétine
# masque le bit 7 (perception.gd, INCHANGÉ). Un même StaticBody3D est donc à la fois solide-pour-le-corps
# et visible-pour-la-rétine, et porte meta "retina_color".

const OBSTACLE_LAYER := 1 << 2                    # bit 2 (valeur 4) — couche dédiée au raycast de blocage (libre)
const RETINA_LAYER := 1 << 7                      # bit 7 (128) — perceptible-rétine (idem food/hazard)
const OBSTACLE_COLOR := Color(0.05, 0.7, 0.95)    # cyan saturé (déclaré 2026-07-17) — cos≈0 avec le rouge-bouffe

var _count := 0
var _frac := 0.5                          # position sur le segment spawn→bouffe (0=spawn, 1=bouffe)
var _halfwidth := 0.6                     # demi-largeur du mur (axe PERPENDICULAIRE au trajet)
var _thick := 0.25                        # épaisseur (axe LE LONG du trajet)
var _height := 1.6                        # hauteur (> torse → les rayons rétine horizontaux le touchent)
var _solid := true                        # SYLVAN_OBSTACLE_SOLID=0 → rendu MAIS traversable (baseline
                                          # « obstacle-aveugle/passable » du G3, ET contrôle A/B de G1 :
                                          # le corps DOIT être arrêté quand solide, traverser sinon)
var _color := OBSTACLE_COLOR
var _rng := RandomNumberGenerator.new()

var _material: StandardMaterial3D
var _bodies: Array[StaticBody3D] = []     # 1 mur solide + perceptible par obstacle
var _centers: Array[Vector3] = []         # centres (monde) du dernier placement


func _init() -> void:
	_count = int(_env("SYLVAN_OBSTACLE_COUNT", "0"))
	_frac = _envf("SYLVAN_OBSTACLE_FRAC", _frac)
	_halfwidth = _envf("SYLVAN_OBSTACLE_HALFWIDTH", _halfwidth)
	_thick = _envf("SYLVAN_OBSTACLE_THICK", _thick)
	_height = _envf("SYLVAN_OBSTACLE_HEIGHT", _height)
	_solid = _env("SYLVAN_OBSTACLE_SOLID", "1") != "0"
	var hue := OS.get_environment("SYLVAN_OBSTACLE_HUE")   # override d'apparence = test de pureté (la réaction survit-elle ?)
	if hue != "":
		_color = Color.from_hsv(float(hue), 0.8, 0.75)


func active() -> bool:
	return _count > 0


func set_seed(value: int) -> void:
	_rng.seed = value


# Construit UNE FOIS, par obstacle, un MUR SOLIDE (StaticBody3D + BoxShape) qui est à la fois bloquant
# (couche bit 2, lu par le raycast du corps) et perceptible (couche bit 7 + meta retina_color, lu par
# la rétine). Idempotent. Positionné/orienté par begin_episode.
func _ensure_built() -> void:
	if not active() or not _bodies.is_empty():
		return
	_material = StandardMaterial3D.new()
	_material.albedo_color = _color
	_material.emission_enabled = true
	_material.emission = _color * 0.35
	var size := Vector3(_halfwidth * 2.0, _height, _thick)
	for i in range(_count):
		var body := StaticBody3D.new()
		# solide → bit 2 (bloquant) + bit 7 (perceptible) ; non-solide → bit 7 seul (rendu mais traversable)
		body.collision_layer = (OBSTACLE_LAYER | RETINA_LAYER) if _solid else RETINA_LAYER
		body.collision_mask = 0                                # ne détecte rien lui-même (statique)
		body.set_meta("retina_color", _color)                  # RGB lu par le raycast couleur de la rétine
		var cs := CollisionShape3D.new()
		var box := BoxShape3D.new()
		box.size = size
		cs.shape = box
		body.add_child(cs)
		var mesh := MeshInstance3D.new()
		var bm := BoxMesh.new()
		bm.size = size
		mesh.mesh = bm
		mesh.material_override = _material
		body.add_child(mesh)
		body.visible = false
		add_child(body)
		_bodies.append(body)


# Placer les murs pour le nouvel épisode. Obstacle 0 = SUR le segment spawn→bouffe-la-plus-proche à
# _frac, orienté en TRAVERS (face large perpendiculaire au trajet). Obstacles supplémentaires (count>1)
# = dispersés. Appelé depuis main.gd (hook local non stagé, comme hazard_manager.begin_episode).
func begin_episode(episode_index: int, spawn_pos: Vector3, food_positions: Array) -> void:
	_centers.clear()
	if not active():
		return
	_ensure_built()
	var nearest: Vector3 = spawn_pos
	var best := INF
	for p in food_positions:
		var d: float = spawn_pos.distance_to(p)
		if d < best:
			best = d
			nearest = p
	for i in range(_bodies.size()):
		var center: Vector3
		var heading: float
		if i == 0 and best < INF:
			center = spawn_pos.lerp(nearest, _frac)
			var dir := nearest - spawn_pos
			heading = atan2(dir.x, dir.z)   # local Z (épaisseur) le long du trajet → face large en travers
		else:
			var a := _rng.randf_range(0.0, TAU)
			var r := _rng.randf_range(2.0, 6.0)
			center = spawn_pos + Vector3(cos(a) * r, 0.0, sin(a) * r)
			heading = a
		center.y = _height * 0.5            # posé sur le sol → s'étend en y sur [0, _height]
		_bodies[i].global_transform = Transform3D(Basis(Vector3.UP, heading), center)
		_bodies[i].visible = true
		_centers.append(center)


func get_positions() -> Array[Vector3]:
	return _centers


func get_halfwidth() -> float:
	return _halfwidth


func _env(key: String, dflt: String) -> String:
	var v := OS.get_environment(key)
	return v if v != "" else dflt


func _envf(key: String, dflt: float) -> float:
	var v := OS.get_environment(key)
	return float(v) if v != "" else dflt
