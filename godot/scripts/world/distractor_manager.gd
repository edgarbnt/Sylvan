extends Node3D
class_name DistractorManager

# ANIMAUX NON COMESTIBLES — DISTRACTEURS (2026-07-24, docs/design_foret_complete.md §2.9).
#
# POURQUOI. Aujourd'hui les seules choses qui BOUGENT sont les proies, et elles sont TOUTES de la
# nourriture. « ça bouge donc c'est de la nourriture » est donc un raccourci GRATUIT : l'agent n'a
# jamais à discriminer, et le prédicteur du WM peut apprendre « mouvement → repas » au lieu de
# « apparence → repas ». Des oiseaux/écureuils qui bougent et qu'on ne peut PAS manger cassent ce
# raccourci : le lien apparence→comestible doit être APPRIS. Même famille que les types arbitraires
# (la seule à avoir passé le filtre §1). Coût faible, valeur élevée (§2.9).
#
# CE QU'ILS SONT. Des Area3D PERCEPTIBLES (bit 7 + meta retina_color, comme les pastilles de bouffe,
# donc vus par le raycast couleur de la rétine — perception.gd collide_with_areas=true), NON
# BLOQUANTS (aucune collision : ce ne sont pas des obstacles), NON CONSOMMABLES (main ne les passe
# JAMAIS à try_consume). Ils VAGUENT (direction persistante + dérive + réflexion aux bords), exactement
# comme les proies, mais leur COULEUR est HORS des cônes ressource.
#
# 🚨 COULEUR HORS DES CÔNES (§3, la leçon du tronc-brun). Le slot détecte une ressource par le COSINUS
# de la couleur du rayon avec sa requête (rouge=bouffe, bleu=eau), seuil 0,55, et EXCLUT en dur les
# rayons sous le seuil (slot_head.py). Un distracteur dont le cos-rouge dépasse 0,55 serait lu comme
# de la NOURRITURE et corromprait la localisation → on choisit une couleur dont cos-rouge ET cos-bleu
# sont < 0,55 (vert par défaut). Le manager AVERTIT bruyamment si ce n'est pas le cas (ne jamais
# dégrader en silence, §6bis).
#
# OPT-IN STRICT : SYLVAN_DISTRACTOR_COUNT=0 par défaut → aucun nœud, aucun mouvement, monde
# bit-identique. Flux de hasard DÉDIÉ (set_seed) → n'ajoute AUCUN tirage au flux des commandes, donc
# le rejeu bit-identique tient (§6quater F : tout nouveau consommateur de RNG prend son propre flux).

const RETINA_LAYER := 1 << 7                       # bit 7 (128) — perceptible-rétine (idem food/hazard)
const DEFAULT_COLOR := Color(0.30, 0.70, 0.25)     # vert vif « bestiole » : cos-rouge 0.37, cos-bleu 0.31

var _count := 0
var _speed := 0.006                                # m/tick ; ils vaguent, pas besoin d'être rapides
var _turn := 0.02                                  # rad/tick de dérive (persistance du transversal)
var _radius_min := 2.0
var _radius_max := 10.0
var _sphere_r := 0.30                              # rayon visuel + perception
var _color := DEFAULT_COLOR

var _rng := RandomNumberGenerator.new()
var _material: StandardMaterial3D
var _areas: Array[Area3D] = []
var _meshes: Array[MeshInstance3D] = []
var _positions: Array[Vector3] = []
var _dirs: Array[Vector3] = []
var _travel := 0.0                                 # §6bis : distance MESURÉE parcourue par épisode
var _ticks := 0


func _init() -> void:
	_count = int(_env("SYLVAN_DISTRACTOR_COUNT", "0"))
	_speed = _envf("SYLVAN_DISTRACTOR_SPEED", _speed)
	_turn = _envf("SYLVAN_DISTRACTOR_TURN", _turn)
	_radius_min = _envf("SYLVAN_DISTRACTOR_MIN_RADIUS", _radius_min)
	_radius_max = _envf("SYLVAN_DISTRACTOR_SPAWN_RADIUS", _radius_max)
	_sphere_r = _envf("SYLVAN_DISTRACTOR_RADIUS", _sphere_r)
	var hue := OS.get_environment("SYLVAN_DISTRACTOR_HUE")
	if hue != "":
		var p := hue.split(",")
		if p.size() == 3:
			_color = Color(float(p[0]), float(p[1]), float(p[2]))


func active() -> bool:
	return _count > 0


func set_seed(value: int) -> void:
	_rng.seed = value


func get_positions() -> Array[Vector3]:
	return _positions


func _ensure_built() -> void:
	if not active() or not _meshes.is_empty():
		return
	# GARDE §3 : refuser SILENCIEUSEMENT un distracteur qui serait lu comme une ressource serait la
	# faute même que ce garde traque. On AVERTIT bruyamment (le monde ne ment pas).
	var n := Vector3(_color.r, _color.g, _color.b)
	if n.length() > 0.0:
		var cos_red := _color.r / n.length()
		var cos_blue := _color.b / n.length()
		if cos_red > 0.55 or cos_blue > 0.55:
			push_warning("[distractor] COULEUR DANS UN CÔNE RESSOURCE (cos_rouge %.2f cos_bleu %.2f > 0.55) : le slot la lira comme une ressource et corrompra le foraging (§3)" % [cos_red, cos_blue])
	_material = StandardMaterial3D.new()
	_material.albedo_color = _color
	_material.emission_enabled = true
	_material.emission = _color * 0.25
	for _i in range(_count):
		var mesh := MeshInstance3D.new()
		var sphere := SphereMesh.new()
		sphere.radius = _sphere_r
		sphere.height = _sphere_r * 2.0
		mesh.mesh = sphere
		mesh.material_override = _material
		# Area3D PERCEPTIBLE, NON bloquante (bit 7 seulement, mask 0 : elle ne détecte rien elle-même).
		var area := Area3D.new()
		area.collision_layer = RETINA_LAYER
		area.collision_mask = 0
		area.set_meta("retina_color", _color)     # RGB lu par le raycast couleur de la rétine
		var cs := CollisionShape3D.new()
		var shape := SphereShape3D.new()
		shape.radius = _sphere_r
		cs.shape = shape
		area.add_child(cs)
		mesh.add_child(area)
		mesh.visible = false
		add_child(mesh)
		_meshes.append(mesh)
		_areas.append(area)
	print("[distractor] %d animaux NON comestibles | couleur=%s | vitesse=%.4f m/tick | non bloquants, non consommables"
		% [_count, str(_color), _speed])


# Placement au nouvel épisode : dans un anneau autour du spawn, directions aléatoires. Émet AVANT de
# réinitialiser la distance de l'épisode ÉCOULÉ (§6bis : ce qui a RÉELLEMENT bougé, mesuré, pas demandé).
func begin_episode(_episode_index: int, spawn_pos: Vector3) -> void:
	if not active():
		return
	if _ticks > 0:
		print("[distractor] episode : distance MESUREE %.2f m sur %d ticks (%.5f m/tick moyen)"
			% [_travel, _ticks, _travel / float(_ticks)])
	_travel = 0.0
	_ticks = 0
	_ensure_built()
	_positions.clear()
	_dirs.clear()
	for i in range(_meshes.size()):
		var a := _rng.randf_range(0.0, TAU)
		var r := _rng.randf_range(_radius_min, _radius_max)
		var p := spawn_pos + Vector3(cos(a) * r, 0.0, sin(a) * r)
		p.y = _sphere_r
		_positions.append(p)
		var da := _rng.randf_range(0.0, TAU)
		_dirs.append(Vector3(cos(da), 0.0, sin(da)))
		_meshes[i].global_position = p
		_meshes[i].visible = true


# Un pas de déambulation, appelé chaque tick par main.gd (indépendant de l'homéostasie → ils bougent
# dans TOUS les régimes de collecte). Vague comme les proies : direction persistante + dérive lente +
# réflexion aux bords. IGNORE l'agent (ne fuit pas, ne poursuit pas : ce sont des distracteurs).
func advance(_delta: float) -> void:
	if not active() or _positions.is_empty():
		return
	for i in range(_positions.size()):
		var ang := _rng.randf_range(-_turn, _turn)
		var d: Vector3 = _dirs[i]
		var nd := Vector3(cos(ang) * d.x - sin(ang) * d.z, 0.0, sin(ang) * d.x + cos(ang) * d.z)
		var p: Vector3 = _positions[i] + nd * _speed
		var rad := Vector2(p.x, p.z).length()
		if rad > _radius_max:                       # réflexion : ne part pas à l'infini
			nd = -nd
			p = _positions[i] + nd * _speed
		_dirs[i] = nd
		_positions[i] = p
		_meshes[i].global_position = p
		_travel += _speed
	_ticks += 1


func _env(key: String, dflt: String) -> String:
	var v := OS.get_environment(key)
	return v if v != "" else dflt


func _envf(key: String, dflt: float) -> float:
	var v := OS.get_environment(key)
	return float(v) if v != "" else dflt
