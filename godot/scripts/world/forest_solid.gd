extends Node3D
class_name ForestSolid

# FORÊT SOLIDE (2026-07-21) — des ARBRES qui BLOQUENT le mouvement et OCCULTENT la vue.
#
# POURQUOI. Le monde était un plan vide : `forest_manager.gd` est explicitement « VISUAL-ONLY forest
# decor [...] NO collision, NO physics » et n'existe qu'en mode visuel — en headless l'entité vivait
# sur une surface nue. Conséquences MESURÉES : la survie sature (9 épisodes sur 10 au plafond), la
# géométrie suffit à survivre, le « vu-puis-perdu » n'existe pas (G0 mémoire : never_seen = 0), et un
# critique appris n'a rien à apprendre « par construction ». Les arbres cassent DEUX hypothèses d'un
# coup : « tout ce qui compte est visible » (occlusion → la mémoire devient load-bearing) et « se
# déplacer = ligne droite » (détour → le modèle interne du planner devient faux de façon VISIBLE).
#
# OPT-IN STRICT : SYLVAN_FOREST_COUNT=0 par défaut → aucun nœud créé, monde bit-identique.
#
# COULEUR CHOISIE PAR LA MESURE, PAS PAR LE GOÛT (diag_foret_g0.py). Les requêtes-couleur du WM sont
# rouge (bouffe) et bleu (eau), seuil 0.55. La « fuite » d'une apparence = ce qui reste de son
# affinité APRÈS seuil ; une fuite > 0 signifie que l'arbre active un slot-ressource, donc que
# l'entité peut confondre un arbre avec de la nourriture. Mesuré :
#     vert foncé (0.13,0.35,0.13) → 0.0000   ← RETENU
#     vert sombre                 → 0.0000
#     gris                        → 0.0424
#     brun foncé  (0.36,0.25,0.15)→ 0.2271   ← le PIRE : un tronc brun est perceptuellement rougeâtre
# Le brun « naturel » est donc précisément le mauvais choix. Bonus mesuré : le prédicteur d'affordance
# (entraîné sur cyan) juge déjà le vert foncé BLOQUANT à 0.985 → il transfère aux arbres sans
# ré-entraînement. (Il reste à corriger pour l'EAU, qu'il juge bloquante à 1.00 — indépendant d'ici.)
#
# DENSITÉ BORNÉE PAR LA MESURE. Toujours diag_foret_g0 : l'erreur de position rapportée par le slot
# passe de 0.00 m (0 % d'arbres) à 0.29 m (30 %) puis 1.43 m (60 %). Avec un rayon de capture de 1 m,
# une forêt dense rend le closing IMPOSSIBLE. Défaut visé ≈ 30 % d'occupation de rétine.
#
# COUCHES : identiques à obstacle_manager (contrat déjà validé G1, 0 pénétration mesurée) —
# bit 2 = bloquant (lu par le raycast du corps cinématique), bit 7 = perceptible-rétine.
# On réutilise ce contrat au lieu d'écrire une nouvelle physique.

const OBSTACLE_LAYER := 1 << 2                     # bit 2 (4) — couche de blocage lue par le corps
const RETINA_LAYER := 1 << 7                       # bit 7 (128) — perceptible par la rétine
const TREE_COLOR := Color(0.13, 0.35, 0.13)        # vert foncé — fuite MESURÉE = 0.0000

var _count := 0
var _radius_min := 2.5                             # anneau de dispersion : on garde le centre dégagé
var _radius_max := 11.0
var _trunk_r := 0.35                               # rayon du tronc (collision + rétine)
var _height := 2.0                                 # > torse → les rayons rétine horizontaux le touchent
var _clear_r := 2.0                                # rayon dégagé autour du spawn (ne pas emmurer l'agent)
var _keepout := 1.4                                # distance mini à une ressource (ne pas la rendre inatteignable)
var _color := TREE_COLOR
var _rng := RandomNumberGenerator.new()

var _material: StandardMaterial3D
var _bodies: Array[StaticBody3D] = []
var _centers: Array[Vector3] = []


func _init() -> void:
	_count = int(_env("SYLVAN_FOREST_COUNT", "0"))
	_radius_min = _envf("SYLVAN_FOREST_RADIUS_MIN", _radius_min)
	_radius_max = _envf("SYLVAN_FOREST_RADIUS_MAX", _radius_max)
	_trunk_r = _envf("SYLVAN_FOREST_TRUNK_R", _trunk_r)
	_height = _envf("SYLVAN_FOREST_HEIGHT", _height)
	_clear_r = _envf("SYLVAN_FOREST_CLEAR_R", _clear_r)
	_keepout = _envf("SYLVAN_FOREST_KEEPOUT", _keepout)
	# Override d'apparence = TEST DE PURETÉ : la réaction survit-elle à un changement de couleur ?
	# (même contrat que SYLVAN_OBSTACLE_HUE). ⚠️ toute couleur ≠ verte RÉINTRODUIT de la fuite — c'est
	# justement ce qu'on veut pouvoir mesurer.
	var hue := OS.get_environment("SYLVAN_FOREST_HUE")
	if hue != "":
		var p := hue.split(",")
		if p.size() == 3:
			_color = Color(float(p[0]), float(p[1]), float(p[2]))


func active() -> bool:
	return _count > 0


func set_seed(value: int) -> void:
	_rng.seed = value


# Construit UNE FOIS `count` troncs cylindriques, à la fois bloquants (bit 2) et perceptibles
# (bit 7 + meta retina_color). Idempotent — le placement se fait dans begin_episode.
func _ensure_built() -> void:
	if not active() or not _bodies.is_empty():
		return
	_material = StandardMaterial3D.new()
	_material.albedo_color = _color
	_material.emission_enabled = true
	_material.emission = _color * 0.25
	for i in range(_count):
		var body := StaticBody3D.new()
		body.collision_layer = OBSTACLE_LAYER | RETINA_LAYER
		body.collision_mask = 0                              # statique : ne détecte rien lui-même
		body.set_meta("retina_color", _color)                # RGB lu par le raycast couleur de la rétine
		var cs := CollisionShape3D.new()
		var cyl := CylinderShape3D.new()
		cyl.radius = _trunk_r
		cyl.height = _height
		cs.shape = cyl
		body.add_child(cs)
		var mesh := MeshInstance3D.new()
		var cm := CylinderMesh.new()
		cm.top_radius = _trunk_r * 0.75                      # léger fuselage : lit mieux en low-poly
		cm.bottom_radius = _trunk_r
		cm.height = _height
		mesh.mesh = cm
		mesh.material_override = _material
		body.add_child(mesh)
		body.visible = false
		add_child(body)
		_bodies.append(body)
	# BANNIERE (anti-log-qui-ment) : le log doit PROUVER ce qui est reellement construit et servi.
	print("[forest] %d arbres SOLIDES construits | couleur=%s | tronc r=%.2f h=%.2f | anneau %.1f-%.1f m"
		% [_count, str(_color), _trunk_r, _height, _radius_min, _radius_max])


# Disperse les arbres pour le nouvel épisode. GARDES : jamais dans le rayon dégagé autour du spawn
# (sinon l'agent démarre emmuré), jamais à moins de `_keepout` d'une ressource (sinon elle devient
# inatteignable et on mesurerait un échec du MONDE, pas de l'entité — §2).
func begin_episode(_episode_index: int, spawn_pos: Vector3, resource_positions: Array) -> void:
	_centers.clear()
	if not active():
		return
	_ensure_built()
	for i in range(_bodies.size()):
		var center := Vector3.ZERO
		var ok := false
		for _try in range(40):
			var a := _rng.randf_range(0.0, TAU)
			var r := _rng.randf_range(_radius_min, _radius_max)
			center = Vector3(cos(a) * r, 0.0, sin(a) * r)
			if center.distance_to(spawn_pos) < _clear_r:
				continue
			var clash := false
			for p in resource_positions:
				if center.distance_to(p) < _keepout:
					clash = true
					break
			if not clash:
				ok = true
				break
		if not ok:
			# aucune place trouvée en 40 essais → on n'invente pas une position douteuse : cet arbre
			# reste masqué (mieux vaut une forêt un peu moins dense qu'une ressource emmurée).
			_bodies[i].visible = false
			continue
		center.y = _height * 0.5                              # posé sur le sol
		_bodies[i].global_transform = Transform3D(Basis(), center)
		_bodies[i].visible = true
		_centers.append(center)
	print("[forest] episode : %d/%d arbres places (keep-out %.1f m autour de %d ressources)"
		% [_centers.size(), _bodies.size(), _keepout, resource_positions.size()])


func get_positions() -> Array[Vector3]:
	return _centers


func _env(key: String, dflt: String) -> String:
	var v := OS.get_environment(key)
	return v if v != "" else dflt


func _envf(key: String, dflt: float) -> float:
	var v := OS.get_environment(key)
	return float(v) if v != "" else dflt
