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
var _keepout := 1.4                                # marge MINI ajoutée au rayon effectif (voir _effective_r)
var _min_gap := 0.0                                # ESPACEMENT MINIMAL entre deux arbres. 0 = aucun
                                                   # contrôle (historique) → les troncs pouvaient se
                                                   # COLLER, voire se superposer : on ne testait que la
                                                   # distance au spawn et aux ressources, jamais entre
                                                   # arbres. Env: SYLVAN_FOREST_MIN_GAP.
var _clump := 1                                    # troncs par MASSIF. 1 = arbre isolé (historique,
                                                   # bit-identique). >1 = bosquet : plusieurs troncs
                                                   # groupés sous un MÊME corps → occulteur LARGE.
var _clump_r := 1.0                                # rayon de dispersion des troncs dans le massif
# ── ARRANGEMENT ÉCOLOGIQUE (2026-07-24) ────────────────────────────────────────────────────────
# Une vraie forêt n'est PAS un tirage uniforme : les graines tombent près du parent (PEUPLEMENTS),
# les arbres se concurrencent (ESPACEMENT MINIMAL), et un arbre tombé ouvre une CLAIRIÈRE. On
# implémente le processus de Neyman-Scott/Thomas, standard en écologie spatiale : des centres de
# peuplement, puis des arbres dispersés autour selon une gaussienne.
# POURQUOI ÇA SERT ICI, au-delà du réalisme : une forêt STRUCTURÉE produit une occlusion NON
# UNIFORME — couloirs de visibilité, écrans denses, ouvertures. C'est ce qui rend l'affût signifiant.
# Un semis uniforme ne serait qu'un brouillard homogène, où aucune position ne vaut mieux qu'une autre.
# OFF (_stands <= 0) → tirage uniforme historique, bit-identique.
var _stands := 0                                   # nb de PEUPLEMENTS (0 = uniforme, historique)
var _stand_sigma := 3.0                            # dispersion des arbres autour de leur peuplement
var _clearings := 0                                # nb de CLAIRIÈRES (zones sans arbres)
var _clearing_r := 4.0                             # rayon d'une clairière
var _stand_centers: Array[Vector3] = []
var _clearing_centers: Array[Vector3] = []
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
	# MASSIFS (2026-07-21) : MESURÉ que 40 troncs fins ne créent AUCUNE situation de mémoire --
	# 12 pertes sur 13 sont des sorties de PORTÉE, pas des occlusions, parce qu'un tronc de 0,35 m
	# défile en quelques pas. Il faut des occulteurs PLUS GROS, pas plus NOMBREUX : une ressource
	# cachée DURABLEMENT. D'où le massif = amas de troncs sous un seul corps.
	_min_gap = _envf("SYLVAN_FOREST_MIN_GAP", _min_gap)
	_clump = int(_env("SYLVAN_FOREST_CLUMP", "1"))
	_clump_r = _envf("SYLVAN_FOREST_CLUMP_R", _clump_r)
	_stands = int(_env("SYLVAN_FOREST_STANDS", "0"))
	_stand_sigma = _envf("SYLVAN_FOREST_STAND_SIGMA", _stand_sigma)
	_clearings = int(_env("SYLVAN_FOREST_CLEARINGS", "0"))
	_clearing_r = _envf("SYLVAN_FOREST_CLEARING_R", _clearing_r)
	# Override d'apparence = TEST DE PURETÉ : la réaction survit-elle à un changement de couleur ?
	# (même contrat que SYLVAN_OBSTACLE_HUE). ⚠️ toute couleur ≠ verte RÉINTRODUIT de la fuite — c'est
	# justement ce qu'on veut pouvoir mesurer.
	var hue := OS.get_environment("SYLVAN_FOREST_HUE")
	if hue != "":
		var p := hue.split(",")
		if p.size() == 3:
			_color = Color(float(p[0]), float(p[1]), float(p[2]))


# Rayon EFFECTIF de l'occulteur. ⚠️ Le keep-out doit en dépendre : avec des massifs, un centre à
# 1,4 m d'une ressource l'ENGLOUTIRAIT. Sans ça on mesurerait un échec du MONDE, pas de l'entité (§2).
func _effective_r() -> float:
	return _trunk_r + (_clump_r if _clump > 1 else 0.0)


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
		# MASSIF : `_clump` troncs sous un MÊME corps. clump=1 → un arbre isolé (historique).
		# Les troncs sont disposés en couronne + un au centre : occulteur LARGE et sans trou,
		# tout en restant du low-poly (des cylindres, pas un mesh importé).
		for j in range(max(_clump, 1)):
			var off := Vector3.ZERO
			if _clump > 1 and j > 0:
				var ang := TAU * float(j - 1) / float(_clump - 1)
				off = Vector3(cos(ang), 0.0, sin(ang)) * _clump_r
			var cs := CollisionShape3D.new()
			var cyl := CylinderShape3D.new()
			cyl.radius = _trunk_r
			cyl.height = _height
			cs.shape = cyl
			cs.position = off
			body.add_child(cs)
			var mesh := MeshInstance3D.new()
			var cm := CylinderMesh.new()
			cm.top_radius = _trunk_r * 0.75                  # léger fuselage : lit mieux en low-poly
			cm.bottom_radius = _trunk_r
			cm.height = _height
			mesh.mesh = cm
			mesh.material_override = _material
			mesh.position = off
			body.add_child(mesh)
			# HOUPPIER (cosmétique). ⚠️ Il ne change RIEN à la perception : la rétine lit le meta
			# `retina_color` du corps, pas le maillage. On garde donc le MÊME vert mesuré (fuite 0.0000)
			# pour que ce que voit l'OWNER corresponde à ce que voit l'ENTITÉ — un houppier brun ferait
			# joli et mentirait sur la perception.
			var can := MeshInstance3D.new()
			var cone := CylinderMesh.new()
			cone.top_radius = 0.0
			cone.bottom_radius = _trunk_r * 1.25   # discret : un houppier large masque la SCENE a
			cone.height = _height * 0.45           # l'observateur, or ce visuel sert a JUGER
			can.mesh = cone
			can.material_override = _material
			can.position = off + Vector3(0.0, _height * 0.55, 0.0)
			body.add_child(can)
		body.visible = false
		add_child(body)
		_bodies.append(body)
	# BANNIERE (anti-log-qui-ment) : le log doit PROUVER ce qui est reellement construit et servi.
	print("[forest] %d massifs SOLIDES (%d troncs chacun) | couleur=%s | tronc r=%.2f | rayon EFFECTIF %.2f m | anneau %.1f-%.1f m"
		% [_count, max(_clump, 1), str(_color), _trunk_r, _effective_r(), _radius_min, _radius_max])


# Disperse les arbres pour le nouvel épisode. GARDES : jamais dans le rayon dégagé autour du spawn
# (sinon l'agent démarre emmuré), jamais à moins de `_keepout` d'une ressource (sinon elle devient
# inatteignable et on mesurerait un échec du MONDE, pas de l'entité — §2).
func begin_episode(_episode_index: int, spawn_pos: Vector3, resource_positions: Array) -> void:
	_centers.clear()
	if not active():
		return
	_ensure_built()
	_sample_structure()
	for i in range(_bodies.size()):
		var center := Vector3.ZERO
		var ok := false
		for _try in range(40):
			center = _propose_position()
			if center.distance_to(spawn_pos) < _clear_r + _effective_r():
				continue
			var in_clearing := false
			for cc in _clearing_centers:              # un arbre ne pousse pas dans une clairière
				if center.distance_to(cc) < _clearing_r:
					in_clearing = true
					break
			if in_clearing:
				continue
			var clash := false
			for p in resource_positions:
				if center.distance_to(p) < _keepout + _effective_r():
					clash = true
					break
			if not clash and _min_gap > 0.0:
				for c in _centers:                    # espacement ARBRE-ARBRE (sinon ils se collent)
					if center.distance_to(c) < _min_gap:
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
	if _min_gap > 0.0:
		print("[forest] espacement mini entre arbres : %.2f m" % _min_gap)
	if _stands > 0 or _clearings > 0:
		# PROUVER l'arrangement au lieu de l'affirmer (règle de méthode du projet).
		print("[forest] structure : %d peuplements (sigma %.1f m), %d clairieres (r %.1f m) | "
			% [_stand_centers.size(), _stand_sigma, _clearing_centers.size(), _clearing_r]
			+ "Clark-Evans MESURE %.2f (<1 = groupe, 1 = aleatoire)" % _clark_evans())


func _sample_structure() -> void:
	# Centres de PEUPLEMENT et de CLAIRIÈRE, tirés une fois par épisode (déterministe via _rng).
	_stand_centers.clear()
	_clearing_centers.clear()
	for _i in range(_stands):
		var a := _rng.randf_range(0.0, TAU)
		var r := sqrt(_rng.randf()) * _radius_max     # uniforme en AIRE (sinon ça s'entasse au centre)
		_stand_centers.append(Vector3(cos(a) * r, 0.0, sin(a) * r))
	for _i in range(_clearings):
		var a2 := _rng.randf_range(0.0, TAU)
		var r2 := sqrt(_rng.randf()) * _radius_max
		_clearing_centers.append(Vector3(cos(a2) * r2, 0.0, sin(a2) * r2))


func _propose_position() -> Vector3:
	# Sans peuplement : tirage uniforme historique. Avec : gaussienne autour d'un peuplement tiré au
	# sort (processus de Thomas) — c'est ce qui produit des massifs et des trouées plutôt qu'un semis
	# régulier, donc une occlusion NON uniforme.
	if _stand_centers.is_empty():
		var a := _rng.randf_range(0.0, TAU)
		var r := _rng.randf_range(_radius_min, _radius_max)
		return Vector3(cos(a) * r, 0.0, sin(a) * r)
	var c: Vector3 = _stand_centers[_rng.randi() % _stand_centers.size()]
	var p := c + Vector3(_rng.randfn(0.0, _stand_sigma), 0.0, _rng.randfn(0.0, _stand_sigma))
	var d := Vector2(p.x, p.z).length()
	if d > _radius_max:                                # rabattu dans l'arène, jamais inventé dehors
		p = Vector3(p.x / d * _radius_max, 0.0, p.z / d * _radius_max)
	return p


func _clark_evans() -> float:
	# INDICE DE CLARK-EVANS = (distance moyenne au plus proche voisin) / (attendue sous Poisson).
	# < 1 = GROUPÉ (peuplements), = 1 = aléatoire, > 1 = régulier. C'est la statistique standard en
	# écologie spatiale : elle PROUVE que l'arrangement est groupé au lieu qu'on l'affirme.
	var n := _centers.size()
	if n < 3:
		return 1.0
	var tot := 0.0
	for i in range(n):
		var best := INF
		for j in range(n):
			if i != j:
				best = minf(best, _centers[i].distance_to(_centers[j]))
		tot += best
	var observed := tot / float(n)
	var density := float(n) / (PI * _radius_max * _radius_max)
	return observed / (0.5 / sqrt(density))


func get_positions() -> Array[Vector3]:
	return _centers


func _env(key: String, dflt: String) -> String:
	var v := OS.get_environment(key)
	return v if v != "" else dflt


func _envf(key: String, dflt: float) -> float:
	var v := OS.get_environment(key)
	return float(v) if v != "" else dflt
