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
var _use_mesh := false         # habillage KayKit (visuel seul) ; défaut OFF = primitives, bit-identique
var _mesh_cache = null         # modèle chargé UNE fois puis dupliqué (un glTF par arbre serait absurde)
var _mesh_pool: Array = []     # les essences disponibles ; l'arbre i prend _mesh_pool[i % taille]
var _mesh_tried := false       # on ne retente pas un chargement qui a échoué à chaque arbre
var _grass_root: Node3D = null # racine du sous-bois VISUEL ; null = jamais construit (headless)
var _grass_per_tree := 10      # touffes semées par arbre dans son disque de traînée
var _grass_cache = null        # modèle d'herbe chargé une fois puis dupliqué
var _grass_tried := false
var _grass_radius := 2.5       # = SYLVAN_TERRAIN_RADIUS, le disque RÉELLEMENT compté
var _rng_grass := RandomNumberGenerator.new()  # flux DÉDIÉ : le décor ne doit pas décaler
                                               # le tirage des commandes (§6quater F)
var _count_min := 0            # borne basse de l'effectif par épisode ; == _count → variation OFF
var _episode_count := 0        # effectif RÉELLEMENT servi cet épisode (tiré dans [_count_min, _count])
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
# APPARENCE VARIABLE DU NON-NOURRITURE (§2.8) — OPT-IN SYLVAN_FOREST_APPEARANCE_VAR, défaut 0 = OFF.
# §2.8 : « faire varier TOUT, pas seulement la nourriture ». Un encodeur entraîné sur des troncs TOUS
# identiques n'alloue aucune capacité à l'apparence des troncs. En variant la teinte PAR ARBRE (stable
# dans une vie, re-tirée PAR ÉPISODE → variable entre objets ET entre épisodes, la règle §2.8), on
# force l'encodeur à représenter aussi l'apparence du non-nourriture. 🚨 GARDE §3 (tronc-brun) : la
# teinte jitterée est CLAMPÉE hors des cônes ressource (cos-rouge/cos-bleu < seuil) — un tronc qui
# dériverait vers le rouge serait lu comme de la NOURRITURE. Défaut 0 → couleur unique, bit-identique.
var _appearance_var := 0.0
# TAILLE VARIABLE PAR ARBRE (2026-07-28) — OPT-IN SYLVAN_FOREST_RADIUS_VAR, défaut 0 = OFF
# bit-identique. MÊME RAISONNEMENT QUE §2.8 POUR LA COULEUR, appliqué à la GÉOMÉTRIE : un encodeur
# qui n'a jamais vu que des troncs de 0,35 m n'a aucune raison d'encoder « rayon » comme une
# variable — il peut le figer dans ses poids. Un arbre d'une autre taille devient alors
# hors-distribution, et l'entité ne peut PAS s'y adapter : non parce qu'elle en serait incapable,
# mais parce qu'on ne lui a jamais montré deux tailles. C'est exactement le mécanisme du verrou A1
# (l'encodeur ne représente que ce qui a VARIÉ), transposé de la teinte à la forme.
# Le rayon est tiré PAR ARBRE et RE-TIRÉ à chaque épisode : stable dans une vie (prévisible), varié
# entre objets ET entre vies (informatif) — la règle §2.8 mot pour mot.
# ⚠️ Contrairement à la couleur, le rayon change la COLLISION *et* la rétine : c'est de la géométrie,
# donc ça doit entrer dans la collecte, jamais après.
var _radius_var := 0.0
var _rad_lo := 0.0
var _rad_hi := 0.0
var _trunk_r_of: Array[float] = []
var _shapes: Array = []       # CylinderShape3D par arbre (clump=1) — redimensionnées par épisode
var _cyls: Array = []         # CylinderMesh par arbre — le visuel suit la collision
var _rng := RandomNumberGenerator.new()

var _material: StandardMaterial3D
var _mats: Array[StandardMaterial3D] = []     # un matériau PAR arbre (apparence variable, §2.8)
var _bodies: Array[StaticBody3D] = []
var _centers: Array[Vector3] = []
# §6bis : étendue d'apparence RÉELLEMENT servie, mesurée sur les arbres placés cet épisode.
var _app_cosred_lo := 1.0
var _app_cosred_hi := 0.0


func _init() -> void:
	_count = int(_env("SYLVAN_FOREST_COUNT", "0"))
	# Défaut = _count : sans variable servie, le tirage est dégénéré et le monde bit-identique.
	# Clampé à [0, _count] parce que la borne haute est le nombre de corps réellement CONSTRUITS —
	# un minimum supérieur au maximum produirait un randi_range inversé, donc un crash ou pire.
	_count_min = clampi(int(_env("SYLVAN_FOREST_COUNT_MIN", str(_count))), 0, _count)
	_use_mesh = _env("SYLVAN_FOREST_MESH", "0") == "1"   # habillage : visuel seul, jamais en collecte
	# Le sous-bois se dessine sur le rayon RÉELLEMENT servi au ralentissement, jamais sur un chiffre
	# recopié : sinon l'image montrerait un disque de traînée qui n'est pas celui qu'on applique.
	_grass_radius = _envf("SYLVAN_TERRAIN_RADIUS", _grass_radius)
	_grass_per_tree = int(_env("SYLVAN_FOREST_UNDERGROWTH", "0"))
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
	_appearance_var = _envf("SYLVAN_FOREST_APPEARANCE_VAR", 0.0)
	_radius_var = clampf(_envf("SYLVAN_FOREST_RADIUS_VAR", 0.0), 0.0, 0.9)


# Rayon EFFECTIF de l'occulteur. ⚠️ Le keep-out doit en dépendre : avec des massifs, un centre à
# 1,4 m d'une ressource l'ENGLOUTIRAIT. Sans ça on mesurerait un échec du MONDE, pas de l'entité (§2).
func _effective_r() -> float:
	# Rayon MAXIMAL possible, pas le nominal : avec des troncs de taille variable, réserver l'espace
	# du rayon moyen laisserait deux gros arbres se chevaucher — et le keep-out autour des ressources
	# cesserait de garantir ce qu'il promet.
	return _trunk_r * (1.0 + _radius_var) + (_clump_r if _clump > 1 else 0.0)


func active() -> bool:
	return _count > 0


func set_seed(value: int) -> void:
	_rng.seed = value


# Construit UNE FOIS `count` troncs cylindriques, à la fois bloquants (bit 2) et perceptibles
# (bit 7 + meta retina_color). Idempotent — le placement se fait dans begin_episode.
func _ensure_built() -> void:
	if not active() or not _bodies.is_empty():
		return
	# Racine du sous-bois : créée UNIQUEMENT en mode visuel. En headless elle reste null et
	# _build_undergrowth() sort au premier test → la collecte ne paie rien et ne change pas.
	if _grass_per_tree > 0 and DisplayServer.get_name() != "headless":
		_rng_grass.seed = 20260729
		_grass_root = Node3D.new()
		_grass_root.name = "SousBois"
		add_child(_grass_root)
	_material = StandardMaterial3D.new()
	_material.albedo_color = _color
	_material.emission_enabled = true
	_material.emission = _color * 0.25
	for i in range(_count):
		var body := StaticBody3D.new()
		body.collision_layer = OBSTACLE_LAYER | RETINA_LAYER
		body.collision_mask = 0                              # statique : ne détecte rien lui-même
		body.set_meta("retina_color", _color)                # RGB lu par le raycast couleur de la rétine
		# Matériau PAR ARBRE (apparence variable §2.8). Défaut = _color → OFF strictement bit-identique
		# (même couleur pour tous, comme le matériau partagé historique). La teinte par-arbre est
		# posée par _apply_tree_appearance() dans begin_episode quand _appearance_var > 0.
		var body_mat := _material
		if _appearance_var > 0.0:
			body_mat = StandardMaterial3D.new()
			body_mat.albedo_color = _color
			body_mat.emission_enabled = true
			body_mat.emission = _color * 0.25
		_mats.append(body_mat)
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
			if j == 0:
				_shapes.append(cyl)
			cyl.height = _height
			cs.shape = cyl
			cs.position = off
			body.add_child(cs)
			var mesh := MeshInstance3D.new()
			var cm := CylinderMesh.new()
			cm.top_radius = _trunk_r * 0.75                  # léger fuselage : lit mieux en low-poly
			cm.bottom_radius = _trunk_r
			if j == 0:
				_cyls.append(cm)
			cm.height = _height
			mesh.mesh = cm
			mesh.material_override = body_mat
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
			can.material_override = body_mat
			can.position = off + Vector3(0.0, _height * 0.55, 0.0)
			body.add_child(can)
			# HABILLAGE (2026-07-29, opt-in SYLVAN_FOREST_MESH=1, mode VISUEL uniquement).
			# On remplace la silhouette cylindre+cône par un vrai modèle KayKit (CC0), et on cache
			# les primitives. Trois invariants tenus, sans lesquels ce serait un habillage malhonnête :
			#   1. la COLLISION ne bouge pas (le CollisionShape3D reste le cylindre) → la profondeur
			#      lue par la rétine est bit-identique, donc le WM en cours d'entraînement reste valide ;
			#   2. `retina_color` ne bouge pas, et le modèle est TEINTÉ vers cette même couleur →
			#      l'owner continue de voir ce que l'entité perçoit (§2.1, cf. le houppier ci-dessus) ;
			#   3. défaut OFF + repli sur les primitives si le pack manque (il est git-ignoré, donc
			#      absent pour quiconque clone le dépôt) → aucune régression possible.
			if _use_mesh:
				var pretty := _load_tree_model(body_mat, i)
				if pretty != null:
					# 🚨 LES ARBRES VOLAIENT (repéré à l'œil par l'owner, 2026-07-29). Le CORPS est
					# posé à y = _height/2 parce qu'un CylinderMesh est centré sur son origine et
					# doit donc remonter d'une demi-hauteur pour toucher le sol. Le modèle glTF, lui,
					# a son origine à sa BASE : le placer sur l'origine du corps le décollait
					# d'exactement une demi-hauteur, soit 1 m. On redescend donc de la même quantité.
					# La collision, elle, n'a jamais bougé — c'est bien pourquoi seul l'œil pouvait
					# le voir, et pourquoi aucune mesure ne l'aurait signalé.
					pretty.position = off + Vector3(0.0, -_height * 0.5, 0.0)
					body.add_child(pretty)
					mesh.visible = false
					can.visible = false
		body.visible = false
		add_child(body)
		_bodies.append(body)
	# BANNIERE (anti-log-qui-ment) : le log doit PROUVER ce qui est reellement construit et servi.
	print("[forest] %d massifs SOLIDES (%d troncs chacun) | couleur=%s | tronc r=%.2f | rayon EFFECTIF %.2f m | anneau %.1f-%.1f m"
		% [_count, max(_clump, 1), str(_color), _trunk_r, _effective_r(), _radius_min, _radius_max])


# Disperse les arbres pour le nouvel épisode. GARDES : jamais dans le rayon dégagé autour du spawn
# (sinon l'agent démarre emmuré), jamais à moins de `_keepout` d'une ressource (sinon elle devient
# inatteignable et on mesurerait un échec du MONDE, pas de l'entité — §2).
# Charge le modèle d'arbre UNE fois et le teinte à la couleur perçue. Renvoie un duplicata prêt à
# être posé, ou null (pack absent, mode headless, chargement raté) → l'appelant garde les primitives.
func _load_tree_model(mat: StandardMaterial3D, idx: int) -> Node3D:
	if not _mesh_tried:
		_mesh_tried = true
		# HEADLESS = on ne charge RIEN. Les workers de collecte ne doivent payer ni le temps de
		# chargement ni la mémoire, et surtout le monde qu'ils enregistrent doit rester le même.
		if DisplayServer.get_name() == "headless":
			return null
		var dir := ProjectSettings.globalize_path("res://../ForestLowPolyAssets/Assets/gltf/")
		# PLUSIEURS ESSENCES, pas une (2026-07-29). Quarante exemplaires du même modèle se lisent
		# comme un motif répété, pas comme une forêt — l'œil repère la copie avant de voir l'arbre.
		# Les variantes du pack coûtent un chargement chacune et rien de plus : elles sont mises en
		# cache une fois et dupliquées ensuite, comme le modèle unique l'était déjà.
		for f in ["Tree_1_A_Color1.gltf", "Tree_2_A_Color1.gltf", "Tree_1_C_Color1.gltf",
				  "Tree_2_B_Color1.gltf"]:
			var doc := GLTFDocument.new()
			var st := GLTFState.new()
			if doc.append_from_file(dir + f, st) == OK:
				var n := doc.generate_scene(st)
				if n != null:
					_mesh_pool.append(n)
		if not _mesh_pool.is_empty():
			_mesh_cache = _mesh_pool[0]
		print("[forest] habillage ARBRES : %s" % ("%d essences KayKit" % _mesh_pool.size()
			if _mesh_cache != null else "REPLI primitif (pack introuvable a %s)" % dir))
	if _mesh_cache == null:
		return null
	# Essence choisie par l'INDICE de l'arbre, pas au hasard : stable d'un épisode à l'autre pour un
	# même arbre, donc la scène ne scintille pas entre deux vies alors que rien n'a bougé.
	var src: Node3D = _mesh_pool[idx % _mesh_pool.size()]
	var inst: Node3D = src.duplicate()
	# Le modèle KayKit fait ~1 unité de large ; on l'amène au gabarit du tronc SERVI pour que la
	# silhouette rendue corresponde à l'obstacle réel. Un arbre dessiné plus fin que sa collision
	# ferait croire à un passage qui n'existe pas — le mensonge inverse de celui du houppier brun.
	var s := _height / 2.0
	inst.scale = Vector3(s, s, s)
	_tint(inst, mat)
	return inst


# Applique au modèle LE MATÉRIAU DU CORPS — surtout pas un matériau neuf.
#
# 🚨 CORRIGÉ AUSSITÔT QU'ÉCRIT (2026-07-29) : ma première version fabriquait un StandardMaterial3D
# figé sur la couleur GLOBALE. Or `_apply_tree_appearance()` donne à CHAQUE arbre une teinte jittée
# (appearance_var) et l'écrit à la fois dans sa méta `retina_color` et dans `_mats[i]`. Le modèle
# habillé ignorait donc la variation : l'entité percevait 40 verts distincts pendant que l'owner
# voyait 40 arbres identiques — exactement le mensonge visuel reproché au houppier brun, retourné.
# En partageant `_mats[i]`, la variation par-arbre se propage seule, et le visuel gagne au passage
# la richesse qu'il avait perdue : la diversité qu'on affiche est celle qui EXISTE.
func _tint(n: Node, mat: StandardMaterial3D) -> void:
	if n is MeshInstance3D:
		(n as MeshInstance3D).material_override = mat
	for c in n.get_children():
		_tint(c, mat)


# Un arbre ABSENT doit l'être pour TOUT LE MONDE : le rendu, la collision, et la rétine. Séparer ces
# trois vérités est exactement la façon dont un monde se met à mentir en silence.
func _hide_tree(i: int) -> void:
	_bodies[i].visible = false
	_bodies[i].collision_layer = 0


func _show_tree(i: int) -> void:
	_bodies[i].visible = true
	_bodies[i].collision_layer = OBSTACLE_LAYER | RETINA_LAYER


func begin_episode(_episode_index: int, spawn_pos: Vector3, resource_positions: Array) -> void:
	_centers.clear()
	if not active():
		return
	_ensure_built()
	# EFFECTIF TIRÉ PAR ÉPISODE (§2.8 : stable dans l'épisode, variable entre épisodes → apprenable).
	# But : que le WM apprenne une FAMILLE de forêts et pas une constante, sinon changer la densité
	# après coup le met hors-distribution et coûte une collecte + un retrain entiers. La borne haute
	# est le nombre de corps construits (_count) ; la borne basse vient du preset. Défaut _count_min
	# = _count → tirage dégénéré, monde bit-identique, et surtout AUCUN tirage consommé.
	_episode_count = _count
	if _count_min < _count:
		_episode_count = _rng.randi_range(_count_min, _count)
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
		if i >= _episode_count or not ok:
			# Deux cas se rejoignent ici : l'arbre est hors de l'effectif tiré pour cet épisode, ou
			# aucune place n'a été trouvée en 40 essais (on n'invente pas une position douteuse —
			# mieux vaut une forêt un peu moins dense qu'une ressource emmurée).
			# 🚨 `visible = false` NE SUFFIT PAS, et c'était un défaut latent : un StaticBody3D masqué
			# garde sa collision ET sa couche rétine. L'arbre restait donc un MUR INVISIBLE, et un
			# obstacle que la rétine voit sans que rien ne soit rendu — le pire des deux mondes.
			# Inoffensif tant que 191/191 arbres étaient placés ; catastrophique dès qu'on en retire
			# la moitié par épisode. On éteint les DEUX couches, et on les rallume au placement.
			_hide_tree(i)
			continue
		center.y = _height * 0.5                              # posé sur le sol
		_bodies[i].global_transform = Transform3D(Basis(), center)
		_show_tree(i)
		_centers.append(center)
	# La bannière rapporte l'effectif TIRÉ, pas seulement le nombre de corps construits : sans ça une
	# variation par épisode serait invisible dans les logs et un corpus ne se décrirait pas lui-même.
	print("[forest] episode : %d/%d arbres places (effectif tire %d dans [%d,%d]) "
		% [_centers.size(), _episode_count, _episode_count, _count_min, _count]
		+ "(keep-out %.1f m autour de %d ressources)" % [_keepout, resource_positions.size()])
	if _min_gap > 0.0:
		print("[forest] espacement mini entre arbres : %.2f m" % _min_gap)
	# PROUVER l'arrangement au lieu de l'affirmer (règle de méthode du projet).
	# ⚠️ IMPRIMÉ EN PERMANENCE, pas seulement en mode peuplements : le gate demande « Clark-Evans < 1
	# en peuplements, ≈ 1 en UNIFORME ». Tant que la ligne n'était émise que si _stands > 0, le TÉMOIN
	# uniforme était structurellement inobservable — on ne pouvait pas passer le gate, seulement
	# l'affirmer.
	print("[forest] structure : %d peuplements (sigma %.1f m), %d clairieres (r %.1f m) | "
		% [_stand_centers.size(), _stand_sigma, _clearing_centers.size(), _clearing_r]
		+ "n=%d ppv_moyen MESURE %.3f m | aire %.1f m2 | Clark-Evans MESURE %.3f (<1 = groupe, 1 = aleatoire)"
		% [_centers.size(), _mean_nn(), _support_area(), _clark_evans()])
	_apply_tree_radius()
	_apply_tree_appearance()
	_build_undergrowth()
	if _radius_var > 0.0:
		print("[forest] taille : var %.2f | rayon des troncs MESURE %.3f..%.3f m (nominal %.2f) — la geometrie VARIE, donc l encodeur peut l apprendre"
			% [_radius_var, _rad_lo, _rad_hi, _trunk_r])
	if _appearance_var > 0.0:
		# §6bis : prouver l'étendue d'apparence RÉELLEMENT servie. cos-rouge lo..hi > 0 = ça VARIE ;
		# hi < 0.55 = tous les arbres restent HORS du cône bouffe (garde §3, aucun tronc lu comme bouffe).
		print("[forest] apparence : var %.2f | cos-rouge des arbres MESURE %.3f..%.3f (hi < 0.55 = tous hors cone bouffe)"
			% [_appearance_var, _app_cosred_lo, _app_cosred_hi])


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


# Distance MOYENNE au plus proche voisin, en mètres. C'est la grandeur BRUTE, sans hypothèse de
# normalisation : c'est elle qu'on compare entre le tirage uniforme et les peuplements. L'indice de
# Clark-Evans ci-dessous en dérive, mais il dépend d'une aire de référence — donc d'un choix.
func _mean_nn() -> float:
	var n := _centers.size()
	if n < 3:
		return 0.0
	var tot := 0.0
	for i in range(n):
		var best := INF
		for j in range(n):
			if i != j:
				best = minf(best, _centers[i].distance_to(_centers[j]))
		tot += best
	return tot / float(n)


# AIRE DE RÉFÉRENCE = le support RÉELLEMENT échantillonné, pas le disque commode.
# ⚠️ Le code d'origine divisait toujours par PI*rmax^2. En tirage uniforme les arbres sortent d'un
# ANNEAU [rmin, rmax] : surestimer l'aire sous-estime la densité, donc SURESTIME la distance
# attendue, donc SOUS-ESTIME l'indice — le témoin « uniforme » aurait paru groupé sans qu'aucun
# arbre ne soit groupé. En mode peuplements les gaussiennes remplissent le disque entier (elles
# peuvent tomber sous rmin), donc le support y est bien le disque.
func _support_area() -> float:
	if _stand_centers.is_empty():
		return PI * (_radius_max * _radius_max - _radius_min * _radius_min)
	return PI * _radius_max * _radius_max


func _clark_evans() -> float:
	# INDICE DE CLARK-EVANS = (distance moyenne au plus proche voisin) / (attendue sous Poisson).
	# < 1 = GROUPÉ (peuplements), = 1 = aléatoire, > 1 = régulier. C'est la statistique standard en
	# écologie spatiale : elle PROUVE que l'arrangement est groupé au lieu qu'on l'affirme.
	# ⚠️ DEUX BIAIS CONNUS, à garder en tête en lisant le chiffre : (a) EFFET DE BORD — dans un
	# domaine borné les arbres du pourtour n'ont pas de voisin au-delà, ce qui GONFLE l'indice ;
	# (b) le tirage uniforme historique tire le rayon en randf_range(rmin, rmax), donc uniformément
	# EN RAYON et non en AIRE, ce qui laisse une densité en 1/r et GROUPE légèrement vers l'intérieur.
	# ⇒ le discriminant honnête reste la comparaison des ppv_moyen mesurés entre les deux modes.
	var n := _centers.size()
	if n < 3:
		return 1.0
	var density := float(n) / _support_area()
	return _mean_nn() / (0.5 / sqrt(density))


func get_positions() -> Array[Vector3]:
	return _centers


# Cosinus de la couleur `c` avec la requête (r,g,b) — même mesure que le slot (perception normalisée).
func _cos_to(c: Color, r: float, g: float, b: float) -> float:
	var n := sqrt(c.r * c.r + c.g * c.g + c.b * c.b)
	var m := sqrt(r * r + g * g + b * b)
	if n <= 0.0 or m <= 0.0:
		return 0.0
	return (c.r * r + c.g * g + c.b * b) / (n * m)


# Teinte perturbée autour de `base` (HSV), PUIS ramenée hors des cônes ressource si le jitter l'y a
# poussée (garde §3 : un tronc lu comme rouge/bleu serait de la nourriture/eau). Déterministe (_rng).
func _jitter_out_of_cone(base: Color) -> Color:
	var h := fposmod(base.h + _rng.randf_range(-_appearance_var, _appearance_var), 1.0)
	var s := clampf(base.s + _rng.randf_range(-_appearance_var, _appearance_var), 0.2, 1.0)
	var v := clampf(base.v + _rng.randf_range(-0.5 * _appearance_var, 0.5 * _appearance_var), 0.15, 1.0)
	var c := Color.from_hsv(h, s, v)
	for _k in range(8):   # au plus 8 rappels vers le vert de base : converge bien avant
		if _cos_to(c, 1.0, 0.0, 0.0) < 0.5 and _cos_to(c, 0.0, 0.0, 1.0) < 0.5:
			break
		c = c.lerp(base, 0.3)
	return c


# Pose une teinte PAR ARBRE (visible) au nouvel épisode : matériau + meta retina_color. Stable dans la
# vie (posée une fois par épisode), variable entre arbres ET entre épisodes (§2.8). OFF → ne fait rien.
func _apply_tree_radius() -> void:
	# Rayon PAR ARBRE, re-tiré à chaque épisode (règle §2.8 : stable dans une vie, varié entre objets
	# et entre vies). On redimensionne la COLLISION et le MAILLAGE ensemble — les dissocier ferait
	# exactement le mensonge visuel qu'on vient de corriger sur l'habillage.
	if _radius_var <= 0.0:
		return
	_trunk_r_of.clear()
	_rad_lo = 1e9
	_rad_hi = 0.0
	for i in range(_shapes.size()):
		var f := 1.0 + _rng.randf_range(-_radius_var, _radius_var)
		var r := maxf(0.05, _trunk_r * f)
		_trunk_r_of.append(r)
		(_shapes[i] as CylinderShape3D).radius = r
		(_cyls[i] as CylinderMesh).bottom_radius = r
		(_cyls[i] as CylinderMesh).top_radius = r * 0.75
		_rad_lo = minf(_rad_lo, r)
		_rad_hi = maxf(_rad_hi, r)


func _apply_tree_appearance() -> void:
	if _appearance_var <= 0.0:
		return
	_app_cosred_lo = 1.0
	_app_cosred_hi = 0.0
	for i in range(_bodies.size()):
		if not _bodies[i].visible:
			continue
		var c := _jitter_out_of_cone(_color)
		_bodies[i].set_meta("retina_color", c)
		_mats[i].albedo_color = c
		_mats[i].emission = c * 0.25
		var cr := _cos_to(c, 1.0, 0.0, 0.0)
		_app_cosred_lo = minf(_app_cosred_lo, cr)
		_app_cosred_hi = maxf(_app_cosred_hi, cr)


# FACTEUR DE VITESSE DU TERRAIN à la position `pos` (docs/design_foret_complete.md §2.3).
# Le sous-bois pousse là où les arbres sont denses ; y avancer est plus lent. On approxime le
# sous-bois par la DENSITÉ LOCALE d'arbres — déjà perceptible (les arbres sont sur la rétine), donc
# le §1 filtre est respecté sans ajouter d'objet invisible. `strength` = pente du ralentissement par
# arbre proche. Retourne 1.0 (dégagé) à `floor_v` (le plus lent) ; 1.0 exact si strength <= 0 (OFF).
# ⚠️ C'est DISTINCT de la collision (bit 2) : un tronc ARRÊTE net, le sous-bois RALENTIT en continu.
func speed_multiplier_at(pos: Vector3, strength: float, radius: float, floor_v: float) -> float:
	if strength <= 0.0 or _centers.is_empty():
		return 1.0
	var n := 0
	var r2 := radius * radius
	for c in _centers:
		if Vector2(pos.x - c.x, pos.z - c.z).length_squared() < r2:
			n += 1
	return maxf(floor_v, 1.0 / (1.0 + strength * float(n)))


# SOUS-BOIS VISIBLE (2026-07-29, opt-in SYLVAN_FOREST_UNDERGROWTH, mode VISUEL seulement).
#
# POURQUOI. `speed_multiplier_at` ci-dessus ralentit l'entité selon le NOMBRE d'arbres dans un rayon
# de `terrain_radius` — c'est du sous-bois, et c'est lourd : le facteur terrain mesuré (0,635) fait
# passer le budget de trajet de 84,9 m à 53,9 m par vie, l'une des constantes les plus lourdes du
# monde. Or il n'avait AUCUN rendu : le sol paraissait uniformément plat pendant qu'elle pataugeait.
#
# CE QUE ÇA MONTRE, ET CE QUE ÇA NE MONTRE PAS. On sème une nappe par arbre, du rayon exact du disque
# de traînée : là où les disques se recouvrent, la végétation s'épaissit d'elle-même, donc l'image
# dit la vérité sur l'endroit où la vitesse tombe. En revanche l'entité, elle, ne VOIT pas ce
# sous-bois : il n'est ni sur la couche 8 ni porteur de `retina_color`. Elle le SUBIT sans le
# percevoir. Ce n'est pas un mensuel visuel — c'est le rendu d'une force réellement vécue — mais
# c'est une ASYMÉTRIE qu'il faut connaître, et c'est une question de conception ouverte : faut-il
# qu'elle puisse voir le terrain qui la freine ? Les touffes sont donc rendues MATES et BASSES,
# quand tout ce qu'elle perçoit est émissif : impossible de les confondre à l'œil avec un objet.
func _build_undergrowth() -> void:
	if _grass_root == null:
		return
	for c in _grass_root.get_children():
		c.queue_free()                       # les arbres bougent à chaque épisode : on re-sème
	if _centers.is_empty():
		return
	# 🚨 PREMIÈRE VERSION REJETÉE À L'ŒIL (owner, 2026-07-29) : dix cônes sombres serrés dans 2,5 m ne
	# se lisaient pas comme de la végétation mais comme une TACHE brune au sol. L'erreur était de
	# dessiner de la DENSITÉ au lieu de dessiner des PLANTES — on voyait la statistique, pas le motif.
	# On sème donc de vraies touffes du pack, plus petites, plus nombreuses et bien plus dispersées.
	var mat := StandardMaterial3D.new()
	# Assez CLAIR pour se détacher de l'humus (0.17,0.26,0.13) : à 0.24 il s'en distinguait à peine et
	# se lisait comme une ombre. Reste MAT, quand tout ce que l'entité perçoit est émissif.
	mat.albedo_color = Color(0.38, 0.52, 0.24)
	mat.roughness = 1.0
	var fallback := CylinderMesh.new()           # repli si le pack manque (il est git-ignoré)
	fallback.top_radius = 0.0
	fallback.bottom_radius = 0.06
	fallback.height = 0.35
	for c in _centers:
		for _k in range(_grass_per_tree):
			var a := _rng_grass.randf_range(0.0, TAU)
			# uniforme EN AIRE dans le disque de traînée : la densité dessinée suit la densité
			# réellement comptée par speed_multiplier_at, au lieu de s'entasser près du tronc.
			# rayon = SYLVAN_TERRAIN_RADIUS : on dessine le disque SERVI, pas un joli disque.
			var r := sqrt(_rng_grass.randf()) * _grass_radius
			var node := _load_grass_model(mat)
			if node == null:
				var m := MeshInstance3D.new()
				m.mesh = fallback
				m.material_override = mat
				node = m
			node.position = Vector3(c.x + cos(a) * r, 0.0, c.z + sin(a) * r)
			node.rotate_y(_rng_grass.randf_range(0.0, TAU))
			var sc := _rng_grass.randf_range(1.1, 2.3)     # tailles variées : un semis régulier
			node.scale = Vector3(sc, sc, sc)               # se lit comme une texture, pas une plante
			_grass_root.add_child(node)


# Touffe d'herbe du pack, chargée une fois puis dupliquée. null → l'appelant prend son repli.
func _load_grass_model(mat: StandardMaterial3D) -> Node3D:
	if not _grass_tried:
		_grass_tried = true
		if DisplayServer.get_name() == "headless":
			return null
		var dir := ProjectSettings.globalize_path("res://../ForestLowPolyAssets/Assets/gltf/")
		var doc := GLTFDocument.new()
		var st := GLTFState.new()
		if doc.append_from_file(dir + "Grass_1_A_Color1.gltf", st) == OK:
			_grass_cache = doc.generate_scene(st)
		print("[forest] habillage SOUS-BOIS : %s" % ("modele KayKit" if _grass_cache != null
			else "REPLI primitif (pack introuvable)"))
	if _grass_cache == null:
		return null
	var inst: Node3D = _grass_cache.duplicate()
	_tint(inst, mat)
	return inst


func _env(key: String, dflt: String) -> String:
	var v := OS.get_environment(key)
	return v if v != "" else dflt


func _envf(key: String, dflt: float) -> float:
	var v := OS.get_environment(key)
	return float(v) if v != "" else dflt
