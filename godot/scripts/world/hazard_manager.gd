extends Node3D
class_name HazardManager

# ZONE NOCIVE (2026-07-15) — premier enrichissement du monde : une région qui ABÎME la santé.
# BUT (docs/etat_critique.md, fork owner) : créer de la PLACE pour que l'expérience compte. Le coût
# codé-main du planner est « rapproche-toi de la ressource » — il n'a AUCUN terme de danger. Donc dès
# qu'une zone nocive est sur le chemin, l'entité fonce dedans en aveugle → coût INÉVITABLE par la
# perception/décision actuelles. Cet échec est PROUVABLE, pas espéré : c'est le gate qui casse la boucle.
#
# ÉTAGE : c'est du MONDE (comme food_manager/water_manager). PERCEPTIBLE VIA UN CHAMP DE FINS PILIERS VERTS
# (2026-07-16) — PAS un cylindre plein. Leçon (A/B 8a4f087) : un cylindre plein OCCULTE la bouffe (rétine
# horizontale → tout objet à hauteur-des-rayons bloque la ligne de vue) → l'entité ne voit plus sa nourriture,
# meurt de faim (11/12), n'entre plus dans le danger → baseline cassé + évitement intestable. FIX : la zone
# DE DÉGÂTS reste un disque au sol (damage_at par distance, inchangé) ; la VISIBILITÉ vient de N fins piliers
# verts (poteaux de signalisation) — la rétine voit le vert MAIS les rayons passent ENTRE les piliers → la
# bouffe derrière reste visible. Un disque plat au sol ne marcherait PAS : la rétine scanne à l'horizontale
# (perception.gd) et ne voit jamais le sol. retina_color VERT (seul canal libre : cos<0.55 rouge/bleu).
#
# Tout est OPT-IN (SYLVAN_HAZARD_COUNT=0 par défaut → module inerte, zéro visuel, zéro régression). Placement :
# un disque sur le segment spawn→bouffe-la-plus-proche, à HAZARD_FRAC du trajet → « aller tout droit » le
# traverse par construction. Auto-logge ses stats par épisode (pas de dépendance à main.gd pour le log).

const PILLAR_HEIGHT := 1.5                 # hauteur des piliers (> torse → les rayons rétine horizontaux les touchent)
# VERT : le seul canal libre (rouge=bouffe, bleu=eau). Choisi pour la REQUÊTE-COULEUR du slot (seuil cosinus
# 0.55, cf slot_head.py) : vert normalisé → cos 0.11 avec rouge, 0.23 avec bleu (les 2 < 0.55 = aucune fuite
# dans les slots bouffe/eau), 0.97 avec une requête verte. Le violet (0.6,0.12,0.85) fuyait dans les DEUX
# (cos 0.57 rouge / 0.81 bleu) → aurait corrompu la perception existante. Sonde slot-couleur, 2026-07-15.
const HAZARD_COLOR := Color(0.1, 0.9, 0.15)   # émissif pour ressortir de l'herbe (mate, non-perceptible)

var _discs: Array[Vector3] = []          # centres des zones (monde)
var _radius := 1.3
var _damage := 0.5                        # santé/pas dedans. 0.5 = niveau LÉTAL choisi au gate (2026-07-15) :
                                          # traverser vide la barre (100 dégâts) → 7/12 vies aveugles TUÉES
                                          # par le danger (vs 0 sans) ; éviter = retour au régime normal. En
                                          # dessous (0.1-0.35) la santé est du slack (la faim tue avant) → non
                                          # conséquent. diagnostics/diag_hazard_gate.py.
var _frac := 0.55                         # position sur le segment spawn→bouffe (0=spawn, 1=bouffe)
var _count := 0
var _pillars := 7                         # nb de piliers par zone (1 centre + anneau) — assez pour VOIR, peu pour ne PAS occulter
var _pillar_r := 0.08                     # rayon d'un pilier (fin → les rayons passent entre)
var _rng := RandomNumberGenerator.new()

var _material: StandardMaterial3D
var _visuals: Array[Node3D] = []          # 1 conteneur (champ de piliers) perceptible par zone

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
	_pillars = int(_env("SYLVAN_HAZARD_PILLARS", "7"))       # sweep visibilité↔occlusion
	_pillar_r = _envf("SYLVAN_HAZARD_PILLAR_R", _pillar_r)


func set_seed(value: int) -> void:
	_rng.seed = value


func active() -> bool:
	return _count > 0


# Construit UNE FOIS, par zone, un CHAMP DE FINS PILIERS verts (1 conteneur Node3D → _pillars piliers).
# Chaque pilier = mesh fin + Area3D couche-8 (jamais bloquante, mask 0) portant meta "retina_color" — lu
# par le raycast couleur de la rétine. Fins + espacés → perceptibles MAIS les rayons passent entre → la
# bouffe derrière reste visible (fix occlusion, cf header). Les dégâts restent le DISQUE (damage_at), pas
# les piliers : un rayon qui rate tous les piliers ne voit pas le danger, mais l'entité DEDANS prend quand
# même les dégâts — réaliste (une zone signalée par des poteaux).
func _ensure_built() -> void:
	if not active() or not _visuals.is_empty():
		return
	if _material == null:
		_material = StandardMaterial3D.new()
		_material.albedo_color = HAZARD_COLOR
		_material.emission_enabled = true
		_material.emission = HAZARD_COLOR * 0.6   # émissif → ressort de l'herbe (mate, non-perceptible)
	for i in range(_count):
		var zone := Node3D.new()                  # 1 conteneur = 1 champ de piliers par zone
		add_child(zone)
		for j in range(max(_pillars, 1)):
			# pilier 0 = centre ; les autres sur un anneau à 0.7·rayon → marque la zone sans mur plein
			var off := Vector3.ZERO
			if j > 0:
				var a := TAU * float(j - 1) / float(max(_pillars - 1, 1))
				off = Vector3(cos(a), 0.0, sin(a)) * (_radius * 0.7)
			var m := MeshInstance3D.new()
			var cyl := CylinderMesh.new()
			cyl.top_radius = _pillar_r
			cyl.bottom_radius = _pillar_r
			cyl.height = PILLAR_HEIGHT
			m.mesh = cyl
			m.material_override = _material
			m.position = off + Vector3(0.0, PILLAR_HEIGHT * 0.5, 0.0)   # pose sur le sol
			zone.add_child(m)
			var area := Area3D.new()
			area.collision_layer = 1 << 7   # couche 8 = "perceptible-rétine" (idem food_manager)
			area.collision_mask = 0
			area.set_meta("retina_color", HAZARD_COLOR)
			var cs := CollisionShape3D.new()
			var shape := CylinderShape3D.new()
			shape.radius = _pillar_r
			shape.height = PILLAR_HEIGHT
			cs.shape = shape
			area.add_child(cs)
			m.add_child(area)
		zone.visible = false
		_visuals.append(zone)


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
