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
# BUISSON NEUTRE (chantier attribution de crédit, docs/design_attribution_credit.md) — FOOD only.
# Objet PERCEPTIBLE (rétine layer 8) mais SANS drive, SANS consommation, SANS physique (jamais dans
# _positions → try_consume l'ignore). Co-localisé avec la baie (prob _bush_p) + buissons DISPERSÉS
# seuls (_bush_alone) + baies parfois seules (1−_bush_p) = décorrélation (identifiabilité, cf design).
# Opt-in SYLVAN_FOOD_BUSH=1 ; absent = rien construit, OFF bit-identique. Teinte DÉCLARÉE (propriété
# du monde), choisie séparable (loin de rouge/vert/bleu) — G1 le vérifie, ajuste le MONDE si besoin.
var _bush_enabled := false
var _bush_hue := 0.45          # teinte cible du buisson (déclarée ; entre vert 0.33 et bleu 0.61, loin du rouge)
var _bush_p := 0.9             # prob qu'une baie soit DANS un buisson (co-occurrence)
var _bush_alone := 4           # nombre de buissons DISPERSÉS (sans baie)
var _bush_meshes: Array[MeshInstance3D] = []
var _bush_areas: Array = []
var _bush_material: StandardMaterial3D
# COLLECTE EAT-RICHE (vers 🅑) — leviers de RÉGIME, gated env, défaut = comportement actuel inchangé.
# eat_hunger_max < 1 : ne consommer une pastille QUE si l'énergie (fraction) est sous ce seuil → chaque
# repas a une vraie MARGE (le WM voit la bosse +energy_per_food non écrêtée à 100) → apprend l'eat-dynamics.
# C'est une régime de COLLECTE (comme le babbling overridable), PAS un relâchement du critère d'éval (§2).
var eat_hunger_max := 1.0
var respawn_min := 2.0
var respawn_max := 4.5
# ─────────────────────────────────────────────────────────────────────────────────────────────
# MODE BOSQUETS (docs/design_monde_bosquets.md §9) — opt-in SYLVAN_<PREFIX>_PATCHES, défaut 0 = OFF
# bit-identique. Il remplace le PERPETUAL FIELD (respawn autour de l'AGENT, qui rend la mémoire
# structurellement inutile : la ressource suit l'entité, mesurée à 3,2-4,1 m médian, jamais > 10,17 m)
# par des bosquets à position FIXE qui s'ÉPUISENT et REPOUSSENT sur une horloge.
#
# Ce qui crée l'ignorance ici n'est PAS de l'occlusion (réfutée : il faudrait couvrir ~100 % du sol)
# mais de l'ALIASING PERCEPTUEL, et il ÉMERGE de la géométrie au lieu d'être codé :
#   • le buisson-marqueur fait 1,5 m → sous-tend 21° à 8 m → 2 rayons le touchent de façon fiable
#   • une baie fait 0,35 m → sous-tend 5° = une demi-inter-rayon (36 rayons à 10°) → touchée ou non
#     selon l'angle exact
# ⇒ de loin, l'entité perçoit « il y a un bosquet là » et PAS « il lui reste des baies ». Un bosquet
# vidé et un bosquet plein renvoient le même vecteur. Seule la mémoire de ce qu'on a mangé, et quand,
# les sépare — c'est la condition formelle du POMDP, pas un réglage.
# ⚠️ Rien ici ne ment sur l'état : on ne masque aucune donnée, on place l'information décisive à une
# échelle que le capteur ne résout pas. Coder « si distance > X alors cacher le stock » serait un
# échafaudage ; ceci n'en est pas un.
var _patch_count := 0             # 0 = OFF (perpetual field inchangé)
# DEUX CONTRAINTES, dans cet ordre d'importance :
#  1. les baies doivent tomber HORS du buisson-marqueur (couronne, cf _patch_berry_pos), sinon il les
#     ENGLOBE et la rétine ne voit jamais que du vert. C'est la cause RÉELLE, mesurée : à 0,6 m de
#     rayon, 87 % des baies étaient dedans -> 0 % de localisation, l'entité mourait aveugle à côté
#     de sa nourriture, 3/5 vies au plancher de famine.
#  2. le rayon externe doit rester < eat_radius (1.0 m) : le slot est un soft-argmax sur les rayons
#     (slot_head.py:138), il rend le BARYCENTRE de la couronne — arriver là doit capturer les baies.
# ⚠️ HYPOTHÈSE RÉFUTÉE, gardée pour mémoire : j'ai d'abord attribué l'échec au seul barycentre et
# resserré 1,2 -> 0,6 m. Ça n'a rien corrigé et a CAUSÉ la cécité totale. Le barycentre était un
# facteur secondaire ; l'occlusion par le marqueur était la cause.
var _patch_radius := 0.95         # rayon EXTERNE de la couronne de baies (< eat_radius 1.0)
var _patch_spacing_max := 0.0     # 0 = pas de borne haute (comportement d'origine)
var _patch_spacing := 9.0         # distance mini entre deux centres (traversée = 818 ticks = 41 pts d'énergie)
var _regrow_ticks := 2000         # une baie repousse SUR PLACE après ce délai
var _patch_centres: Array[Vector3] = []
var _alive: Array[bool] = []      # une baie consommée devient invisible et non-consommable
var _regrow_at: Array[int] = []   # tick de vie auquel elle réapparaît (-1 = vivante)
# PÉRISSABLE (levier CONSÉQUENCE, 2026-07-23) : une baie vivante non mangée PERISH ticks après sa
# naissance disparaît (comme cueillie mais SANS énergie) et repousse plus tard. Rend la
# récupérabilité COÛTEUSE : perdre du temps sur un mauvais choix fait périr la baie avant le retour.
# Opt-in SYLVAN_<PREFIX>_PERISH (0 = OFF, bit-identique). GRATUIT côté WM (règle de monde, la baie
# reste perçue par la rétine comme avant tant qu'elle est là).
var _perish_ticks := 0
var _ripe_cue := false          # indice de MATURITÉ VISIBLE (luminosité du buisson) — opt-in, OFF = bit-identique
var _ripe_decay := 0.0          # la maturité BAISSE la valeur nutritive : 0 = OFF ; 0.75 = une baie
                                # sur le point de se relocaliser ne rend plus que 25 % de son énergie.
# PROIE (2026-07-24) : la nourriture SE DÉPLACE au lieu d'attendre. Spéc. issue du test gratuit
# `diagnostics/diag_prey_interception.py` : le levier n'existe QUE si la proie (a) a du mouvement
# TRANSVERSAL et (b) va à >= 0,9x la vitesse de l'agent (0,011 m/tick mesuré).
# ⚠️ ELLE NE FUIT PAS. Une proie qui fuit converge vers une trajectoire RADIALE, contre laquelle
# l'angle d'avance est nul PAR CONSTRUCTION : poursuite et interception coïncident, gain nul (mesuré).
# Elle VAQUE : direction quasi constante + dérive lente, réflexion aux bords de l'arène.
var _prey_speed := 0.0          # m/tick ; 0 = OFF, bit-identique
var _prey_dir: Array[Vector3] = []
var _prey_turn := 0.01          # rad/tick de dérive (persistance : garde le transversal)
var _prey_travel := 0.0         # distance cumulée parcourue par la proie (PREUVE de ce qui est servi)
# TYPES ARBITRAIRES (2026-07-24). Chaque proie a un TYPE visible (teinte) dont la valeur nutritive
# est ARBITRAIRE : rien dans la physique perceptible ne la prédit, il faut en avoir mangé une.
# C'est la seule condition MESURÉE (diag_arbitrary_headroom.py) où un critique devient NÉCESSAIRE :
# formule ajustée 49,5 % de la marge oracle contre 69,7 % pour un appris — aucune formule ne peut
# contenir une table de correspondance arbitraire.
# ⚠️ TYPES CODÉS EN LUMINOSITÉ, PAS EN TEINTE — décidé sur MESURE (2026-07-24). Une première palette
# variait la TEINTE dans le cône bouffe : le type y était lisible à 82,9 % depuis la RÉTINE mais
# seulement 29,5 % après l'ENCODEUR du WM et 27,3 % dans le latent, contre 44,2 % de majorité — donc
# l'encodeur DÉTRUIT la teinte (il a été entraîné sur un monde à une seule couleur de nourriture,
# cette variation lui est hors-distribution). La LUMINOSITÉ, elle, SURVIT : l'indice de maturité est
# lisible à R² 0,65. On encode donc le type dans l'amplitude, à direction RGB CONSTANTE.
# Conséquence prouvée : cos(rouge)=0,928 et cos(bleu)=0,206 sont IDENTIQUES pour les 4 types, et
# l'affinité du slot est un cosinus -> localisation rigoureusement inchangée (invariance déjà mesurée
# à 0,00000000 m). Écart RGB minimal 0,175 = distinguables dans la rétine.
const TYPE_COLORS := [Color(0.900, 0.300, 0.200), Color(0.648, 0.216, 0.144),
					  Color(0.450, 0.150, 0.100), Color(0.288, 0.096, 0.064)]
var _n_types := 0               # 0 = OFF, bit-identique
var _type_values: Array[float] = []   # multiplicateur de valeur nutritive PAR TYPE (arbitraire)
var _type_of: Array[int] = []
var _type_hues: Array[Color] = []     # palette SÉPARABLE opt-in (SYLVAN_<PREFIX>_TYPE_HUES) ; vide = TYPE_COLORS

# ── FLAQUES (2026-07-24, docs/design_foret_complete.md §2.12 + §2.12bis) — OPT-IN _PUDDLE_PERIOD ──
# L'eau comme flaques DISPERSÉES à disponibilité VARIABLE. La variation est la valeur d'apprentissage,
# pas la 2ᵉ pulsion (§2.12 : l'arbitrage faim/soif est déjà tranché par un coût analytique).
# 🚨 RÈGLE §2.12bis : l'incertitude doit être OBSERVABLE et GRADUELLE, jamais instantanée et cachée.
# Le perish RELOCALISE (saut aléatoire) = MAUVAIS format (le WM déterministe ne peut pas l'anticiper,
# anomalie A4). Une flaque, elle, RÉTRÉCIT en douceur : son niveau suit un cosinus surélevé, sa taille
# VISUELLE ET son empreinte RÉTINE rétrécissent ENSEMBLE (le mesh porte l'Area de perception → jamais
# de mensonge visuel, §2.1), et les flaques sont DÉSYNCHRONISÉES (au même instant, certaines pleines,
# d'autres sèches -> il y a un CHOIX). Boire est gaté : une flaque trop sèche ne désaltère pas.
var _puddle_period := 0.0             # ticks d'un cycle sec->plein->sec ; 0 = OFF, bit-identique
var _puddle_floor := 0.15             # taille mini (flaque presque sèche, encore un peu visible)
var _puddle_drink := 0.4              # niveau mini pour pouvoir boire (en dessous = trop sec)
var _puddle_lvl: Array[float] = []
var _pud_prev := PackedFloat32Array()   # §6bis : accumulateurs MESURÉS sur l'épisode
var _pud_max_step := 0.0
var _pud_min := 1.0
var _pud_max := 0.0
var _pud_desync_sum := 0.0
var _pud_ticks := 0
var _born_at: Array[int] = []     # tick de vie où la baie (re)devient vivante
var _patch_meshes: Array[MeshInstance3D] = []
var _patch_areas: Array = []
# Vert clair, CHOISI PAR RECHERCHE NUMÉRIQUE contre les requêtes RÉELLES des deux WM vivants, pas
# à l'œil. Le buisson ne doit déclencher AUCUN slot : c'est un repère perceptible, pas une ressource.
# Marges mesurées (seuil − cosinus, toutes doivent être > 0) :
#   wm_objcentric_kin   (requêtes canaux purs, seuil 0.55) : rouge +0.142 · bleu +0.099
#   wm_objcentric_kin_typed (requêtes = couleurs rendues mesurées, seuils 0.808/0.859/0.920)
#                                            : bouffe +0.024 · eau +0.025 · danger +0.023
# ⚠️ La teinte « naturelle » (0.20,0.30,0.20) était sûre sur le WM vivant (+0.065) mais déclenchait
# le slot BOUFFE sur le WM typé (marge −0.032) : l'entité aurait essayé de MANGER les buissons.
# Mesuré avant de lancer quoi que ce soit — c'est ce qui a évité un A/B gaspillé.
# Compromis accepté : 9.8° seulement de l'arbre (0.13,0.35,0.13). Sans conséquence dans la boucle
# vivante (le buisson n'est pas une ressource, l'arbre est un obstacle détecté par COLLISION, pas
# par couleur), mais à re-mesurer si un jour une tête apprend à les distinguer par la teinte.
# ⚠️ Marges du WM typé mesurées sur l'ANCIENNE palette : ajouter un type d'objet au monde impose
# de les re-mesurer (build_typed_slots.py — une mesure, pas un ré-entraînement).
const PATCH_BUSH_COLOR := Color(0.47, 0.93, 0.53)
const PATCH_BUSH_R := 0.55        # 1,1 m de large = 16° à 4 m : repere fiable, et assez petit
                                  # pour que la couronne de baies [0.60, 0.95] reste DEHORS (visible)

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
	if i >= _areas.size():
		return
	if _n_types <= 0 and _appearance_var <= 0.0:
		return
	# Le TYPE fixe la teinte (elle DOIT rester lisible : c'est le seul indice de la valeur).
	var c: Color = _jitter(_albedo)
	if _n_types > 0 and i < _type_of.size():
		# Palette séparable si fournie (SYLVAN_<PREFIX>_TYPE_HUES), sinon les TYPE_COLORS historiques.
		if not _type_hues.is_empty():
			c = _type_hues[_type_of[i] % _type_hues.size()]
		else:
			c = TYPE_COLORS[_type_of[i] % TYPE_COLORS.size()]
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
	# CALIBRAGE DE VIE (2026-07-21) : `energy_per_food` 40 est un reglage de COLLECTE
	# ("smaller meals -> eat OFTEN -> many eat-events for the WM"), pas un reglage de VIE —
	# meme erreur que le drain 0.15 corrige en juin, de l'autre cote du bilan.
	# ARITHMETIQUE MESUREE : trajet vers une ressource a d metres = 100*d pas (vitesse 0.0100
	# m/tick MESUREE) -> coute 5*d a CHAQUE jauge (drain 0.05). Sur 2 cycles alternes, le bilan
	# net d'une jauge = restore - 10*d -> equilibre a d = restore/10. Avec restore=40 :
	# equilibre a 4 m, or les ressources spawnent 2-8 m (mediane ~5) -> bilan NEGATIF/nul ->
	# marche aleatoire a derive nulle -> mort certaine, survie dominee par la VARIANCE (mesure :
	# budget/cycle -1.9..+0.2, 46-51 % de cycles gagnants). Aucune competence ne peut s'y voir.
	# Override opt-in ; defaut 40 INCHANGE (collecte WM et resultats passes intacts).
	var _epf_env := OS.get_environment("SYLVAN_%s_ENERGY_PER" % _prefix)
	if _epf_env != "":
		energy_per_food = maxf(1.0, float(_epf_env))
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
		# BUISSON NEUTRE (chantier attribution de crédit) — opt-in, FOOD only.
		if OS.get_environment("SYLVAN_FOOD_BUSH") == "1":
			_bush_enabled = true
		var _bhue_env := OS.get_environment("SYLVAN_FOOD_BUSH_HUE")
		if _bhue_env != "":
			_bush_hue = fposmod(float(_bhue_env), 1.0)
		var _bp_env := OS.get_environment("SYLVAN_FOOD_BUSH_P")
		if _bp_env != "":
			_bush_p = clampf(float(_bp_env), 0.0, 1.0)
		var _balone_env := OS.get_environment("SYLVAN_FOOD_BUSH_ALONE")
		if _balone_env != "":
			_bush_alone = maxi(0, int(_balone_env))
	if _material == null:
		_material = StandardMaterial3D.new()
		_material.albedo_color = _albedo
		_material.emission_enabled = true
		_material.emission = _emission
	_read_patch_env()
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


func _read_patch_env() -> void:
	var pc := OS.get_environment("SYLVAN_%s_PATCHES" % _prefix)
	if pc != "":
		_patch_count = maxi(0, int(pc))
	var pr := OS.get_environment("SYLVAN_%s_PATCH_RADIUS" % _prefix)
	if pr != "":
		_patch_radius = maxf(0.1, float(pr))
	var ps := OS.get_environment("SYLVAN_%s_PATCH_SPACING" % _prefix)
	if ps != "":
		_patch_spacing = maxf(0.0, float(ps))
	var pm := OS.get_environment("SYLVAN_%s_PATCH_SPACING_MAX" % _prefix)
	if pm != "":
		_patch_spacing_max = maxf(0.0, float(pm))
	var rg := OS.get_environment("SYLVAN_%s_REGROW" % _prefix)
	if rg != "":
		_regrow_ticks = maxi(1, int(rg))
	var nt := OS.get_environment("SYLVAN_%s_TYPES" % _prefix)
	if nt != "":
		_n_types = clampi(int(nt), 0, TYPE_COLORS.size())
	# PALETTE DE TYPES SÉPARABLE — opt-in SYLVAN_<PREFIX>_TYPE_HUES="r,g,b;r,g,b;..." (2026-07-24).
	# 🚨 POURQUOI. Les TYPE_COLORS gelées sont des MULTIPLES SCALAIRES d'une même direction (mesuré
	# diag_foret_g5_palette.py : cosinus mutuel 1,0000, sonde directionnelle 26% ≈ hasard). La
	# perception NORMALISE la couleur (cosinus), donc elle ne voit que la DIRECTION : des teintes qui
	# ne diffèrent que par la luminosité sont, pour elle, IDENTIQUES → c'est la racine des 29,5 %
	# (verrou A1). Cet override sert une palette ÉTALÉE EN DIRECTION dans le cône bouffe, validée PASS
	# par la même sonde (cos rouge > 0,55, hors eau, écart 19-33°, sonde 98%). Défaut vide = les
	# TYPE_COLORS historiques, bit-identique. On NE mute PAS la constante gelée (repro historique).
	_type_hues.clear()
	var th := OS.get_environment("SYLVAN_%s_TYPE_HUES" % _prefix)
	if th != "":
		for grp in th.split(";"):
			var p := grp.split(",")
			if p.size() == 3:
				_type_hues.append(Color(float(p[0]), float(p[1]), float(p[2])))
	var tv := OS.get_environment("SYLVAN_%s_TYPE_VALUES" % _prefix)
	_type_values.clear()
	if tv != "":
		for part in tv.split(","):
			_type_values.append(maxf(0.0, float(part)))
	while _type_values.size() < _n_types:
		_type_values.append(1.0)
	var pspd := OS.get_environment("SYLVAN_%s_PREY_SPEED" % _prefix)
	if pspd != "":
		_prey_speed = maxf(0.0, float(pspd))
	# FLAQUES (§2.12) : cycle sec/plein. Défaut 0 = OFF, bit-identique.
	var pp := OS.get_environment("SYLVAN_%s_PUDDLE_PERIOD" % _prefix)
	if pp != "":
		_puddle_period = maxf(0.0, float(pp))
	var pfl := OS.get_environment("SYLVAN_%s_PUDDLE_FLOOR" % _prefix)
	if pfl != "":
		_puddle_floor = clampf(float(pfl), 0.0, 1.0)
	var pdk := OS.get_environment("SYLVAN_%s_PUDDLE_DRINK" % _prefix)
	if pdk != "":
		_puddle_drink = clampf(float(pdk), 0.0, 1.0)
	var pt := OS.get_environment("SYLVAN_%s_PREY_TURN" % _prefix)
	if pt != "":
		_prey_turn = maxf(0.0, float(pt))
	var rd2 := OS.get_environment("SYLVAN_%s_RIPE_DECAY" % _prefix)
	if rd2 != "":
		_ripe_decay = clampf(float(rd2), 0.0, 1.0)
	var rc := OS.get_environment("SYLVAN_%s_RIPE_CUE" % _prefix)
	if rc != "":
		_ripe_cue = rc != "0"
	var pe := OS.get_environment("SYLVAN_%s_PERISH" % _prefix)
	if pe != "":
		_perish_ticks = maxi(0, int(pe))


func _sample_patch_centres() -> void:
	# Centres tirés UNIFORMÉMENT EN AIRE (r = √u), pas linéairement en r : le code historique
	# échantillonne `randf_range(min, max)`, ce qui concentre la densité en 1/r vers le centre.
	# Espacement mini imposé par rejet — sinon deux bosquets fusionnent et le choix disparaît.
	_patch_centres.clear()
	for _i in range(_patch_count):
		for _try in range(80):
			var ang := _rng.randf_range(0.0, TAU)
			var u := _rng.randf()
			var r := sqrt(min_radius * min_radius + u * (spawn_radius * spawn_radius - min_radius * min_radius))
			var c := Vector3(cos(ang) * r, food_y, sin(ang) * r)
			var ok := true
			var nearest := INF
			for other in _patch_centres:
				var d := c.distance_to(other)
				nearest = minf(nearest, d)
				if d < _patch_spacing:
					ok = false
					break
			# BORNE HAUTE : le voisin le plus proche doit rester à portée de traversée. Sans elle, le
			# rejet écarte les bosquets au maximum de l'anneau — mesuré 12,53 m pour 9,0 demandés, ce
			# qui fait passer le prix d'une traversée de 41 à 57 points d'énergie et change la nature
			# du problème. On borne la STRUCTURE du monde ; on ne vise pas l'agent.
			if ok and _patch_spacing_max > 0.0 and not _patch_centres.is_empty() and nearest > _patch_spacing_max:
				ok = false
			if ok:
				_patch_centres.append(c)
				break
	if _patch_centres.is_empty():                    # espacement infaisable : ne pas mentir, le dire
		push_warning("[patch] aucun centre placé (spacing=%.1f trop grand pour l'anneau)" % _patch_spacing)


func _patch_berry_pos(i: int, patch_idx: int = -1) -> Vector3:
	# COURONNE, pas disque : les baies doivent tomber HORS du buisson-marqueur, sinon celui-ci les
	# ENGLOBE et le raycast de la rétine frappe le buisson en premier — la baie devient invisible.
	# Mesuré : à rayon 1,2 m, 22 % des baies étaient englobées -> 73 % localisées ; à 0,6 m, 87 %
	# englobées -> 0 % localisées, l'entité mourait aveugle à côté de sa nourriture. La géométrie
	# prédit la mesure au point près, c'est bien le buisson qui masquait.
	# Borne haute : rayon < eat_radius (1.0 m) pour qu'arriver au centre capture toute la couronne.
	# patch_idx >= 0 force le bosquet (relocalisation périssable) ; défaut = bosquet attitré i%N.
	var pi := (i % _patch_centres.size()) if patch_idx < 0 else (patch_idx % _patch_centres.size())
	var c: Vector3 = _patch_centres[pi]
	var ang := _rng.randf_range(0.0, TAU)
	var inner := PATCH_BUSH_R + 0.05
	var outer := maxf(inner + 0.05, _patch_radius)
	var u := _rng.randf()
	var r := sqrt(inner * inner + u * (outer * outer - inner * inner))   # uniforme en aire
	return Vector3(c.x + cos(ang) * r, food_y, c.z + sin(ang) * r)


func _update_ripeness_cue() -> void:
	# MATURITÉ VISIBLE (2026-07-24) : la LUMINOSITÉ du buisson-marqueur encode l'âge de SA baie —
	# vif = fraîche, sombre = sur le point de se relocaliser (ou bosquet vide).
	#
	# POURQUOI LE BUISSON ET PAS LA BAIE. Le slot pondère ses rayons par une saillance
	# `max(RGB) − min(RGB)` : teinter la BAIE ferait mécaniquement préférer la plus fraîche au slot,
	# donc la PERCEPTION arbitrerait à la place du critique et `-min_dist` en profiterait aussi —
	# raccourci câblé, interdit (§2/§3). Le buisson, lui, est à cos 0,40 du rouge et 0,45 du bleu,
	# donc SOUS le seuil 0,55 : ses rayons sont exclus EN DUR des deux slots. Et l'affinité est un
	# COSINUS, invariant par changement d'échelle — mesuré : cos reste 0,402/0,453 de x1,0 à x0,2.
	# ⇒ l'indice est prouvablement INVISIBLE aux slots (position inchangée) et VISIBLE dans la
	# rétine brute : seul un critique lisant la scène (le latent) peut s'en servir, jamais -min_dist.
	if not _ripe_cue or _perish_ticks <= 0 or _patch_centres.is_empty():
		return
	for k in range(_patch_meshes.size()):
		if k >= _patch_centres.size():
			continue
		var age := 1.0                                  # 1 = aucune baie vivante ici -> éteint
		for i in range(_positions.size()):
			if _alive[i] and _nearest_patch(i) == k:
				var a := float(_life_tick - _born_at[i]) / float(_perish_ticks)
				age = minf(age, clampf(a, 0.0, 1.0))
		var s := 1.0 - 0.8 * age                        # x1,0 (fraîche) -> x0,2 (imminente)
		var c := Color(PATCH_BUSH_COLOR.r * s, PATCH_BUSH_COLOR.g * s, PATCH_BUSH_COLOR.b * s)
		var mat: StandardMaterial3D = _patch_meshes[k].material_override
		if mat != null:
			mat.albedo_color = c
			mat.emission = c * 0.3
		if k < _patch_areas.size():
			_patch_areas[k].set_meta("retina_color", c)   # la rétine lit CE champ, pas le mesh


func _nearest_patch(i: int) -> int:
	# Bosquet le plus proche de la baie i (pour la relocalisation périssable : sauter AILLEURS).
	var best := 0
	var bd := INF
	for k in range(_patch_centres.size()):
		var d: float = _positions[i].distance_to(_patch_centres[k])
		if d < bd:
			bd = d
			best = k
	return best


func reset(_episode_index: int = 0) -> void:
	_ensure_built()
	# §6bis — flush des flaques de l'épisode ÉCOULÉ : on rapporte ce qui a été SERVI, mesuré par tick
	# (gradualité = plus gros pas de niveau ; amplitude = min..max ; désync = écart moyen entre flaques
	# = preuve qu'il y avait un CHOIX). Émis AVANT de réinitialiser, sauté au tout 1er reset.
	if _puddle_period > 0.0 and _pud_ticks > 0:
		print("[puddle] %s : cycle %d ticks | niveau MESURE %.2f..%.2f | plus gros pas/tick %.4f (graduel) | desync moyen %.3f (choix) | boire si >= %.2f"
			% [_prefix, int(_puddle_period), _pud_min, _pud_max, _pud_max_step,
			   _pud_desync_sum / float(_pud_ticks), _puddle_drink])
	_pud_max_step = 0.0
	_pud_min = 1.0
	_pud_max = 0.0
	_pud_desync_sum = 0.0
	_pud_ticks = 0
	_puddle_lvl.clear()
	_pud_prev = PackedFloat32Array()
	consumed_this_episode = 0
	_life_tick = 0
	_swapped = false
	_positions.clear()
	_alive.clear()
	_regrow_at.clear()
	_born_at.clear()
	_prey_dir.clear()
	_type_of.clear()
	if _patch_count > 0:
		_sample_patch_centres()
	for i in range(food_count):
		var p := _random_pos() if _patch_centres.is_empty() else _patch_berry_pos(i)
		_positions.append(p)
		_alive.append(true)
		_regrow_at.append(-1)
		# Périssable : stagger les naissances sur [-perish, 0] pour que les baies ne périssent PAS
		# toutes au même tick (sinon relocalisation synchronisée). OFF -> 0, bit-identique.
		_born_at.append(-(_rng.randi() % _perish_ticks) if _perish_ticks > 0 else 0)
		var _pa := _rng.randf_range(0.0, TAU)
		_prey_dir.append(Vector3(cos(_pa), 0.0, sin(_pa)))
		_type_of.append(_rng.randi() % _n_types if _n_types > 0 else 0)
		_meshes[i].global_position = p
		_meshes[i].visible = true
		_apply_appearance(i)
	if _patch_count > 0:
		_place_patch_bushes()
		_log_patches()
	if _bush_enabled:
		_place_bushes()


func _log_patches() -> void:
	# PROUVER ce qui est servi (règle de méthode : trois fois un réglage a semblé appliqué sans
	# l'être). On rapporte le nombre de centres RÉELLEMENT placés — pas celui demandé — et
	# l'espacement minimal MESURÉ, pas déclaré.
	var gap := INF
	for i in range(_patch_centres.size()):
		for j in range(i + 1, _patch_centres.size()):
			gap = minf(gap, _patch_centres[i].distance_to(_patch_centres[j]))
	var gap_s := "n/a" if _patch_centres.size() < 2 else "%.2f m" % gap
	print("[patch] %s : %d/%d bosquets placés | %d baies | espacement voisin MESURÉ %s (demandé %.1f-%.1f) | repousse %d ticks | buisson r=%.2f couleur=(%.2f,%.2f,%.2f)" % [
		_prefix, _patch_centres.size(), _patch_count, food_count, gap_s, _patch_spacing, _patch_spacing_max,
		_regrow_ticks, PATCH_BUSH_R, PATCH_BUSH_COLOR.r, PATCH_BUSH_COLOR.g, PATCH_BUSH_COLOR.b])
	# §6bis — prouver la PALETTE de types réellement servie (la source des couleurs, pas juste leur nb).
	if _n_types > 0:
		var use_hues := not _type_hues.is_empty()
		var src := ("SYLVAN_%s_TYPE_HUES (separable)" % _prefix) if use_hues else "TYPE_COLORS (historique, multiples scalaires)"
		var s := ""
		for t in range(_n_types):
			var col: Color = _type_hues[t % _type_hues.size()] if use_hues else TYPE_COLORS[t % TYPE_COLORS.size()]
			s += " (%.2f,%.2f,%.2f)" % [col.r, col.g, col.b]
		print("[patch] %s : %d types SERVIS depuis %s |%s" % [_prefix, _n_types, src, s])


func _place_patch_bushes() -> void:
	# Un buisson-marqueur LARGE par bosquet, toujours visible même quand le bosquet est vidé.
	# C'est lui qui porte l'aliasing : il dit « il y a un bosquet ici » et rien sur le stock.
	if _patch_meshes.is_empty():
		for _i in range(_patch_count):
			var bm := MeshInstance3D.new()
			var bs := SphereMesh.new()
			bs.radius = PATCH_BUSH_R
			bs.height = PATCH_BUSH_R * 2.0
			bm.mesh = bs
			var mat := StandardMaterial3D.new()
			mat.albedo_color = PATCH_BUSH_COLOR
			bm.material_override = mat
			add_child(bm)
			var ba := Area3D.new()
			ba.collision_layer = 1 << 7            # couche 8 = perceptible-rétine (comme les baies)
			ba.collision_mask = 0                  # jamais bloquant : le corps traverse le bosquet
			ba.set_meta("retina_color", PATCH_BUSH_COLOR)
			var bcs := CollisionShape3D.new()
			var bsh := SphereShape3D.new()
			bsh.radius = PATCH_BUSH_R
			bcs.shape = bsh
			ba.add_child(bcs)
			bm.add_child(ba)
			_patch_meshes.append(bm)
			_patch_areas.append(ba)
	for i in range(_patch_meshes.size()):
		if i < _patch_centres.size():
			_patch_meshes[i].global_position = Vector3(_patch_centres[i].x, PATCH_BUSH_R, _patch_centres[i].z)
			_patch_meshes[i].visible = true
		else:
			_patch_meshes[i].visible = false


func _build_bushes() -> void:
	# Construit les meshes/areas BUISSON une fois (perceptibles layer 8, NON consommables — jamais
	# dans _positions). Idempotent. OFF (_bush_enabled=false) → no-op, bit-identique.
	if not _bush_enabled or not _bush_meshes.is_empty():
		return
	var bc := Color.from_hsv(_bush_hue, 0.7, 0.5)
	_bush_material = StandardMaterial3D.new()
	_bush_material.albedo_color = bc
	_bush_material.emission_enabled = true
	_bush_material.emission = bc * 0.3
	for i in range(food_count + _bush_alone):      # 1 co-localisé par pastille + _bush_alone dispersés
		var bm := MeshInstance3D.new()
		var bs := SphereMesh.new()
		bs.radius = 0.20                           # petit (≈ baie) → n'occulte pas la baie voisine
		bs.height = 0.40
		bm.mesh = bs
		bm.material_override = _bush_material
		add_child(bm)
		var ba := Area3D.new()
		ba.collision_layer = 1 << 7                # layer 8 = perceptible-rétine (comme la baie)
		ba.collision_mask = 0
		ba.set_meta("retina_color", bc)
		var bcs := CollisionShape3D.new()
		var bsh := SphereShape3D.new()
		bsh.radius = 0.28
		bcs.shape = bsh
		ba.add_child(bcs)
		bm.add_child(ba)
		bm.visible = false
		_bush_meshes.append(bm)
		_bush_areas.append(ba)


func _place_bushes() -> void:
	# Décorrélation (identifiabilité) : chaque baie DANS un buisson avec prob _bush_p (co-occurrence,
	# buisson offset ~0.5 m → baie et buisson sur des rayons voisins DISTINCTS), sinon baie SEULE ;
	# + _bush_alone buissons DISPERSÉS seuls. Positions ré-échantillonnées à chaque appel (reset +
	# respawn) via _rng (déterministe). Le buisson ne change RIEN à la consommation (hors _positions).
	_build_bushes()
	for bm in _bush_meshes:
		bm.visible = false
	var bi := 0
	for i in range(_positions.size()):
		if bi >= _bush_meshes.size():
			break
		if _rng.randf() < _bush_p:                 # baie DANS un buisson
			var p: Vector3 = _positions[i]
			var ox := _rng.randf_range(-0.30, 0.30)   # serré → buisson fiablement co-perçu avec la baie
			var oz := _rng.randf_range(-0.30, 0.30)
			_bush_meshes[bi].global_position = Vector3(p.x + ox, food_y, p.z + oz)
			_bush_meshes[bi].visible = true
			bi += 1
		# sinon : baie SEULE (pas de buisson) → décorrélation
	var left := _bush_alone
	while bi < _bush_meshes.size() and left > 0:   # buissons DISPERSÉS (sans baie)
		_bush_meshes[bi].global_position = _random_pos()
		_bush_meshes[bi].visible = true
		bi += 1
		left -= 1


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
	_tick_regrowth()
	# Régime eat-riche : ne pas consommer tant qu'on n'est pas assez affamé (seuil eat_hunger_max).
	# energy_frac = énergie/max. Défaut 1.0 + seuil 1.0 → mange toujours (inchangé).
	if energy_frac > eat_hunger_max:
		return 0.0
	var restored := 0.0
	var ground := Vector3(agent_pos.x, food_y, agent_pos.z)
	for i in range(_positions.size()):
		if _patch_count > 0 and not _alive[i]:
			continue                     # baie déjà cueillie : elle repoussera SUR PLACE
		if ground.distance_to(_positions[i]) <= eat_radius:
			# FLAQUE TROP SÈCHE (§2.12) : disponibilité variable. Le niveau est OBSERVABLE (la flaque a
			# visiblement rétréci) donc le gate n'est pas caché — l'entité peut apprendre à ne pas
			# viser une flaque sèche. OFF (_puddle_period=0) : jamais bloqué, bit-identique.
			if _puddle_period > 0.0 and i < _puddle_lvl.size() and _puddle_lvl[i] < _puddle_drink:
				continue
			# MATURITÉ -> VALEUR NUTRITIVE (2026-07-24). Le seul signal du monde qui soit à la fois
			# PERCEPTIBLE (la luminosité du buisson l'annonce), NON-GÉOMÉTRIQUE (indépendant de la
			# distance) et PRÉDICTIBLE (fonction déterministe de l'âge, contrairement au saut
			# aléatoire de la relocalisation, que le rêve ne peut pas anticiper).
			# ⚠️ L'indice VU et la valeur OBTENUE dérivent du MÊME âge : le monde ne ment pas.
			# OFF (_ripe_decay = 0) -> facteur 1.0 exactement, bit-identique.
			var _age := 0.0
			if _perish_ticks > 0:
				_age = clampf(float(_life_tick - _born_at[i]) / float(_perish_ticks), 0.0, 1.0)
			var _tmul := 1.0
			if _n_types > 0 and i < _type_of.size() and _type_of[i] < _type_values.size():
				_tmul = _type_values[_type_of[i]]     # ARBITRAIRE : rien ne le prédit, il faut goûter
			restored += energy_per_food * (1.0 - _ripe_decay * _age) * _tmul
			consumed_this_episode += 1
			if _patch_count > 0:
				# MODE BOSQUETS : la baie disparaît LÀ OÙ ELLE ÉTAIT et repousse sur une horloge.
				# Le bosquet s'épuise donc réellement, et son buisson-marqueur reste visible —
				# c'est ce couple qui rend « je l'ai vidé » impossible à lire et nécessaire à retenir.
				_alive[i] = false
				_regrow_at[i] = _life_tick + _regrow_ticks
				_meshes[i].visible = false
			else:
				# PERPETUAL FIELD (historique) : respawn autour de l'AGENT → la ressource le suit,
				# donc elle est toujours à 2-4,5 m et aucune mémoire ne sert. Conservé par défaut.
				_positions[i] = _respawn_near(agent_pos)
				_meshes[i].global_position = _positions[i]
				_apply_appearance(i)
	if _bush_enabled and restored > 0.0:
		_place_bushes()          # ré-échantillonne la co-localisation baie/buisson après un respawn
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


func _tick_regrowth() -> void:
	_update_puddles()   # FLAQUES (§2.12) : rétrécissement graduel, indépendant du régime bosquets
	# Une baie cueillie réapparaît SUR SON BOSQUET après _regrow_ticks. Position ré-échantillonnée
	# dans le bosquet (une baie ne repousse pas exactement au même millimètre) — jamais autour de
	# l'agent. OFF (_patch_count == 0) : boucle jamais exécutée, bit-identique.
	if _patch_count <= 0:
		return
	# PÉRISSABLE = attaque de la RÉCUPÉRABILITÉ (levier conséquence, 2026-07-23). Une baie vivante
	# trop vieille NE DISPARAÎT PAS (ce monde n'a que 2 baies -> disparaître = famine, mesuré 0 repas)
	# : elle se RELOCALISE sur un AUTRE bosquet. Le compte de baies vivantes reste invariant (survie
	# préservée, §2), mais la baie que l'agent visait n'est plus là où il allait -> un choix trop lent
	# (hésitation, virage inutile) perd son trajet. La densité est constante ; seule la DÉCISION coûte.
	if _perish_ticks > 0 and not _patch_centres.is_empty():
		for i in range(_positions.size()):
			if _alive[i] and _life_tick - _born_at[i] >= _perish_ticks:
				var nb := _patch_centres.size()
				# saute vers un bosquet DIFFÉRENT de l'actuel (si >1 bosquet), sinon même bosquet
				var cur := _nearest_patch(i)
				var pk := cur if nb <= 1 else ((cur + 1 + (_rng.randi() % (nb - 1))) % nb)
				_born_at[i] = _life_tick
				_positions[i] = _patch_berry_pos(i, pk)
				_meshes[i].global_position = _positions[i]
				_apply_appearance(i)
	for i in range(_positions.size()):
		if _alive[i] or _regrow_at[i] < 0 or _life_tick < _regrow_at[i]:
			continue
		_alive[i] = true
		_regrow_at[i] = -1
		_born_at[i] = _life_tick
		if _n_types > 0 and i < _type_of.size():
			_type_of[i] = _rng.randi() % _n_types      # nouveau tirage : le type n'est pas figé à vie
		if not _patch_centres.is_empty():
			_positions[i] = _patch_berry_pos(i)
			_meshes[i].global_position = _positions[i]
		_meshes[i].visible = true
		_apply_appearance(i)
	_move_prey()
	_update_ripeness_cue()


# Niveau d'une flaque dans [_puddle_floor, 1], lisse dans le temps, DÉSYNCHRONISÉ entre flaques.
# Cosinus surélevé -> aucun saut ; la phase par flaque étale les cycles pour qu'à tout instant il y
# ait des pleines ET des sèches (donc un choix). Déterministe (fonction de _life_tick) -> rejeu OK.
func _puddle_level(i: int) -> float:
	if _puddle_period <= 0.0:
		return 1.0
	var n := maxi(1, _positions.size())
	var phase := float(i) / float(n)
	var frac := fposmod(float(_life_tick) / _puddle_period + phase, 1.0)
	var wet := 0.5 * (1.0 - cos(TAU * frac))          # [0,1] lisse, dérivée bornée
	return _puddle_floor + (1.0 - _puddle_floor) * wet


func _update_puddles() -> void:
	if _puddle_period <= 0.0:
		return
	var n := _positions.size()
	while _puddle_lvl.size() < n:
		_puddle_lvl.append(1.0)
	if _pud_prev.size() < n:
		_pud_prev.resize(n)
	var s := 0.0
	var s2 := 0.0
	for i in range(n):
		var lvl := _puddle_level(i)
		if _pud_ticks > 0:
			_pud_max_step = maxf(_pud_max_step, absf(lvl - _pud_prev[i]))   # §6bis : gradualité MESURÉE
		_pud_prev[i] = lvl
		_puddle_lvl[i] = lvl
		# UNE seule opération : scaler le MESH scale AUSSI l'Area de perception (enfant) → la flaque
		# rétrécit pour l'OWNER et pour l'ENTITÉ ensemble. Pas de mensonge visuel (§2.1).
		_meshes[i].scale = Vector3(lvl, lvl, lvl)
		_pud_min = minf(_pud_min, lvl)
		_pud_max = maxf(_pud_max, lvl)
		s += lvl
		s2 += lvl * lvl
	if n > 0:
		var mean := s / float(n)
		_pud_desync_sum += sqrt(maxf(0.0, s2 / float(n) - mean * mean))   # écart entre flaques = choix
	_pud_ticks += 1


func _move_prey() -> void:
	# La nourriture VAQUE : direction quasi constante (persistance -> mouvement TRANSVERSAL conservé,
	# ce qui est la condition mesurée du levier) + dérive lente, et réflexion aux bords de l'arène
	# pour qu'elle ne parte pas à l'infini. Elle IGNORE l'agent : ne pas fuir est délibéré (une fuite
	# rend la poursuite et l'interception mathématiquement identiques -> gain nul).
	if _prey_speed <= 0.0:
		return
	for i in range(_positions.size()):
		if i >= _prey_dir.size() or not _alive_or_field(i):
			continue
		var a := _rng.randf_range(-_prey_turn, _prey_turn)
		var d: Vector3 = _prey_dir[i]
		var nd := Vector3(cos(a) * d.x - sin(a) * d.z, 0.0, sin(a) * d.x + cos(a) * d.z)
		var p: Vector3 = _positions[i] + nd * _prey_speed
		var r := Vector2(p.x, p.z).length()
		if r > spawn_radius or r < min_radius:            # réflexion : rebrousse vers l'arène
			nd = -nd
			p = _positions[i] + nd * _prey_speed
		_prey_travel += _positions[i].distance_to(p)
		_prey_dir[i] = nd
		_positions[i] = p
		_meshes[i].global_position = p
	# PREUVE (règle de méthode : trois fois un réglage a semblé appliqué sans l'être). On rapporte la
	# vitesse RÉELLEMENT parcourue par la proie, pas celle demandée.
	if _life_tick == 500 and _positions.size() > 0:
		print("[prey] vitesse MESURÉE %.5f m/tick (demandée %.5f) sur %d baies, %d ticks" % [
			_prey_travel / float(_life_tick * _positions.size()), _prey_speed, _positions.size(), _life_tick])


func _alive_or_field(i: int) -> bool:
	# En mode bosquets une baie cueillie est invisible (elle ne doit pas bouger) ; en champ perpétuel
	# _alive n'est pas utilisé et toutes les pastilles sont vivantes.
	return _alive[i] if _patch_count > 0 else true


func alive_count() -> int:
	if _patch_count <= 0:
		return _positions.size()
	var n := 0
	for a in _alive:
		if a:
			n += 1
	return n


func get_positions() -> Array:
	# En mode bosquets, seules les baies VIVANTES sont des ressources : une baie cueillie ne doit
	# ni compter dans le radar, ni dans le keep-out. Hors mode bosquets, _alive est tout-vrai →
	# retourne exactement _positions, bit-identique.
	if _patch_count <= 0:
		return _positions
	var out: Array[Vector3] = []
	for i in range(_positions.size()):
		if _alive[i]:
			out.append(_positions[i])
	return out


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
