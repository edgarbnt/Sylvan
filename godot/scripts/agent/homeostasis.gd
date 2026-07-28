extends RefCounted
class_name Homeostasis

@export var max_energy := 100.0
@export var max_health := 100.0
# 2ᵉ PULSION (2026-06-18) — SOIF, parallèle à l'énergie. Symétrique pour un arbitrage PROPRE
# (les deux drainent au même rythme par défaut → la priorité émerge de l'URGENCE + la géométrie,
# pas d'un biais de drain). L'eau (water_manager) restaure la soif comme la bouffe restaure l'énergie.
@export var max_thirst := 100.0
@export var passive_thirst_drain := 0.15
var health_regen := 0.0      # MONDE v2 : regen lente de santé (SYLVAN_HEALTH_REGEN, défaut 0 = OFF)
var thirst_enabled := false  # la soif ne draine/tue QUE si l'eau existe (main.gd l'active selon SYLVAN_WATER_COUNT) —
                             # sinon un run bouffe-seule mourrait de soif sans pouvoir boire.
# PHASE C — the intrinsic DRIVE is now ON. Energy drains every step (metabolism); the
# agent dies at 0 (is_critical → episode done). Tuned so an agent that NEVER eats starves
# around step ~600, while eating food (restore_energy) extends survival → "j'ai faim → agir"
# becomes the binding pressure. This is the LeCun intrinsic cost: behaviour minimises the
# discomfort of low energy, not a hand-coded reward.
@export var passive_energy_drain := 0.15  # WM-DATA economy: faster hunger → eats MORE OFTEN (the WM needs
                                          # an agent that never eats starves ~830 steps → it gets TIME to
                                          # LEARN to navigate before dying. The fall=100% in forage4/5 was
                                          # STARVATION, not toppling (it never physically falls — owner
                                          # confirmed visually): the old economy made only DENSE food
                                          # survivable, so sparse food = unreachable before death = no
                                          # learning signal. Paired with energy_per_food 35→100.

var energy := max_energy
var health := max_health
var thirst := max_thirst


func reset_state() -> void:
	# PLAFOND DES JAUGES RÉGLABLE (2026-07-28) — mesuré : un repas sert 84 points sur une jauge de
	# 100, donc l'entité ne peut l'encaisser en entier que tombée à 16, un niveau qu'un planner qui
	# cherche à survivre ne s'autorise jamais. Sondé sur 3 densités : elle mange entre 73 et 51, et
	# n'encaisse que 21 à 32 points — 25 à 38 % du repas. La valeur servie était donc une valeur que
	# le monde ne pouvait pas délivrer, et tout le calibrage « 10 événements/vie » reposait dessus.
	# Agrandir le PLAFOND (sans toucher au départ) rend le repas encaissable sans rien adoucir : le
	# plancher de famine reste init/drain. Défaut 100 = comportement d'origine bit-identique.
	# Les deux jauges bougent ENSEMBLE : un plafond asymétrique ferait mourir une pulsion avant
	# l'autre sans que rien ne l'ait décidé (même raison que l'égalité des drains, ci-dessus).
	var _me := OS.get_environment("SYLVAN_MAX_ENERGY")
	if _me != "":
		max_energy = maxf(1.0, float(_me))
	var _mt := OS.get_environment("SYLVAN_MAX_THIRST")
	if _mt != "":
		max_thirst = maxf(1.0, float(_mt))
	energy = max_energy
	health = max_health
	thirst = max_thirst
	# Diag arbitrage (2ᵉ pulsion) : forcer les niveaux de départ pour créer un BESOIN contrôlé
	# (ex. soif critique + faim pleine) → tester quelle ressource l'agent priorise.
	var _ie := OS.get_environment("SYLVAN_INIT_ENERGY")
	if _ie != "":
		energy = clampf(float(_ie), 0.0, max_energy)
	var _it := OS.get_environment("SYLVAN_INIT_THIRST")
	if _it != "":
		thirst = clampf(float(_it), 0.0, max_thirst)
	# Métabolisme réglable (2026-06-18) : le drain 0.15 = réglage COLLECTE-DE-DONNÉES ("eat often"),
	# pas un métabolisme de VIE. La capacité naviguer+manger est prouvée (89%), donc ralentir le drain
	# corrige un paramètre, ne masque aucune lacune. Défaut inchangé.
	var _ed := OS.get_environment("SYLVAN_ENERGY_DRAIN")
	if _ed != "":
		passive_energy_drain = maxf(0.0, float(_ed))
	var _td := OS.get_environment("SYLVAN_THIRST_DRAIN")
	if _td != "":
		passive_thirst_drain = maxf(0.0, float(_td))
	# MONDE v2 (2026-07-16, décision owner) : RÉGÉNÉRATION lente de santé — la santé devient une
	# ÉCONOMIE cyclique (encaisser un sprint douloureux, récupérer, recommencer) au lieu d'un budget
	# à sens unique. ~10× plus lent que les dégâts hazard (0.05 vs 0.5/pas → un sprint de ~27 dégâts
	# se récupère en ~540 pas). Défaut 0 = OFF, corps inchangé.
	var _hr := OS.get_environment("SYLVAN_HEALTH_REGEN")
	if _hr != "":
		health_regen = maxf(0.0, float(_hr))


func apply_metabolism(effort_cost: float = 0.0) -> void:
	energy = maxf(0.0, energy - passive_energy_drain - effort_cost)
	if thirst_enabled:
		thirst = maxf(0.0, thirst - passive_thirst_drain)  # soif draine passivement (pas d'effort-cost : boire ≠ marcher)
	if effort_cost > 0.9:
		health = maxf(0.0, health - (effort_cost - 0.9) * 0.2)
	if health_regen > 0.0 and health > 0.0:
		health = minf(max_health, health + health_regen)   # regen lente (monde v2) ; un mort ne régénère pas


func spend_locomotion(cost: float) -> void:
	# ÉVENTAIL DE VITESSE (§2.13) — prélèvement SÉPARÉ de apply_metabolism, délibérément. Le
	# paramètre effort_cost de apply_metabolism abîme la SANTÉ au-dessus de 0.9 (héritage des pattes) :
	# y verser le coût de vitesse coupleraient silencieusement « sprinter » et « se blesser », deux
	# mécaniques distinctes dont l'une (la blessure) est un chantier DIFFÉRÉ (§6quinquies B, échec
	# P2-bis mesuré).
	#
	# 🚨 RÉPARTI SUR LES DEUX JAUGES (2026-07-26) — corrige un déséquilibre STRUCTUREL mesuré. Cette
	# fonction ne débitait que l'ÉNERGIE, alors que la calibration du monde traitait le coût comme
	# réparti sur les deux (D_total = énergie + soif). Conséquence mesurée au premier gate
	# closed-loop : au trot l'énergie se vidait en 250 ticks quand la soif tenait 750 — rapport 3,0x,
	# 6,6x au sprint — et l'entité mourait de FAIM 11 fois sur 12, la soif encore à 38-79.
	# Courir donne faim ET soif : le coût est distribué AU PRORATA des drains passifs, de sorte que
	# la somme prélevée vaut exactement `cost`. Les identités du preset (vx* = sqrt(D_total/k),
	# événements par vie) redeviennent VRAIES au lieu d'être rapiécées.
	# Mono-pulsion (soif désactivée) : tout va à l'énergie — sinon un monde sans soif encaisserait
	# silencieusement une locomotion moitié prix, et ses chiffres cesseraient d'être comparables.
	if cost <= 0.0:
		return
	if not thirst_enabled:
		energy = maxf(0.0, energy - cost)
		return
	var total := passive_energy_drain + passive_thirst_drain
	var share_e := 0.5 if total <= 0.0 else passive_energy_drain / total
	energy = maxf(0.0, energy - cost * share_e)
	thirst = maxf(0.0, thirst - cost * (1.0 - share_e))


func restore_energy(amount: float) -> void:
	# Eating food refills energy (capped at max). The positive side of the homeostatic
	# drive: metabolism drains, food restores — the gap is what the agent must learn to close.
	energy = minf(max_energy, energy + amount)


func restore_thirst(amount: float) -> void:
	# Drinking water refills thirst (capped at max). Symétrique à restore_energy.
	thirst = minf(max_thirst, thirst + amount)


func apply_damage(amount: float) -> void:
	health = maxf(0.0, health - amount)


func is_critical() -> bool:
	return energy <= 0.0 or (thirst_enabled and thirst <= 0.0) or health <= 0.0
