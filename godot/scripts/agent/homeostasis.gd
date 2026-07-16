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
