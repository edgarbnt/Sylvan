extends RefCounted
class_name Perception

# EGOCENTRIC FOOD RADAR — the first perception modality (the [V] slot of the blueprint).
# For each angular SECTOR around the agent's heading, report the proximity (1 - dist/range,
# in [0,1]) of the NEAREST food pellet in that sector; 0 if none within range. This is the
# resource-sensing equivalent of a horizontal raycast fan, computed analytically from the
# food positions (no colliders needed → cheap). It is EGOCENTRIC: sector 0 is centred on the
# agent's FORWARD (+z) direction, so the signal is heading-relative ("food on my left/right/
# ahead") — exactly what the policy needs to turn and walk toward food. Geometric raycasts
# (walls/terrain/danger) come later, when the world has obstacles to perceive.

const NUM_SECTORS := 12
const MAX_RANGE := 10.0

# RÉTINE (perception apprise) — raycast PHYSIQUE depth+couleur, REMPLACE à terme le radar-oracle.
# N rayons sur 360° depuis la tête ; chaque rayon renvoie [depth_norm, R, G, B] de l'objet le plus proche
# qu'il touche. Contrairement à food_radar (oracle : "bouffe = secteur 7"), la rétine ne pré-associe RIEN :
# l'agent doit APPRENDRE couleur→ressource et trianguler la position. Couleur lue via meta "retina_color"
# (tout objet perceptible la déclare → futur objet = poser collider+meta, rien à re-coder ici). Les objets
# perceptibles vivent sur la COUCHE 8 (dédiée) → la requête ne touche jamais l'agent ni la physique du gait.
const RETINA_RAYS := 36
const RETINA_LAYER_MASK := 1 << 7   # couche de collision 8 = "perceptible-rétine"


# Renvoie RETINA_RAYS*4 floats = [depth_norm, R, G, B] par rayon. Rayon 0 = droit devant (forward),
# bearing croissant vers la DROITE (même convention de signe que food_radar). Miss → [1.0, 0,0,0].
# space_state = agent.get_world_3d().direct_space_state (état physique courant).
static func retina(space_state: PhysicsDirectSpaceState3D, origin: Vector3, forward: Vector3, num_rays: int = RETINA_RAYS, max_range: float = MAX_RANGE) -> Array:
	var out: Array = []
	out.resize(num_rays * 4)
	out.fill(0.0)
	var fwd := Vector3(forward.x, 0.0, forward.z)
	if space_state == null or fwd.length() < 0.001:
		for k in range(num_rays):
			out[k * 4] = 1.0  # tout en miss (depth max, couleur 0)
		return out
	fwd = fwd.normalized()
	var right := Vector3(fwd.z, 0.0, -fwd.x)  # forward tourné -90° autour de +y → bearing signé
	var ray_step := TAU / float(num_rays)
	for k in range(num_rays):
		var b := ray_step * float(k)  # rayon 0 = forward
		var dir := (fwd * cos(b) + right * sin(b)).normalized()
		var q := PhysicsRayQueryParameters3D.create(origin, origin + dir * max_range)
		q.collision_mask = RETINA_LAYER_MASK
		q.collide_with_areas = true
		q.collide_with_bodies = true
		var hit := space_state.intersect_ray(q)
		var base := k * 4
		if hit.is_empty():
			out[base] = 1.0  # rien sur ce rayon → depth max, RGB reste 0
		else:
			var d: float = origin.distance_to(hit.position)
			out[base] = clampf(d / max_range, 0.0, 1.0)
			var collider = hit.get("collider")
			if collider != null and collider.has_meta("retina_color"):
				var col: Color = collider.get_meta("retina_color")
				out[base + 1] = col.r
				out[base + 2] = col.g
				out[base + 3] = col.b
	return out


# Returns num_sectors proximities in [0,1]. forward = agent's facing (torso +z basis).
# num_sectors defaults to NUM_SECTORS (12, the WM-trained radar); pass a finer count (e.g. 36)
# for A1 planner food-localisation (finer bearing) WITHOUT changing the WM's input.
static func food_radar(agent_pos: Vector3, forward: Vector3, food_positions: Array, num_sectors: int = NUM_SECTORS) -> Array:
	var radar: Array = []
	radar.resize(num_sectors)
	radar.fill(0.0)
	var fwd := Vector3(forward.x, 0.0, forward.z)
	if fwd.length() < 0.001:
		return radar
	fwd = fwd.normalized()
	var right := Vector3(fwd.z, 0.0, -fwd.x)  # forward rotated -90° about +y → signed bearing
	var sector_size := TAU / float(num_sectors)
	for fp in food_positions:
		var to := Vector3(fp.x - agent_pos.x, 0.0, fp.z - agent_pos.z)
		var dist := to.length()
		if dist > MAX_RANGE or dist < 0.001:
			continue
		var dir := to / dist
		var bearing := atan2(dir.dot(right), dir.dot(fwd))  # [-PI, PI], 0 = straight ahead
		var sector := int(floor((bearing + PI) / sector_size)) % num_sectors
		var proximity := 1.0 - dist / MAX_RANGE
		if proximity > radar[sector]:
			radar[sector] = proximity
	return radar


# HEADING-CONTROL command: encode a single TARGET direction as a bump in the SAME 12-sector
# egocentric format as food_radar (so a policy trained on this command transfers to the real food
# radar). The target's egocentric bearing lights its sector (+ neighbours, for a smooth turning
# gradient). Used by SYLVAN_HEADING_MODE to teach "walk toward the lit sector".
static func heading_command(forward: Vector3, target_dir: Vector3, magnitude: float = 0.8) -> Array:
	var radar: Array = []
	radar.resize(NUM_SECTORS)
	radar.fill(0.0)
	var fwd := Vector3(forward.x, 0.0, forward.z)
	var tgt := Vector3(target_dir.x, 0.0, target_dir.z)
	if fwd.length() < 0.001 or tgt.length() < 0.001:
		return radar
	fwd = fwd.normalized()
	tgt = tgt.normalized()
	var right := Vector3(fwd.z, 0.0, -fwd.x)
	var bearing := atan2(tgt.dot(right), tgt.dot(fwd))
	var sector_size := TAU / float(NUM_SECTORS)
	# SMOOTH symmetric bump on the CONTINUOUS bearing (no hard floor-binning, which made a dead-ahead
	# target asymmetric ~15° off-axis and let the heading command fight a perfectly symmetric policy).
	# Each sector value = Gaussian of the angular distance from its CENTRE to the target bearing, so a
	# dead-ahead command lights sectors 5 & 6 (∓15°) EQUALLY → mirror-invariant with the s<->11-s map.
	var sigma := sector_size
	for s in range(NUM_SECTORS):
		var center := s * sector_size - PI + sector_size * 0.5
		var d := wrapf(bearing - center, -PI, PI)
		radar[s] = magnitude * exp(-(d * d) / (2.0 * sigma * sigma))
	return radar
