extends Node3D
class_name ForestManager

# VISUAL-ONLY forest decor + day/night cycle (Couche 1 du passage en forêt, 2026-06-08).
# Scatters KayKit Forest Nature Pack glTF models (CC0) as pure cosmetic meshes (NO collision,
# NO physics, NO contract change) so the owner SEES the quadruped living in a forest. Loaded at
# RUNTIME via GLTFDocument (the assets live outside res://, so they're not in Godot's import
# pipeline). main.gd only instantiates this in the VISUAL mode (DisplayServer != headless) →
# the headless training workers never pay the load cost and the learning is untouched.

# Asset pack lives at <repo>/ForestLowPolyAssets (sibling of godot/). res:// = godot/, so:
const GLTF_DIR := "res://../ForestLowPolyAssets/Assets/gltf/"

# Scatter plan: [prefix, count, min_radius, max_radius, min_scale, max_scale]
const SCATTER := [
	["Tree", 28, 6.0, 24.0, 0.9, 1.6],
	["Rock", 22, 4.0, 22.0, 0.6, 1.5],
	["Bush", 26, 3.0, 18.0, 0.7, 1.3],
	["Grass", 70, 2.0, 16.0, 0.7, 1.4],
]
const CLEAR_RADIUS := 2.5   # keep the very centre clear so the agent stays visible

var _rng := RandomNumberGenerator.new()
var _cache := {}            # path -> generated scene Node3D (loaded once, duplicated per instance)
var _decor_root: Node3D
var _dir := ""              # absolute path to the glTF dir (res:// can't escape the project for DirAccess)

# Day/night cycle (visual only)
var _sun: DirectionalLight3D
var _env: Environment
var _day_period := 120.0    # seconds for a full day->night->day cycle (visible while watching)
var _time := 0.0
@export var enabled := true

# Third-person follow camera (visual only)
var _camera: Camera3D
var _agent: Node
var _cam_offset := Vector3(0.0, 3.0, -5.5)   # above + behind (-z = behind the +z front)


func setup(world_node: Node, agent_node: Node = null, seed_value: int = 0) -> void:
	_rng.seed = seed_value
	_agent = agent_node
	_dir = ProjectSettings.globalize_path(GLTF_DIR)   # -> absolute /…/ForestLowPolyAssets/Assets/gltf/
	_decor_root = Node3D.new()
	_decor_root.name = "ForestDecor"
	add_child(_decor_root)

	# Grab the sun + environment + camera from the world scene.
	if world_node != null:
		_sun = world_node.get_node_or_null("DirectionalLight3D")
		var we = world_node.get_node_or_null("WorldEnvironment")
		if we != null and "environment" in we:
			_env = we.environment
		_camera = world_node.get_node_or_null("Camera3D")
		if _camera != null:
			_camera.current = true
		_make_ground_grass(world_node)

	var files := _list_gltf()
	if files.is_empty():
		push_warning("ForestManager: no glTF files found at %s" % _dir)
		return
	for entry in SCATTER:
		var prefix: String = entry[0]
		var pool := files.filter(func(f): return f.begins_with(prefix))
		if pool.is_empty():
			continue
		for i in range(int(entry[1])):
			_place_one(pool, entry[2], entry[3], entry[4], entry[5])

	# Free the cached template nodes (orphans, not in the tree) now that every instance is a
	# duplicate holding its own node tree (shared mesh/material resources stay alive). Reduces the
	# "Leaked instance dependency" noise Godot prints at exit for runtime-loaded glTF.
	for t in _cache.values():
		t.free()
	_cache.clear()


# Paint the flat ground grass-green (visual only; the headless trainers never run this).
func _make_ground_grass(world_node: Node) -> void:
	var gmesh := world_node.get_node_or_null("Ground/MeshInstance3D") as MeshInstance3D
	if gmesh == null:
		return
	var grass := StandardMaterial3D.new()
	grass.albedo_color = Color(0.27, 0.5, 0.21)   # grass green
	grass.roughness = 0.95
	gmesh.set_surface_override_material(0, grass)


func _list_gltf() -> Array:
	var out: Array = []
	var d := DirAccess.open(_dir)
	if d == null:
		return out
	d.list_dir_begin()
	var fn := d.get_next()
	while fn != "":
		if fn.ends_with(".gltf"):
			out.append(fn)
		fn = d.get_next()
	d.list_dir_end()
	return out


func _load_model(file_name: String) -> Node3D:
	if _cache.has(file_name):
		return _cache[file_name]
	var doc := GLTFDocument.new()
	var state := GLTFState.new()
	var err := doc.append_from_file(_dir + file_name, state)
	if err != OK:
		return null
	var node := doc.generate_scene(state)
	if node != null:
		_cache[file_name] = node
	return node


func _place_one(pool: Array, min_r: float, max_r: float, min_s: float, max_s: float) -> void:
	var template := _load_model(pool[_rng.randi() % pool.size()])
	if template == null:
		return
	var inst := template.duplicate()
	var angle := _rng.randf_range(0.0, TAU)
	var radius := _rng.randf_range(maxf(min_r, CLEAR_RADIUS), max_r)
	inst.position = Vector3(cos(angle) * radius, 0.0, sin(angle) * radius)
	inst.rotation.y = _rng.randf_range(0.0, TAU)
	inst.scale = Vector3.ONE * _rng.randf_range(min_s, max_s)
	_decor_root.add_child(inst)


func _process(delta: float) -> void:
	if not enabled:
		return
	_update_follow_camera(delta)
	if _sun == null:
		return
	_time += delta
	var phase := fmod(_time / _day_period, 1.0)   # [0,1) one full day
	var theta := phase * TAU                        # sun arc angle
	var elevation := sin(theta)                     # >0 day, <0 night
	# Sweep the sun pitch around so light comes from changing directions through the day.
	_sun.rotation = Vector3(-theta, 0.6, 0.0)
	# Bright at noon, ~dark at night; warm (orange) near the horizon, white at noon.
	_sun.light_energy = clampf(elevation * 1.3, 0.0, 1.3)
	var warmth := clampf(1.0 - maxf(elevation, 0.0), 0.0, 1.0)
	_sun.light_color = Color(1.0, lerpf(1.0, 0.55, warmth), lerpf(1.0, 0.30, warmth))
	if _env != null:
		# Dim the ambient at night so it actually gets dark.
		_env.ambient_light_energy = lerpf(0.05, 0.6, clampf(elevation, 0.0, 1.0))
		if "background_energy_multiplier" in _env:
			_env.background_energy_multiplier = lerpf(0.15, 1.0, clampf(elevation * 0.5 + 0.5, 0.0, 1.0))


# Third-person follow: keep the camera at a fixed offset above/behind the trunk, smoothly tracking
# it as it roams, always looking at it. Fixed world offset (not heading-relative) = stable, no nausea.
func _update_follow_camera(delta: float) -> void:
	if _camera == null or _agent == null or not ("bodies" in _agent):
		return
	var torso = _agent.bodies.get("torso")
	if torso == null:
		return
	var tp: Vector3 = torso.global_position
	var desired: Vector3 = tp + _cam_offset
	var k := clampf(delta * 2.5, 0.0, 1.0)   # follow smoothing
	_camera.global_position = _camera.global_position.lerp(desired, k)
	_camera.look_at(tp + Vector3(0.0, 0.4, 0.0), Vector3.UP)
