extends Node3D

const EPISODE_MANAGER_SCRIPT = preload("res://scripts/world/episode_manager.gd")
const WORLD_MANAGER_SCRIPT = preload("res://scripts/world/world_manager.gd")
const SPAWN_MANAGER_SCRIPT = preload("res://scripts/world/spawn_manager.gd")
const OBSERVATION_BUILDER_SCRIPT = preload("res://scripts/agent/observation_builder.gd")
const REWARD_MANAGER_SCRIPT = preload("res://scripts/rl/reward_manager.gd")
const ROLLOUT_WRITER_SCRIPT = preload("res://scripts/rl/rollout_writer.gd")
const RL_BRIDGE_SCRIPT = preload("res://scripts/rl/rl_bridge.gd")
const ACTION_ADAPTER_SCRIPT = preload("res://scripts/agent/action_adapter.gd")
const HOMEOSTASIS_SCRIPT = preload("res://scripts/agent/homeostasis.gd")
const POLICY_PLAYER_SCRIPT = preload("res://scripts/control/policy_player.gd")
const FOOD_MANAGER_SCRIPT = preload("res://scripts/world/food_manager.gd")
const PERCEPTION_SCRIPT = preload("res://scripts/agent/perception.gd")
const FOREST_MANAGER_SCRIPT = preload("res://scripts/world/forest_manager.gd")

@export var world_scene: PackedScene
@export var agent_scene: PackedScene

var world_instance: Node3D
var agent_instance: Node3D
var episode_manager = EPISODE_MANAGER_SCRIPT.new()
var world_manager = WORLD_MANAGER_SCRIPT.new()
var spawn_manager = SPAWN_MANAGER_SCRIPT.new()
var observation_builder = OBSERVATION_BUILDER_SCRIPT.new()
var reward_manager = REWARD_MANAGER_SCRIPT.new()
var rollout_writer = ROLLOUT_WRITER_SCRIPT.new()
var rl_bridge = RL_BRIDGE_SCRIPT.new()
var action_adapter = ACTION_ADAPTER_SCRIPT.new()
var homeostasis = HOMEOSTASIS_SCRIPT.new()
var policy_player = POLICY_PLAYER_SCRIPT.new()
var food_manager = FOOD_MANAGER_SCRIPT.new()
var water_manager = FOOD_MANAGER_SCRIPT.new()  # 2ᵉ pulsion: eau (même classe, préfixe WATER)
var _water_enabled := false
var collecting := false
var target_num_episodes := 0
var completed_episodes := 0
var collector_mode := "babbling"
var seed_value := 42
var max_episode_steps := 512
var exploration_noise_initial := 0.0
var exploration_noise_final := 0.0
var action_repeat := 1
var current_action_repeat_step := 0
var latest_obs: Dictionary
var latest_action: Array[float]
var accumulated_reward := 0.0

# Perturbation curriculum: random horizontal shoves to force active balance.
var perturbation_strength := 0.0       # impulse magnitude (N·s), from SYLVAN_PERTURBATION_STRENGTH
var steps_since_perturbation := 0
var next_perturbation_steps := 60


func _ready() -> void:
	randomize()
	if world_scene != null:
		world_instance = world_scene.instantiate()
		add_child(world_instance)

	if agent_scene != null:
		agent_instance = agent_scene.instantiate()
		add_child(agent_instance)
		action_adapter.action_dim = agent_instance.action_dim  # sync (salamander=13) so the spine isn't sliced off
		_apply_curriculum()

	# Survival world: food pellets (visible + consumable). Added in every mode so the
	# owner sees them in the viewer too; positions reset per episode below.
	add_child(food_manager)
	food_manager.reset(0)  # initial layout (so pellets show even before the episode loop)
	# 2ᵉ PULSION (2026-06-18): l'EAU = un 2ᵉ FoodManager (même machinerie), préfixe d'env WATER,
	# pastilles bleues. Planner-only (le WM ne la voit pas en étage 1). Activé seulement si demandé
	# (SYLVAN_WATER_COUNT défini) → les runs mono-ressource existants restent identiques.
	_water_enabled = OS.get_environment("SYLVAN_WATER_COUNT") != ""
	homeostasis.thirst_enabled = _water_enabled  # pas d'eau → pas de pression de soif (sinon mort de soif imbuvable)
	if _water_enabled:
		water_manager.configure("WATER", "SYLVAN_DRINK_RADIUS", Color(0.2, 0.5, 0.95), Color(0.05, 0.18, 0.5))
		add_child(water_manager)
		water_manager.reset(0)

	# VISUAL-ONLY forest decor + day/night cycle. Gated on a real display so the headless
	# training workers (--headless = dummy renderer) never load it → zero impact on learning.
	if DisplayServer.get_name() != "headless":
		var forest = FOREST_MANAGER_SCRIPT.new()
		forest.name = "ForestManager"
		add_child(forest)
		forest.setup(world_instance, agent_instance, OS.get_environment("SYLVAN_SEED").to_int())

	collecting = OS.get_environment("SYLVAN_COLLECT") == "1"
	if not collecting:
		return

	target_num_episodes = int(OS.get_environment("SYLVAN_NUM_EPISODES"))
	max_episode_steps = int(OS.get_environment("SYLVAN_MAX_EPISODE_STEPS"))
	seed_value = int(OS.get_environment("SYLVAN_SEED"))
	collector_mode = OS.get_environment("SYLVAN_COLLECTOR_MODE")
	exploration_noise_initial = float(OS.get_environment("SYLVAN_POLICY_EXPLORATION_STD_INITIAL"))
	exploration_noise_final = float(OS.get_environment("SYLVAN_POLICY_EXPLORATION_STD_FINAL"))
	# Frame-skip: decide an action once every N physics steps and HOLD it (physics still runs
	# every step). Halves the per-step policy-server round-trips → faster collection. 30Hz
	# control (action_repeat=2) is standard for legged RL. Defaults to 1 if unset.
	var ar := OS.get_environment("SYLVAN_ACTION_REPEAT")
	action_repeat = maxi(1, int(ar)) if ar != "" else 1
	episode_manager.max_episode_steps = max_episode_steps
	world_manager.set_seed(seed_value)
	spawn_manager.set_seed(seed_value)
	food_manager.set_seed(seed_value)
	if _water_enabled:
		water_manager.set_seed(seed_value + 777)  # décalage → l'eau ne spawn pas SUR la bouffe
	add_child(rollout_writer)
	var output_dir := OS.get_environment("SYLVAN_RUN_DIR")
	if output_dir.is_empty():
		push_error("SYLVAN_RUN_DIR is required for collection")
		get_tree().quit(1)
		return
	rollout_writer.configure_output_directory(output_dir)
	var policy_path := OS.get_environment("SYLVAN_POLICY_JSON")
	if not policy_path.is_empty():
		policy_player.load_policy(policy_path)
	var policy_host := OS.get_environment("SYLVAN_POLICY_HOST")
	var policy_port := int(OS.get_environment("SYLVAN_POLICY_PORT"))
	if not policy_host.is_empty() and policy_port > 0:
		policy_player.connect_to_server(policy_host, policy_port)
	_start_episode()


func _apply_curriculum() -> void:
	# Balance crutches driven by the orchestrator (decayed over cycles).
	# Applied in every mode (collection AND validation) so the agent is evaluated
	# under the same conditions it trained in; both fade to 0 → pure emergence.
	var assist_ratio := 0.0
	var assist_str := OS.get_environment("SYLVAN_ASSIST_RATIO")
	if not assist_str.is_empty():
		assist_ratio = float(assist_str)
	if agent_instance.has_method("set_assistance"):
		agent_instance.set_assistance(assist_ratio)

	var reflex := 0.0
	var reflex_str := OS.get_environment("SYLVAN_REFLEX_STRENGTH")
	if not reflex_str.is_empty():
		reflex = float(reflex_str)
	if agent_instance.has_method("set_reflex_strength"):
		agent_instance.set_reflex_strength(reflex)

	var pert_str := OS.get_environment("SYLVAN_PERTURBATION_STRENGTH")
	if not pert_str.is_empty():
		perturbation_strength = float(pert_str)

	if assist_ratio > 0.0 or reflex > 0.0 or perturbation_strength > 0.0:
		print("[Godot] Curriculum: gravity_assist=%.2f reflex=%.2f perturbation=%.1f" % [assist_ratio, reflex, perturbation_strength])


# HEADING-CONTROL training (SYLVAN_HEADING_MODE=1): replace the food radar with a clean directional
# COMMAND bump — a random target heading that changes every ~120 steps — and reward walking toward it.
# Teaches the policy to STEER toward the lit vision sector; deployed with the food radar in the same
# channel, that skill becomes "walk toward food" (the missing motor primitive: controlled turning).
var _heading_mode := false
var _heading_inited := false
var _heading_target := Vector3.ZERO
var _heading_steps := 0
const _HEADING_PERIOD := 120
var _heading_rng := RandomNumberGenerator.new()
var _no_homeostasis := false  # heading-training regime → no metabolism/starvation-death (clean MDP)
var _blank_vision := false    # SYLVAN_BLANK_VISION=1 → vision channel is all zeros (pure-proprio locomotion)
var _food_as_command := false # SYLVAN_FOOD_AS_COMMAND=1 → deploy: vision = a clean heading-bump toward the NEAREST
                              # pellet (matches the training command exactly), NOT the saturated multi-pellet radar
var _ep_start_pos := Vector3.ZERO  # torso position at episode start (CPG straightness/forward-progress diagnostic)
# Phase 3 AUTOPILOT: a trivial hand P-controller omega = clamp(K * bearing_to_nearest_food) feeding the
# CPG command — closed-loop point-to-point navigation, ZERO learning. Proves command-space nav is tractable.
var _cpg_autopilot := false
# Phase 5 PLANNER mode (SYLVAN_CPG_PLANNER=1): the command (vx, omega) comes from the WM planner
# server each step instead of a sampler/autopilot formula. Godot sends the REAL food radar (not the
# command) in the vision channel, and reads back {action, command} — set_cpg_command(command) so the
# CPG steers by construction toward the planner's choice. This is the Mode-2 brain driving the body.
var _cpg_planner := false
var _retina_debug := false   # SYLVAN_RETINA_DEBUG=1 → log les 1ers ticks de rétine (sanity étage 0, gratuit)
var _retina_dbg_left := 3
var _retina_planner := false # SYLVAN_RETINA_PLANNER=1 → envoie la rétine live au serveur (localisation apprise)
var _autopilot_k := 1.5
var _autopilot_wmax := 0.8
var _dbg_bearing := 0.0  # last bearing-to-food (deg) for autopilot diagnosis
# Phase 2b: per-episode command sampler — vary omega so the residual learns to PROPEL while TURNING
# (the Phase-2 residual was straight-only). The command also enters the obs (vision) so it knows the turn.
var _cpg_sample_cmd := false
var _cpg_sample_wmax := 0.6
# Phase A (2026-06-16): SCHEDULED command CURRICULUM. Sample COUPLED (vx, omega) per episode; the |omega|
# range grows 0 -> _cmd_wmax over _curric_episodes (driven by this worker's episode_index, monotone with
# training, no trainer<->Godot plumbing). Research: uniform large-range command sampling makes RL freeze
# the turn; start near zero and widen. ~15% straight episodes for clean fast-straight data.
var _cmd_curric := false
var _curric_episodes := 400.0
var _cmd_vx_max := 0.5
var _cmd_wmax := 0.6
var _curric_rng := RandomNumberGenerator.new()
# SOTA anti-wobble (legged_gym/WTW): penalise residual jitter via action_rate Σ(Δa)² and
# action_smoothness_2 Σ(a_t − 2a_{t-1} + a_{t-2})² (2nd derivative of the targets). Computed at each
# decision and injected into reward_metrics for locomotion_track_v2. History over decisions only.
var _prev_action: Array[float] = []
var _prev_prev_action: Array[float] = []
var _action_rate_pen := 0.0
var _action_smooth2_pen := 0.0
# Phase 4 WM collection (SYLVAN_WM_COLLECT=1): piecewise-constant command babbling + a per-step
# "wm" block in the JSONL (cmd, torso pose, food radar at t and t+1, eat amount). The policy's
# obs contract is UNTOUCHED (vision still carries the command) — the extra ground truth rides
# alongside so the command-space world model has targets the obs can't provide.
var _wm_collect := false
var _wm_rng := RandomNumberGenerator.new()
var _wm_resample_in := 0      # decision steps until the next (vx, omega) draw
var _wm_straight_ep := false  # ~15% of episodes hold omega=0 (clean straight-line data)
var _wm_t0 := {}              # cmd/torso/radar captured at decision time (start of the action-repeat window)
var _wm_ate := 0.0            # energy restored by eating within the current window
# Babbling command ranges for WM collection. Defaults match the original Phase-4 quad collect;
# overridable via SYLVAN_WM_VX_MIN/MAX/WMAX so each body can be sampled in ITS clean operating
# regime (e.g. the hexapod walks straight + turns both ways at vx~0.7, not at the 0.3-0.6 default).
var _WM_VX_MIN := 0.3
var _WM_VX_MAX := 0.6
var _WM_WMAX := 0.6
# Scripted-turn collection (SYLVAN_WM_TURN_SCRIPT=1): instead of random-sign piecewise omega, hold a
# SUSTAINED single-sign omega for a LONG block so the agent orbits (clean arc R=vx/omega) and its heading
# sweeps continuously → every 360°-placed target crosses behind↔front each loop = the acquisition event the
# dream must learn. Random-sign babbling reverses the arc before a sweep completes (1.0 crossing/ep) — this
# scripts the sweep. vx is fixed mid-clean; sign alternates each block for L/R symmetry.
var _wm_turn_script := false
var _wm_turn_sign := 1.0


var _diag_heading := ""  # SYLVAN_DIAG_HEADING = "cw"/"ccw": force a CONSTANT egocentric-90° turn command
                         # to measure if the body can yaw both ways symmetrically (turn-asymmetry probe).


func _update_heading() -> void:
	if not _heading_inited:
		_heading_inited = true
		_diag_heading = OS.get_environment("SYLVAN_DIAG_HEADING")
		_heading_mode = OS.get_environment("SYLVAN_HEADING_MODE") == "1" or _diag_heading != ""
		# Clean-MDP for the heading skill: food is NOT in the observation here, so leaving metabolism +
		# starvation-death active makes every episode a guaranteed ~600-step death whose -1 terminal
		# (with a zeroed GAE bootstrap) poisons the value fn and gives NO gradient for sustained steering.
		# Disable homeostasis (auto in heading mode, or via SYLVAN_DISABLE_HOMEOSTASIS=1) so episodes run
		# to truncation and the ONLY terminal is a real physical fall.
		_no_homeostasis = _heading_mode or OS.get_environment("SYLVAN_DISABLE_HOMEOSTASIS") == "1"
		if _no_homeostasis:
			print("[Godot] HOMEOSTASIS DISABLED (clean MDP) | done = physical fall only")
		# PHASE-1 symmetric-walker training: blank the vision channel so the gait is learned from
		# proprioception alone, with NO radar/command confound — isolates "is the drift dead?" from
		# "does it steer?". Deployed/heading runs leave this off.
		_retina_debug = OS.get_environment("SYLVAN_RETINA_DEBUG") == "1"
		_retina_planner = OS.get_environment("SYLVAN_RETINA_PLANNER") == "1"
		_blank_vision = OS.get_environment("SYLVAN_BLANK_VISION") == "1"
		if _blank_vision:
			print("[Godot] VISION BLANKED (zeros) | pure-proprioceptive locomotion")
		_food_as_command = OS.get_environment("SYLVAN_FOOD_AS_COMMAND") == "1"
		if _food_as_command:
			print("[Godot] FOOD-AS-COMMAND | vision = clean heading bump toward nearest pellet (matches training)")
		# RE-ARCH Phase 0: command-conditioned CPG base. SYLVAN_CPG=1 enables the analytic walk+turn
		# controller; (vx, omega) come from SYLVAN_CMD_VX / SYLVAN_CMD_OMEGA; SYLVAN_RESIDUAL_GAIN=0 =>
		# pure open-loop CPG (no learning). Later phases drive (vx,omega) from a sampler / the WM planner.
		if OS.get_environment("SYLVAN_CPG") == "1":
			var _cmd_vx := OS.get_environment("SYLVAN_CMD_VX").to_float()
			var _cmd_omega := OS.get_environment("SYLVAN_CMD_OMEGA").to_float()
			var _res_gain := OS.get_environment("SYLVAN_RESIDUAL_GAIN").to_float()
			agent_instance.enable_cpg(true, _res_gain)
			agent_instance.set_cpg_command(_cmd_vx, _cmd_omega)
			# FULLY-LEARNED mode (2026-06-14): bypass the CPG motor (policy outputs the 12 targets
			# directly), but KEEP cpg_enabled=true so the command plumbing (sampling, obs, reward cmd)
			# all keeps working. Only step_agent's motor application changes.
			if OS.get_environment("SYLVAN_LEARNED") == "1":
				agent_instance.learned_mode = true
				var _las := OS.get_environment("SYLVAN_LEARNED_ACTION_SCALE")
				if _las != "": agent_instance.learned_action_scale = _las.to_float()
				var _lbl := OS.get_environment("SYLVAN_LEARNED_BLEND")
				if _lbl != "": agent_instance.learned_blend = _lbl.to_float()
				print("[Godot] LEARNED MODE | action_scale=%.2f blend=%.2f (blend<1 = CPG-bootstrap)" % [agent_instance.learned_action_scale, agent_instance.learned_blend])
			# Phase-1 tuning: optional CPG param overrides via env (empty = keep the @export default).
			var _ov := OS.get_environment("SYLVAN_CPG_STEP")
			if _ov != "": agent_instance.cpg_step_amp = _ov.to_float()
			_ov = OS.get_environment("SYLVAN_CPG_LIFT")
			if _ov != "": agent_instance.cpg_lift_amp = _ov.to_float()
			_ov = OS.get_environment("SYLVAN_CPG_TURNK")
			if _ov != "": agent_instance.cpg_turn_k = _ov.to_float()
			_ov = OS.get_environment("SYLVAN_CPG_TURNAMP")
			if _ov != "": agent_instance.cpg_turn_amp = _ov.to_float()
			_ov = OS.get_environment("SYLVAN_CPG_YAWLAT")
			if _ov != "": agent_instance.cpg_yaw_lat = _ov.to_float()
			_ov = OS.get_environment("SYLVAN_CPG_SPINETURN")  # salamander spine curvature-offset gain (turn)
			if _ov != "": agent_instance.cpg_spine_turn = _ov.to_float()
			_ov = OS.get_environment("SYLVAN_CPG_SPINEAMP")   # CONTACT-driven undulation amplitude
			if _ov != "": agent_instance.cpg_spine_amp = _ov.to_float()
			_ov = OS.get_environment("SYLVAN_CPG_SPINESIGN") # undulation phase sign (+1/-1, set empirically)
			if _ov != "": agent_instance.cpg_spine_sign = _ov.to_float()
			_ov = OS.get_environment("SYLVAN_CPG_SPEEDCAD")  # gait frequency scales with commanded vx
			if _ov != "": agent_instance.cpg_speed_cadence_k = _ov.to_float()
			_ov = OS.get_environment("SYLVAN_CPG_LIFTPH")
			if _ov != "": agent_instance.cpg_lift_phase = _ov.to_float()
			_ov = OS.get_environment("SYLVAN_CPG_PERIOD")
			if _ov != "": agent_instance.gait_cycle_period = _ov.to_float()
			_ov = OS.get_environment("SYLVAN_CPG_DUTY")
			if _ov != "": agent_instance.cpg_duty = _ov.to_float()
			# Voluntary gait modulation knobs (big/small steps, run, knee bend) — dynamic; env now, the
			# JEPA planner / a policy drives them later. Empty = 1.0 (nominal). The body is not a prison.
			var _ms := OS.get_environment("SYLVAN_MOD_STRIDE")
			var _mc := OS.get_environment("SYLVAN_MOD_CADENCE")
			var _ml := OS.get_environment("SYLVAN_MOD_LIFT")
			agent_instance.set_cpg_modulation(
				_ms.to_float() if _ms != "" else 1.0,
				_mc.to_float() if _mc != "" else 1.0,
				_ml.to_float() if _ml != "" else 1.0)
			# Phase-1 TRACTION tuning: foot friction (anti-slip grip) + PD stiffness kp (push force).
			# The compliant kp=4 was for bipedal balance; a quad CPG wants stiffer legs that actually push.
			var _kp := OS.get_environment("SYLVAN_KP")
			if _kp != "": agent_instance.kp = _kp.to_float()
			var _ff := OS.get_environment("SYLVAN_FOOT_FRICTION")
			if _ff != "":
				for _fn in ["fl_lower", "fr_lower", "bl_lower", "br_lower"]:
					var _fb = agent_instance.bodies.get(_fn)
					if _fb != null and _fb.physics_material_override != null:
						_fb.physics_material_override.friction = _ff.to_float()
			_cpg_planner = OS.get_environment("SYLVAN_CPG_PLANNER") == "1"
			if _cpg_planner:
				print("[Godot] PLANNER MODE ON | command (vx,omega) from the WM planner server each step")
			_cpg_autopilot = OS.get_environment("SYLVAN_CPG_AUTOPILOT") == "1"
			var _ak := OS.get_environment("SYLVAN_AUTOPILOT_K")
			if _ak != "": _autopilot_k = _ak.to_float()
			var _aw := OS.get_environment("SYLVAN_AUTOPILOT_WMAX")
			if _aw != "": _autopilot_wmax = _aw.to_float()
			_cpg_sample_cmd = OS.get_environment("SYLVAN_CPG_SAMPLE_CMD") == "1"
			var _csw := OS.get_environment("SYLVAN_CPG_SAMPLE_WMAX")
			if _csw != "": _cpg_sample_wmax = _csw.to_float()
			if _cpg_sample_cmd:
				print("[Godot] CPG CMD SAMPLER ON | per-episode omega in [-%.2f,+%.2f] -> command in vision obs" % [_cpg_sample_wmax, _cpg_sample_wmax])
			_cmd_curric = OS.get_environment("SYLVAN_CMD_CURRIC") == "1"
			var _ce := OS.get_environment("SYLVAN_CURRIC_EPISODES")
			if _ce != "": _curric_episodes = maxf(1.0, _ce.to_float())
			var _cvm := OS.get_environment("SYLVAN_CMD_VX_MAX")
			if _cvm != "": _cmd_vx_max = _cvm.to_float()
			var _cwm := OS.get_environment("SYLVAN_CMD_WMAX")
			if _cwm != "": _cmd_wmax = _cwm.to_float()
			if _cmd_curric:
				_curric_rng.seed = seed_value + 9191
				print("[Godot] CMD CURRICULUM ON | this-iter range: vx in [0.15,%.2f], |omega|<=%.2f (ramped per-iter by train_ppo) | 15%% straight" % [_cmd_vx_max, _cmd_wmax])
			_wm_collect = OS.get_environment("SYLVAN_WM_COLLECT") == "1"
			if _wm_collect:
				_wm_rng.seed = seed_value + 4242
				var _wvmin := OS.get_environment("SYLVAN_WM_VX_MIN")
				if _wvmin != "": _WM_VX_MIN = _wvmin.to_float()
				var _wvmax := OS.get_environment("SYLVAN_WM_VX_MAX")
				if _wvmax != "": _WM_VX_MAX = _wvmax.to_float()
				var _wwm := OS.get_environment("SYLVAN_WM_WMAX")
				if _wwm != "": _WM_WMAX = _wwm.to_float()
				_wm_turn_script = OS.get_environment("SYLVAN_WM_TURN_SCRIPT") == "1"
				if _wm_turn_script:
					print("[Godot] WM COLLECT ON | SCRIPTED-TURN: vx fixed=%.2f, omega = sign*%.2f sustained, sign flips every 120-200 decisions (orbit sweep, behind->front acquisition)" % [(_WM_VX_MIN + _WM_VX_MAX) * 0.5, _WM_WMAX])
				else:
					print("[Godot] WM COLLECT ON | piecewise cmd: vx in [%.2f,%.2f], omega in [-%.2f,+%.2f], redraw every 40-80 decisions | wm block logged per step" % [_WM_VX_MIN, _WM_VX_MAX, _WM_WMAX, _WM_WMAX])
			if _cpg_autopilot:
				print("[Godot] AUTOPILOT ON | omega = clamp(%.1f * bearing_to_nearest_food, +-%.2f)" % [_autopilot_k, _autopilot_wmax])
			print("[Godot] CPG ON | vx=%.2f omega=%.2f res=%.2f | step=%.2f lift=%.2f liftph=%.2f turnk=%.2f period=%.2f | mod[stride=%.2f cadence=%.2f lift=%.2f]" % [_cmd_vx, _cmd_omega, _res_gain, agent_instance.cpg_step_amp, agent_instance.cpg_lift_amp, agent_instance.cpg_lift_phase, agent_instance.cpg_turn_k, agent_instance.gait_cycle_period, agent_instance.cpg_stride_scale, agent_instance.cpg_cadence_scale, agent_instance.cpg_lift_scale])
		_heading_rng.seed = seed_value + 777
		if _heading_mode:
			print("[Godot] HEADING-CONTROL mode ON | diag=", _diag_heading)
	if not _heading_mode:
		return
	# DIAG: command a heading 90° to the agent's left/right EVERY step → it must keep yawing one way.
	if _diag_heading != "":
		var torso: Node3D = agent_instance.bodies.get("torso")
		if torso != null:
			var f := torso.global_transform.basis.z
			# +90° about Y: (x,z)->(z,-x) ; -90°: (x,z)->(-z,x)
			_heading_target = Vector3(f.z, 0.0, -f.x) if _diag_heading == "cw" else Vector3(-f.z, 0.0, f.x)
		return
	_heading_steps += 1
	if _heading_target == Vector3.ZERO or _heading_steps >= _HEADING_PERIOD:
		var a := _heading_rng.randf_range(0.0, TAU)
		_heading_target = Vector3(cos(a), 0.0, sin(a))
		_heading_steps = 0


func _physics_process(delta: float) -> void:
	if not collecting or agent_instance == null:
		return
	if completed_episodes >= target_num_episodes:
		get_tree().quit()
		return

	_update_heading()

	# Phase 3 AUTOPILOT (closed-loop heading): steer the CPG's omega toward the nearest food pellet.
	# A trivial proportional controller — no learning. The CPG turns by construction toward the bearing.
	if _cpg_autopilot and agent_instance.cpg_enabled:
		var _ptorso: Node3D = agent_instance.bodies.get("torso")
		if _ptorso != null:
			var _fdir: Vector3 = food_manager.nearest_dir(_ptorso.global_position)
			var _omega := 0.0
			if _fdir.length_squared() > 0.0001:
				var _f := _ptorso.global_transform.basis.z   # +z forward
				var _bearing := atan2(_f.z * _fdir.x - _f.x * _fdir.z, _f.x * _fdir.x + _f.z * _fdir.z)
				_dbg_bearing = rad_to_deg(_bearing)
				# NOTE: naive omega=K*bearing ORBITS the target (forward-only vehicle can't turn in place).
				# Needs a proper pursuit law (lookahead / slow vx when far off-heading) — or let the JEPA learn nav.
				_omega = clampf(_autopilot_k * _bearing, -_autopilot_wmax, _autopilot_wmax)
			agent_instance.set_cpg_command(agent_instance.cpg_command.x, _omega)

	# Phase 4 WM collection: redraw the command on a 40-80 decision cadence, at decision boundaries
	# ONLY — so the command the vision obs carries always matches what the CPG executes this window.
	if _wm_collect and agent_instance.cpg_enabled and current_action_repeat_step == 0:
		_wm_resample_in -= 1
		if _wm_resample_in <= 0:
			if _wm_turn_script:
				# SCRIPTED sweep: single-sign omega block, fixed mid-clean vx, sign alternates each block.
				# Block length tunable via SYLVAN_WM_TURN_BLOCK (decisions); shorter = S-curves (un-trap the
				# orbit so behind targets actually sweep to front), longer = orbit.
				var _blk := 120
				var _eblk := OS.get_environment("SYLVAN_WM_TURN_BLOCK")
				if _eblk != "": _blk = _eblk.to_int()
				_wm_resample_in = _wm_rng.randi_range(_blk, _blk + 40)
				_wm_turn_sign = -_wm_turn_sign
				var _tvx := (_WM_VX_MIN + _WM_VX_MAX) * 0.5
				agent_instance.set_cpg_command(_tvx, _wm_turn_sign * _WM_WMAX)
			else:
				_wm_resample_in = _wm_rng.randi_range(40, 80)
				var _wvx := _wm_rng.randf_range(_WM_VX_MIN, _WM_VX_MAX)
				var _wom := 0.0 if _wm_straight_ep else _wm_rng.randf_range(-_WM_WMAX, _WM_WMAX)
				agent_instance.set_cpg_command(_wvx, _wom)

	if current_action_repeat_step == 0:
		latest_obs = observation_builder.build_observation(agent_instance, homeostasis.energy, homeostasis.health, _compute_vision())
		if _retina_debug and _retina_dbg_left > 0:
			_retina_dbg_left -= 1
			var _rt: Node3D = agent_instance.bodies.get("torso")
			if _rt != null:
				var _ss := agent_instance.get_world_3d().direct_space_state
				var _ret: Array = PERCEPTION_SCRIPT.retina(_ss, _rt.global_position, _rt.global_transform.basis.z)
				var _hits := 0
				var _line := ""
				for _k in range(PERCEPTION_SCRIPT.RETINA_RAYS):
					var _b := _k * 4
					if _ret[_b] < 0.999:  # hit (depth < max)
						_hits += 1
						_line += " ray%d[d=%.2f rgb=%.2f,%.2f,%.2f]" % [_k, _ret[_b], _ret[_b + 1], _ret[_b + 2], _ret[_b + 3]]
				var _fpos: Array = food_manager.get_positions()
				var _fd := 999.0
				if _fpos.size() > 0:
					_fd = Vector3(_rt.global_position.x, _fpos[0].y, _rt.global_position.z).distance_to(_fpos[0])
				print("[Retina] hits=%d/%d food0_dist=%.2f |%s" % [_hits, PERCEPTION_SCRIPT.RETINA_RAYS, _fd, _line])
		if _cpg_planner and agent_instance.cpg_enabled:
			# A1 (perception): also send a FINER 36-sector food radar for planner localisation only
			# (food bearing to ~±5° instead of ±15°). The WM still gets its trained 12-sector vision;
			# this extra field is used server-side for food_xz reconstruction, nothing else.
			var _ptorso2: Node3D = agent_instance.bodies.get("torso")
			if _ptorso2 != null:
				latest_obs["vision_fine"] = PERCEPTION_SCRIPT.food_radar(_ptorso2.global_position, _ptorso2.global_transform.basis.z, food_manager.get_positions(), 36)
				# RÉTINE étage 1 : rayons couleur BRUTS live → le serveur localise via la tête APPRISE
				# (remplace l'oracle food_xz). Envoyé seulement quand demandé (SYLVAN_RETINA_PLANNER=1).
				if _retina_planner:
					latest_obs["retina"] = PERCEPTION_SCRIPT.retina(agent_instance.get_world_3d().direct_space_state, _ptorso2.global_position, _ptorso2.global_transform.basis.z)
				# 2ᵉ pulsion: radar EAU 36-secteurs (planner-only) + niveau de soif. Hors WM (étage 1).
				if _water_enabled:
					latest_obs["vision_water"] = PERCEPTION_SCRIPT.food_radar(_ptorso2.global_position, _ptorso2.global_transform.basis.z, water_manager.get_positions(), 36)
					latest_obs["thirst"] = homeostasis.thirst
			# Phase 5: the WM planner server returns BOTH the residual action and the chosen command.
			var resp: Dictionary = policy_player.predict_planner(latest_obs)
			var cmd: Array = resp.get("command", [])
			if cmd.size() == 2:
				agent_instance.set_cpg_command(float(cmd[0]), float(cmd[1]))
			var act: Array = resp.get("action", [])
			var typed_action: Array[float] = []
			for v in act:
				typed_action.append(float(v))
			latest_action = typed_action
		else:
			latest_action = _build_action(latest_obs)
		# SOTA anti-wobble penalties on the residual output (1st + 2nd difference over decisions).
		if latest_action.size() > 0 and _prev_action.size() == latest_action.size():
			var ar := 0.0
			var sm := 0.0
			var have2: bool = _prev_prev_action.size() == latest_action.size()
			for _i in range(latest_action.size()):
				var d1: float = latest_action[_i] - _prev_action[_i]
				ar += d1 * d1
				if have2:
					var d2: float = latest_action[_i] - 2.0 * _prev_action[_i] + _prev_prev_action[_i]
					sm += d2 * d2
			_action_rate_pen = ar
			_action_smooth2_pen = sm if have2 else 0.0
		_prev_prev_action = _prev_action
		_prev_action = latest_action.duplicate()
		agent_instance.apply_action(latest_action)
		if _wm_collect:
			_wm_t0 = _wm_snapshot()
			_wm_ate = 0.0

	agent_instance.step_agent(delta)

	# Perturbation curriculum: random horizontal shove every ~0.75-1.5 s once the
	# agent has settled. Forces active balance / stepping recovery (passive standing
	# no longer survives). Skipped during the post-reset settle window.
	if perturbation_strength > 0.0 and agent_instance.reset_timer <= 0.0:
		steps_since_perturbation += 1
		if steps_since_perturbation >= next_perturbation_steps:
			var ang := randf() * TAU
			agent_instance.apply_perturbation(Vector3(cos(ang), 0.0, sin(ang)) * perturbation_strength)
			steps_since_perturbation = 0
			next_perturbation_steps = randi_range(25, 50)

	var torso: Node3D = agent_instance.bodies.get("torso")
	if not _no_homeostasis:
		homeostasis.apply_metabolism(agent_instance.effort * 0.08)
		# Eat any food pellet the agent walks over → restore energy (the homeostatic reward loop).
		if torso != null:
			var restored: float = food_manager.try_consume(torso.global_position, homeostasis.energy / homeostasis.max_energy)
			if restored > 0.0:
				homeostasis.restore_energy(restored)
				_wm_ate += restored
			# 2ᵉ pulsion: boire l'eau → restaure la soif (symétrique à manger).
			if _water_enabled:
				var drank: float = water_manager.try_consume(torso.global_position)
				if drank > 0.0:
					homeostasis.restore_thirst(drank)
		if agent_instance.has_fallen:
			homeostasis.apply_damage(3.0)

	current_action_repeat_step += 1
	var done: bool = agent_instance.has_fallen if _no_homeostasis else (homeostasis.is_critical() or agent_instance.has_fallen)
	var _rmetrics: Dictionary = agent_instance.get_locomotion_metrics()
	_rmetrics["action_rate"] = _action_rate_pen        # inject the anti-jitter penalties (computed above)
	_rmetrics["action_smooth2"] = _action_smooth2_pen  # so the reward (incl. locomotion_omni_v1) can use them
	accumulated_reward += reward_manager.compute_reward(_rmetrics, homeostasis.energy, homeostasis.health, done)

	if current_action_repeat_step >= action_repeat or done:
		var truncated: bool = episode_manager.should_truncate()
		var next_obs: Dictionary = observation_builder.build_observation(agent_instance, homeostasis.energy, homeostasis.health, _compute_vision())
		var reward: float = accumulated_reward / float(current_action_repeat_step)
		var step_id: int = episode_manager.next_step_id()
		
		if step_id % 10 == 0:
			var _yaw := 0.0
			var _disp := 0.0
			var _foodd := -1.0
			var _brg := _dbg_bearing  # fallback (autopilot sets it); recomputed fresh below so PLANNER mode logs it too
			if torso != null:
				_yaw = rad_to_deg(atan2(torso.global_transform.basis.z.x, torso.global_transform.basis.z.z))
				var _d := torso.global_position - _ep_start_pos
				_disp = Vector2(_d.x, _d.z).length()
				_foodd = food_manager.nearest_distance(torso.global_position)
				var _fd: Vector3 = food_manager.nearest_dir(torso.global_position)
				if _fd.length_squared() > 0.0001:
					var _ff := torso.global_transform.basis.z   # +z forward
					_brg = rad_to_deg(atan2(_ff.z * _fd.x - _ff.x * _fd.z, _fd.x * _ff.x + _fd.z * _ff.z))
			var _waterd := -1.0
			if _water_enabled and torso != null:
				_waterd = water_manager.nearest_distance(torso.global_position)
			print("[Godot] Episode %d | Step %d | Energy: %.1f | Thirst: %.1f | Health: %.1f | Reward: %.3f | Yaw: %.0f | fwd_v: %.2f | disp: %.2f | food_d: %.2f | water_d: %.2f | om: %.2f | brg: %.0f" % [completed_episodes, step_id, homeostasis.energy, homeostasis.thirst, homeostasis.health, reward, _yaw, agent_instance.forward_velocity, _disp, _foodd, _waterd, agent_instance.cpg_command.y, _brg])

		var transition: Dictionary = rl_bridge.build_transition(
			latest_obs,
			latest_action,
			reward,
			next_obs,
			done,
			truncated,
			episode_manager.current_episode_id,
			step_id,
			seed_value,
			agent_instance.agent_version
		)
		# BC bootstrap collection (SYLVAN_BC_COLLECT=1): log the FINAL applied joint targets so the
		# learned policy can be behavior-cloned on the GOOD CPG+residual gait (obs → applied).
		if OS.get_environment("SYLVAN_BC_COLLECT") == "1":
			transition["applied"] = agent_instance.latest_applied_action
		# Phase 4 WM collection: ride the ground truth the obs can't carry (vision = command in CPG
		# mode) alongside the transition. t0 = decision time (matches obs), t1 = now (matches next_obs).
		if _wm_collect:
			var _t1 := _wm_snapshot()
			transition["wm"] = {
				"cmd": _wm_t0.get("cmd", [0.0, 0.0]),
				"torso0": _wm_t0.get("torso", [0.0, 0.0, 0.0]),
				"radar0": _wm_t0.get("radar", []),
				"torso1": _t1.get("torso", [0.0, 0.0, 0.0]),
				"radar1": _t1.get("radar", []),
				"ate": _wm_ate,
				# RÉTINE étage 1 : rayons bruts t0 + label position vraie (cible de la tête 🅐). t0 suffit
				# pour la tête (perception statique) ; le radar0/1 reste pour le WM (étage 2 séparé).
				"retina0": _wm_t0.get("retina", []),
				"food_rel0": _wm_t0.get("food_rel", []),
				"water_rel0": _wm_t0.get("water_rel", []),
			}
		rollout_writer.write_transition(transition)
		
		current_action_repeat_step = 0
		accumulated_reward = 0.0
		
		if done or truncated:
			_finish_episode("done" if done else "truncated")


func _start_episode() -> void:
	var episode_index: int = episode_manager.episode_index
	spawn_manager.begin_episode(episode_index)
	world_manager.reset_world(episode_index)
	food_manager.reset(episode_index)
	if _water_enabled:
		water_manager.reset(episode_index)
	homeostasis.reset_state()
	agent_instance.reset_agent(
		spawn_manager.get_agent_spawn_position(),
		spawn_manager.get_agent_spawn_yaw(),
		spawn_manager.get_agent_spawn_impulse()
	)
	var _t0: Node3D = agent_instance.bodies.get("torso")
	_ep_start_pos = _t0.global_position if _t0 != null else Vector3.ZERO
	# Phase A: SCHEDULED command curriculum (coupled vx+omega). The |omega| range (_cmd_wmax) is ramped
	# PER ITERATION by train_ppo via SYLVAN_CMD_WMAX — the SAME pattern as SYLVAN_TARGET_VELOCITY — because
	# Godot RELAUNCHES every iteration and re-reads the env in _ready (so episode_index resets each iter and
	# CANNOT drive the ramp). Here we just sample the CURRENT range. ~15% straight eps for clean fast data.
	if _cmd_curric and agent_instance.cpg_enabled:
		var _cvx := _curric_rng.randf_range(0.15, maxf(0.2, _cmd_vx_max))
		var _com := 0.0
		if _curric_rng.randf() >= 0.15:   # 85% turning episodes
			_com = _curric_rng.randf_range(-_cmd_wmax, _cmd_wmax)
		agent_instance.set_cpg_command(_cvx, _com)
	# Phase 2b: resample the turn command each episode so the residual sees the full omega range.
	elif _cpg_sample_cmd and agent_instance.cpg_enabled:
		agent_instance.set_cpg_command(agent_instance.cpg_command.x, _heading_rng.randf_range(-_cpg_sample_wmax, _cpg_sample_wmax))
	# Phase 4 WM collection: pick the episode's regime and force a command draw on the first decision.
	if _wm_collect and agent_instance.cpg_enabled:
		_wm_straight_ep = _wm_rng.randf() < 0.15
		_wm_resample_in = 0
		_wm_t0 = {}
		_wm_ate = 0.0

	episode_manager.start_episode()
	rollout_writer.begin_episode(episode_manager.current_episode_id)
	current_action_repeat_step = 0
	accumulated_reward = 0.0
	steps_since_perturbation = 0
	next_perturbation_steps = randi_range(25, 50)
	_heading_target = Vector3.ZERO  # re-anchor the commanded heading each episode (it persisted across boundaries)
	_heading_steps = 0
	_prev_action = []  # reset action history so the smoothness penalty ignores the episode boundary
	_prev_prev_action = []
	_action_rate_pen = 0.0
	_action_smooth2_pen = 0.0
	# BC LOGGER (Task 2) : signaler au serveur planner le début d'épisode → rotation de fichier JSONL.
	# Seulement quand _cpg_planner est actif (vrai dès le 2ème épisode, après _update_heading).
	# Le 1er épisode (avant _update_heading) est couvert par l'auto-open dans serve_planner_command.py.
	if _cpg_planner and policy_player.is_server_ready():
		policy_player.send_reset()


func _finish_episode(reason: String) -> void:
	episode_manager.finish_episode(reason)
	rollout_writer.end_episode()
	completed_episodes += 1
	if completed_episodes >= target_num_episodes:
		get_tree().quit()
	else:
		_start_episode()


func _wm_snapshot() -> Dictionary:
	# Ground truth for the command-space world model: the active command, the torso pose
	# (x, z, yaw rad) and the REAL food radar (the vision obs carries the command in CPG mode).
	var torso: Node3D = agent_instance.bodies.get("torso")
	if torso == null:
		return {}
	var f := torso.global_transform.basis.z
	# RÉTINE (étage 1) : on logge les rayons couleur bruts À CÔTÉ de l'oracle radar, + le LABEL = la vraie
	# position de la ressource la plus proche en frame agent (même convention (x_right, z_fwd) que
	# food_xz_from_radar). C'est la cible supervisée de la tête de perception apprise (🅐). Vérité du
	# simulateur (pas l'oracle radar) → label propre ; l'oracle reste débranché à l'éval (honnêteté §2).
	var _ss := agent_instance.get_world_3d().direct_space_state
	var _snap := {
		"cmd": [agent_instance.cpg_command.x, agent_instance.cpg_command.y],
		"torso": [torso.global_position.x, torso.global_position.z, atan2(f.x, f.z)],
		"radar": PERCEPTION_SCRIPT.food_radar(torso.global_position, f, food_manager.get_positions()),
		"retina": PERCEPTION_SCRIPT.retina(_ss, torso.global_position, f),
		"food_rel": _nearest_rel(food_manager.get_positions(), torso.global_position, f),
	}
	if _water_enabled:
		_snap["water_rel"] = _nearest_rel(water_manager.get_positions(), torso.global_position, f)
	return _snap


# Vraie position (x_right, z_fwd) de la ressource la plus proche en frame agent + présence (1/0).
# Même convention de signe que Perception.food_radar / food_xz_from_radar. present=0 si rien dans la portée.
func _nearest_rel(positions: Array, origin: Vector3, fwd_basis: Vector3) -> Array:
	var fwd := Vector3(fwd_basis.x, 0.0, fwd_basis.z)
	if fwd.length() < 0.001:
		return [0.0, 0.0, 0.0]
	fwd = fwd.normalized()
	var right := Vector3(fwd.z, 0.0, -fwd.x)
	var best := 1e9
	var bx := 0.0
	var bz := 0.0
	for p in positions:
		var to := Vector3(p.x - origin.x, 0.0, p.z - origin.z)
		var d := to.length()
		if d < best:
			best = d
			bz = to.dot(fwd)
			bx = to.dot(right)
	if best > PERCEPTION_SCRIPT.MAX_RANGE:
		return [0.0, 0.0, 0.0]
	return [bx, bz, 1.0]


func _compute_vision() -> Array:
	# Egocentric food radar (the [V] perception slot). Empty if the body isn't ready.
	if _blank_vision:
		return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	var torso: Node3D = agent_instance.bodies.get("torso")
	if torso == null:
		return []
	if _heading_mode:
		return PERCEPTION_SCRIPT.heading_command(torso.global_transform.basis.z, _heading_target)
	if _food_as_command:
		# Deploy the cap as food-nav: emit the SAME clean directional bump it trained on, aimed at the
		# nearest pellet — so "walk toward the commanded heading" composes into "walk toward food".
		return PERCEPTION_SCRIPT.heading_command(torso.global_transform.basis.z, food_manager.nearest_dir(torso.global_position))
	if agent_instance.cpg_enabled:
		if _cpg_planner:
			# Phase 5: send the REAL food radar so the WM planner (server-side) can locate food.
			# The server returns the command; the residual's command-in-vision is rebuilt server-side.
			return PERCEPTION_SCRIPT.food_radar(torso.global_position, torso.global_transform.basis.z, food_manager.get_positions())
		# CPG mode (Phase 2b residual / autopilot): the vision channel carries the COMMAND (vx, omega) so the
		# residual policy KNOWS the asked turn and can keep propelling while the CPG steers. Rest zero-padded.
		return [agent_instance.cpg_command.x, agent_instance.cpg_command.y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	return PERCEPTION_SCRIPT.food_radar(torso.global_position, torso.global_transform.basis.z, food_manager.get_positions())


func _build_action(obs: Dictionary) -> Array[float]:
	if collector_mode == "policy_server" and policy_player.is_server_ready():
		return action_adapter.normalize_action(
			_apply_exploration_noise(policy_player.predict_observation(obs))
		)
	if collector_mode == "policy_json" and policy_player.is_ready():
		var state := _observation_to_state(obs)
		return action_adapter.normalize_action(
			_apply_exploration_noise(policy_player.predict(state))
		)
	var action: Array[float] = []
	for index in range(agent_instance.action_dim):
		# Default fallback: try to stand perfectly straight (action 0.0) with very tiny noise to explore
		action.append(randfn(0.0, 0.05))
	return action_adapter.normalize_action(action)


func _observation_to_state(obs: Dictionary) -> Array[float]:
	var state: Array[float] = []
	var proprio: Array = obs.get("proprio", []) as Array
	for value in proprio:
		state.append(float(value))
	var metrics: Dictionary = obs.get("metrics", {}) as Dictionary
	for key in ["uprightness", "forward_velocity", "torso_tilt", "height", "ground_contact", "effort"]:
		state.append(float(metrics.get(key, 0.0)))
	return state


func _apply_exploration_noise(action: Array[float]) -> Array[float]:
	var sigma := _current_exploration_std()
	if sigma <= 0.0:
		return action
	var noisy_action: Array[float] = []
	for value in action:
		noisy_action.append(float(value) + randfn(0.0, sigma))
	return noisy_action


func _current_exploration_std() -> float:
	if max_episode_steps <= 1:
		return exploration_noise_final
	var progress := clampf(
		float(episode_manager.current_step_id) / float(max_episode_steps - 1),
		0.0,
		1.0
	)
	return lerpf(exploration_noise_initial, exploration_noise_final, progress)
