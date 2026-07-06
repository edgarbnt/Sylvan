extends Node3D
class_name SylvanAgent

@export var action_dim := 18  # HEXAPOD: 6 legs×3 (hip_x,hip_z,knee). No spine. Tripod gait (2026-06-16).
@export var agent_version := "hexapod_v1_18dof"
# QUADRUPED REDESIGN (2026-06-08): the body changed from a bipedal humanoid to a DOG/LIZARD-like
# quadruped. Rationale (owner's call): bipedal balance was a money-pit — the entire project history
# (firststeps→J0→J_walk→survival v1-v7) was dominated by ONE problem, "it falls", and every survival
# objective collapsed because falling is terminal and the gait diffuses. A quadruped has a low COM +
# 4-point support → it does NOT fall → a stable reward gradient → locomotion learns fast → we finally
# reach the NORTH STAR (emergent foraging / survival / the JEPA world-model deciding "hungry→food→go").
# The main trunk is still keyed "torso" so main.gd (food/perception, forward=+basis.z) is untouched.
#
# COMPLIANCE (anti-robocop, kept from the humanoid): joints are back-drivable spring-dampers — a soft
# motor the policy MODULATES, not a rigid position-servo that snaps and holds (uncorrectable "robocop").
@export var motor_max_impulse := 55.0  # back-drivable compliant joints
@export var motor_max_speed := 8.0
@export var kp := 4.0 # Proportional gain — softened for a compliant, springy servo
# PD-TORQUE control (2026-06-16 actuation redesign): τ = kp_t·(target−current) − kd_t·joint_vel, applied
# as a torque (the fast-robot paradigm), emulated via the motor's force-limit so the calibrated signs are
# reused. Enables dynamic/ballistic motion the position-velocity servo can't. env SYLVAN_TORQUE_MODE/KP_T/KD_T.
@export var torque_mode := false
@export var kp_t := 60.0
@export var kd_t := 3.0
@export var action_response_hz := 10.0
@export var settle_time := 0.5 # Grace before fall detection / perturbations kick in
@export var gait_cycle_period := 0.6 # Seconds for one full gait cycle (quadruped trot is quicker)
@export var angular_damp := 0.3
@export var fall_height_threshold := 0.22 # Trunk much lower than the humanoid (rests ~0.45-0.50);
                                          # a real collapse drops the trunk below ~0.22.
@export var fall_uprightness_threshold := 0.3 # Trunk up-axis still points up when standing/walking

# Curriculum "balance reflex" (fading crutch) — kept but a quadruped barely needs it.
#   1) bias joint targets toward the neutral standing pose (angle 0), and
#   2) an ACTIVE upright restoring torque on the trunk.
var reflex_strength := 0.0
@export var balance_restore_gain := 260.0
@export var balance_damp_gain := 40.0

# --- CPG (Central Pattern Generator): command-conditioned walk+turn BY CONSTRUCTION (re-arch Phase 0) ---
# A closed analytic function of (command, gait_phase) -> joint targets. Owns the rhythm AND steering;
# turning = differential stride length (left vs right legs), so it is geometric, not a learned reward
# gradient. When cpg_enabled, step_agent uses cpg_reference(vx,omega) and treats the policy output as a
# small bounded residual (residual_gain, e.g. 0.15) for balance only. residual_gain=0 => pure CPG.
@export var cpg_step_amp := 0.6   # fore-aft hip swing amplitude (action units) at vx = cpg_vx_ref
@export var cpg_lift_amp := 0.6   # knee-lift amplitude (action units; 0..1 -> 0..1.4 rad flexion)
@export var cpg_turn_k := 0.6     # LEGACY skid-steer turn gain: per-side stride MULTIPLIER (1 +/- omega*turn_k),
                                  # coupled to vx, both sides step forward → gentle ARC only (~8.6°/s ceiling).
@export var cpg_turn_amp := 0.8   # decoupled differential-stride amplitude (steps legs so the body can
                                  # rotate). SALAMANDER: 0.8 + spine bend → ~27-34°/s in pure CPG (2× old).
                                  # NEW (2026-06-13) decoupled TANK-turn amplitude: an ADDITIVE per-side stride
                                  # (+/- cpg_turn_amp*omega) INDEPENDENT of vx. Large enough → inner legs step
                                  # BACKWARD (negative stride) → fast pivot, turns even at vx=0. 0 = legacy only.
@export var cpg_yaw_lat := 0.0    # NEW (2026-06-14) RAIBERT lateral-turn gain: hip_z abduction =
                                  # cpg_yaw_lat*omega*lat_side, front/rear legs OPPOSED → yaw couple from
                                  # lateral foot placement (the SOTA turn channel). 0 = hip_z neutral (legacy).
# SALAMANDER spine (2026-06-15): the PRIMARY turn channel. cpg_spine_turn = curvature OFFSET gain —
# the spine bends by (cpg_spine_turn*omega) (action units, ±1 → ±0.6 rad), a STATIC bias toward the
# turn side (research: a constant offset X on the spine steers, sign=direction, ~linear in curvature).
# This bends the whole front segment + its legs around → real yaw, not foot-scrub skid. cpg_spine_amp =
# optional lateral undulation amplitude (S-wave, phase-locked); 0 to start (high amplitude hurts steering).
@export var cpg_spine_turn := 1.5
@export var cpg_spine_amp := 0.0   # CONTACT-driven standing-wave undulation amplitude (2026-06-16). >0 bends
                                   # the fore-girdle toward the stance-side foreleg (anchored foot = pivot →
                                   # longer stride). Driven by REAL foot contact, NOT the gait clock.
@export var cpg_spine_sign := 1.0  # flips the undulation phase (empirical: +1 or -1 — the one that EXTENDS
                                   # stride/disp, not scrubs). env SYLVAN_CPG_SPINESIGN.
@export var cpg_vx_ref := 0.4     # reference forward speed at which stride == cpg_step_amp
@export var cpg_lift_phase := 0.75 # (legacy sinusoid knob — unused by the stance/swing CPG below)
@export var cpg_duty := 0.6        # STANCE fraction of the gait cycle (foot planted, pushing). The rest is
                                  # SWING (foot lifted, returning). Explicit stance/swing = real propulsion.
var cpg_enabled := false
var residual_gain := 0.0
# FULLY-LEARNED mode (2026-06-14): bypass the CPG entirely — the policy outputs the 12 joint targets
# DIRECTLY (scaled around the neutral standing pose, legged_gym style). cpg_enabled stays true so the
# command plumbing (sampling, obs, reward cmd) keeps working; only the motor application changes.
var learned_mode := false
@export var learned_action_scale := 0.8   # offset amplitude around neutral (legged_gym uses ~0.25; our servo needs more)
# IMITATION-BOOTSTRAP blend (2026-06-14, deep-research #4): applied = (1-blend)*CPG + blend*learned.
# blend=0 → pure CPG (fluid walker prior), blend=1 → pure learned (no ceiling). Curriculum 0.3→1.0
# (warm-started) bootstraps the learned policy from the CPG gait → escapes the from-scratch standstill.
@export var learned_blend := 1.0
# BC bootstrap: the FINAL applied joint targets (the 12 values step_agent feeds the PD), exposed so a
# residual7 rollout can be logged as (obs → applied) for behavior-cloning the learned policy on the
# GOOD CPG+residual gait (the open-loop CPG alone is too wobbly to clone).
var latest_applied_action: Array[float] = []
# DeepMimic-style gait imitation: how close the learned policy's action is to the CPG reference gait
# (a MOVING target). exp(-meanSE/sigma) ∈ (0,1]; 1 = matches the CPG gait, low = deviates (e.g. standing
# still doesn't match the moving gait → kills the standstill drift). A reward term anchors fluidity.
var gait_imitation := 1.0
const GAIT_IMIT_SIGMA := 0.1
var cpg_command := Vector2.ZERO   # (vx, omega) command, set per episode/segment by main.gd

# KINEMATIC body (pivot corps différentiel, 2026-07-06) — court-circuite le CPG/pattes : le corps GLISSE
# rigidement à (vx, omega) (roues invisibles). Les pattes restent gelées en pose neutre (statue qui glisse)
# → proprio 132 cohérente (angles neutres, vitesses jointes nulles), tout le contrat obs/WM/torso préservé.
# Locomotion = prérequis DONNÉ (pas apprise). Gate SYLVAN_KINEMATIC ; vitesse/rotation tunables.
var kinematic_mode := false
var kin_yaw := 0.0                 # cap intégré (rad)
var kin_speed := 0.5               # m/s par unité vx (SYLVAN_KIN_SPEED ; ~régime hexapode à vx≈1)
var kin_turn := 1.5                # rad/s par unité omega (SYLVAN_KIN_TURN ; ~86°/s à omega≈1)
# Voluntary gait MODULATION (the body is NOT a prison): dynamic multipliers the brain (env now, the JEPA
# planner / a policy later) sets to take big/small steps, run, or change knee bend — like an animal that
# modulates its CPG, not a fixed clip. Steering-by-construction is INVARIANT to these (it lives in the
# L/R stride differential), so this freedom costs nothing. Defaults 1.0 = the tuned nominal gait.
var cpg_stride_scale := 1.0    # big <-> small steps
var cpg_cadence_scale := 1.0   # slow walk <-> run (gait-clock rate)
@export var cpg_speed_cadence_k := 0.0  # SPEED (2026-06-16): when >0, gait FREQUENCY scales with commanded
                                        # vx so stride AND frequency rise TOGETHER (the natural speed path —
                                        # cranking cadence alone destabilised). env SYLVAN_CPG_SPEEDCAD.
var cpg_lift_scale := 1.0      # knee bend / foot clearance

var latest_action: Array[float] = []
var latest_proprio: Array[float] = []
var filtered_action: Array[float] = []
var forward_velocity := 0.0
var horizontal_speed_metric := 0.0  # |trunk horizontal velocity| — leading instability indicator
var com_support_offset_metric := 0.0  # horizontal COM<->feet-centroid distance — accumulating drift
var foot_contacts: Array[float] = []  # per-foot ground contact for the 4 feet [fl,fr,bl,br] (reward-only)
var foot_heights: Array[float] = []   # per-foot height (global y) — reward-only, gait clearance
var foot_speeds_xy: Array[float] = [] # per-foot horizontal speed — reward-only, foot-slip (drag) detection
var left_foot_contact := 0.0   # legacy (kept declared; unused for the quad)
var right_foot_contact := 0.0
var current_joint_angles: Array[float] = []  # the dof angles (reward-only)
var forward_lean := 0.0        # how much the trunk pitches FORWARD (reward-only); >0 = nose-diving
# --- Phase-clock gait state (periodic-gait reward; available for the quad walk reward) ---
var gait_phase := 0.0          # [0,1); advances delta/gait_cycle_period each step (dt-robust)
# legacy bipedal gait-timing vars (kept declared so reset_agent doesn't break; unused for the quad)
var l_foot_speed := 0.0
var r_foot_speed := 0.0
var l_foot_speed_xy := 0.0
var r_foot_speed_xy := 0.0
var l_air_time := 0.0
var r_air_time := 0.0
var l_touchdown := 0.0
var r_touchdown := 0.0
var prev_left_contact := 1.0
var prev_right_contact := 1.0
var center_height := 1.0
var effort := 0.0
var has_fallen := false
var uprightness_metric := 1.0
var torso_tilt_metric := 0.0
var ground_contact_ratio := 1.0

var bodies := {}
var joints := {}
var initial_transforms := {}
var body_material: StandardMaterial3D
var joint_material: StandardMaterial3D
var front_material: StandardMaterial3D  # red front marker (purely visual)

var dof_config = [] # Maps index to [joint_name, axis ("x"/"z"), min_angle, max_angle, neutral_angle]

# HEXAPOD (2026-06-16): a single low elongated trunk "torso" carrying 6 legs in 3 pairs (front/mid/rear).
# 6 contact-monitored feet (the lower leg segments). Order = l1,r1,l2,r2,l3,r3 (left/right × front/mid/rear).
const FEET := ["l1_lower", "r1_lower", "l2_lower", "r2_lower", "l3_lower", "r3_lower"]
# "torso" stays the single trunk (FIRST) — heading + food-radar reference (main.gd uses torso +basis.z).
# 13 bodies = trunk + 6 legs × (upper, lower).
const BODY_NAMES := ["torso", "l1_upper", "l1_lower", "r1_upper", "r1_lower", "l2_upper", "l2_lower", "r2_upper", "r2_lower", "l3_upper", "l3_lower", "r3_upper", "r3_lower"]
# Leg layout: [prefix, x (left -, right +), z (front +, mid 0, back -), parent]. All 6 ride the single trunk.
const LEG_SPECS := [["l1", -0.15, 0.20, "torso"], ["r1", 0.15, 0.20, "torso"], ["l2", -0.15, 0.0, "torso"], ["r2", 0.15, 0.0, "torso"], ["l3", -0.15, -0.20, "torso"], ["r3", 0.15, -0.20, "torso"]]
# SPRAWLING (gecko/salamander) posture: at rest the upper leg is abducted OUTWARD by sprawl_splay rad
# (rolled about the fore-aft axis) so knees/elbows point out to the sides and the belly sits low —
# NOT the dog-like vertical legs. The hip sits on the body SIDE (±SPRAWL_HIPX). Propulsion is still the
# hip PITCH (sweeps the splayed leg fore-aft); the hip ROLL holds the splay; the knee flexes.
var sprawl_splay := 0.45  # lateral abduction at rest (sprawl angle, rad); env SYLVAN_SPRAWL_SPLAY.
                          # Open-loop sweep suggested less splay (0.22) = faster, BUT warm-starting the
                          # sprawled-body policy onto it FELL 20-40% (omni_v7) — the narrow stance needs
                          # FROM-SCRATCH balance training. Reverted to the stable 0.45; de-sprawl is a
                          # from-scratch redesign experiment, not a warm-start tweak.
const SPRAWL_HIPX := 0.10
var mass_scale := 1.0      # global body-mass multiplier (env SYLVAN_MASS_SCALE); <1 = lighter = snappier
var leg_mass_scale := 1.0  # EXTRA multiplier on leg segments (env SYLVAN_LEG_MASS_SCALE); light limbs = fast swing
var SPRAWL_LUP := 0.20    # upper-leg length (env SYLVAN_LEG_UP). Kept stock: open-loop, longer legs added
                          # heavy yaw drift (taller COM) that cancelled the de-sprawl speed gain. De-sprawl alone
                          # is the clean lever; revisit leg length once RL stabilises the de-sprawled body.
var SPRAWL_LLOW := 0.15   # lower-leg length (env SYLVAN_LEG_LOW)

func _ready() -> void:
	# SERVO env-overrides for the turn-ceiling investigation (2026-06-14) — read BEFORE _build_quadruped
	# (motor_max_impulse is baked into the joints at build). Faster/stiffer servo = legs reposition faster
	# → more yaw-per-gait-cycle (the prime universal suspect for the ~14°/s cap). Defaults = original.
	var _e := OS.get_environment("SYLVAN_MOTOR_IMPULSE")
	if _e != "": motor_max_impulse = _e.to_float()
	_e = OS.get_environment("SYLVAN_KP")
	if _e != "": kp = _e.to_float()
	_e = OS.get_environment("SYLVAN_MOTOR_SPEED")
	if _e != "": motor_max_speed = _e.to_float()
	_e = OS.get_environment("SYLVAN_SPRAWL_SPLAY")  # gecko sprawl angle (rad); lower = legs more under body
	if _e != "": sprawl_splay = _e.to_float()
	_e = OS.get_environment("SYLVAN_LEG_UP")    # upper-leg length (redesign speed lever 2026-06-16)
	if _e != "": SPRAWL_LUP = _e.to_float()
	_e = OS.get_environment("SYLVAN_LEG_LOW")   # lower-leg length
	if _e != "": SPRAWL_LLOW = _e.to_float()
	_e = OS.get_environment("SYLVAN_TORQUE_MODE")
	if _e != "": torque_mode = (_e == "1")
	_e = OS.get_environment("SYLVAN_KP_T")
	if _e != "": kp_t = _e.to_float()
	_e = OS.get_environment("SYLVAN_KD_T")
	if _e != "": kd_t = _e.to_float()
	_e = OS.get_environment("SYLVAN_MASS_SCALE")
	if _e != "": mass_scale = _e.to_float()
	_e = OS.get_environment("SYLVAN_LEG_MASS_SCALE")
	if _e != "": leg_mass_scale = _e.to_float()
	_setup_materials()
	_build_quadruped()
	_store_initial_states()

	# DOF mapping (action index -> joint + axis). 3 DOF per leg × 4 legs = 12:
	#   hip_x = sagittal swing (the main locomotion DOF, leg fore/aft),
	#   hip_z = lateral abduction (splay in/out — frontal-plane stability),
	#   knee_x = flexion (hinge). neutral=0 everywhere → action=0 = straight vertical legs = a stable
	# 4-legged "table" that does not topple (the whole point of the morphology change). The walk
	# reward bends them; "do nothing" stands.
	dof_config = []
	for spec in LEG_SPECS:
		var p: String = spec[0]
		# SPRAWLED rest pose: hip ROLL neutral = -sgn·splay (holds the leg abducted out), knee neutral =
		# splay (the upper-vs-lower included angle at rest). hip PITCH neutral 0 (fore-aft swing centred).
		var roll_neutral := -signf(spec[1]) * sprawl_splay
		dof_config.append([p + "_hip", "x", -0.8, 0.8, 0.0])
		dof_config.append([p + "_hip", "z", -1.3, 1.3, roll_neutral])
		dof_config.append([p + "_knee", "x", 0.0, 1.6, sprawl_splay])
	# HEXAPOD: 6 legs × 3 = 18 DOF, no spine.
	_rebuild_proprioception()

func _setup_materials() -> void:
	body_material = StandardMaterial3D.new()
	body_material.albedo_color = Color(0.12, 0.12, 0.14)
	body_material.roughness = 0.5
	body_material.metallic = 0.2

	joint_material = StandardMaterial3D.new()
	joint_material.albedo_color = Color(0.0, 0.65, 1.0)
	joint_material.roughness = 0.2
	joint_material.emission_enabled = true
	joint_material.emission = Color(0.0, 0.4, 0.8)
	joint_material.emission_energy_multiplier = 0.8

	# Red marker for the visual FRONT (+z) of the trunk, so devant/derrière is obvious.
	front_material = StandardMaterial3D.new()
	front_material.albedo_color = Color(0.9, 0.05, 0.05)
	front_material.roughness = 0.3
	front_material.emission_enabled = true
	front_material.emission = Color(0.6, 0.0, 0.0)
	front_material.emission_energy_multiplier = 0.9

func _build_quadruped() -> void:
	# SALAMANDER trunk: TWO segments (front "torso" + rear "torso_back") along z, joined by a lateral
	# spine joint at the middle (z=0). Each segment is short (low yaw inertia); bending the spine swings
	# the front segment + its legs around → turning by spine flexion (breaks the skid yaw ceiling). Legs
	# at the corners hang straight down → wide low stable stance. COM stays low (~0.35) over a 0.30×0.48 base.
	var _sw := OS.get_environment("SYLVAN_STANCE_SCALE")
	var stance := _sw.to_float() if _sw != "" else 1.0
	# GECKO/SALAMANDER trunk: LOW (belly near ground), ELONGATED, NARROW — so the legs sprawl out to the
	# sides (not hang underneath). Two segments joined by the lateral spine; centre height ~0.30.
	# HEXAPOD trunk: ONE low elongated box spanning the 3 leg pairs (z = -0.20..+0.20). No spine.
	bodies["torso"] = _create_body("torso", "box", Vector3(0.18, 0.10, 0.50), 4.0, Vector3(0, 0.30, 0), body_material)

	# Visual-only red disc on the trunk FRONT (+z). Decorative MeshInstance child — no collision/mass.
	var front_disc := MeshInstance3D.new()
	front_disc.name = "front_marker"
	var disc_mesh := CylinderMesh.new()
	disc_mesh.top_radius = 0.07
	disc_mesh.bottom_radius = 0.07
	disc_mesh.height = 0.02
	disc_mesh.material = front_material
	front_disc.mesh = disc_mesh
	# Cylinder axis is local Y; rotate +90deg about X so the flat face points +z (front).
	front_disc.transform = Transform3D(Basis(Vector3(1, 0, 0), PI / 2.0), Vector3(0, 0.0, 0.16))
	bodies["torso"].add_child(front_disc)

	# Eyes: a mount node on the trunk FRONT for the perception slice (vision not wired yet; the food
	# radar in main.gd already uses torso +basis.z). Rotated PI about Y so a child Camera3D's local
	# -z faces the agent's visual front (+z).
	var eyes := Marker3D.new()
	eyes.name = "eyes"
	eyes.transform = Transform3D(Basis(Vector3.UP, PI), Vector3(0, 0.12, 0.15))
	bodies["torso"].add_child(eyes)

	# Build the 4 SPRAWLED legs (upper + lower). The upper leg is abducted OUTWARD by sprawl_splay
	# (knee/elbow points out to the side, gecko-style); the lower leg drops straight down to the foot.
	for spec in LEG_SPECS:
		var p: String = spec[0]
		var lz: float = spec[2]
		var hip := Vector3(signf(spec[1]) * SPRAWL_HIPX * stance, 0.30, lz)
		var udir := Vector3(signf(spec[1]) * sin(sprawl_splay), -cos(sprawl_splay), 0.0)  # hip -> knee
		# upper: a capsule lying along udir (rolled about z by ±splay), from the hip outward+down.
		bodies[p + "_upper"] = _create_body(p + "_upper", "capsule", Vector2(0.045, SPRAWL_LUP), 1.0, hip + udir * (SPRAWL_LUP * 0.5), body_material)
		bodies[p + "_upper"].rotation = Vector3(0, 0, signf(spec[1]) * sprawl_splay)
		# lower (the foot): vertical capsule from the knee straight down, tip ~ground.
		var knee := hip + udir * SPRAWL_LUP
		bodies[p + "_lower"] = _create_body(p + "_lower", "capsule", Vector2(0.04, SPRAWL_LLOW), 0.6, knee + Vector3(0, -SPRAWL_LLOW * 0.5, 0), joint_material)

	for foot_name in FEET:
		var foot: RigidBody3D = bodies[foot_name]
		foot.contact_monitor = true
		foot.max_contacts_reported = 4

	var rot_zero = Vector3.ZERO
	# Knee hinge axis: same orientation as the humanoid sagittal knee (-90deg about Y) so the lower
	# leg swings fore/aft in the sagittal plane. All 4 knees bend the same way for the first version
	# (front/back asymmetry can be refined later from the owner's visual).
	var rot_hinge = Vector3(0, -PI / 2, 0)

	for spec in LEG_SPECS:
		var p: String = spec[0]
		var lz: float = spec[2]
		var parent_seg: RigidBody3D = bodies[spec[3]]  # front legs -> "torso", rear legs -> "torso_back"
		var hip := Vector3(signf(spec[1]) * SPRAWL_HIPX * stance, 0.30, lz)
		var udir := Vector3(signf(spec[1]) * sin(sprawl_splay), -cos(sprawl_splay), 0.0)
		var knee := hip + udir * SPRAWL_LUP
		# Hip: 6DOF on the body SIDE. x=PITCH (fore-aft swing = propulsion of the splayed leg),
		# z=ROLL (lateral abduction — holds the sprawl; range widened to ±1.3 to fit splay 0.7 + swing).
		joints[p + "_hip"] = _create_6dof_joint(p + "_hip", parent_seg, bodies[p + "_upper"], hip, rot_zero, -0.8, 0.8, -1.3, 1.3)
		# Knee: hinge at the elbow/knee (outer end of the upper leg).
		joints[p + "_knee"] = _create_hinge_joint(p + "_knee", bodies[p + "_upper"], bodies[p + "_lower"], knee, rot_hinge, 0.0, 1.6)

	# HEXAPOD: no spine joint (single rigid trunk).

func _create_body(b_name: String, shape_type: String, size, mass: float, pos: Vector3, mat: StandardMaterial3D) -> RigidBody3D:
	var body = RigidBody3D.new()
	body.name = b_name
	body.transform.origin = pos
	# MASS scaling (2026-06-16 fluidity redesign): the body was ~12.4 kg with HALF the mass in the legs →
	# poor power-to-weight (pâteux) + huge leg inertia (slow swing). Scale down globally + extra on legs
	# (light limbs = fast cadence). env SYLVAN_MASS_SCALE / SYLVAN_LEG_MASS_SCALE.
	var m := mass * mass_scale
	if b_name.ends_with("_upper") or b_name.ends_with("_lower"):
		m *= leg_mass_scale
	body.mass = m
	body.can_sleep = false
	body.collision_layer = 2
	body.collision_mask = 1
	body.linear_damp = 0.05
	# angular_damp directly resists ALL rotation (incl. yaw/turning) — prime suspect for the universal
	# ~15°/s turn ceiling (caps every mechanism). Env-overridable to test (2026-06-14).
	var _ad := OS.get_environment("SYLVAN_ANGULAR_DAMP")
	body.angular_damp = _ad.to_float() if _ad != "" else angular_damp
	body.continuous_cd = true

	var physics_material = PhysicsMaterial.new()
	physics_material.bounce = 0.0
	# Foot friction (env-overridable for the turn/body physics investigation 2026-06-14): high friction
	# (2.4) makes feet GRIP → resists the foot-slip that skid turning needs. Lower it to test if friction
	# is a turn-rate limiter. SYLVAN_FOOT_FRICTION overrides the foot value; body stays 1.0.
	var _ff := OS.get_environment("SYLVAN_FOOT_FRICTION")
	var foot_friction := _ff.to_float() if _ff != "" else 2.4
	physics_material.friction = foot_friction if b_name.ends_with("lower") else 1.0
	body.physics_material_override = physics_material

	var col_shape = CollisionShape3D.new()
	var mesh_inst = MeshInstance3D.new()

	if shape_type == "capsule":
		var cap = CapsuleShape3D.new()
		cap.radius = size.x
		cap.height = size.y
		col_shape.shape = cap
		var cap_mesh = CapsuleMesh.new()
		cap_mesh.radius = size.x
		cap_mesh.height = size.y
		cap_mesh.material = mat
		mesh_inst.mesh = cap_mesh
	elif shape_type == "box":
		var box = BoxShape3D.new()
		box.size = size
		col_shape.shape = box
		var box_mesh = BoxMesh.new()
		box_mesh.size = size
		box_mesh.material = mat
		mesh_inst.mesh = box_mesh

	body.add_child(col_shape)
	body.add_child(mesh_inst)
	add_child(body)
	return body

func _create_hinge_joint(j_name: String, body_a: RigidBody3D, body_b: RigidBody3D, pos: Vector3, rot: Vector3, limit_min: float, limit_max: float) -> HingeJoint3D:
	var joint = HingeJoint3D.new()
	joint.name = j_name
	joint.transform.origin = pos
	joint.rotation = rot
	add_child(joint)
	joint.node_a = joint.get_path_to(body_a)
	joint.node_b = joint.get_path_to(body_b)
	joint.set_flag(HingeJoint3D.FLAG_USE_LIMIT, true)
	joint.set_param(HingeJoint3D.PARAM_LIMIT_LOWER, limit_min)
	joint.set_param(HingeJoint3D.PARAM_LIMIT_UPPER, limit_max)
	joint.set_param(HingeJoint3D.PARAM_LIMIT_SOFTNESS, 0.9)
	joint.set_flag(HingeJoint3D.FLAG_ENABLE_MOTOR, true)
	joint.set_param(HingeJoint3D.PARAM_MOTOR_MAX_IMPULSE, motor_max_impulse)
	joint.set_param(HingeJoint3D.PARAM_MOTOR_TARGET_VELOCITY, 0.0)
	return joint

func _create_6dof_joint(j_name: String, body_a: RigidBody3D, body_b: RigidBody3D, pos: Vector3, rot: Vector3, lx_min: float, lx_max: float, lz_min: float, lz_max: float) -> Generic6DOFJoint3D:
	var joint = Generic6DOFJoint3D.new()
	joint.name = j_name
	joint.transform.origin = pos
	joint.rotation = rot
	add_child(joint)
	joint.node_a = joint.get_path_to(body_a)
	joint.node_b = joint.get_path_to(body_b)

	# Lock linear movements
	joint.set_flag_x(Generic6DOFJoint3D.FLAG_ENABLE_LINEAR_LIMIT, true)
	joint.set_param_x(Generic6DOFJoint3D.PARAM_LINEAR_LOWER_LIMIT, 0)
	joint.set_param_x(Generic6DOFJoint3D.PARAM_LINEAR_UPPER_LIMIT, 0)
	joint.set_flag_y(Generic6DOFJoint3D.FLAG_ENABLE_LINEAR_LIMIT, true)
	joint.set_param_y(Generic6DOFJoint3D.PARAM_LINEAR_LOWER_LIMIT, 0)
	joint.set_param_y(Generic6DOFJoint3D.PARAM_LINEAR_UPPER_LIMIT, 0)
	joint.set_flag_z(Generic6DOFJoint3D.FLAG_ENABLE_LINEAR_LIMIT, true)
	joint.set_param_z(Generic6DOFJoint3D.PARAM_LINEAR_LOWER_LIMIT, 0)
	joint.set_param_z(Generic6DOFJoint3D.PARAM_LINEAR_UPPER_LIMIT, 0)

	# X-axis (Pitch / sagittal swing)
	joint.set_flag_x(Generic6DOFJoint3D.FLAG_ENABLE_ANGULAR_LIMIT, true)
	joint.set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_LOWER_LIMIT, lx_min)
	joint.set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_UPPER_LIMIT, lx_max)
	joint.set_flag_x(Generic6DOFJoint3D.FLAG_ENABLE_MOTOR, true)
	joint.set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_FORCE_LIMIT, motor_max_impulse)

	# Z-axis (Roll / lateral abduction)
	joint.set_flag_z(Generic6DOFJoint3D.FLAG_ENABLE_ANGULAR_LIMIT, true)
	joint.set_param_z(Generic6DOFJoint3D.PARAM_ANGULAR_LOWER_LIMIT, lz_min)
	joint.set_param_z(Generic6DOFJoint3D.PARAM_ANGULAR_UPPER_LIMIT, lz_max)
	joint.set_flag_z(Generic6DOFJoint3D.FLAG_ENABLE_MOTOR, true)
	joint.set_param_z(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_FORCE_LIMIT, motor_max_impulse)

	# Y-axis (Yaw) - Locked
	joint.set_flag_y(Generic6DOFJoint3D.FLAG_ENABLE_ANGULAR_LIMIT, true)
	joint.set_param_y(Generic6DOFJoint3D.PARAM_ANGULAR_LOWER_LIMIT, 0)
	joint.set_param_y(Generic6DOFJoint3D.PARAM_ANGULAR_UPPER_LIMIT, 0)

	return joint

# SALAMANDER spine joint: a 6DOF joint between the two trunk segments with ONLY the Y (yaw / lateral
# bend) angular axis free + motorised; X and Z angular and all 3 linear axes are locked. Measured via
# _get_joint_angle's 6DOF "y" branch (relative-forward yaw between the segments).
func _create_spine_joint(j_name: String, body_a: RigidBody3D, body_b: RigidBody3D, pos: Vector3, y_min: float, y_max: float) -> Generic6DOFJoint3D:
	var joint = Generic6DOFJoint3D.new()
	joint.name = j_name
	joint.transform.origin = pos
	add_child(joint)
	joint.node_a = joint.get_path_to(body_a)
	joint.node_b = joint.get_path_to(body_b)

	# Lock all linear axes.
	joint.set_flag_x(Generic6DOFJoint3D.FLAG_ENABLE_LINEAR_LIMIT, true)
	joint.set_param_x(Generic6DOFJoint3D.PARAM_LINEAR_LOWER_LIMIT, 0)
	joint.set_param_x(Generic6DOFJoint3D.PARAM_LINEAR_UPPER_LIMIT, 0)
	joint.set_flag_y(Generic6DOFJoint3D.FLAG_ENABLE_LINEAR_LIMIT, true)
	joint.set_param_y(Generic6DOFJoint3D.PARAM_LINEAR_LOWER_LIMIT, 0)
	joint.set_param_y(Generic6DOFJoint3D.PARAM_LINEAR_UPPER_LIMIT, 0)
	joint.set_flag_z(Generic6DOFJoint3D.FLAG_ENABLE_LINEAR_LIMIT, true)
	joint.set_param_z(Generic6DOFJoint3D.PARAM_LINEAR_LOWER_LIMIT, 0)
	joint.set_param_z(Generic6DOFJoint3D.PARAM_LINEAR_UPPER_LIMIT, 0)

	# Lock X (pitch) and Z (roll) angular → no dorso-ventral or twist bend (lateral-only spine for now).
	joint.set_flag_x(Generic6DOFJoint3D.FLAG_ENABLE_ANGULAR_LIMIT, true)
	joint.set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_LOWER_LIMIT, 0)
	joint.set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_UPPER_LIMIT, 0)
	joint.set_flag_z(Generic6DOFJoint3D.FLAG_ENABLE_ANGULAR_LIMIT, true)
	joint.set_param_z(Generic6DOFJoint3D.PARAM_ANGULAR_LOWER_LIMIT, 0)
	joint.set_param_z(Generic6DOFJoint3D.PARAM_ANGULAR_UPPER_LIMIT, 0)

	# Y (yaw / lateral bend) = the only free axis, motorised like the hips.
	joint.set_flag_y(Generic6DOFJoint3D.FLAG_ENABLE_ANGULAR_LIMIT, true)
	joint.set_param_y(Generic6DOFJoint3D.PARAM_ANGULAR_LOWER_LIMIT, y_min)
	joint.set_param_y(Generic6DOFJoint3D.PARAM_ANGULAR_UPPER_LIMIT, y_max)
	joint.set_flag_y(Generic6DOFJoint3D.FLAG_ENABLE_MOTOR, true)
	# STRONGER than a leg joint: the spine must hold its commanded curvature (incl. straight=0) firmly
	# against the trot's propulsive/lateral forces, else it goes floppy → the body jackknifes, drifts,
	# and the legs' forward thrust dissipates into spine wobble instead of translation. Env-tunable mult.
	var _sf := OS.get_environment("SYLVAN_SPINE_FORCE")
	var spine_mult := _sf.to_float() if _sf != "" else 12.0  # ×12 holds straightest (sweep 2026-06-15)
	joint.set_param_y(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_FORCE_LIMIT, motor_max_impulse * spine_mult)

	return joint

func _store_initial_states() -> void:
	for b_name in bodies:
		var body: RigidBody3D = bodies[b_name]
		body.force_update_transform()
		initial_transforms[body] = body.global_transform

var reset_timer := 0.0
var previous_joint_angles: Array[float] = []
var joint_velocities: Array[float] = []
var current_pose_error := 0.0

func set_assistance(ratio: float) -> void:
	# Gravity crutch: ratio 0 = full gravity, 1 = weightless. Decayed over cycles.
	var clamped := clampf(ratio, 0.0, 1.0)
	for b_name in bodies:
		var body: RigidBody3D = bodies[b_name]
		body.gravity_scale = 1.0 - clamped


func set_reflex_strength(value: float) -> void:
	reflex_strength = clampf(value, 0.0, 1.0)


# External disturbance: a horizontal impulse on the trunk (perturbation curriculum).
func apply_perturbation(impulse: Vector3) -> void:
	if not bodies.has("torso"):
		return
	var torso: RigidBody3D = bodies["torso"]
	torso.apply_central_impulse(impulse)

func reset_agent(position: Vector3, yaw: float = 0.0, impulse: Vector3 = Vector3.ZERO) -> void:
	global_position = position
	latest_action.clear()
	filtered_action.clear()
	effort = 0.0
	has_fallen = false
	uprightness_metric = 1.0
	torso_tilt_metric = 0.0
	ground_contact_ratio = 1.0
	reset_timer = settle_time
	previous_joint_angles.clear()
	previous_joint_angles.resize(action_dim)
	previous_joint_angles.fill(0.0)
	joint_velocities.clear()
	joint_velocities.resize(action_dim)
	joint_velocities.fill(0.0)
	# Random initial gait phase so the policy sees every instant of the cycle early.
	gait_phase = randf()
	var yaw_basis := Basis(Vector3.UP, yaw)

	for b_name in bodies:
		var body: RigidBody3D = bodies[b_name]
		body.freeze = true

	for b_name in bodies:
		var body: RigidBody3D = bodies[b_name]
		var rid = body.get_rid()
		var base_transform: Transform3D = initial_transforms[body]
		var rotated_origin := yaw_basis * base_transform.origin
		var rotated_basis := yaw_basis * base_transform.basis
		var t := Transform3D(rotated_basis, rotated_origin + position)
		PhysicsServer3D.body_set_state(rid, PhysicsServer3D.BODY_STATE_TRANSFORM, t)
		PhysicsServer3D.body_set_state(rid, PhysicsServer3D.BODY_STATE_LINEAR_VELOCITY, Vector3.ZERO)
		PhysicsServer3D.body_set_state(rid, PhysicsServer3D.BODY_STATE_ANGULAR_VELOCITY, Vector3.ZERO)
		body.global_transform = t
		body.linear_velocity = Vector3.ZERO
		body.angular_velocity = Vector3.ZERO

	for j_name in joints:
		var joint = joints[j_name]
		if joint is HingeJoint3D:
			joint.set_param(HingeJoint3D.PARAM_MOTOR_TARGET_VELOCITY, 0.0)
		elif joint is Generic6DOFJoint3D:
			joint.set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, 0.0)
			joint.set_param_y(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, 0.0)
			joint.set_param_z(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, 0.0)

	for b_name in bodies:
		var body: RigidBody3D = bodies[b_name]
		body.freeze = false

	# KINEMATIC : re-geler tout le rig en mode KINEMATIC (piloté par transform, pas par la physique) →
	# les pattes ne tombent plus / ne traînent plus, l'assemblage suit le glissement (vx, omega) du corps.
	if kinematic_mode:
		kin_yaw = yaw
		for b_name in bodies:
			var body: RigidBody3D = bodies[b_name]
			body.freeze_mode = RigidBody3D.FREEZE_MODE_KINEMATIC
			body.freeze = true

	if impulse.length_squared() > 0.0:
		var torso: RigidBody3D = bodies["torso"]
		torso.apply_central_impulse(impulse)

	_rebuild_proprioception()

func apply_action(action: Array[float]) -> void:
	latest_action = action.duplicate()

func enable_cpg(enabled: bool, res_gain: float) -> void:
	cpg_enabled = enabled
	residual_gain = clampf(res_gain, 0.0, 1.0)

func enable_kinematic(enabled: bool, speed: float, turn: float) -> void:
	kinematic_mode = enabled
	if speed > 0.0:
		kin_speed = speed
	if turn > 0.0:
		kin_turn = turn

func set_cpg_command(vx: float, omega: float) -> void:
	cpg_command = Vector2(vx, omega)

# Voluntary gait modulation — big/small steps, run, knee bend. Set dynamically by the brain (the JEPA
# planner / a policy) the same way (vx, omega) is. 1.0 = nominal. This is what stops the CPG being a cage.
func set_cpg_modulation(stride_scale: float, cadence_scale: float, lift_scale: float) -> void:
	cpg_stride_scale = maxf(0.0, stride_scale)
	cpg_cadence_scale = maxf(0.0, cadence_scale)
	cpg_lift_scale = maxf(0.0, lift_scale)

# Command-conditioned CPG: (vx, omega) + the running gait_phase -> 12 joint TARGETS in action space
# [-1,1]. Diagonal trot (FL+BR in phase, FR+BL half a cycle later); hip and knee in quadrature so each
# foot traces a swing/stance ellipse. STEERING BY CONSTRUCTION: omega scales left vs right stride length
# (skid-steer) — the sign is structural, only the gain (cpg_turn_k) needs calibrating. No neural net.
func cpg_reference(vx: float, omega: float) -> Array[float]:
	var a: Array[float] = []
	a.resize(action_dim)
	a.fill(0.0)
	# HEXAPOD TRIPOD gait. dof_config leg order = [l1,r1,l2,r2,l3,r3] (left/right × front/mid/rear), 3 DOF each.
	# Two ANTIPHASE tripods (always 3 feet down → statically stable, no balance bottleneck):
	#   A = {l1, r2, l3} (offset 0.0),  B = {r1, l2, r3} (offset 0.5).
	var phase_offset: Array[float] = [0.0, 0.5, 0.5, 0.0, 0.0, 0.5]
	var side: Array[float] = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]   # left = +1, right = -1
	var stride := cpg_step_amp * cpg_stride_scale * (vx / maxf(0.01, cpg_vx_ref))
	for li in range(6):
		var ph := TAU * (gait_phase + phase_offset[li])
		# SKID steering: differential stride length L vs R — both sides still step FORWARD (no tank/pivot)
		# → the body arcs while advancing. cpg_turn_k is the only turn gain here (no spine on the hexapod).
		var leg_stride := stride * (1.0 - omega * cpg_turn_k * side[li])
		var hip_x := leg_stride * sin(ph)                                    # fore-aft swing = propulsion
		var lift := cpg_lift_amp * cpg_lift_scale * maxf(0.0, cos(ph + TAU * cpg_lift_phase)) # knee lift in swing
		var base := li * 3
		a[base + 0] = clampf(hip_x, -1.0, 1.0)   # hip_x (sagittal swing)
		# hip_z: optional Raibert lateral-placement turn aid (phase-coupled sine ∝ omega); 0 by default.
		a[base + 1] = clampf(cpg_yaw_lat * omega * side[li] * sin(ph), -1.0, 1.0)
		a[base + 2] = clampf(lift, 0.0, 1.0)     # knee (flexion, positive only)
	return a

# Robust, decoupled joint-angle measurement (geometric, not Euler — avoids gimbal lock):
#   - knee (hinge): unsigned flexion = angle between the upper/lower leg long axes,
#   - hip (6DOF): tilt of the child segment's up-axis in the parent frame, split into
#     pitch (sagittal, X) and roll (lateral/frontal, Z).
# At the rest pose every reading is 0, matching the neutral standing target.
func _get_joint_angle(joint, axis_name: String) -> float:
	var body_a = joint.get_node(joint.node_a) as Node3D
	var body_b = joint.get_node(joint.node_b) as Node3D
	if joint is HingeJoint3D:
		var d: float = clampf(body_a.global_transform.basis.y.dot(body_b.global_transform.basis.y), -1.0, 1.0)
		return acos(d) # knee flexion >= 0
	# 6DOF yaw: rotation about vertical leaves the up-axis fixed, so measure the child FORWARD-axis.
	if axis_name == "y":
		var f: Vector3 = body_a.global_transform.basis.inverse() * body_b.global_transform.basis.z
		return atan2(f.x, f.z) # yaw
	# 6DOF: child up-axis in the parent's local frame (rest = (0,1,0)).
	var u: Vector3 = body_a.global_transform.basis.inverse() * body_b.global_transform.basis.y
	if axis_name == "x":
		return atan2(-u.z, u.y) # pitch (forward/back)
	return atan2(u.x, u.y)      # roll (sideways / lateral)

# COM-frame body velocity (2026-06-16): forward speed along the MEAN heading of the two trunk
# segments + mean yaw rate. Averaging front+back cancels the antiphase spine-undulation wobble that
# corrupts the single front-segment reading. Returns Vector2(forward_speed, yaw_rate).
func _com_metrics() -> Vector2:
	if not (bodies.has("torso") and bodies.has("torso_back")):
		return Vector2(forward_velocity, bodies["torso"].angular_velocity.y if bodies.has("torso") else 0.0)
	var tf: RigidBody3D = bodies["torso"]
	var tb: RigidBody3D = bodies["torso_back"]
	var com_vel := 0.5 * (tf.linear_velocity + tb.linear_velocity)
	var heading := (tf.global_transform.basis.z + tb.global_transform.basis.z)
	heading.y = 0.0
	heading = heading.normalized()
	return Vector2(com_vel.dot(heading), 0.5 * (tf.angular_velocity.y + tb.angular_velocity.y))

func _kinematic_step(delta: float) -> void:
	# Glisse l'assemblage entier rigidement à (vx, omega) — réutilise le placement de reset_agent
	# (PhysicsServer3D.body_set_state + initial_transforms). Poses relatives FIGÉES (pose neutre) → la
	# proprio 132 reste cohérente (angles neutres, vitesses jointes nulles), le WM voit une créature qui glisse.
	var moving := reset_timer <= 0.0
	if not moving:
		reset_timer -= delta
	else:
		kin_yaw += kin_turn * cpg_command.y * delta
	var yaw_basis := Basis(Vector3.UP, kin_yaw)
	var forward := (yaw_basis * Vector3(0.0, 0.0, 1.0)).normalized()
	var vel := (forward * (kin_speed * cpg_command.x)) if moving else Vector3.ZERO
	var angvel := Vector3(0.0, kin_turn * cpg_command.y, 0.0) if moving else Vector3.ZERO
	if moving:
		global_position += vel * delta
	for b_name in bodies:
		var body: RigidBody3D = bodies[b_name]
		var base_transform: Transform3D = initial_transforms[body]
		var t := Transform3D(yaw_basis * base_transform.basis, yaw_basis * base_transform.origin + global_position)
		var rid = body.get_rid()
		PhysicsServer3D.body_set_state(rid, PhysicsServer3D.BODY_STATE_TRANSFORM, t)
		PhysicsServer3D.body_set_state(rid, PhysicsServer3D.BODY_STATE_LINEAR_VELOCITY, vel)
		PhysicsServer3D.body_set_state(rid, PhysicsServer3D.BODY_STATE_ANGULAR_VELOCITY, angvel)
		body.global_transform = t
		body.linear_velocity = vel
		body.angular_velocity = angvel
	# Bookkeeping (corps droit, toujours debout — pas de chute possible).
	forward_velocity = kin_speed * cpg_command.x
	center_height = bodies["torso"].global_position.y - global_position.y
	uprightness_metric = 1.0
	torso_tilt_metric = 0.0
	forward_lean = 0.0
	horizontal_speed_metric = Vector2(vel.x, vel.z).length()
	ground_contact_ratio = 1.0
	has_fallen = false
	# Horloge de démarche COSMÉTIQUE (pilote l'anim du loup + les 2 dims proprio gait).
	if moving:
		var _cad := cpg_cadence_scale
		if cpg_speed_cadence_k > 0.0:
			_cad *= maxf(0.4, 1.0 + cpg_speed_cadence_k * (cpg_command.x / maxf(0.01, cpg_vx_ref) - 1.0))
		gait_phase = fmod(gait_phase + delta * _cad / maxf(0.01, gait_cycle_period), 1.0)
	_rebuild_proprioception()

func step_agent(delta: float) -> void:
	if filtered_action.size() != action_dim:
		filtered_action.resize(action_dim)
		filtered_action.fill(0.0)

	# CORPS CINÉMATIQUE : court-circuite entièrement le CPG + PD + pattes. Le corps obéit exactement
	# à (vx, omega). Le CPG (cpg_reference), le mix résidu et la boucle PD ci-dessous ne sont jamais atteints.
	if kinematic_mode:
		_kinematic_step(delta)
		return

	# Effective joint command: pure policy action, OR (CPG mode) the analytic CPG reference plus a small
	# bounded policy residual. Either way it flows through the SAME LPF + compliant-PD law below.
	var action: Array[float] = []
	action.resize(action_dim)
	action.fill(0.0)
	if learned_mode:
		# FULLY-LEARNED (+ imitation-bootstrap blend): applied = (1-blend)*CPG + blend*learned_target.
		# blend=1 → pure learned (the goal); blend<1 → blended with the CPG gait prior so the agent
		# walks from iter 0 (escapes the from-scratch standstill). Grow blend across warm-started runs.
		var cpg := cpg_reference(cpg_command.x, cpg_command.y)
		var _imit_se := 0.0
		for i in range(action_dim):
			var a := float(latest_action[i]) if i < latest_action.size() else 0.0
			var learned_t := clampf(learned_action_scale * a, -1.0, 1.0)
			action[i] = clampf((1.0 - learned_blend) * cpg[i] + learned_blend * learned_t, -1.0, 1.0)
			_imit_se += (action[i] - cpg[i]) * (action[i] - cpg[i])
		gait_imitation = exp(-(_imit_se / float(action_dim)) / GAIT_IMIT_SIGMA)  # 1 = on the CPG gait
	elif cpg_enabled:
		var cpg := cpg_reference(cpg_command.x, cpg_command.y)
		# TURN-FADED residual (2026-06-15): the RL residual reliably learns to walk STRAIGHT but also to
		# SUPPRESS turning (its deterministic mean freezes when a turn is commanded — confirmed across many
		# runs). The CPG itself turns well by construction (+18°/s). So fade the residual's authority to ~0
		# as |omega| grows: RL refines the straight walk, the CPG owns the turn (pivot-then-go). env SYLVAN_TURN_FADE.
		var _tf := OS.get_environment("SYLVAN_TURN_FADE")
		var turn_fade := _tf.to_float() if _tf != "" else 0.5   # |omega| at which residual gain → 0
		var eff_gain := residual_gain
		if turn_fade > 0.0:
			eff_gain = residual_gain * (1.0 - clampf(absf(cpg_command.y) / turn_fade, 0.0, 1.0))
		for i in range(action_dim):
			var r := float(latest_action[i]) if i < latest_action.size() else 0.0
			action[i] = clampf(cpg[i] + eff_gain * r, -1.0, 1.0)
	else:
		for i in range(action_dim):
			if i < latest_action.size():
				action[i] = float(latest_action[i])

	latest_applied_action = action.duplicate()  # BC: the final applied targets (cpg+residual or learned)

	# PHYSICS PROBE (2026-06-14): apply a direct yaw torque to the torso to measure the body's pure
	# rotational ceiling, independent of any controller. If yaw saturates ~15°/s even at high torque →
	# the body/contact physically caps turning (no RL helps → redesign). If it scales → body can turn
	# fast, the cap is learning/control. SYLVAN_YAW_TORQUE (N·m), default off.
	var _yt := OS.get_environment("SYLVAN_YAW_TORQUE")
	if _yt != "" and bodies.has("torso"):
		bodies["torso"].apply_torque(Vector3(0.0, _yt.to_float(), 0.0))

	if reset_timer > 0.0:
		reset_timer -= delta

	var alpha := clampf(1.0 - exp(-delta * TAU * action_response_hz), 0.0, 1.0)
	effort = 0.0

	# PD controllers (generic over dof_config)
	for i in range(mini(action_dim, dof_config.size())):
		var cfg = dof_config[i]
		var j_name = cfg[0]
		var axis_name = cfg[1]
		var min_angle = cfg[2]
		var max_angle = cfg[3]
		var neutral_angle = cfg[4]
		var joint = joints[j_name]

		var target_action = clampf(float(action[i]), -1.0, 1.0)
		filtered_action[i] = lerpf(filtered_action[i], target_action, alpha)

		# Map [-1,+1] around the NEUTRAL standing angle: action=0 -> neutral, +1 -> max, -1 -> min.
		var act_centered = filtered_action[i]
		var target_angle = neutral_angle + (
			act_centered * (max_angle - neutral_angle) if act_centered >= 0.0
			else act_centered * (neutral_angle - min_angle)
		)

		# Balance reflex (curriculum crutch): pull the target toward neutral proportionally.
		if reflex_strength > 0.0:
			target_angle = lerpf(target_angle, 0.0, reflex_strength)

		var current_angle = _get_joint_angle(joint, axis_name)

		var error = target_angle - current_angle
		var motor_velocity = clampf(error * kp, -motor_max_speed, motor_max_speed)

		if reset_timer > 0.0:
			motor_velocity = 0.0

		if torque_mode:
			# PD-TORQUE: τ = kp_t·error − kd_t·joint_vel, emulated by saturating the motor velocity in τ's
			# direction with the force-limit = |τ| (reuses the calibrated signs, incl. spine-y negation).
			var jvel: float = float(joint_velocities[i]) if i < joint_velocities.size() else 0.0
			var tau: float = 0.0 if reset_timer > 0.0 else (kp_t * error - kd_t * jvel)
			var flim := clampf(absf(tau), 0.0, motor_max_impulse)
			var vdir := motor_max_speed * signf(tau)
			if joint is HingeJoint3D:
				joint.set_param(HingeJoint3D.PARAM_MOTOR_TARGET_VELOCITY, vdir)
				joint.set_param(HingeJoint3D.PARAM_MOTOR_MAX_IMPULSE, flim)
			elif joint is Generic6DOFJoint3D:
				if axis_name == "x":
					joint.set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, vdir)
					joint.set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_FORCE_LIMIT, flim)
				elif axis_name == "y":
					joint.set_param_y(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, -vdir)
					joint.set_param_y(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_FORCE_LIMIT, flim)
				else:
					joint.set_param_z(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, vdir)
					joint.set_param_z(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_FORCE_LIMIT, flim)
			effort += flim / maxf(0.001, motor_max_impulse)
		elif joint is HingeJoint3D:
			joint.set_param(HingeJoint3D.PARAM_MOTOR_TARGET_VELOCITY, motor_velocity)
			effort += absf(motor_velocity) / maxf(0.001, motor_max_speed)
		elif joint is Generic6DOFJoint3D:
			if axis_name == "x":
				joint.set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, motor_velocity)
			elif axis_name == "y":
				# Y (spine yaw): the 6DOF Y-motor's +rotation DECREASES the measured angle atan2(f.x,f.z),
				# so the PD sign must be NEGATED — otherwise it's positive feedback and the spine runs to
				# its limit (the −34°≈−0.6rad drift that bent the body, 2026-06-15).
				joint.set_param_y(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, -motor_velocity)
			else:
				joint.set_param_z(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, motor_velocity)
			effort += absf(motor_velocity) / maxf(0.001, motor_max_speed)

	effort /= max(1.0, float(action_dim))

	var torso: RigidBody3D = bodies["torso"]
	var torso_up := torso.global_transform.basis.y.normalized()
	# +z is the visual front (consistent with main.gd's food radar using torso.basis.z).
	var torso_forward := (torso.global_transform.basis.z).normalized()

	center_height = torso.global_position.y - global_position.y
	forward_velocity = torso.linear_velocity.dot(torso_forward)
	# Forward pitch: torso_forward points horizontal when level, DOWN (negative y) when nose-diving.
	forward_lean = maxf(0.0, -torso_forward.y)
	horizontal_speed_metric = Vector2(torso.linear_velocity.x, torso.linear_velocity.z).length()
	uprightness_metric = clampf(torso_up.dot(Vector3.UP), 0.0, 1.0)
	torso_tilt_metric = 1.0 - uprightness_metric

	# Active balance crutch torque (faded via reflex_strength; a quad barely needs it).
	if reflex_strength > 0.0:
		var restore_axis := torso_up.cross(Vector3.UP)
		var assist_torque := restore_axis * balance_restore_gain - torso.angular_velocity * balance_damp_gain
		torso.apply_torque(reflex_strength * assist_torque)

	# Per-foot ground contact (4 feet) + support centroid.
	foot_contacts.clear()
	foot_heights.clear()
	foot_speeds_xy.clear()
	var n_down := 0.0
	var feet_centroid := Vector3.ZERO
	for fn in FEET:
		var foot: RigidBody3D = bodies[fn]
		var c := 1.0 if foot.get_contact_count() > 0 else 0.0
		foot_contacts.append(c)
		foot_heights.append(foot.global_position.y)
		foot_speeds_xy.append(Vector2(foot.linear_velocity.x, foot.linear_velocity.z).length())
		n_down += c
		feet_centroid += foot.global_position
	feet_centroid /= float(FEET.size())
	ground_contact_ratio = n_down / float(FEET.size())

	# Advance the gait phase in REAL time (dt-robust) once past the settle window.
	if reset_timer <= 0.0:
		# SPEED-coupled cadence: gait frequency rises with commanded vx (stride already scales with vx in
		# cpg_reference) so BOTH rise together — the natural, stable speed path (research 2026-06-16).
		var _cad := cpg_cadence_scale
		if cpg_speed_cadence_k > 0.0:
			_cad *= maxf(0.4, 1.0 + cpg_speed_cadence_k * (cpg_command.x / maxf(0.01, cpg_vx_ref) - 1.0))
		gait_phase = fmod(gait_phase + delta * _cad / maxf(0.01, gait_cycle_period), 1.0)

	# Horizontal offset of the COM proxy (trunk) from the support centroid — accumulates as the
	# body drifts/leans, a leading topple signal that a corrective step reduces.
	var com_off := torso.global_position - feet_centroid
	com_support_offset_metric = Vector2(com_off.x, com_off.z).length()

	# Joint velocities + pose error (generic over dof_config)
	current_pose_error = 0.0
	for i in range(mini(action_dim, dof_config.size())):
		var joint = joints[dof_config[i][0]]
		var current_angle = _get_joint_angle(joint, dof_config[i][1])
		joint_velocities[i] = (current_angle - previous_joint_angles[i]) / delta
		previous_joint_angles[i] = current_angle
		current_pose_error += absf(current_angle)
	current_pose_error /= float(max(1, action_dim))

	if reset_timer <= 0.0:
		has_fallen = center_height < fall_height_threshold or uprightness_metric < fall_uprightness_threshold
	else:
		has_fallen = false

	_rebuild_proprioception()

func get_proprioception() -> Array[float]:
	return latest_proprio

func get_locomotion_metrics() -> Dictionary:
	var fc: Array = foot_contacts if foot_contacts.size() == 4 else [0.0, 0.0, 0.0, 0.0]
	var fh: Array = foot_heights if foot_heights.size() == 4 else [0.0, 0.0, 0.0, 0.0]
	var fv: Array = foot_speeds_xy if foot_speeds_xy.size() == 4 else [0.0, 0.0, 0.0, 0.0]
	return {
		# 7-key contract (must match LOCOMOTION_METRIC_KEYS in python/sylvan/constants.py)
		"uprightness": uprightness_metric,
		"forward_velocity": forward_velocity,
		"torso_tilt": torso_tilt_metric,
		"height": center_height,
		"ground_contact": ground_contact_ratio,
		"effort": effort,
		"pose_error": current_pose_error,
		# reward-only SCALARS (NOT arrays — buffer stores metrics as float)
		# Base velocities for the SOTA command-tracking reward (legged_gym/WTW penalties):
		"yaw_rate": bodies["torso"].angular_velocity.y if bodies.has("torso") else 0.0,           # ω_z (yaw, FRONT segment)
		# COM-FRAME velocities (2026-06-16): the front-segment forward_velocity/yaw_rate are whipped by the
		# spine S-wave (the "use COM not IMU" pitfall — research_omnidirectional_locomotion.md). Averaging the
		# two trunk segments cancels the antiphase wobble → a clean body-frame signal for the tracking reward.
		"com_fwd_v": _com_metrics().x,
		"com_yaw_rate": _com_metrics().y,
		# COMMAND (vx, omega) — REQUIRED by the tracking rewards (track_v2/sprawl_v1). These were MISSING
		# → metrics.get("cmd_omega") silently returned 0.0 → the reward always wanted yaw=0 (go straight)
		# even when a turn was commanded → turning could never be learned. (Bug found 2026-06-15.)
		"cmd_vx": cpg_command.x,
		"cmd_omega": cpg_command.y,
		"spine_angle": 0.0,     # HEXAPOD: no spine
		"lin_vel_z": bodies["torso"].linear_velocity.y if bodies.has("torso") else 0.0,           # vertical bob
		"ang_vel_x": bodies["torso"].angular_velocity.x if bodies.has("torso") else 0.0,          # roll rate
		"ang_vel_z": bodies["torso"].angular_velocity.z if bodies.has("torso") else 0.0,          # pitch rate
		"horizontal_speed": horizontal_speed_metric,
		"com_support_offset": com_support_offset_metric,
		"forward_lean": forward_lean,
		"gait_phase": gait_phase,
		"gait_imitation": gait_imitation,   # DeepMimic: 1 = learned action matches the CPG gait
		"n_feet_down": ground_contact_ratio * 4.0,
		"fl_contact": fc[0],
		"fr_contact": fc[1],
		"bl_contact": fc[2],
		"br_contact": fc[3],
		"fl_foot_h": fh[0], "fr_foot_h": fh[1], "bl_foot_h": fh[2], "br_foot_h": fh[3],
		"fl_foot_vxy": fv[0], "fr_foot_vxy": fv[1], "bl_foot_vxy": fv[2], "br_foot_vxy": fv[3],
	}

func _rebuild_proprioception() -> void:
	latest_proprio.clear()
	var torso: RigidBody3D = bodies["torso"]

	latest_proprio.append(center_height)
	latest_proprio.append(torso.linear_velocity.x)
	latest_proprio.append(torso.linear_velocity.y)
	latest_proprio.append(torso.linear_velocity.z)
	latest_proprio.append(torso.angular_velocity.x)
	latest_proprio.append(torso.angular_velocity.y)
	latest_proprio.append(torso.angular_velocity.z) # 7

	# Trunk stays FIRST (forward_lean readers depend on proprio[11] = trunk forward.y proxy).
	# 9 bodies × 6 = 54 dims.
	for b in BODY_NAMES:
		var node: RigidBody3D = bodies[b]
		var basis = node.global_transform.basis
		latest_proprio.append(basis.y.x)
		latest_proprio.append(basis.y.y)
		latest_proprio.append(basis.y.z)
		latest_proprio.append(-basis.z.x)
		latest_proprio.append(-basis.z.y)
		latest_proprio.append(-basis.z.z) # 54

	# Per-foot ground contact (4 feet).
	for fn in FEET:
		var foot: RigidBody3D = bodies[fn]
		latest_proprio.append(1.0 if foot.get_contact_count() > 0 else 0.0) # 4

	# Centre of mass relative to the spawn root.
	var com := Vector3.ZERO
	var total_mass := 0.0
	for b_name in bodies:
		var node: RigidBody3D = bodies[b_name]
		com += node.global_position * node.mass
		total_mass += node.mass
	if total_mass > 0.0:
		com /= total_mass
	com -= global_position
	latest_proprio.append(com.x)
	latest_proprio.append(com.y)
	latest_proprio.append(com.z) # 3

	# Joint angles (12) — also cached for reward use.
	current_joint_angles.clear()
	for cfg in dof_config:
		var ang := _get_joint_angle(joints[cfg[0]], cfg[1])
		current_joint_angles.append(ang)
		latest_proprio.append(ang) # 12

	for v in joint_velocities:
		latest_proprio.append(v) # 12

	# Gait phase clock as a POLICY INPUT (sin/cos avoids the wrap discontinuity), appended LAST.
	latest_proprio.append(sin(TAU * gait_phase))
	latest_proprio.append(cos(TAU * gait_phase)) # 2

	# Total = 7 + 54 + 4 + 3 + 12 + 12 + 2 = 94
