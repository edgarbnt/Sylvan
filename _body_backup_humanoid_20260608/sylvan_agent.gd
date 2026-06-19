extends Node3D
class_name SylvanAgent

@export var action_dim := 18
@export var agent_version := "ragdoll_v7_fullbody_18dof"
# COMPLIANCE (anti-robocop): the joints were rigid position-SERVOS — a very strong motor
# (impulse 150 on a ~29 kg body) snapping each joint to its target angle and HOLDING it
# rigidly → "robocop", uncorrectable by any reward. Lowered so the legs are BACK-DRIVABLE:
# they yield to gravity / ground reaction / momentum (a spring-damper) and the policy only
# MODULATES torque. Knees now settle slightly bent under load = relaxed, fluid, shock-absorbing.
@export var motor_max_impulse := 55.0  # was 150 — back-drivable compliant joints
@export var motor_max_speed := 8.0 # Slower max speed to prevent breakdance jumping
@export var kp := 4.0 # Proportional gain — softened (was 6) for a compliant, springy servo
@export var action_response_hz := 10.0
@export var settle_time := 0.5 # Grace before fall detection / perturbations kick in
@export var gait_cycle_period := 0.8 # Seconds for one full gait cycle (L swing + R swing).
                                     # Drives the phase-clock periodic walk reward (Siekmann/Cassie).
@export var angular_damp := 0.3
@export var fall_height_threshold := 0.35 # Relaxed (was 0.45) to avoid instant deaths
@export var fall_uprightness_threshold := 0.2 # Relaxed (was 0.3)

# Curriculum "balance reflex" (the fading crutch). Two combined effects, both
# scaled by reflex_strength and decayed to 0 over cycles by the orchestrator:
#   1) bias joint targets toward the neutral upright pose (angle 0), and
#   2) an ACTIVE upright restoring torque on the torso (an "invisible hand"),
#      because a fixed-pose PD alone cannot balance an inverted pendulum.
# Orthogonal to JEPA: an innate, fading stabiliser; at 0 the agent is on its own.
var reflex_strength := 0.0
@export var balance_restore_gain := 260.0 # Upright restoring torque magnitude
@export var balance_damp_gain := 40.0     # Angular-velocity damping on the torso

var latest_action: Array[float] = []
var latest_proprio: Array[float] = []
var filtered_action: Array[float] = []
var forward_velocity := 0.0
var horizontal_speed_metric := 0.0  # |torso horizontal velocity| — leading balance indicator
var com_support_offset_metric := 0.0  # horizontal COM<->feet-midpoint distance — accumulating drift
var left_foot_contact := 0.0   # per-foot ground contact (reward-only, gait shaping — NOT in the 7-key contract)
var right_foot_contact := 0.0
var current_joint_angles: Array[float] = []  # the 10 dof angles (reward-only, for gait imitation)
var l_foot_fwd_offset := 0.0   # foot position ahead of torso along forward axis (reward-only, "stepped forward")
var r_foot_fwd_offset := 0.0
var forward_lean := 0.0        # how much the torso pitches FORWARD (reward-only); >0 = leaning forward
var l_foot_height := 0.0       # foot height above the base (reward-only); higher = lifted/cleared
var r_foot_height := 0.0
# --- Phase-clock gait state (periodic walk reward, Siekmann/Cassie) ---
# A memoryless MLP policy cannot track phase internally, so the clock is BOTH an
# observation input (sin/cos appended to proprio) AND drives the periodic reward.
var gait_phase := 0.0          # [0,1); advances delta/gait_cycle_period each step (dt-robust)
var l_foot_speed := 0.0        # |left foot 3D velocity| (stance-phase "planted" penalty)
var r_foot_speed := 0.0
var l_foot_speed_xy := 0.0     # |left foot HORIZONTAL velocity| (swing clearance / slip)
var r_foot_speed_xy := 0.0
var l_air_time := 0.0          # seconds the left foot has been airborne (legged_gym feet_air_time)
var r_air_time := 0.0
var l_touchdown := 0.0         # 1.0 the step the left foot first re-contacts (else 0.0)
var r_touchdown := 0.0
var prev_left_contact := 1.0   # previous-step contact for touchdown edge detection
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

var dof_config = [] # Maps index to [joint_name, axis ("x" or "z"), min_angle, max_angle]

func _ready() -> void:
	_setup_materials()
	_build_humanoid()
	_store_initial_states()
	
	# Define DOF mapping (action index to joint and axis)
	# [joint_name, axis, min_angle, max_angle, neutral_angle]
	# neutral_angle = the joint angle of the default pose (action=0 maps here). The LEGS
	# now default to a slightly-flexed ATHLETIC stance (knee 0.3, hip/ankle pitch 0.15;
	# see the block on the leg DOFs below) so the relaxed pose is bent by construction;
	# arms/neck stay at 0 (limbs hanging, head straight). Crucially the knees' range is
	# asymmetric ([0,1.2]) so the old (min+max)/2 mapping put action=0 at 0.6 rad bent →
	# "do nothing" crouched and toppled; centring on the explicit neutral fixes that.
	# LEG DOFs [0..9] — order frozen: the reward-only metrics (j_l_hip_x etc.) read
	# current_joint_angles[0,2,4,5], and diagnose_gait reads knees at proprio angle
	# indices 4,5. The J2 full-body DOFs are APPENDED [10..17] so leg indices never move.
	dof_config = [
		# BENT ATHLETIC NEUTRAL POSE (anti-robocop). A straight standing leg rests its
		# knee against the 0 extension limit — a cheap stable column gravity won't bend,
		# so every policy converges to stiff "robocop" legs. We move the DEFAULT (action=0)
		# pose to a slightly-flexed athletic stance so "relaxed = bent BY CONSTRUCTION":
		# knee +0.3 rad, with hip-pitch +0.15 and ankle-pitch +0.15 each absorbing half the
		# bend. Sign check (_get_joint_angle: pitch = atan2(-u.z, u.y), forward = +z): a
		# torso-vertical crouch tilts thigh and shin FORWARD → both pitches POSITIVE; with
		# L_thigh~=L_shin and equal half-bends the hip stays above the ankle → COM over the
		# (forward-extending) foot. Soft servo (motor_max_impulse 55, kp 4) holds this ready
		# stance instead of snapping the leg straight.
		["left_hip", "x", -0.8, 0.8, 0.15],
		["left_hip", "z", -0.5, 0.5, 0.0],
		["right_hip", "x", -0.8, 0.8, 0.15],
		["right_hip", "z", -0.5, 0.5, 0.0],
		["left_knee", "x", 0.0, 1.2, 0.3],
		["right_knee", "x", 0.0, 1.2, 0.3],
		["left_ankle", "x", -0.5, 0.5, 0.15],
		["left_ankle", "z", -0.3, 0.3, 0.0],
		["right_ankle", "x", -0.5, 0.5, 0.15],
		["right_ankle", "z", -0.3, 0.3, 0.0],
		# --- J2 full body (appended; action_dim 10 -> 18) ---
		# Arms: shoulder pitch (swing fore/aft, the arm-swing that kills the chasse-neige)
		# + shoulder roll (abduct away from body) + elbow flexion (hinge, manipulation-ready).
		["left_shoulder", "x", -1.8, 1.8, 0.0],   # [10]
		["left_shoulder", "z", -0.2, 1.5, 0.0],   # [11]
		["left_elbow", "x", 0.0, 2.4, 0.0],       # [12]
		["right_shoulder", "x", -1.8, 1.8, 0.0],  # [13]
		["right_shoulder", "z", -0.2, 1.5, 0.0],  # [14]
		["right_elbow", "x", 0.0, 2.4, 0.0],      # [15]
		# Neck: pitch (look up/down) + yaw (turn gaze left/right, for future foraging).
		["neck", "x", -0.6, 0.6, 0.0],            # [16] pitch
		["neck", "y", -1.2, 1.2, 0.0]             # [17] yaw
	]
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

	# Red marker for the visual FRONT (+z) of the torso, so devant/derrière is obvious.
	front_material = StandardMaterial3D.new()
	front_material.albedo_color = Color(0.9, 0.05, 0.05)
	front_material.roughness = 0.3
	front_material.emission_enabled = true
	front_material.emission = Color(0.6, 0.0, 0.0)
	front_material.emission_energy_multiplier = 0.9

func _build_humanoid() -> void:
	# Mass distribution favours a LOW centre of mass for intrinsic stability:
	# lighter torso (was 10), heavier legs (was 3/2). J2: arms + head are added on
	# top (light, near the trunk axis) so the COM stays low enough to balance — the
	# hold-neutral sanity check must still survive ~300 steps. Contract grows ONCE
	# here: proprio 74->120, action 10->18 (see constants.py).
	bodies["torso"] = _create_body("torso", "capsule", Vector2(0.15, 0.5), 6.0, Vector3(0, 0.91, 0), body_material)
	# Visual-only red disc on the torso FRONT (+z = the toes/visual front). Decorative
	# MeshInstance child — no collision, no mass, not a RigidBody → proprio/action/COM
	# contract and physics are untouched. Just makes devant/derrière unambiguous.
	var front_disc := MeshInstance3D.new()
	front_disc.name = "front_marker"
	var disc_mesh := CylinderMesh.new()
	disc_mesh.top_radius = 0.06
	disc_mesh.bottom_radius = 0.06
	disc_mesh.height = 0.02
	disc_mesh.material = front_material
	front_disc.mesh = disc_mesh
	# Cylinder axis is local Y; rotate +90 deg about X so the flat face points +z (front).
	front_disc.transform = Transform3D(Basis(Vector3(1, 0, 0), PI / 2.0), Vector3(0, 0, 0.155))
	bodies["torso"].add_child(front_disc)

	bodies["left_thigh"] = _create_body("left_thigh", "capsule", Vector2(0.07, 0.3), 4.0, Vector3(-0.15, 0.51, 0), body_material)
	bodies["left_shin"] = _create_body("left_shin", "capsule", Vector2(0.06, 0.3), 3.0, Vector3(-0.15, 0.21, 0), body_material)
	# Modest feet: a bit longer fore-aft (z) than wide for natural balance, but no
	# clown shoes. The torso restoring torque (not the feet) is the real stabiliser.
	# Feet WIDENED 0.14->0.22 and lengthened 0.28->0.36 (J_walk): standing (double
	# support, feet spread) was fine on narrow feet, but WALKING balances on ONE foot
	# mid-swing — a 14cm-wide foot gave too small a single-support base, so the gait
	# FEET SHRUNK (0.22x0.36 -> 0.17x0.28, -23%; mass 1.8->1.5). The big foot was a PASSIVE
	# stability crutch — a large support polygon lets the agent rest on the foot instead of
	# balancing actively. Smaller feet (esp. shorter fore-aft, the topple direction in walking)
	# force a real ankle/hip balance strategy → more natural, fluid gait. Contract is unchanged
	# (no proprio/action dims), so the walk policy warm-starts and ADAPTS to the smaller base.
	bodies["left_foot"] = _create_body("left_foot", "box", Vector3(0.17, 0.06, 0.28), 1.5, Vector3(-0.15, 0.03, 0.03), joint_material)

	bodies["right_thigh"] = _create_body("right_thigh", "capsule", Vector2(0.07, 0.3), 4.0, Vector3(0.15, 0.51, 0), body_material)
	bodies["right_shin"] = _create_body("right_shin", "capsule", Vector2(0.06, 0.3), 3.0, Vector3(0.15, 0.21, 0), body_material)
	bodies["right_foot"] = _create_body("right_foot", "box", Vector3(0.17, 0.06, 0.28), 1.5, Vector3(0.15, 0.03, 0.03), joint_material)

	for foot_name in ["left_foot", "right_foot"]:
		var foot: RigidBody3D = bodies[foot_name]
		foot.contact_monitor = true
		foot.max_contacts_reported = 4

	# --- J2 full body: arms + head (light, hung from the trunk top) ---
	# Bodies only collide with the ground (layer 2, mask 1), not with each other, so
	# arms next to the torso never jam. Arms hang straight down at neutral (action=0).
	bodies["left_upper_arm"] = _create_body("left_upper_arm", "capsule", Vector2(0.05, 0.28), 1.2, Vector3(-0.22, 0.86, 0), body_material)
	bodies["left_forearm"] = _create_body("left_forearm", "capsule", Vector2(0.045, 0.25), 0.9, Vector3(-0.22, 0.585, 0), joint_material)
	bodies["right_upper_arm"] = _create_body("right_upper_arm", "capsule", Vector2(0.05, 0.28), 1.2, Vector3(0.22, 0.86, 0), body_material)
	bodies["right_forearm"] = _create_body("right_forearm", "capsule", Vector2(0.045, 0.25), 0.9, Vector3(0.22, 0.585, 0), joint_material)
	bodies["head"] = _create_body("head", "capsule", Vector2(0.11, 0.22), 1.2, Vector3(0, 1.34, 0), joint_material)

	var rot_zero = Vector3.ZERO
	# Knee hinge axis FLIPPED (+PI/2 -> -PI/2): with +z as the visual front (see the
	# torso_forward flip), the old axis made the knee bend FORWARD (shin swings +z,
	# bird/backward-knee — verified: mean shin-thigh z = +0.38 over bent frames). The
	# flipped axis bends the knee BACKWARD (heel toward butt, human-like) for +z walking.
	var rot_hinge = Vector3(0, -PI / 2, 0)
	# Elbow bends the OPPOSITE way to the knee (forearm swings FORWARD, not backward),
	# so it needs the opposite hinge axis (+PI/2). Verified from data: with the knee's
	# -PI/2 axis the elbow bent backward (forearm_down.z - upper_down.z = -0.29/-0.38,
	# same sign as the knee = reversed); +PI/2 makes it flex forward (human-like).
	var rot_hinge_arm = Vector3(0, PI / 2, 0)

	joints["left_hip"] = _create_6dof_joint("left_hip", bodies["torso"], bodies["left_thigh"], Vector3(-0.15, 0.66, 0), rot_zero, -0.8, 0.8, -0.5, 0.5)
	joints["right_hip"] = _create_6dof_joint("right_hip", bodies["torso"], bodies["right_thigh"], Vector3(0.15, 0.66, 0), rot_zero, -0.8, 0.8, -0.5, 0.5)
	
	joints["left_knee"] = _create_hinge_joint("left_knee", bodies["left_thigh"], bodies["left_shin"], Vector3(-0.15, 0.36, 0), rot_hinge, 0.0, 1.2)
	joints["right_knee"] = _create_hinge_joint("right_knee", bodies["right_thigh"], bodies["right_shin"], Vector3(0.15, 0.36, 0), rot_hinge, 0.0, 1.2)
	
	joints["left_ankle"] = _create_6dof_joint("left_ankle", bodies["left_shin"], bodies["left_foot"], Vector3(-0.15, 0.06, 0), rot_zero, -0.5, 0.5, -0.3, 0.3)
	joints["right_ankle"] = _create_6dof_joint("right_ankle", bodies["right_shin"], bodies["right_foot"], Vector3(0.15, 0.06, 0), rot_zero, -0.5, 0.5, -0.3, 0.3)

	# --- J2 full body joints ---
	# Shoulders: 6DOF pitch (x, swing fore/aft) + roll (z, abduct). Elbows: hinge,
	# same sagittal axis as the knees. Neck: pitch (x) + yaw (y) via the xy 6DOF helper.
	joints["left_shoulder"] = _create_6dof_joint("left_shoulder", bodies["torso"], bodies["left_upper_arm"], Vector3(-0.20, 1.00, 0), rot_zero, -1.8, 1.8, -0.2, 1.5)
	joints["right_shoulder"] = _create_6dof_joint("right_shoulder", bodies["torso"], bodies["right_upper_arm"], Vector3(0.20, 1.00, 0), rot_zero, -1.8, 1.8, -0.2, 1.5)
	joints["left_elbow"] = _create_hinge_joint("left_elbow", bodies["left_upper_arm"], bodies["left_forearm"], Vector3(-0.22, 0.72, 0), rot_hinge_arm, 0.0, 2.4)
	joints["right_elbow"] = _create_hinge_joint("right_elbow", bodies["right_upper_arm"], bodies["right_forearm"], Vector3(0.22, 0.72, 0), rot_hinge_arm, 0.0, 2.4)
	joints["neck"] = _create_6dof_joint_xy("neck", bodies["torso"], bodies["head"], Vector3(0, 1.20, 0), rot_zero, -0.6, 0.6, -1.2, 1.2)

	# Eyes: a mount node on the head for the perception slice (vision not wired yet).
	# Rotated PI about Y so a child Camera3D's local -z faces the agent's visual front (+z).
	var eyes := Marker3D.new()
	eyes.name = "eyes"
	eyes.transform = Transform3D(Basis(Vector3.UP, PI), Vector3(0, 0.0, 0.11))
	bodies["head"].add_child(eyes)

func _create_body(b_name: String, shape_type: String, size, mass: float, pos: Vector3, mat: StandardMaterial3D) -> RigidBody3D:
	var body = RigidBody3D.new()
	body.name = b_name
	body.transform.origin = pos
	body.mass = mass
	body.can_sleep = false
	body.collision_layer = 2
	body.collision_mask = 1
	body.linear_damp = 0.05
	body.angular_damp = angular_damp
	body.continuous_cd = true

	var physics_material = PhysicsMaterial.new()
	physics_material.bounce = 0.0
	physics_material.friction = 2.4 if b_name.ends_with("foot") else 1.0
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
		
	# X-axis (Pitch)
	joint.set_flag_x(Generic6DOFJoint3D.FLAG_ENABLE_ANGULAR_LIMIT, true)
	joint.set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_LOWER_LIMIT, lx_min)
	joint.set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_UPPER_LIMIT, lx_max)
	joint.set_flag_x(Generic6DOFJoint3D.FLAG_ENABLE_MOTOR, true)
	joint.set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_FORCE_LIMIT, motor_max_impulse)

	# Z-axis (Roll)
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

# Like _create_6dof_joint but frees the X (pitch) and Y (yaw) axes and LOCKS Z.
# Used by the neck so the head can look up/down (x) and turn left/right (y).
func _create_6dof_joint_xy(j_name: String, body_a: RigidBody3D, body_b: RigidBody3D, pos: Vector3, rot: Vector3, lx_min: float, lx_max: float, ly_min: float, ly_max: float) -> Generic6DOFJoint3D:
	var joint = Generic6DOFJoint3D.new()
	joint.name = j_name
	joint.transform.origin = pos
	joint.rotation = rot
	add_child(joint)
	joint.node_a = joint.get_path_to(body_a)
	joint.node_b = joint.get_path_to(body_b)

	# Lock all linear movements
	joint.set_flag_x(Generic6DOFJoint3D.FLAG_ENABLE_LINEAR_LIMIT, true)
	joint.set_flag_y(Generic6DOFJoint3D.FLAG_ENABLE_LINEAR_LIMIT, true)
	joint.set_flag_z(Generic6DOFJoint3D.FLAG_ENABLE_LINEAR_LIMIT, true)
	joint.set_param_x(Generic6DOFJoint3D.PARAM_LINEAR_LOWER_LIMIT, 0)
	joint.set_param_x(Generic6DOFJoint3D.PARAM_LINEAR_UPPER_LIMIT, 0)
	joint.set_param_y(Generic6DOFJoint3D.PARAM_LINEAR_LOWER_LIMIT, 0)
	joint.set_param_y(Generic6DOFJoint3D.PARAM_LINEAR_UPPER_LIMIT, 0)
	joint.set_param_z(Generic6DOFJoint3D.PARAM_LINEAR_LOWER_LIMIT, 0)
	joint.set_param_z(Generic6DOFJoint3D.PARAM_LINEAR_UPPER_LIMIT, 0)

	# X-axis (Pitch) — free
	joint.set_flag_x(Generic6DOFJoint3D.FLAG_ENABLE_ANGULAR_LIMIT, true)
	joint.set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_LOWER_LIMIT, lx_min)
	joint.set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_UPPER_LIMIT, lx_max)
	joint.set_flag_x(Generic6DOFJoint3D.FLAG_ENABLE_MOTOR, true)
	joint.set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_FORCE_LIMIT, motor_max_impulse)

	# Y-axis (Yaw) — free
	joint.set_flag_y(Generic6DOFJoint3D.FLAG_ENABLE_ANGULAR_LIMIT, true)
	joint.set_param_y(Generic6DOFJoint3D.PARAM_ANGULAR_LOWER_LIMIT, ly_min)
	joint.set_param_y(Generic6DOFJoint3D.PARAM_ANGULAR_UPPER_LIMIT, ly_max)
	joint.set_flag_y(Generic6DOFJoint3D.FLAG_ENABLE_MOTOR, true)
	joint.set_param_y(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_FORCE_LIMIT, motor_max_impulse)

	# Z-axis (Roll) — Locked
	joint.set_flag_z(Generic6DOFJoint3D.FLAG_ENABLE_ANGULAR_LIMIT, true)
	joint.set_param_z(Generic6DOFJoint3D.PARAM_ANGULAR_LOWER_LIMIT, 0)
	joint.set_param_z(Generic6DOFJoint3D.PARAM_ANGULAR_UPPER_LIMIT, 0)

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


# External disturbance: a horizontal impulse on the torso. Used by the perturbation
# curriculum to force ACTIVE balance (passive standing no longer suffices). The
# impulse is NOT part of the action vector — it's an unobserved disturbance the
# policy must learn to recover from (incl. by stepping).
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
	# Reference State Initialization (RSI, DeepMimic-style, mocap-free): start at a
	# RANDOM phase so the policy sees every instant of the gait cycle early instead of
	# only learning the start of a stride. Air-time / touchdown accumulators reset.
	gait_phase = randf()
	l_air_time = 0.0
	r_air_time = 0.0
	l_touchdown = 0.0
	r_touchdown = 0.0
	prev_left_contact = 1.0
	prev_right_contact = 1.0
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

	if impulse.length_squared() > 0.0:
		var torso: RigidBody3D = bodies["torso"]
		torso.apply_central_impulse(impulse)

	_rebuild_proprioception()

func apply_action(action: Array[float]) -> void:
	latest_action = action.duplicate()

# Robust, decoupled joint-angle measurement (Euler decomposition gimbal-locked and
# coupled the axes, reporting values far outside the joint limits and breaking the
# leg PD + proprioception). We measure geometrically instead:
#   - knee (hinge): unsigned flexion = angle between the thigh and shin long axes
#     (0 = straight, grows as it bends → matches the [0, 1.2] limit),
#   - hip/ankle (6DOF): tilt of the child segment's up-axis expressed in the parent
#     frame, split into pitch (sagittal, X) and roll (lateral/frontal, Z).
# At the rest pose every reading is 0, matching the neutral upright target.
func _get_joint_angle(joint, axis_name: String) -> float:
	var body_a = joint.get_node(joint.node_a) as Node3D
	var body_b = joint.get_node(joint.node_b) as Node3D
	if joint is HingeJoint3D:
		var d: float = clampf(body_a.global_transform.basis.y.dot(body_b.global_transform.basis.y), -1.0, 1.0)
		return acos(d) # knee flexion >= 0
	# 6DOF yaw (neck): rotation about the vertical leaves the up-axis fixed, so measure
	# the child FORWARD-axis (basis.z) in the parent frame (rest = (0,0,1)).
	if axis_name == "y":
		var f: Vector3 = body_a.global_transform.basis.inverse() * body_b.global_transform.basis.z
		return atan2(f.x, f.z) # yaw (turn left/right)
	# 6DOF: child up-axis in the parent's local frame (rest = (0,1,0)).
	var u: Vector3 = body_a.global_transform.basis.inverse() * body_b.global_transform.basis.y
	if axis_name == "x":
		return atan2(-u.z, u.y) # pitch (forward/back)
	return atan2(u.x, u.y)      # roll (sideways / lateral)

func step_agent(delta: float) -> void:
	if filtered_action.size() != action_dim:
		filtered_action.resize(action_dim)
		filtered_action.fill(0.0)

	var action := latest_action.duplicate()
	if action.size() < action_dim:
		action.resize(action_dim)
		for idx in range(action.size(), action_dim):
			if action[idx] == null:
				action[idx] = 0.0

	if reset_timer > 0.0:
		reset_timer -= delta

	var alpha := clampf(1.0 - exp(-delta * TAU * action_response_hz), 0.0, 1.0)
	effort = 0.0

	# Process PD controllers
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

		# Map [-1,+1] around the NEUTRAL standing angle: action=0 -> neutral (stand),
		# +1 -> max limit, -1 -> min limit. Asymmetric ranges (knees) are handled by
		# scaling each side independently, so 0 is always the upright pose.
		var act_centered = filtered_action[i]
		var target_angle = neutral_angle + (
			act_centered * (max_angle - neutral_angle) if act_centered >= 0.0
			else act_centered * (neutral_angle - min_angle)
		)

		# Balance reflex (curriculum crutch): pull the target toward the neutral
		# upright pose (angle 0 = straight limb) proportionally to reflex_strength.
		# Strong early so the agent stands while the World Model learns the
		# "standing" dynamics; decayed to 0 so control becomes fully learned.
		if reflex_strength > 0.0:
			target_angle = lerpf(target_angle, 0.0, reflex_strength)

		var current_angle = _get_joint_angle(joint, axis_name)

		var error = target_angle - current_angle
		var motor_velocity = clampf(error * kp, -motor_max_speed, motor_max_speed)
		
		if reset_timer > 0.0:
			motor_velocity = 0.0
			
		if joint is HingeJoint3D:
			joint.set_param(HingeJoint3D.PARAM_MOTOR_TARGET_VELOCITY, motor_velocity)
		elif joint is Generic6DOFJoint3D:
			if axis_name == "x":
				joint.set_param_x(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, motor_velocity)
			elif axis_name == "y":
				joint.set_param_y(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, motor_velocity)
			else:
				joint.set_param_z(Generic6DOFJoint3D.PARAM_ANGULAR_MOTOR_TARGET_VELOCITY, motor_velocity)
			
		effort += absf(motor_velocity) / maxf(0.001, motor_max_speed)

	effort /= max(1.0, float(action_dim))

	var torso: RigidBody3D = bodies["torso"]
	var torso_up := torso.global_transform.basis.y.normalized()
	# FLIPPED to +basis.z (was -basis.z): the owner's visual showed the agent walking
	# BACKWARD — "forward" (-z) was the creature's visual BACK, so every locomotion
	# reward had been rewarding backward motion. +z is the visual front. This flips
	# forward_velocity AND the foot_fwd step signal together (consistent). J0/balance
	# is unaffected (it uses speed magnitude + COM offset, not forward sign).
	var torso_forward := (torso.global_transform.basis.z).normalized()
	
	center_height = torso.global_position.y - global_position.y
	forward_velocity = torso.linear_velocity.dot(torso_forward)
	# Forward pitch: torso_forward points horizontal when upright, and DOWN (negative
	# y) when the torso leans forward. forward_lean>0 = leaning forward (owner's visual:
	# "penché en avant en permanence" → COM ahead of feet → topples forward).
	forward_lean = maxf(0.0, -torso_forward.y)
	# Horizontal COM-drift speed (y is up): grows the moment the torso starts to
	# topple, BEFORE uprightness/height react — a leading balance signal.
	horizontal_speed_metric = Vector2(torso.linear_velocity.x, torso.linear_velocity.z).length()
	uprightness_metric = clampf(torso_up.dot(Vector3.UP), 0.0, 1.0)
	torso_tilt_metric = 1.0 - uprightness_metric

	# Active balance crutch: torque that rotates the torso's up-axis back toward
	# world-up (magnitude ~ sin(tilt)), minus angular damping. Faded via reflex_strength.
	if reflex_strength > 0.0:
		var restore_axis := torso_up.cross(Vector3.UP)
		var assist_torque := restore_axis * balance_restore_gain - torso.angular_velocity * balance_damp_gain
		torso.apply_torque(reflex_strength * assist_torque)
	
	var left_foot: RigidBody3D = bodies["left_foot"]
	var right_foot: RigidBody3D = bodies["right_foot"]
	left_foot_contact = 1.0 if left_foot.get_contact_count() > 0 else 0.0
	right_foot_contact = 1.0 if right_foot.get_contact_count() > 0 else 0.0
	ground_contact_ratio = 0.5 * (left_foot_contact + right_foot_contact)

	# --- Phase-clock + gait timing (periodic walk reward) ---
	# Advance the gait phase in REAL time (dt-robust → unaffected by --speedup/--fixed-fps);
	# frozen during the settle window so the clock starts clean when control begins.
	if reset_timer <= 0.0:
		gait_phase = fmod(gait_phase + delta / maxf(0.01, gait_cycle_period), 1.0)
	# Foot velocities: full 3D for the stance "planted" penalty, horizontal for clearance/slip.
	var lv := left_foot.linear_velocity
	var rv := right_foot.linear_velocity
	l_foot_speed = lv.length()
	r_foot_speed = rv.length()
	l_foot_speed_xy = Vector2(lv.x, lv.z).length()
	r_foot_speed_xy = Vector2(rv.x, rv.z).length()
	# feet_air_time (legged_gym): accumulate airborne time, fire a touchdown pulse on the
	# step the foot first re-contacts (reward pays out ∝ how long the swing lasted = stride).
	l_air_time += delta
	r_air_time += delta
	l_touchdown = 1.0 if (left_foot_contact > 0.5 and prev_left_contact < 0.5) else 0.0
	r_touchdown = 1.0 if (right_foot_contact > 0.5 and prev_right_contact < 0.5) else 0.0
	prev_left_contact = left_foot_contact
	prev_right_contact = right_foot_contact
	# Zero the accumulator only once GROUNDED and PAST the touchdown step, so the swing
	# duration is still readable (with touchdown=1) when the reward consumes the metrics.
	if left_foot_contact > 0.5 and l_touchdown < 0.5:
		l_air_time = 0.0
	if right_foot_contact > 0.5 and r_touchdown < 0.5:
		r_air_time = 0.0

	# Forward offset of each foot vs the torso (along torso forward): a LEADING
	# "stepped forward" signal — a real step plants the swing foot AHEAD of the body.
	l_foot_fwd_offset = (left_foot.global_position - torso.global_position).dot(torso_forward)
	r_foot_fwd_offset = (right_foot.global_position - torso.global_position).dot(torso_forward)
	# Foot height above the base (owner's visual: "il traîne les pieds" → no clearance →
	# destabilises). >~0.06 at rest; reward LIFTING the swing foot well above that.
	l_foot_height = left_foot.global_position.y - global_position.y
	r_foot_height = right_foot.global_position.y - global_position.y

	# Horizontal offset of the COM (torso proxy) from the support centre (feet
	# midpoint). Unlike instantaneous speed this ACCUMULATES as the agent leans/
	# drifts, so a slow topple shows up within the world-model's trusted ~30-step
	# horizon (J1: WM faithful to ~30 steps). A corrective STEP — moving a foot
	# under the drifting COM — reduces it, so penalising it rewards active balance.
	var feet_mid := 0.5 * (left_foot.global_position + right_foot.global_position)
	var com_off := torso.global_position - feet_mid
	com_support_offset_metric = Vector2(com_off.x, com_off.z).length()

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
	return {
		"uprightness": uprightness_metric,
		"forward_velocity": forward_velocity,
		"horizontal_speed": horizontal_speed_metric,
		"com_support_offset": com_support_offset_metric,
		"torso_tilt": torso_tilt_metric,
		"height": center_height,
		"ground_contact": ground_contact_ratio,
		"effort": effort,
		"pose_error": current_pose_error,
		# reward-only gait signals (per-foot contact + key joint angles as SCALARS —
		# the buffer stores metrics as float, so NO arrays here); NOT in the 7-key contract.
		"left_contact": left_foot_contact,
		"right_contact": right_foot_contact,
		"j_l_hip_x": current_joint_angles[0] if current_joint_angles.size() > 5 else 0.0,
		"j_r_hip_x": current_joint_angles[2] if current_joint_angles.size() > 5 else 0.0,
		"j_l_knee": current_joint_angles[4] if current_joint_angles.size() > 5 else 0.0,
		"j_r_knee": current_joint_angles[5] if current_joint_angles.size() > 5 else 0.0,
		"l_foot_fwd": l_foot_fwd_offset,
		"r_foot_fwd": r_foot_fwd_offset,
		"forward_lean": forward_lean,
		"l_foot_height": l_foot_height,
		"r_foot_height": r_foot_height,
		# Phase-clock periodic-walk signals (reward-only SCALARS, never arrays).
		"gait_phase": gait_phase,
		"l_foot_speed": l_foot_speed,
		"r_foot_speed": r_foot_speed,
		"l_foot_speed_xy": l_foot_speed_xy,
		"r_foot_speed_xy": r_foot_speed_xy,
		"l_air_time": l_air_time,
		"r_air_time": r_air_time,
		"l_touchdown": l_touchdown,
		"r_touchdown": r_touchdown,
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
	
	# Torso stays FIRST (forward_lean readers depend on proprio[11] = torso forward.y).
	# J2 bodies are APPENDED after the 7 leg bodies → 12 bodies × 6 = 72 dims.
	var body_names = ["torso", "left_thigh", "left_shin", "left_foot", "right_thigh", "right_shin", "right_foot", "head", "left_upper_arm", "left_forearm", "right_upper_arm", "right_forearm"]
	for b in body_names:
		var node: RigidBody3D = bodies[b]
		var basis = node.global_transform.basis
		latest_proprio.append(basis.y.x)
		latest_proprio.append(basis.y.y)
		latest_proprio.append(basis.y.z)
		latest_proprio.append(-basis.z.x)
		latest_proprio.append(-basis.z.y)
		latest_proprio.append(-basis.z.z) # 72 (12 bodies)
		
	var left_foot: RigidBody3D = bodies["left_foot"]
	var right_foot: RigidBody3D = bodies["right_foot"]
	latest_proprio.append(1.0 if left_foot.get_contact_count() > 0 else 0.0)
	latest_proprio.append(1.0 if right_foot.get_contact_count() > 0 else 0.0) # 2

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

	# Local angles for PD (also cached for the gait-imitation reward)
	current_joint_angles.clear()
	for cfg in dof_config:
		var ang := _get_joint_angle(joints[cfg[0]], cfg[1])
		current_joint_angles.append(ang)
		latest_proprio.append(ang) # 18

	for v in joint_velocities:
		latest_proprio.append(v) # 18

	# Gait phase clock as a POLICY INPUT (sin/cos avoids the φ=1→0 wrap discontinuity).
	# APPENDED LAST so every prior index ([79,80] contacts, [88,89] knees, [11] lean) is
	# unchanged — this also lets the Python warm-start zero-pad just these 2 columns. → 122.
	latest_proprio.append(sin(TAU * gait_phase))
	latest_proprio.append(cos(TAU * gait_phase)) # 2

	# Total = 7 + 72 + 2 (contacts [79,80]) + 3 (com) + 18 (angles, knees [88,89]) + 18 = 120
