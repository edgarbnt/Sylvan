extends RefCounted

# Single source of truth for the environment reward (stored in the buffer; the
# World Model's RewardHead learns it; the Controller trains on it in imagination).
#
# Selectable objective via SYLVAN_REWARD_OBJECTIVE (default "locomotion_v4"):
#   - "locomotion_v4"     : J_walk — GAIT shaping (target velocity + single-support
#                           + stance-foot alternation, via per-foot contacts).
#   - "active_balance_v3" : J0 — STATIC balance. Stand still, centred, upright.
#                           Penalises COM drift + horizontal speed (anti-movement).
#   - "locomotion_v2"     : J_walk attempt — forward velocity "faster=better"
#                           (DIVED forward; kept for reference).
#   - "locomotion_v3"     : J_walk — TARGET velocity (~0.6 m/s): hold a walking pace
#                           upright; overshoot earns nothing so the dive dies and
#                           sustained walking is the unique optimum.
# All kept so we never lose an earlier objective (robust / reversible).

var _objective := ""
var _last_sole := -1  # gait state: which foot is the sole ground contact (0=left,1=right,-1=none); reset per episode
var _foot_air := [0.0, 0.0, 0.0, 0.0]    # per-foot consecutive airborne steps (gait-quality v2); reset per episode
var _foot_stance := [0.0, 0.0, 0.0, 0.0] # per-foot consecutive stance steps (gait-quality v2); reset per episode
var _prev_energy := -1.0  # PHASE C survival: last step's energy, to detect EATING (energy rose)
var _prev_food_dist := -1.0  # DIRECTED foraging: last step's distance to nearest food (approach gradient)


func _get_objective() -> String:
	if _objective == "":
		var env_obj := OS.get_environment("SYLVAN_REWARD_OBJECTIVE")
		_objective = env_obj if env_obj != "" else "locomotion_v21"
	return _objective


# Target walking speed (m/s) for the locomotion reward. Env-driven so a training
# CURRICULUM can RAMP it (e.g. 0.1 -> 0.6): a low target makes even a tiny first
# movement score near-max, giving a strong gradient OUT of the standstill basin
# (v4/v5 got stuck standing because at target 0.6 a small step barely paid). Read
# fresh each call so the ramp takes effect per Godot launch.
func _target_velocity() -> float:
	var env_t := OS.get_environment("SYLVAN_TARGET_VELOCITY")
	return float(env_t) if env_t != "" else 0.6


func compute_reward(metrics: Dictionary, energy: float, health: float, done: bool) -> float:
	match _get_objective():
		"active_balance_v3":
			return _reward_active_balance_v3(metrics, done)
		"locomotion_v2":
			return _reward_locomotion_v2(metrics, done)
		"locomotion_v3":
			return _reward_locomotion_v3(metrics, done)
		"locomotion_v4":
			return _reward_locomotion_v4(metrics, done)
		"locomotion_v5":
			return _reward_locomotion_v5(metrics, done)
		"locomotion_v6":
			return _reward_locomotion_v6(metrics, done)
		"locomotion_v7":
			return _reward_locomotion_v7(metrics, done)
		"locomotion_v9":
			return _reward_locomotion_v9(metrics, done)
		"locomotion_v10":
			return _reward_locomotion_v10(metrics, done)
		"locomotion_v11":
			return _reward_locomotion_v11(metrics, done)
		"locomotion_v12":
			return _reward_locomotion_v12(metrics, done)
		"locomotion_v13":
			return _reward_locomotion_v13(metrics, done)
		"locomotion_v16":
			return _reward_locomotion_v16(metrics, done)
		"locomotion_v17":
			return _reward_locomotion_v17(metrics, done)
		"locomotion_v20":
			return _reward_locomotion_v20(metrics, done)
		"locomotion_fb":
			return _reward_locomotion_fb(metrics, done)
		"locomotion_fb_v2":
			return _reward_locomotion_fb_v2(metrics, done)
		"locomotion_periodic":
			return _reward_locomotion_periodic(metrics, done)
		"survival_v1":
			return _reward_survival_v1(metrics, energy, done)
		"locomotion_quad_v1":
			return _reward_locomotion_quad_v1(metrics, done)
		"locomotion_track_v1":
			return _reward_locomotion_track_v1(metrics, done)
		"locomotion_track_v2":
			return _reward_locomotion_track_v2(metrics, done)
		"locomotion_learned_v1":
			return _reward_locomotion_learned_v1(metrics, done)
		"locomotion_spin_v1":
			return _reward_locomotion_spin_v1(metrics, done)
		"locomotion_sprawl_v1":
			return _reward_locomotion_sprawl_v1(metrics, done)
		"locomotion_omni_v1":
			return _reward_locomotion_omni_v1(metrics, done)
		"locomotion_quad_v2":
			return _reward_locomotion_quad_v2(metrics, done)
		"locomotion_quad_v3":
			return _reward_locomotion_quad_v3(metrics, done)
		"survival_quad_v1":
			return _reward_survival_quad_v1(metrics, energy, done)
		"survival_quad_directed":
			return _reward_survival_quad_directed(metrics, energy, done)
		"survival_pure":
			return _reward_survival_pure(metrics, energy, done)
		"heading_quad_v1":
			return _reward_heading_quad_v1(metrics, done)
		"heading_quad_v2":
			return _reward_heading_quad_v2(metrics, done)
		_:
			return _reward_locomotion_v21(metrics, done)


# QUADRUPED locomotion (2026-06-08). The body changed to a 4-legged quadruped (intrinsically
# stable, ~never falls). The bipedal periodic/phase-clock rewards don't apply (they read l/r-foot
# swing timing this body no longer exposes). A quadruped needs almost nothing fancy: reward
# FORWARD progress, gated by staying up, penalise backward + effort. No deep standstill basin to
# escape (walking doesn't risk a fall here), so forward velocity is simply the dominant signal.
func _reward_locomotion_quad_v1(metrics: Dictionary, done: bool) -> float:
	if done:
		return -1.0
	var height := float(metrics.get("height", 0.0))
	var upright := float(metrics.get("uprightness", 0.0))
	var fwd := float(metrics.get("forward_velocity", 0.0))
	var effort := float(metrics.get("effort", 0.0))
	# Gates: trunk not collapsed (rests ~0.50; collapse < ~0.25) and roughly level.
	var h_gate := clampf((height - 0.25) / 0.20, 0.0, 1.0)
	var up_gate := clampf((upright - 0.5) / 0.5, 0.0, 1.0)
	var gate := h_gate * up_gate
	# Forward progress toward a target pace; backward is penalised; standstill ~ neutral.
	var env_v := OS.get_environment("SYLVAN_TARGET_VELOCITY")
	var v_cmd := float(env_v) if env_v != "" else 0.6
	var fwd_clamped := clampf(fwd, -0.2, v_cmd)
	var fwd_reward := 1.2 * fwd_clamped * gate
	var alive := 0.15 * gate
	var effort_pen := 0.12 * clampf(effort, 0.0, 1.0)
	return clampf(alive + fwd_reward - effort_pen, -1.0, 2.0)


# SALAMANDER walk+turn (2026-06-15). ADDITIVE forward + ADDITIVE yaw-tracking — the additive form is
# DELIBERATE: the multiplicative/exp rewards (track_v2, learned_v1) collapsed to ~0 on the wobbly sprawled
# gait → zero gradient → no learning (vloss 0). Linear forward (quad_v1) trained the walk to 0.40 m/s;
# this adds a tent-shaped yaw term peaking at the commanded yaw so the spine+step turn is rewarded WITHOUT
# drowning the forward signal. cmd_omega/yaw_rate read from metrics; forward target from SYLVAN_TARGET_VELOCITY.
const SPRAWL_YAW_CAP := 1.0     # reward yaw_rate up to ~1 rad/s (~57°/s) in the commanded direction
const SPRAWL_W_YAW := 1.5       # yaw weight > forward (turning must pay enough to leave the straight optimum)
const SPRAWL_W_STRAIGHT := 0.6  # penalty on holding the spine bent when NOT turning (kills the bent-curve gait)

func _reward_locomotion_sprawl_v1(metrics: Dictionary, done: bool) -> float:
	if done:
		return -1.0
	var height := float(metrics.get("height", 0.0))
	var upright := float(metrics.get("uprightness", 0.0))
	var fwd := float(metrics.get("forward_velocity", 0.0))
	var yaw_rate := float(metrics.get("yaw_rate", 0.0))
	var cmd_om := float(metrics.get("cmd_omega", 0.0))
	var spine := float(metrics.get("spine_angle", 0.0))
	var effort := float(metrics.get("effort", 0.0))
	var gate := clampf((height - 0.20) / 0.15, 0.0, 1.0) * clampf((upright - 0.5) / 0.5, 0.0, 1.0)
	var env_v := OS.get_environment("SYLVAN_TARGET_VELOCITY")
	var v_cmd := float(env_v) if env_v != "" else 0.5
	var turn_frac := clampf(absf(cmd_om), 0.0, 1.0)   # 0 = straight command, 1 = full turn
	# forward: track the commanded pace, SCALED DOWN when turning hard (can't go full speed in a tight turn).
	var fwd_target := v_cmd * (1.0 - 0.5 * turn_frac)
	var fwd_reward := 1.2 * clampf(fwd, -0.2, fwd_target) * gate
	# yaw: LINEAR reward in the commanded direction (gradient EVERYWHERE → the policy can climb from
	# ~0 toward a fast turn). The previous tent kernel gave 0 reward AND 0 gradient far from the target,
	# so the policy stayed stuck near 0°/s (the turn never learned). Same lesson as linear-forward.
	var yaw_reward: float
	if absf(cmd_om) > 0.05:
		yaw_reward = SPRAWL_W_YAW * clampf(yaw_rate * signf(cmd_om), -0.3, SPRAWL_YAW_CAP) * gate
	else:
		yaw_reward = SPRAWL_W_YAW * clampf(1.0 - absf(yaw_rate), 0.0, 1.0) * gate  # straight: reward NOT turning
	# STRAIGHT-SPINE penalty: when going straight (turn_frac~0) a bent spine is forbidden (it was the
	# bent-curve cheat); when turning, the bend is allowed (penalty fades out with turn_frac).
	var straight_pen := SPRAWL_W_STRAIGHT * absf(spine) * (1.0 - turn_frac)
	var alive := 0.1 * gate
	var effort_pen := 0.1 * clampf(effort, 0.0, 1.0)
	return clampf(alive + fwd_reward + yaw_reward - effort_pen, -1.0, 3.0)


# OMNIDIRECTIONAL command-tracking (2026-06-16, Phase A). Research recipe
# (docs/research_omnidirectional_locomotion.md): EXPONENTIAL kernels on COM-frame velocity tracking
# (lin + yaw), measured at the COM (com_fwd_v/com_yaw_rate) NOT the spine-whipped front segment. The
# exp kernels collapsed BEFORE (sprawl_v1 went linear) ONLY because commands were too hard → err huge
# → exp≈0 everywhere. The scheduled command CURRICULUM (main.gd) keeps |err| achievable so the kernel
# has gradient. sigma kept generous so far-from-target still pulls. A small LINEAR forward bonus hedges
# the speed gradient (the user's priority). turn-fade is OFF for this profile (the residual owns the turn).
const OMNI_SIG_LIN := 0.30      # exp(-err^2/sig); err 0.5 -> 0.43 (still strong gradient)
const OMNI_SIG_YAW := 0.25      # yaw a touch sharper (must commit to the commanded turn)
const OMNI_W_LIN := 1.0
const OMNI_W_YAW := 1.2         # turning is the hard mode + must stay BALANCED vs the speed push so the
                                # policy can't farm reward by going fast-straight & dropping the turn (2026-06-16)
const OMNI_W_LINBONUS := 0.7    # SPEED push (2026-06-16): strong linear "faster is better" gradient —
                                # the exp kernel alone made ~0.2 m/s optimal (faster cost more effort than
                                # it paid). Now reaching the commanded pace dominates. Pairs with low effort_pen.
const OMNI_W_STRAIGHT := 0.4    # forbid a bent spine when going straight (kills the bent-curve cheat)

func _reward_locomotion_omni_v1(metrics: Dictionary, done: bool) -> float:
	if done:
		return -1.0
	var height := float(metrics.get("height", 0.0))
	var upright := float(metrics.get("uprightness", 0.0))
	var fwd := float(metrics.get("com_fwd_v", metrics.get("forward_velocity", 0.0)))
	var yaw := float(metrics.get("com_yaw_rate", metrics.get("yaw_rate", 0.0)))
	var cmd_vx := float(metrics.get("cmd_vx", 0.0))
	var cmd_om := float(metrics.get("cmd_omega", 0.0))
	var spine := float(metrics.get("spine_angle", 0.0))
	var effort := float(metrics.get("effort", 0.0))
	var gate := clampf((height - 0.20) / 0.15, 0.0, 1.0) * clampf((upright - 0.5) / 0.5, 0.0, 1.0)
	# FORWARD: exp tracking of the commanded pace (always on, straight AND turning → forward+turn rewarded).
	var lin_err := fwd - cmd_vx
	var r_lin := exp(-lin_err * lin_err / OMNI_SIG_LIN)
	# TURN tracking + a COUPLING gate that stops the policy farming the speed bonus by ignoring the turn
	# (the user's concern, 2026-06-16). turn_quality = fraction of the COMMANDED yaw achieved (0..1).
	var turning := absf(cmd_om) > 0.05
	var turn_quality := 1.0
	var yaw_reward: float
	if turning:
		turn_quality = clampf(yaw * signf(cmd_om) / maxf(0.05, absf(cmd_om)), 0.0, 1.0)
		yaw_reward = OMNI_W_YAW * turn_quality          # directional: ZERO for not turning when commanded
	else:
		yaw_reward = OMNI_W_YAW * exp(-yaw * yaw / OMNI_SIG_YAW)  # straight: reward NOT turning
	# SPEED bonus GATED by turn_quality: going fast-straight during a commanded turn earns no bonus, so the
	# optimum is forward AND turn TOGETHER (fast-straight-ignore-turn < turn-correct < turn-and-fast).
	var lin_bonus := OMNI_W_LINBONUS * clampf(fwd, -0.2, maxf(cmd_vx, 0.1)) * turn_quality
	var track := (OMNI_W_LIN * r_lin + yaw_reward + lin_bonus) * gate
	var turn_frac := clampf(absf(cmd_om), 0.0, 1.0)
	var straight_pen := OMNI_W_STRAIGHT * absf(spine) * (1.0 - turn_frac)
	var alive := 0.1 * gate
	var effort_pen := 0.02 * clampf(effort, 0.0, 1.0)   # SPEED push: was 0.05 — stop penalising vigour
	# FLUIDITY (2026-06-16): anti-jitter on the residual — penalise the 2nd difference (smoothness_2, the
	# strong lever) + 1st difference (rate). hexapod_v1 looked uncoordinated/"s'emmêle les pinceaux";
	# this pushes a smooth coordinated residual. Moderate weights (don't kill speed); capped.
	var smooth_pen := minf(1.0, 0.18 * float(metrics.get("action_smooth2", 0.0)) \
		+ 0.05 * float(metrics.get("action_rate", 0.0)))
	return clampf(alive + track - effort_pen - straight_pen - smooth_pen, -1.0, 3.5)  # lin_bonus folded into track


# COMMAND-TRACKING locomotion (2026-06-13, Phase 2d-bis). The motor base must FOLLOW the command
# (vx, omega), not just "go fast forward". locomotion_quad_v1 rewards ONLY forward velocity → the
# residual learned to SUPPRESS turning (a sharp turn costs forward speed + effort) → at turn_k=1.0 it
# cancelled the turn entirely (-1.6°/s). Fix: reward TRACKING — forward speed toward vx AND yaw rate
# toward a target proportional to omega. Turning when commanded is now REWARDED, not punished. This is
# motor COMPETENCE (execute the command), NOT a navigation/task prior (the planner's cost stays separate,
# BLUEPRINT §14). yaw_rate = torso.angular_velocity.y (same sign as omega, verified on wm data).
const TRACK_YAW_GAIN := 0.5     # target yaw rate (rad/s) per unit omega: omega=0.6 -> 0.30 rad/s (~17°/s,
                                # ~2× the current 8.6°/s). Monotonic reward up to target so an unreachable
                                # target still pulls toward MAX turn (no "give up" basin).
const TRACK_STRAIGHT_PEN := 1.0 # when omega~0, penalise stray yaw_rate so it walks STRAIGHT.
func _reward_locomotion_track_v1(metrics: Dictionary, done: bool) -> float:
	if done:
		return -1.0
	var height := float(metrics.get("height", 0.0))
	var upright := float(metrics.get("uprightness", 0.0))
	var fwd := float(metrics.get("forward_velocity", 0.0))
	var yaw_rate := float(metrics.get("yaw_rate", 0.0))
	var effort := float(metrics.get("effort", 0.0))
	var cmd_vx := float(metrics.get("cmd_vx", 0.5))
	var cmd_om := float(metrics.get("cmd_omega", 0.0))
	# Gates: trunk up + level (same as v1).
	var gate := clampf((height - 0.25) / 0.20, 0.0, 1.0) * clampf((upright - 0.5) / 0.5, 0.0, 1.0)
	# Forward tracking toward commanded vx (backward penalised, standstill ~neutral).
	var fwd_reward := 1.0 * clampf(fwd, -0.2, cmd_vx) * gate
	# Turn tracking: reward yaw_rate in the COMMANDED direction up to the target; straighten when omega~0.
	var turn_reward := 0.0
	if absf(cmd_om) > 0.05:
		var target := absf(cmd_om) * TRACK_YAW_GAIN
		var aligned := signf(cmd_om) * yaw_rate          # >0 = turning the commanded way
		turn_reward = clampf(aligned / maxf(target, 0.01), -0.3, 1.0)
	else:
		turn_reward = -TRACK_STRAIGHT_PEN * absf(yaw_rate)
	var alive := 0.15 * gate
	var effort_pen := 0.12 * clampf(effort, 0.0, 1.0)
	return clampf(alive + fwd_reward + 1.0 * turn_reward * gate - effort_pen, -1.0, 2.0)


# COMMAND-TRACKING v2 (2026-06-13) — SOTA recipe (legged_gym/Walk-These-Ways), see
# [[sylvan-locomotion-rl-knowledge]]. Two fixes for the wobbly turn that track_v1 (additive) had:
#   (1) MULTIPLICATIVE reward r_total = r_task · exp(-penalties): the task (velocity tracking) is ALWAYS
#       positively rewarded; penalties only DISCOUNT it, never drown it / cause task-abandon (WTW's fix
#       for exactly the ±3 rad/s wobble). (2) exp-kernel velocity tracking (lin + yaw-rate DIRECT) plus
#       the anti-wobble stack — action_rate, the strong action_smoothness_2 (2nd diff of targets),
#       vertical-bob and roll/pitch-rate penalties. Yaw target = cmd_omega · TRK2_YAW_GAIN (rad/s).
const TRK2_SIGMA := 0.25        # linear-velocity tracking kernel width
const TRK2_SIGMA_YAW := 0.35    # yaw kernel a bit MORE tolerant → gentler gradient, less over-correction wobble
const TRK2_YAW_GAIN := 1.0      # target yaw rate (rad/s) per unit omega: omega=0.6 → 0.6 rad/s (~34°/s, ~4× now)
const TRK2_W_LIN := 1.0
const TRK2_W_YAW := 1.0         # equal weight: push turning as hard as forward (turning is our gap)
const TRK2_K_ACTRATE := 0.5     # penalty weights (inside the exp; multiplicative form is robust to exact values)
const TRK2_K_SMOOTH := 1.0      # action_smoothness_2 — the strong anti-wobble lever
const TRK2_K_VZ := 1.0          # vertical bob
const TRK2_K_AXY := 0.3         # roll/pitch rate
func _reward_locomotion_track_v2(metrics: Dictionary, done: bool) -> float:
	if done:
		return -1.0
	var height := float(metrics.get("height", 0.0))
	var upright := float(metrics.get("uprightness", 0.0))
	var fwd := float(metrics.get("forward_velocity", 0.0))
	var yaw_rate := float(metrics.get("yaw_rate", 0.0))
	var cmd_vx := float(metrics.get("cmd_vx", 0.5))
	var cmd_om := float(metrics.get("cmd_omega", 0.0))
	var gate := clampf((height - 0.25) / 0.20, 0.0, 1.0) * clampf((upright - 0.5) / 0.5, 0.0, 1.0)
	# TASK (always ≥0): track forward speed and yaw RATE via exp kernels.
	var lin_err := (fwd - cmd_vx) * (fwd - cmd_vx)
	var yaw_target := cmd_om * TRK2_YAW_GAIN
	var yaw_err := (yaw_rate - yaw_target) * (yaw_rate - yaw_target)
	var r_task := TRK2_W_LIN * exp(-lin_err / TRK2_SIGMA) + TRK2_W_YAW * exp(-yaw_err / TRK2_SIGMA_YAW)
	# PENALTIES (anti-wobble): discount the task multiplicatively, never drown it.
	var vz := float(metrics.get("lin_vel_z", 0.0))
	var wx := float(metrics.get("ang_vel_x", 0.0))
	var wz := float(metrics.get("ang_vel_z", 0.0))
	var pen := TRK2_K_ACTRATE * float(metrics.get("action_rate", 0.0)) \
		+ TRK2_K_SMOOTH * float(metrics.get("action_smooth2", 0.0)) \
		+ TRK2_K_VZ * vz * vz \
		+ TRK2_K_AXY * (wx * wx + wz * wz)
	return gate * r_task * exp(-pen)


# FULLY-LEARNED locomotion (2026-06-14) — NO CPG. The policy outputs the 12 joint targets directly,
# so the GAIT must be generated by the reward (not a CPG). locomotion_learned_v1 = track_v2's command
# tracking (exp kernels fwd→cmd_vx + yaw→cmd_om) and multiplicative anti-wobble, PLUS the gait-generating
# feet_air_time (quad_v2) and a COM-stability anti-fall driver. SOTA recipe (legged_gym/WTW) adapted to
# our velocity servo + multiplicative form. Agility is unlocked by the COMMAND CURRICULUM (yaw range
# expanded at training time), not by the reward. cmd_vx/cmd_omega injected by main.gd (cpg_enabled stays
# true for the command plumbing even though the CPG motor is bypassed in learned_mode).
const LRN_W_FWD := 1.5         # LINEAR forward-progress weight = the from-scratch driver OUT of standstill
                              # (exp kernels over-reward standing still → no gradient with few envs).
const LRN_W_IMIT := 0.4        # DeepMimic imitation: anchors fluidity + kills standstill drift. LOWERED
                              # 1.0→0.4 (2026-06-14 agile-turn run) to FREE the policy to deviate from the
                              # CPG gait and turn harder than the CPG's ~15°/s ceiling (was over-anchored).
const LRN_W_YAW := 1.5         # yaw tracking (exp): RAISED 0.5→1.5 (= forward weight) to push turning as hard
                              # as walking (ft3 only turned ~8°/s; turning is the goal).
const LRN_W_AIR := 0.4         # feet air-time weight (the gait-generating positive term)
const LRN_W_STAB := 0.3        # COM-over-support stability (anti-fall; LOW so it doesn't make standing comfy)
const LRN_K_SLIP := 0.15       # foot-slip penalty (planted foot dragging)
const LRN_AIR_TARGET := 6.0    # target swing duration in physics steps (~0.1 s at 60 Hz)
const LRN_SLIP_NORM := 0.4     # horizontal foot-speed normalisation (m/s)
const LRN_KEYS_C := ["fl_contact", "fr_contact", "bl_contact", "br_contact"]
const LRN_KEYS_V := ["fl_foot_vxy", "fr_foot_vxy", "bl_foot_vxy", "br_foot_vxy"]

# SPIN (2026-06-14) — proof that AGILE turning is achievable (hardware supports >100°/s: body spins to
# 1000°/s under torque, legs peak 180°/s). Past rewards targeted a MODEST yaw (~34°/s) → no incentive to
# go fast. This reward rewards yaw_rate in the COMMANDED direction LINEARLY up to a high cap → strong
# gradient to SUSTAIN fast turning. Gated on upright (don't farm yaw while falling), multiplicative
# anti-wobble. Specialist: low/zero vx (pivot), maximise yaw. If it breaks the ~14°/s ceiling → control
# was the limit, confirmed; then integrate into a full walk+turn controller (or use SAC).
const SPIN_YAW_CAP := 2.0      # rad/s target ceiling (~115°/s) — well within the legs' 180°/s peak
const SPIN_K_VZ := 1.0
const SPIN_K_AXY := 0.2

func _reward_locomotion_spin_v1(metrics: Dictionary, done: bool) -> float:
	if done:
		return -1.0
	var height := float(metrics.get("height", 0.0))
	var upright := float(metrics.get("uprightness", 0.0))
	var gate := clampf((height - 0.25) / 0.20, 0.0, 1.0) * clampf((upright - 0.5) / 0.5, 0.0, 1.0)
	var yaw_rate := float(metrics.get("yaw_rate", 0.0))
	var cmd_om := float(metrics.get("cmd_omega", 0.0))
	var dir := signf(cmd_om) if absf(cmd_om) > 0.01 else 0.0
	# LINEAR reward for turning in the commanded direction (gradient all the way up to the cap), penalise
	# turning the wrong way; when no turn commanded (dir 0), reward NOT turning.
	var spin: float
	if dir != 0.0:
		spin = clampf(yaw_rate * dir, -0.3, SPIN_YAW_CAP)
	else:
		spin = clampf(1.0 - absf(yaw_rate), 0.0, 1.0)
	var vz := float(metrics.get("lin_vel_z", 0.0))
	var wx := float(metrics.get("ang_vel_x", 0.0))
	var wz := float(metrics.get("ang_vel_z", 0.0))
	var pen := SPIN_K_VZ * vz * vz + SPIN_K_AXY * (wx * wx + wz * wz) \
		+ 0.5 * float(metrics.get("action_smooth2", 0.0))
	var alive := 0.1 * gate
	return clampf(alive + gate * spin * exp(-pen), -1.0, 3.0)


func _reward_locomotion_learned_v1(metrics: Dictionary, done: bool) -> float:
	if done:
		_foot_air = [0.0, 0.0, 0.0, 0.0]
		_foot_stance = [0.0, 0.0, 0.0, 0.0]
		return -1.0
	var height := float(metrics.get("height", 0.0))
	var upright := float(metrics.get("uprightness", 0.0))
	var gate := clampf((height - 0.25) / 0.20, 0.0, 1.0) * clampf((upright - 0.5) / 0.5, 0.0, 1.0)
	# TASK (≥0): track commanded forward speed + yaw rate via exp kernels (track_v2 constants).
	var fwd := float(metrics.get("forward_velocity", 0.0))
	var yaw_rate := float(metrics.get("yaw_rate", 0.0))
	var cmd_vx := float(metrics.get("cmd_vx", 0.0))
	var cmd_om := float(metrics.get("cmd_omega", 0.0))
	# FORWARD = LINEAR progress toward the commanded speed (monotone from standstill → strong gradient
	# even with few envs; this is what made the pre-CPG quad walk). Capped at cmd so it doesn't dive past.
	var fwd_reward := LRN_W_FWD * clampf(fwd, -0.2, cmd_vx)
	# YAW = exp kernel tracking the commanded yaw rate (secondary; precise turning once walking).
	var yaw_err := (yaw_rate - cmd_om * TRK2_YAW_GAIN) * (yaw_rate - cmd_om * TRK2_YAW_GAIN)
	var r_task := fwd_reward + LRN_W_YAW * exp(-yaw_err / TRK2_SIGMA_YAW)
	# GAIT: feet air-time forces all 4 legs to CYCLE (no CPG to do it). Pay completed swings at touchdown.
	var air_reward := 0.0
	var slip_pen := 0.0
	for i in range(4):
		var c := float(metrics.get(LRN_KEYS_C[i], 1.0))
		var vxy := float(metrics.get(LRN_KEYS_V[i], 0.0))
		if c < 0.5:
			_foot_air[i] += 1.0
		else:
			if _foot_air[i] > 0.0:
				air_reward += clampf((_foot_air[i] - LRN_AIR_TARGET) / LRN_AIR_TARGET, -1.0, 1.0)
			_foot_air[i] = 0.0
			slip_pen += clampf(vxy / LRN_SLIP_NORM, 0.0, 1.0)
	# Gait bonus earned only WHILE COMMANDED to move (else farming air-time by rearing in place).
	var move_gate := clampf((absf(cmd_vx) + absf(cmd_om)) / 0.15, 0.0, 1.0)
	# STABILITY: keep COM over the foot support (anti-fall — the from-scratch driver).
	var com_off := float(metrics.get("com_support_offset", 0.0))
	var stability := clampf(1.0 - com_off / 0.25, 0.0, 1.0)
	var imitation := float(metrics.get("gait_imitation", 0.0))   # DeepMimic anchor: stay on the CPG gait
	var pos := r_task + LRN_W_IMIT * imitation + LRN_W_AIR * air_reward * move_gate + LRN_W_STAB * stability
	# PENALTIES (multiplicative anti-wobble — discount the task, never drown it).
	var vz := float(metrics.get("lin_vel_z", 0.0))
	var wx := float(metrics.get("ang_vel_x", 0.0))
	var wz := float(metrics.get("ang_vel_z", 0.0))
	var pen := TRK2_K_SMOOTH * float(metrics.get("action_smooth2", 0.0)) \
		+ TRK2_K_ACTRATE * float(metrics.get("action_rate", 0.0)) \
		+ TRK2_K_VZ * vz * vz + TRK2_K_AXY * (wx * wx + wz * wz) + LRN_K_SLIP * slip_pen
	return clampf(gate * pos * exp(-pen), -1.0, 4.0)


# QUADRUPED locomotion v2 (2026-06-09) — KILL THE DRAG. v1 rewards only forward velocity, so the quad
# learned a degenerate front-wheel-drive crawl: the BACK legs never lift (contact ~95%), one held as a
# rigid asymmetric STRUT → a yaw drift symmetry methods could only HALVE (owner saw it: a rear leg
# trailing, extended). v2 == v1 + STANDARD non-chiral gait-quality terms (legged_gym/ANYmal recipe) that
# force all 4 legs to CYCLE, removing the strut (and thus the yaw bias) at the source:
#   - feet AIR-TIME reward (paid at touchdown ∝ swing duration): rewards real strides, punishes hops,
#   - FOOT-SLIP penalty (a planted foot moving horizontally = dragging): punishes a skidding stance foot,
#   - anti-STANCE penalty (a foot in continuous contact too long = a strut): forces it to lift.
const QV2_AIR_TARGET := 6.0    # target swing duration in physics steps (~0.1 s at 60 Hz)
const QV2_SLIP_NORM := 0.4     # horizontal foot-speed normalisation (m/s) for the slip penalty
const QV2_STANCE_MAX := 22.0   # a foot planted longer than this (steps, ~0.37 s) is a strut → penalise
const QV2_KEYS_C := ["fl_contact", "fr_contact", "bl_contact", "br_contact"]
const QV2_KEYS_V := ["fl_foot_vxy", "fr_foot_vxy", "bl_foot_vxy", "br_foot_vxy"]

func _reward_locomotion_quad_v2(metrics: Dictionary, done: bool) -> float:
	if done:
		_foot_air = [0.0, 0.0, 0.0, 0.0]
		_foot_stance = [0.0, 0.0, 0.0, 0.0]
		return -1.0
	var base := _reward_locomotion_quad_v1(metrics, done)   # forward (gated) + alive - effort
	var height := float(metrics.get("height", 0.0))
	var upright := float(metrics.get("uprightness", 0.0))
	var gate := clampf((height - 0.25) / 0.20, 0.0, 1.0) * clampf((upright - 0.5) / 0.5, 0.0, 1.0)
	var air_reward := 0.0
	var slip_pen := 0.0
	var stance_pen := 0.0
	for i in range(4):
		var c := float(metrics.get(QV2_KEYS_C[i], 1.0))
		var vxy := float(metrics.get(QV2_KEYS_V[i], 0.0))
		if c < 0.5:                       # airborne (swing): accumulate air-time
			_foot_air[i] += 1.0
			_foot_stance[i] = 0.0
		else:                              # in contact (stance)
			if _foot_air[i] > 0.0:        # just touched down → pay for the completed swing (signed)
				air_reward += clampf((_foot_air[i] - QV2_AIR_TARGET) / QV2_AIR_TARGET, -1.0, 1.0)
			_foot_air[i] = 0.0
			_foot_stance[i] += 1.0
			slip_pen += clampf(vxy / QV2_SLIP_NORM, 0.0, 1.0)                               # planted but sliding = drag
			stance_pen += clampf((_foot_stance[i] - QV2_STANCE_MAX) / QV2_STANCE_MAX, 0.0, 1.0)  # never-lifting strut
	# Gate ALL gait-quality terms on FORWARD SPEED so they are earned ONLY WHILE WALKING. Without
	# this, the agent farms air-time by REARING UP and waving the front legs in place (fwd_vel→0,
	# front contact→0.15) — a safe non-falling local optimum. With the gate, rearing earns ~0 bonus,
	# so walk-and-cycle (forward term + gated gait-quality) is the unique optimum.
	var fwd := float(metrics.get("forward_velocity", 0.0))
	var fwd_gate := clampf(fwd / 0.25, 0.0, 1.0)
	var gait_quality := 0.4 * air_reward - 0.15 * slip_pen - 0.2 * stance_pen   # air 0.6→0.4: gentler lift (over-lifting destabilised the from-scratch gait)
	# STABILITY (additive, the anti-fall driver): reward keeping the COM over the foot support.
	# From-scratch + gait-quality gave a SYMMETRIC but FALLING gait (fall ~100%); this dense balance
	# term supplies the gradient to stay upright WHILE walking. Gated on upright, NOT on fwd (balance
	# always matters), so it does not reintroduce a standstill/rearing exploit (forward term still pulls).
	var com_off := float(metrics.get("com_support_offset", 0.0))
	var stability := 0.8 * clampf(1.0 - com_off / 0.25, 0.0, 1.0) * gate
	return clampf(base + stability + gait_quality * gate * fwd_gate, -1.0, 2.8)


# QUADRUPED locomotion v3 (2026-06-09) — PROPER DIAGONAL TROT (kill the bound). v2 makes the 4 legs
# cycle but, with no phase schedule, the policy converged to a BOUND (both FRONT legs lift together,
# back legs lazy — owner's visual). v3 adds a phase-clock PERIODIC reward (Siekmann/legged_gym): it
# dictates WHEN each foot swings — diagonal pairs FL+BR vs FR+BL, half a cycle apart. A foot is
# penalised for GROUND CONTACT during its swing window and for SLIDING during its stance window →
# forces the alternating diagonal trot (the policy observes the [sin,cos] clock so it can comply).
# Composed with forward + COM stability so the trot stays upright and moving. legs order [fl,fr,bl,br].
const QV3_KEYS_C := ["fl_contact", "fr_contact", "bl_contact", "br_contact"]
const QV3_KEYS_V := ["fl_foot_vxy", "fr_foot_vxy", "bl_foot_vxy", "br_foot_vxy"]

func _reward_locomotion_quad_v3(metrics: Dictionary, done: bool) -> float:
	if done:
		return -1.0
	var height := float(metrics.get("height", 0.0))
	var upright := float(metrics.get("uprightness", 0.0))
	var fwd := float(metrics.get("forward_velocity", 0.0))
	var effort := float(metrics.get("effort", 0.0))
	var gate := clampf((height - 0.25) / 0.20, 0.0, 1.0) * clampf((upright - 0.5) / 0.5, 0.0, 1.0)
	var env_v := OS.get_environment("SYLVAN_TARGET_VELOCITY")
	var v_cmd := float(env_v) if env_v != "" else 0.3
	var forward_reward := 1.2 * clampf(fwd, -0.2, v_cmd) * gate
	var alive := 0.15 * gate
	var effort_pen := 0.12 * clampf(effort, 0.0, 1.0)
	# Stability: keep the COM over the foot support.
	var com_off := float(metrics.get("com_support_offset", 0.0))
	var stability := 0.6 * clampf(1.0 - com_off / 0.25, 0.0, 1.0) * gate
	# PERIODIC diagonal trot: pair A = FL(0)+BR(3) swings centred on phase 0.0; pair B = FR(1)+BL(2) on 0.5.
	var phase := float(metrics.get("gait_phase", 0.0))
	var swing_a := _swing_indicator(phase, 0.0)
	var swing_b := _swing_indicator(phase, 0.5)
	var c0 := float(metrics.get(QV3_KEYS_C[0], 1.0)); var c1 := float(metrics.get(QV3_KEYS_C[1], 1.0))
	var c2 := float(metrics.get(QV3_KEYS_C[2], 1.0)); var c3 := float(metrics.get(QV3_KEYS_C[3], 1.0))
	var fv0 := float(metrics.get(QV3_KEYS_V[0], 0.0)); var fv1 := float(metrics.get(QV3_KEYS_V[1], 0.0))
	var fv2 := float(metrics.get(QV3_KEYS_V[2], 0.0)); var fv3 := float(metrics.get(QV3_KEYS_V[3], 0.0))
	# Penalise ground contact during a foot's SWING window (it should be airborne then).
	var swing_contact := swing_a * (c0 + c3) + swing_b * (c1 + c2)                      # ~[0,2]
	# Penalise foot horizontal speed during STANCE (a planted foot must NOT slide/drag).
	var stance_slip := clampf(((1.0 - swing_a) * (fv0 + fv3) + (1.0 - swing_b) * (fv1 + fv2)) / 0.4, 0.0, 2.0)
	var periodic := (-0.5 * swing_contact - 0.2 * stance_slip) * gate
	return clampf(alive + forward_reward + stability + periodic - effort_pen, -1.0, 3.0)


# HEADING CONTROL (2026-06-09) — the MISSING motor primitive: turn toward a COMMANDED heading and
# walk to it. With SYLVAN_HEADING_MODE=1 the vision channel carries a target-direction bump (not the
# food radar); `heading_align` = cos between forward and that target. Progress toward the target ≈
# forward_velocity * heading_align (the quad walks along its body forward). Reward = go-toward-target
# + a face-the-target term (bootstraps turning before it can move) + upright/alive - effort. Trained
# alone, then deployed with the food radar in the same channel → "walk toward food" composes for free.
func _reward_heading_quad_v1(metrics: Dictionary, done: bool) -> float:
	if done:
		return -1.0
	var height := float(metrics.get("height", 0.0))
	var upright := float(metrics.get("uprightness", 0.0))
	var fwd_vel := float(metrics.get("forward_velocity", 0.0))
	var effort := float(metrics.get("effort", 0.0))
	var align := float(metrics.get("heading_align", 0.0))   # cos(forward, target) in [-1,1]
	var h_gate := clampf((height - 0.25) / 0.20, 0.0, 1.0)
	var up_gate := clampf((upright - 0.5) / 0.5, 0.0, 1.0)
	var gate := h_gate * up_gate
	var env_v := OS.get_environment("SYLVAN_TARGET_VELOCITY")
	var v_cmd := float(env_v) if env_v != "" else 0.5
	# Progress toward the commanded heading = forward speed × facing. 0 when standing.
	var progress := clampf(fwd_vel, -0.2, v_cmd) * align
	var progress_reward := 1.5 * progress * gate
	var align_reward := 0.2 * align * gate     # small turning gradient to bootstrap facing the target
	# ANTI-FREEZE: the old 0.4*align term could be farmed IMMOBILE (face the target, don't walk) →
	# under symmetry pressure the gait collapsed onto the front legs. Penalise standing so WALKING
	# toward the target is the only way to score.
	var move_pen := (0.35 * gate) if fwd_vel < 0.1 else 0.0
	var alive := 0.05 * gate
	var effort_pen := 0.12 * clampf(effort, 0.0, 1.0)
	return clampf(alive + progress_reward + align_reward - move_pen - effort_pen, -1.0, 2.5)


# HEADING CONTROL v2 (2026-06-09) — train the CAP on the clean fresh2 walker (symmetric, stable, no
# fixed bias). v1 coupled turn-credit to speed (progress=fwd_vel*align → a correct turn that hasn't
# translated yet earned ~0). v2 = a yaw-TRACKER that also PRESERVES the gait: progress-toward-command
# + a speed-independent FACE term (turn credit) + anti-freeze + STABILITY (com) + the gait-quality
# terms (air/slip/stance, fwd-gated) so steering does not erode the cycling gait. Deployed with the
# (smooth) food radar in the same channel → "walk toward food".
func _reward_heading_quad_v2(metrics: Dictionary, done: bool) -> float:
	if done:
		_foot_air = [0.0, 0.0, 0.0, 0.0]
		_foot_stance = [0.0, 0.0, 0.0, 0.0]
		return -1.0
	var height := float(metrics.get("height", 0.0))
	var upright := float(metrics.get("uprightness", 0.0))
	var fwd_vel := float(metrics.get("forward_velocity", 0.0))
	var effort := float(metrics.get("effort", 0.0))
	var align := float(metrics.get("heading_align", 0.0))   # cos(θ_err) in [-1,1], 1 = facing the commanded heading
	var gate := clampf((height - 0.25) / 0.20, 0.0, 1.0) * clampf((upright - 0.5) / 0.5, 0.0, 1.0)
	var env_v := OS.get_environment("SYLVAN_TARGET_VELOCITY")
	var v_cmd := float(env_v) if env_v != "" else 0.3
	# Progress toward the commanded heading (move along body-forward, scaled by facing).
	var progress_reward := 1.2 * clampf(fwd_vel, -0.2, v_cmd) * align * gate
	# Speed-INDEPENDENT face term — pays for TURNING toward the target before it translates (v1's gap).
	var face_reward := 0.4 * align * gate
	# Anti-freeze: don't just face-and-stand.
	var move_pen := (0.3 * gate) if fwd_vel < 0.1 else 0.0
	# STABILITY (carried from the clean walker so steering stays upright).
	var com_off := float(metrics.get("com_support_offset", 0.0))
	var stability := 0.5 * clampf(1.0 - com_off / 0.25, 0.0, 1.0) * gate
	# GAIT-QUALITY (keep the 4 legs cycling while turning), gated on forward speed.
	var air_reward := 0.0
	var slip_pen := 0.0
	var stance_pen := 0.0
	for i in range(4):
		var c := float(metrics.get(QV2_KEYS_C[i], 1.0))
		var vxy := float(metrics.get(QV2_KEYS_V[i], 0.0))
		if c < 0.5:
			_foot_air[i] += 1.0
			_foot_stance[i] = 0.0
		else:
			if _foot_air[i] > 0.0:
				air_reward += clampf((_foot_air[i] - QV2_AIR_TARGET) / QV2_AIR_TARGET, -1.0, 1.0)
			_foot_air[i] = 0.0
			_foot_stance[i] += 1.0
			slip_pen += clampf(vxy / QV2_SLIP_NORM, 0.0, 1.0)
			stance_pen += clampf((_foot_stance[i] - QV2_STANCE_MAX) / QV2_STANCE_MAX, 0.0, 1.0)
	var fwd_gate := clampf(absf(fwd_vel) / 0.25, 0.0, 1.0)
	var gait_quality := (0.4 * air_reward - 0.15 * slip_pen - 0.2 * stance_pen) * gate * fwd_gate
	var alive := 0.05 * gate
	var effort_pen := 0.12 * clampf(effort, 0.0, 1.0)
	return clampf(alive + progress_reward + face_reward + stability + gait_quality - move_pen - effort_pen, -1.0, 3.0)


# QUADRUPED survival = quad walk anchor + an eating spike (same philosophy as survival_v1, but on
# the stable body so it should NOT collapse into freeze). Used once the walk is solid.
func _reward_survival_quad_v1(metrics: Dictionary, energy: float, done: bool) -> float:
	if done:
		_prev_energy = -1.0
		return -1.0
	var gait := _reward_locomotion_quad_v1(metrics, done)
	var eat_bonus := 0.0
	if _prev_energy >= 0.0 and energy > _prev_energy + 0.5:
		eat_bonus = 2.0
	_prev_energy = energy
	return clampf(gait + eat_bonus, -1.0, 4.0)


# DIRECTED FORAGING (2026-06-08) — the first real spark of "hungry -> food there -> GO". On the
# stable quad we can finally pursue this without the freeze trap. The key is a DENSE approach
# gradient: reward SHRINKING the distance to the nearest food (so the agent learns to TURN and
# steer toward it using its food radar), NOT a fixed-direction forward term (that would fight
# turning). Plus the eat spike, an upright gate, a small effort penalty. There is no standstill
# basin: standing earns ~0 approach, so moving toward food strictly dominates.
func _reward_survival_quad_directed(metrics: Dictionary, energy: float, done: bool) -> float:
	if done:
		_prev_energy = -1.0
		_prev_food_dist = -1.0
		return -1.0
	var height := float(metrics.get("height", 0.0))
	var upright := float(metrics.get("uprightness", 0.0))
	var effort := float(metrics.get("effort", 0.0))
	var h_gate := clampf((height - 0.25) / 0.20, 0.0, 1.0)
	var up_gate := clampf((upright - 0.5) / 0.5, 0.0, 1.0)
	var gate := h_gate * up_gate

	# Did it just eat? (energy rose) — a strong immediate reinforcement.
	var ate := _prev_energy >= 0.0 and energy > _prev_energy + 0.5
	var eat_bonus := 2.0 if ate else 0.0
	_prev_energy = energy

	# Approach gradient: positive when the nearest pellet got CLOSER this step. Clamp the per-step
	# change so a food RESPAWN (the eaten pellet relocates far → nearest jumps) can't spike the
	# reward; and on the eat step force it non-negative (don't punish the relocation that eating causes).
	var food_dist := float(metrics.get("nearest_food_dist", -1.0))
	var approach := 0.0
	if _prev_food_dist >= 0.0 and food_dist >= 0.0:
		approach = clampf(_prev_food_dist - food_dist, -0.03, 0.05)
		if ate:
			approach = maxf(approach, 0.0)
	_prev_food_dist = food_dist
	var approach_reward := 25.0 * approach * gate   # dense steering signal (~0.2/step toward food)

	# HEADING term: reward FACING the nearest food (cos alignment in [-1,1]), gated. Dense per-step
	# signal that pays for TURNING toward food IMMEDIATELY (not only after the distance drops) — fixes
	# the "runs straight, turns late" reflex. Modest weight so it orients but still has to GO eat
	# (approach + eat_bonus dominate; energy drain forbids just spinning in place facing food).
	var food_align := float(metrics.get("food_align", 0.0))
	var heading_reward := 0.5 * food_align * gate   # bumped 0.35→0.5: with SPARSE food the radar points at
	                                                # ONE clear target, so paying to FACE it early (turning the
	                                                # correct way) matters more — and a right-only turn that
	                                                # overshoots a left target now loses this term every step.

	var alive := 0.05 * gate
	var effort_pen := 0.1 * clampf(effort, 0.0, 1.0)
	return clampf(approach_reward + heading_reward + eat_bonus + alive - effort_pen, -1.0, 4.0)


# PHASE C — SURVIVAL (intrinsic homeostatic drive). The reward is the agent's COMFORT,
# derived from ENERGY (the LeCun intrinsic cost), NOT a hand-crafted locomotion bonus:
#   - dense comfort = energy level, GATED by being upright (it must stand/walk to forage),
#   - a SPIKE when it just ate (energy rose this step) → reinforces reaching food,
#   - a small alive bonus; death (fall OR starvation) is penalised.
# There is NO fixed-direction forward term (that would fight turning toward food). The walk
# is inherited from the warm-start and kept usable by the upright gate; the agent must learn,
# via its food radar (vision), to STEER toward pellets and eat before energy hits zero.
func _reward_survival_v1(metrics: Dictionary, energy: float, done: bool) -> float:
	if done:
		_prev_energy = -1.0
		return -1.0
	var height = float(metrics.get("height", 0.0))
	var height_gate = clampf((height - 0.45) / 0.40, 0.0, 1.0)
	var energy_frac = clampf(energy / 100.0, 0.0, 1.0)
	var eat_bonus = 0.0
	if _prev_energy >= 0.0 and energy > _prev_energy + 0.5:
		eat_bonus = 2.0   # just ate a pellet — strong immediate reinforcement
	_prev_energy = energy
	var alive = 0.3
	var comfort = 0.7 * energy_frac * height_gate   # high energy + upright = comfortable
	var reward = alive + comfort + eat_bonus
	return clampf(reward, -1.0, 5.0)


# PURE SURVIVAL — Mode-1 (2026-06-28). The ONLY axiom is "stay alive": +1 per step.
# Death (is_critical → done=true) removes all future +1, so maximising lifespan is the
# unique optimum — no explicit death penalty needed (GAE bootstraps terminal with 0).
# OPTIONAL bootstrap crutch: SYLVAN_PAIN_SHAPING_W (default 0.0) adds an annealable
# (1−level)² discomfort term so early training has a dense gradient toward full drives.
# Thirst is injected into metrics["thirst"] by main.gd; if absent (water disabled or old
# code), defaults to 100.0 → penalty = 0 → backward-compatible, no surprise penalty.
# Set w→0 as the policy matures; at w=0 the reward is the pure +1/step axiom.
func _reward_survival_pure(metrics: Dictionary, energy: float, _done: bool) -> float:
	var reward := 1.0
	var w_env := OS.get_environment("SYLVAN_PAIN_SHAPING_W")
	var w := float(w_env) if w_env != "" else 0.0
	if w > 0.0:
		var e := clampf(energy / 100.0, 0.0, 1.0)
		var t_raw := float(metrics.get("thirst", 100.0))
		var t := clampf(t_raw / 100.0, 0.0, 1.0)
		reward -= w * ((1.0 - e) * (1.0 - e) + (1.0 - t) * (1.0 - t))
	return reward


# J2 FULL-BODY walk — PHASE-CLOCK PERIODIC REWARD (Siekmann et al., ICRA 2021, the Cassie
# recipe), reference-free. The observable-state shaping line (v7→fb_v2) hit a ceiling:
# speed/stability frontier, stiff knees, no crisp rhythm — because a memoryless MLP given no
# CLOCK converges to the phase-AVERAGE (neutral legs). Here the agent observes a phase clock
# (sin/cos in proprio) and the reward imposes a PERIODIC contact schedule:
#   - each foot has a SWING window (be airborne: penalise ground contact) and a STANCE window
#     (be planted: penalise foot speed), offset half a cycle between left and right,
#   - feet_air_time (legged_gym): pay out at touchdown ∝ swing duration → real STRIDES, the
#     single biggest anti-"robocop" lever,
#   - swing-foot CLEARANCE toward an apex height (lift the foot, don't drag),
#   - forward-velocity tracking (exp kernel), and a LOWERED base-height target that
#     geometrically forces bent knees (a straight-legged stance can't satisfy it).
# Because the policy can SEE where it is in the cycle, it can comply → a crisp, periodic,
# bent-knee gait. Warm-start from the full-body stander (zero-padded to the 122-d clock
# contract) with boosted exploration.
const PR_SWING := 0.4        # swing fraction of the cycle (40% airborne, 60% stance per foot)
const PR_GATE_K := 0.1       # smoothness of the periodic swing indicator
const PR_V_NORM := 0.5       # foot-speed normalisation for the stance "planted" penalty (m/s)
const PR_T_AIR := 0.35       # target swing/air time (s); longer steps pay, hops are penalised
const PR_H_CLEAR := 0.12     # swing-foot apex height target above the base (m)
const PR_V_CMD := 0.4        # commanded forward speed (m/s)
const PR_SIGMA_V := 0.12     # forward-velocity tracking kernel width
const PR_H_TARGET := 0.78    # LOWERED base-height target (standing ~0.85) → forces knee flexion

# Smooth, fully-periodic swing indicator: ~1 when foot phase is within ±PR_SWING/2 of `center`
# (a pulse of width PR_SWING), ~0 otherwise. Uses cos() so there is NO wrap discontinuity at
# φ=1→0 (a hard `p < PR_SWING` test would cut the left-foot swing in half).
func _swing_indicator(p: float, center: float) -> float:
	var thresh := cos(PI * PR_SWING)
	return 1.0 / (1.0 + exp(-(cos(TAU * (p - center)) - thresh) / PR_GATE_K))

func _reward_locomotion_periodic(metrics: Dictionary, done: bool) -> float:
	if done:
		_last_sole = -1
		return -1.0
	var height = float(metrics.get("height", 0.0))
	var forward_velocity = float(metrics.get("forward_velocity", 0.0))
	var left_c = float(metrics.get("left_contact", 0.0))
	var right_c = float(metrics.get("right_contact", 0.0))
	var phase := float(metrics.get("gait_phase", 0.0))
	var l_speed := float(metrics.get("l_foot_speed", 0.0))
	var r_speed := float(metrics.get("r_foot_speed", 0.0))
	var l_speed_xy := float(metrics.get("l_foot_speed_xy", 0.0))
	var r_speed_xy := float(metrics.get("r_foot_speed_xy", 0.0))
	var l_foot_h := float(metrics.get("l_foot_height", 0.0))
	var r_foot_h := float(metrics.get("r_foot_height", 0.0))
	var l_air := float(metrics.get("l_air_time", 0.0))
	var r_air := float(metrics.get("r_air_time", 0.0))
	var l_td := float(metrics.get("l_touchdown", 0.0))
	var r_td := float(metrics.get("r_touchdown", 0.0))

	# NOT-FALLEN gate (height, not uprightness): lets it lean/wobble (controlled fall) and earn.
	var height_gate = clampf((height - 0.45) / 0.40, 0.0, 1.0)

	# Per-foot swing/stance schedule (right foot offset half a cycle).
	var swing_l = _swing_indicator(phase, 0.0)
	var swing_r = _swing_indicator(phase, 0.5)

	# (1) PERIODIC backbone. Force/contact gate: penalise ground contact during SWING (foot
	# should be airborne). Speed gate: penalise foot speed during STANCE (foot should be planted).
	var r_frc = -(swing_l * left_c + swing_r * right_c)                       # [-2, 0]
	var q_spd_l = 1.0 - exp(-l_speed / PR_V_NORM)
	var q_spd_r = 1.0 - exp(-r_speed / PR_V_NORM)
	var r_spd = -((1.0 - swing_l) * q_spd_l + (1.0 - swing_r) * q_spd_r)      # [-2, 0]
	var r_phase = (0.25 * r_frc + 0.25 * r_spd) * height_gate                 # [-1, 0]

	# (2) feet_air_time — the stride/anti-robocop lever. Signed: swing longer than target pays,
	# short hops are penalised. Only fires the step a foot touches down.
	var r_air_raw = l_td * (l_air - PR_T_AIR) + r_td * (r_air - PR_T_AIR)
	var air_reward = 2.0 * clampf(r_air_raw, -0.5, 0.5) * height_gate

	# (3) swing-foot CLEARANCE: keep the airborne foot near an apex height WHILE it swings
	# (× horizontal speed so it can't cheat by parking the foot high and still). Penalty.
	var clear_err = swing_l * pow(l_foot_h - PR_H_CLEAR, 2.0) * l_speed_xy \
		+ swing_r * pow(r_foot_h - PR_H_CLEAR, 2.0) * r_speed_xy
	var clearance_penalty = 3.0 * clampf(clear_err, 0.0, 0.3) * height_gate

	# (4) forward-velocity tracking (exp kernel) — commanded pace from the CURRICULUM
	# (SYLVAN_TARGET_VELOCITY, ramped low->high by train_ppo) so the gait is first rewarded
	# for a gentle, balance-preserving speed and only later pushed faster. Fixes the earlier
	# periodic run that lunged at a fixed 0.4 m/s and toppled. Falls back to PR_V_CMD when no
	# curriculum env is set.
	var env_v := OS.get_environment("SYLVAN_TARGET_VELOCITY")
	var v_cmd := float(env_v) if env_v != "" else PR_V_CMD
	var vel_kernel = exp(-pow(forward_velocity - v_cmd, 2.0) / PR_SIGMA_V)
	var forward_reward = 0.7 * vel_kernel * height_gate

	# (5) lowered base-height target → geometrically forces bent knees (anti-robocop at the source).
	var height_penalty = 2.0 * pow(height - PR_H_TARGET, 2.0)

	var alive_bonus = 0.3
	var effort = float(metrics.get("effort", 0.0))
	var smoothness_penalty = 0.25 * clampf(effort, 0.0, 1.0)

	var reward = alive_bonus + r_phase + air_reward + forward_reward \
		- clearance_penalty - height_penalty - smoothness_penalty
	return clampf(reward, -1.0, 5.0)


# J_walk v21 — EMBRACE THE CONTROLLED FALL (owner's insight: walking = project the COM
# forward, let yourself fall, CATCH with the other leg; the agent is "tétanisé de lever
# une jambe" because nearly every reward was multiplied by uprightness, so the instant of
# instability when lifting a foot CUT its rewards → it learned to drag both feet, safe but
# static). v21 inverts the philosophy:
#   - gate rewards by HEIGHT (not-fallen), NOT uprightness — so leaning/wobbling while
#     stepping still pays; only a REAL fall (collapsed height) kills the reward,
#   - DROP the lean & tilt penalties (allow the forward projection that walking IS),
#   - STRONGLY reward the lift+catch cycle (single-support, foot clearance, foot planted
#     ahead), NO longer choked by uprightness,
#   - keep done=-1 as the ONLY stability anchor (don't actually fall, but dare the wobble).
func _reward_locomotion_v21(metrics: Dictionary, done: bool) -> float:
	if done:
		_last_sole = -1
		return -1.0
	var height = float(metrics.get("height", 0.0))
	var forward_velocity = float(metrics.get("forward_velocity", 0.0))
	var left_c = float(metrics.get("left_contact", 0.0))
	var right_c = float(metrics.get("right_contact", 0.0))
	var l_knee := float(metrics.get("j_l_knee", 0.0))
	var r_knee := float(metrics.get("j_r_knee", 0.0))
	var l_foot_fwd := float(metrics.get("l_foot_fwd", 0.0))
	var r_foot_fwd := float(metrics.get("r_foot_fwd", 0.0))
	var l_foot_h := float(metrics.get("l_foot_height", 0.0))
	var r_foot_h := float(metrics.get("r_foot_height", 0.0))

	# NOT-FALLEN gate (height, not uprightness): 1 when standing tall, 0 when collapsed.
	# Lets it LEAN/WOBBLE (the controlled fall) and still earn — only a real fall zeroes it.
	var height_gate := clampf((height - 0.45) / (0.85 - 0.45), 0.0, 1.0)

	var knee_alt := clampf(absf(l_knee - r_knee) / 0.4, 0.0, 1.0)
	var knee_flex_reward = 0.7 * knee_alt * height_gate

	# Project forward (toward target) — the controlled fall.
	var vel_score := clampf(1.0 - absf(forward_velocity - 0.30) / 0.30, 0.0, 1.0)
	var forward_reward = 1.2 * height_gate * vel_score
	var backward_penalty = 0.6 * clampf(-forward_velocity / 0.3, 0.0, 1.0)

	# LIFT + CATCH cycle (NOT uprightness-gated — daring the wobble must pay):
	var swing_advance := 0.0   # catch: swing foot planted ahead
	if left_c < 0.5:
		swing_advance = maxf(swing_advance, clampf(l_foot_fwd / 0.25, 0.0, 1.0))
	if right_c < 0.5:
		swing_advance = maxf(swing_advance, clampf(r_foot_fwd / 0.25, 0.0, 1.0))
	var step_advance_reward = 1.0 * swing_advance * height_gate

	var clearance := 0.0       # lift: swing foot off the ground
	if left_c < 0.5:
		clearance = maxf(clearance, clampf((l_foot_h - 0.08) / 0.12, 0.0, 1.0))
	if right_c < 0.5:
		clearance = maxf(clearance, clampf((r_foot_h - 0.08) / 0.12, 0.0, 1.0))
	var clearance_reward = 1.0 * clearance * height_gate

	var single_support = 1.0 if absf((left_c + right_c) - 1.0) < 0.5 else 0.0
	var swing_reward = 0.7 * single_support * height_gate   # STRONG, ungated by uprightness
	var alternation_reward = 0.0
	if single_support > 0.5:
		var sole := 0 if left_c > 0.5 else 1
		if _last_sole >= 0 and sole != _last_sole:
			alternation_reward = 0.7 * height_gate
		_last_sole = sole

	var alive_bonus = 0.2
	# NO lean penalty, NO tilt penalty (allow the forward projection). Keep a light smoothness.
	var effort = float(metrics.get("effort", 0.0))
	var smoothness_penalty = 0.2 * clampf(effort, 0.0, 1.0)

	var reward = alive_bonus + knee_flex_reward + forward_reward + step_advance_reward + clearance_reward + swing_reward + alternation_reward - backward_penalty - smoothness_penalty
	return clampf(reward, -1.0, 5.0)


# J2 FULL-BODY walk — v21 + ANTI-ROBOCOP. Owner's visual: the agent keeps its legs RIGIDLY
# LOCKED straight (robocop), which blocks an efficient gait. v21 rewarded knee ALTERNATION,
# but both knees could still sit near 0 (locked) and weakly satisfy it. fb adds, FROM iter 0:
#   (1) a LOCK PENALTY when the mean knee is near-straight while upright → forces a compliant,
#       slightly-bent "ready" posture (the owner's repeated concern, fixed at the source),
#   (2) a stronger knee-flexion/alternation reward (0.7 -> 0.9).
# Everything else = v21 (height-gated controlled fall, forward dominant, lift+catch, single-
# support + alternation). Warm-start from the full-body stander with BOOSTED exploration so it
# escapes the locked-knee standstill basin. Arms swing emergently (already observed) — not rewarded.
func _reward_locomotion_fb(metrics: Dictionary, done: bool) -> float:
	if done:
		_last_sole = -1
		return -1.0
	var height = float(metrics.get("height", 0.0))
	var forward_velocity = float(metrics.get("forward_velocity", 0.0))
	var left_c = float(metrics.get("left_contact", 0.0))
	var right_c = float(metrics.get("right_contact", 0.0))
	var l_knee := float(metrics.get("j_l_knee", 0.0))
	var r_knee := float(metrics.get("j_r_knee", 0.0))
	var l_foot_fwd := float(metrics.get("l_foot_fwd", 0.0))
	var r_foot_fwd := float(metrics.get("r_foot_fwd", 0.0))
	var l_foot_h := float(metrics.get("l_foot_height", 0.0))
	var r_foot_h := float(metrics.get("r_foot_height", 0.0))

	# NOT-FALLEN gate (height, not uprightness): lets it LEAN/WOBBLE and still earn.
	var height_gate := clampf((height - 0.45) / (0.85 - 0.45), 0.0, 1.0)

	# Knee ALTERNATION (stepping mechanic) — boosted vs v21 (0.7 -> 0.9).
	var knee_alt := clampf(absf(l_knee - r_knee) / 0.4, 0.0, 1.0)
	var knee_flex_reward = 0.9 * knee_alt * height_gate

	# ANTI-ROBOCOP: penalise LOCKED straight legs while upright. mean knee < ~0.18 rad
	# (~10 deg) = locked → penalised; >= 0.18 = compliant ready stance → no penalty.
	# height-gated so a falling agent isn't double-penalised.
	var mean_knee := 0.5 * (l_knee + r_knee)
	var lock_penalty = 0.6 * clampf((0.18 - mean_knee) / 0.18, 0.0, 1.0) * height_gate

	# Project forward (toward target ~0.30 m/s) — the controlled fall.
	var vel_score := clampf(1.0 - absf(forward_velocity - 0.30) / 0.30, 0.0, 1.0)
	var forward_reward = 1.2 * height_gate * vel_score
	var backward_penalty = 0.6 * clampf(-forward_velocity / 0.3, 0.0, 1.0)

	# LIFT + CATCH cycle (NOT uprightness-gated):
	var swing_advance := 0.0
	if left_c < 0.5:
		swing_advance = maxf(swing_advance, clampf(l_foot_fwd / 0.25, 0.0, 1.0))
	if right_c < 0.5:
		swing_advance = maxf(swing_advance, clampf(r_foot_fwd / 0.25, 0.0, 1.0))
	var step_advance_reward = 1.0 * swing_advance * height_gate

	var clearance := 0.0
	if left_c < 0.5:
		clearance = maxf(clearance, clampf((l_foot_h - 0.08) / 0.12, 0.0, 1.0))
	if right_c < 0.5:
		clearance = maxf(clearance, clampf((r_foot_h - 0.08) / 0.12, 0.0, 1.0))
	var clearance_reward = 1.0 * clearance * height_gate

	var single_support = 1.0 if absf((left_c + right_c) - 1.0) < 0.5 else 0.0
	var swing_reward = 0.7 * single_support * height_gate
	var alternation_reward = 0.0
	if single_support > 0.5:
		var sole := 0 if left_c > 0.5 else 1
		if _last_sole >= 0 and sole != _last_sole:
			alternation_reward = 0.7 * height_gate
		_last_sole = sole

	var alive_bonus = 0.2
	var effort = float(metrics.get("effort", 0.0))
	var smoothness_penalty = 0.2 * clampf(effort, 0.0, 1.0)

	var reward = alive_bonus + knee_flex_reward + forward_reward + step_advance_reward + clearance_reward + swing_reward + alternation_reward - backward_penalty - smoothness_penalty - lock_penalty
	return clampf(reward, -1.0, 5.0)


# J2 FULL-BODY walk v2 — SUPER-STABLE pass (owner: "make it super stable; knees still too straight").
# fb walked but stayed bancale (fall >=57% over 250 iters) and stiff. v2 keeps the controlled-fall
# spirit (height gate, no uprightness GATE → no "tetanised") but rebalances toward STABILITY:
#   - SLOWER target (0.30 -> 0.18 m/s): a slow step is far more stable; forward weight 1.2 -> 1.0.
#   - ACTIVE-BALANCE term (additive, NOT a gate): keep the COM over the foot support
#     (com_support_offset low) — rewards balancing while stepping without freezing the gait.
#   - real BENT-KNEE TARGET ~0.35 rad (a loaded-spring stance) instead of a weak anti-lock penalty
#     → properly bent knees = lower COM + shock absorption = more stable AND less robocop.
#   - stronger smoothness penalty (cleaner, less thrashing).
# Pair with annealed exploration at train time (init-log-std -0.8, entropy 0.001) so the std settles
# into a crisp cyclic gait instead of wobbling. Warm-start from the MOST STABLE walk checkpoint.
func _reward_locomotion_fb_v2(metrics: Dictionary, done: bool) -> float:
	if done:
		_last_sole = -1
		return -1.0
	var height = float(metrics.get("height", 0.0))
	var forward_velocity = float(metrics.get("forward_velocity", 0.0))
	var left_c = float(metrics.get("left_contact", 0.0))
	var right_c = float(metrics.get("right_contact", 0.0))
	var l_knee := float(metrics.get("j_l_knee", 0.0))
	var r_knee := float(metrics.get("j_r_knee", 0.0))
	var l_foot_fwd := float(metrics.get("l_foot_fwd", 0.0))
	var r_foot_fwd := float(metrics.get("r_foot_fwd", 0.0))
	var l_foot_h := float(metrics.get("l_foot_height", 0.0))
	var r_foot_h := float(metrics.get("r_foot_height", 0.0))
	var com_off := float(metrics.get("com_support_offset", 0.0))

	# NOT-FALLEN gate (height, not uprightness): lets it LEAN/WOBBLE and still earn.
	var height_gate := clampf((height - 0.45) / (0.85 - 0.45), 0.0, 1.0)

	# ACTIVE BALANCE (additive, the stability driver): keep the COM over the support.
	# ~0 when balanced, grows as it topples; reward the inverse. NOT a gate on the gait.
	var stability_reward = 0.8 * clampf(1.0 - com_off / 0.20, 0.0, 1.0) * height_gate

	# Real BENT-KNEE target ~0.35 rad (loaded spring) — addresses the stiff legs AND lowers the COM.
	var mean_knee := 0.5 * (l_knee + r_knee)
	var bend_reward = 0.7 * clampf(1.0 - absf(mean_knee - 0.35) / 0.35, 0.0, 1.0) * height_gate
	# Knee ALTERNATION (stepping mechanic).
	var knee_alt := clampf(absf(l_knee - r_knee) / 0.4, 0.0, 1.0)
	var knee_alt_reward = 0.6 * knee_alt * height_gate

	# SLOWER forward target (0.18 m/s) — a slow gait is far more stable.
	var vel_score := clampf(1.0 - absf(forward_velocity - 0.18) / 0.18, 0.0, 1.0)
	var forward_reward = 1.0 * height_gate * vel_score
	var backward_penalty = 0.6 * clampf(-forward_velocity / 0.2, 0.0, 1.0)

	# LIFT + CATCH cycle (not uprightness-gated):
	var swing_advance := 0.0
	if left_c < 0.5:
		swing_advance = maxf(swing_advance, clampf(l_foot_fwd / 0.22, 0.0, 1.0))
	if right_c < 0.5:
		swing_advance = maxf(swing_advance, clampf(r_foot_fwd / 0.22, 0.0, 1.0))
	var step_advance_reward = 0.8 * swing_advance * height_gate

	var clearance := 0.0
	if left_c < 0.5:
		clearance = maxf(clearance, clampf((l_foot_h - 0.08) / 0.12, 0.0, 1.0))
	if right_c < 0.5:
		clearance = maxf(clearance, clampf((r_foot_h - 0.08) / 0.12, 0.0, 1.0))
	var clearance_reward = 0.7 * clearance * height_gate

	var single_support = 1.0 if absf((left_c + right_c) - 1.0) < 0.5 else 0.0
	var swing_reward = 0.6 * single_support * height_gate
	var alternation_reward = 0.0
	if single_support > 0.5:
		var sole := 0 if left_c > 0.5 else 1
		if _last_sole >= 0 and sole != _last_sole:
			alternation_reward = 0.6 * height_gate
		_last_sole = sole

	var alive_bonus = 0.2
	var effort = float(metrics.get("effort", 0.0))
	var smoothness_penalty = 0.35 * clampf(effort, 0.0, 1.0)

	var reward = alive_bonus + stability_reward + bend_reward + knee_alt_reward + forward_reward \
		+ step_advance_reward + clearance_reward + swing_reward + alternation_reward \
		- backward_penalty - smoothness_penalty
	return clampf(reward, -1.0, 5.0)


# J_walk v20 — LIFT THE FEET (foot clearance). v17/v19 walk but DRAG the feet (owner's
# visual: "il traîne les pieds, il ne lève pas ses pieds → déstabilisation"): single_support
# was rewarded (one foot with no contact) but a foot can lose contact while barely off the
# ground (dragging). v20 == v17 + a STRONG reward for the AIRBORNE foot's HEIGHT (real ground
# clearance), so it actually picks the foot up to swing it. Rest foot height ~0.03; reward
# lifting the swing foot to ~0.08-0.20.
func _reward_locomotion_v20(metrics: Dictionary, done: bool) -> float:
	if done:
		_last_sole = -1
		return -1.0
	var uprightness = float(metrics.get("uprightness", 0.0))
	var torso_tilt = float(metrics.get("torso_tilt", 0.0))
	var forward_velocity = float(metrics.get("forward_velocity", 0.0))
	var left_c = float(metrics.get("left_contact", 0.0))
	var right_c = float(metrics.get("right_contact", 0.0))
	var l_knee := float(metrics.get("j_l_knee", 0.0))
	var r_knee := float(metrics.get("j_r_knee", 0.0))
	var l_foot_fwd := float(metrics.get("l_foot_fwd", 0.0))
	var r_foot_fwd := float(metrics.get("r_foot_fwd", 0.0))
	var fwd_lean := float(metrics.get("forward_lean", 0.0))
	var l_foot_h := float(metrics.get("l_foot_height", 0.0))
	var r_foot_h := float(metrics.get("r_foot_height", 0.0))
	var upright_gate = clampf(uprightness, 0.0, 1.0)

	var knee_alt := clampf(absf(l_knee - r_knee) / 0.4, 0.0, 1.0)
	var knee_flex_reward = 0.9 * knee_alt * upright_gate

	var target_v := 0.30
	var vel_score := clampf(1.0 - absf(forward_velocity - target_v) / target_v, 0.0, 1.0)
	var forward_reward = 1.2 * upright_gate * vel_score
	var backward_penalty = 0.6 * clampf(-forward_velocity / 0.3, 0.0, 1.0)

	var swing_advance := 0.0
	if left_c < 0.5:
		swing_advance = maxf(swing_advance, clampf(l_foot_fwd / 0.25, 0.0, 1.0))
	if right_c < 0.5:
		swing_advance = maxf(swing_advance, clampf(r_foot_fwd / 0.25, 0.0, 1.0))
	var step_advance_reward = 1.0 * swing_advance * upright_gate

	# FOOT CLEARANCE: the airborne foot (no contact) must be LIFTED HIGH (not dragged).
	var clearance := 0.0
	if left_c < 0.5:
		clearance = maxf(clearance, clampf((l_foot_h - 0.08) / 0.12, 0.0, 1.0))
	if right_c < 0.5:
		clearance = maxf(clearance, clampf((r_foot_h - 0.08) / 0.12, 0.0, 1.0))
	var clearance_reward = 0.9 * clearance * upright_gate

	var single_support = 1.0 if absf((left_c + right_c) - 1.0) < 0.5 else 0.0
	var swing_reward = 0.4 * single_support * upright_gate
	var alternation_reward = 0.0
	if single_support > 0.5:
		var sole := 0 if left_c > 0.5 else 1
		if _last_sole >= 0 and sole != _last_sole:
			alternation_reward = 0.5 * upright_gate
		_last_sole = sole

	var alive_bonus = 0.3
	var tilt_penalty = 0.2 * torso_tilt
	var effort = float(metrics.get("effort", 0.0))
	var smoothness_penalty = 0.3 * clampf(effort, 0.0, 1.0)
	var lean_penalty = 1.0 * clampf((fwd_lean - 0.2) / 0.4, 0.0, 1.0)

	var reward = alive_bonus + knee_flex_reward + forward_reward + step_advance_reward + clearance_reward + swing_reward + alternation_reward - tilt_penalty - backward_penalty - smoothness_penalty - lean_penalty
	return clampf(reward, -1.0, 5.0)


# J_walk v17 — STOP the forward lean. v16 reassembled a real forward bent-knee gait
# but the owner's visual showed the torso "penché en avant en permanence" → COM ahead
# of the feet → it topples forward (its main fall cause). v17 == v16 + a penalty on
# EXCESSIVE forward pitch (forward_lean = -torso_forward.y): no penalty for a slight
# walking lean (<0.2), ramping strongly above it, so it walks more UPRIGHT. Warm-start
# from v16 (keep the gait, straighten the posture).
func _reward_locomotion_v17(metrics: Dictionary, done: bool) -> float:
	if done:
		_last_sole = -1
		return -1.0
	var uprightness = float(metrics.get("uprightness", 0.0))
	var torso_tilt = float(metrics.get("torso_tilt", 0.0))
	var forward_velocity = float(metrics.get("forward_velocity", 0.0))
	var left_c = float(metrics.get("left_contact", 0.0))
	var right_c = float(metrics.get("right_contact", 0.0))
	var l_knee := float(metrics.get("j_l_knee", 0.0))
	var r_knee := float(metrics.get("j_r_knee", 0.0))
	var l_foot_fwd := float(metrics.get("l_foot_fwd", 0.0))
	var r_foot_fwd := float(metrics.get("r_foot_fwd", 0.0))
	var fwd_lean := float(metrics.get("forward_lean", 0.0))
	var upright_gate = clampf(uprightness, 0.0, 1.0)

	var knee_alt := clampf(absf(l_knee - r_knee) / 0.4, 0.0, 1.0)
	var knee_flex_reward = 0.9 * knee_alt * upright_gate

	var target_v := 0.30
	var vel_score := clampf(1.0 - absf(forward_velocity - target_v) / target_v, 0.0, 1.0)
	var forward_reward = 1.2 * upright_gate * vel_score
	var backward_penalty = 0.6 * clampf(-forward_velocity / 0.3, 0.0, 1.0)

	var swing_advance := 0.0
	if left_c < 0.5:
		swing_advance = maxf(swing_advance, clampf(l_foot_fwd / 0.25, 0.0, 1.0))
	if right_c < 0.5:
		swing_advance = maxf(swing_advance, clampf(r_foot_fwd / 0.25, 0.0, 1.0))
	var step_advance_reward = 1.0 * swing_advance * upright_gate

	var single_support = 1.0 if absf((left_c + right_c) - 1.0) < 0.5 else 0.0
	var swing_reward = 0.4 * single_support * upright_gate
	var alternation_reward = 0.0
	if single_support > 0.5:
		var sole := 0 if left_c > 0.5 else 1
		if _last_sole >= 0 and sole != _last_sole:
			alternation_reward = 0.5 * upright_gate
		_last_sole = sole

	var alive_bonus = 0.3
	var tilt_penalty = 0.2 * torso_tilt
	var effort = float(metrics.get("effort", 0.0))
	var smoothness_penalty = 0.3 * clampf(effort, 0.0, 1.0)
	# Forward-lean penalty: allow a slight walking lean (<0.2), punish the excess.
	var lean_penalty = 1.0 * clampf((fwd_lean - 0.2) / 0.4, 0.0, 1.0)

	var reward = alive_bonus + knee_flex_reward + forward_reward + step_advance_reward + swing_reward + alternation_reward - tilt_penalty - backward_penalty - smoothness_penalty - lean_penalty
	return clampf(reward, -1.0, 5.0)


# J_walk v16 — forward (+z, flipped/stable) WITH real bent-knee steps. v15 restarted
# from J0 and regressed to stiff micro-steps (knee_alternation 0.14 vs v13's 0.26):
# the reduced knee reward (0.4) couldn't rebuild knee flexion from a straight-legged
# stander. v16 restores the STRONG knee-flexion reward (0.9, the v10 value that first
# cracked knee bending) + strong step-advance, warm-started from v15 (keep its forward
# +z direction & stability, re-impose real steps on top). Peaked speed 0.30 (calm) +
# smoothness + backward penalty kept.
func _reward_locomotion_v16(metrics: Dictionary, done: bool) -> float:
	if done:
		_last_sole = -1
		return -1.0
	var uprightness = float(metrics.get("uprightness", 0.0))
	var torso_tilt = float(metrics.get("torso_tilt", 0.0))
	var forward_velocity = float(metrics.get("forward_velocity", 0.0))
	var left_c = float(metrics.get("left_contact", 0.0))
	var right_c = float(metrics.get("right_contact", 0.0))
	var l_knee := float(metrics.get("j_l_knee", 0.0))
	var r_knee := float(metrics.get("j_r_knee", 0.0))
	var l_foot_fwd := float(metrics.get("l_foot_fwd", 0.0))
	var r_foot_fwd := float(metrics.get("r_foot_fwd", 0.0))
	var upright_gate = clampf(uprightness, 0.0, 1.0)

	# STRONG alternating knee flexion (rebuild real bent-knee steps).
	var knee_alt := clampf(absf(l_knee - r_knee) / 0.4, 0.0, 1.0)
	var knee_flex_reward = 0.9 * knee_alt * upright_gate

	# Forward progress peaked at a stable 0.30 m/s (calm pace, no rush) + backward penalty.
	var target_v := 0.30
	var vel_score := clampf(1.0 - absf(forward_velocity - target_v) / target_v, 0.0, 1.0)
	var forward_reward = 1.2 * upright_gate * vel_score
	var backward_penalty = 0.6 * clampf(-forward_velocity / 0.3, 0.0, 1.0)

	# STRONG step-forward: swing foot planted ahead of the torso (+z).
	var swing_advance := 0.0
	if left_c < 0.5:
		swing_advance = maxf(swing_advance, clampf(l_foot_fwd / 0.25, 0.0, 1.0))
	if right_c < 0.5:
		swing_advance = maxf(swing_advance, clampf(r_foot_fwd / 0.25, 0.0, 1.0))
	var step_advance_reward = 1.0 * swing_advance * upright_gate

	var single_support = 1.0 if absf((left_c + right_c) - 1.0) < 0.5 else 0.0
	var swing_reward = 0.4 * single_support * upright_gate
	var alternation_reward = 0.0
	if single_support > 0.5:
		var sole := 0 if left_c > 0.5 else 1
		if _last_sole >= 0 and sole != _last_sole:
			alternation_reward = 0.5 * upright_gate
		_last_sole = sole

	var alive_bonus = 0.3
	var tilt_penalty = 0.2 * torso_tilt
	var effort = float(metrics.get("effort", 0.0))
	var smoothness_penalty = 0.3 * clampf(effort, 0.0, 1.0)

	var reward = alive_bonus + knee_flex_reward + forward_reward + step_advance_reward + swing_reward + alternation_reward - tilt_penalty - backward_penalty - smoothness_penalty
	return clampf(reward, -1.0, 5.0)


# J_walk v13 — PEAK the speed at a stable pace (v12's CAP failed: the agent rushed to
# 0.4+ m/s and fell 70%, because capping the reward removed the bonus for speed but
# never PENALISED it). v13 uses a TRIANGULAR speed reward peaking at 0.30 m/s and
# DECLINING above it — so going faster than 0.30 actively pays less, pulling the
# fast-walking warm-start (from v12) DOWN toward a stable pace. step_advance halved
# (it drove the rush). Gait terms (knee, single-support, alternation) kept.
func _reward_locomotion_v13(metrics: Dictionary, done: bool) -> float:
	if done:
		_last_sole = -1
		return -1.0
	var uprightness = float(metrics.get("uprightness", 0.0))
	var torso_tilt = float(metrics.get("torso_tilt", 0.0))
	var forward_velocity = float(metrics.get("forward_velocity", 0.0))
	var left_c = float(metrics.get("left_contact", 0.0))
	var right_c = float(metrics.get("right_contact", 0.0))
	var l_knee := float(metrics.get("j_l_knee", 0.0))
	var r_knee := float(metrics.get("j_r_knee", 0.0))
	var l_foot_fwd := float(metrics.get("l_foot_fwd", 0.0))
	var r_foot_fwd := float(metrics.get("r_foot_fwd", 0.0))
	var upright_gate = clampf(uprightness, 0.0, 1.0)

	var knee_alt := clampf(absf(l_knee - r_knee) / 0.4, 0.0, 1.0)
	var knee_flex_reward = 0.4 * knee_alt * upright_gate

	# Triangular speed reward: peak at 0.30 m/s, 0 at standstill AND at >=0.60 — so
	# rushing past 0.30 actively earns LESS, pulling the fast walker down to a stable pace.
	var target_v := 0.30
	var vel_score := clampf(1.0 - absf(forward_velocity - target_v) / target_v, 0.0, 1.0)
	var forward_reward = 1.4 * upright_gate * vel_score
	var backward_penalty = 0.6 * clampf(-forward_velocity / 0.3, 0.0, 1.0)

	var swing_advance := 0.0
	if left_c < 0.5:
		swing_advance = maxf(swing_advance, clampf(l_foot_fwd / 0.25, 0.0, 1.0))
	if right_c < 0.5:
		swing_advance = maxf(swing_advance, clampf(r_foot_fwd / 0.25, 0.0, 1.0))
	var step_advance_reward = 0.5 * swing_advance * upright_gate   # halved (drove the rush)

	var single_support = 1.0 if absf((left_c + right_c) - 1.0) < 0.5 else 0.0
	var swing_reward = 0.3 * single_support * upright_gate
	var alternation_reward = 0.0
	if single_support > 0.5:
		var sole := 0 if left_c > 0.5 else 1
		if _last_sole >= 0 and sole != _last_sole:
			alternation_reward = 0.4 * upright_gate
		_last_sole = sole

	var alive_bonus = 0.3
	var tilt_penalty = 0.2 * torso_tilt
	# Smoothness (owner's visual: "pas fluide ni calme, mouvements rapides/saccadés"):
	# a mild penalty on motor effort encourages a CALMER, lower-energy gait instead of
	# frantic high-frequency corrections. Kept small so it never causes the freeze
	# (J0 lesson) — the forward/gait rewards dominate.
	var effort = float(metrics.get("effort", 0.0))
	var smoothness_penalty = 0.3 * clampf(effort, 0.0, 1.0)

	var reward = alive_bonus + knee_flex_reward + forward_reward + step_advance_reward + swing_reward + alternation_reward - tilt_penalty - backward_penalty - smoothness_penalty
	return clampf(reward, -1.0, 5.0)


# J_walk v12 — STABILISE the walk. v11 assembled real bent-knee forward stepping
# (single_support 31%, alternations 2.3, knees bent) but RUSHED to ~0.35-0.39 m/s and
# fell 70-90% — the forward target (0.6) rewarded ever-faster speed. v12 caps the
# forward target at a STABLE 0.30 m/s (no reward for going faster), so the agent stops
# rushing and can spend capacity on staying up at a moderate pace. Otherwise == v11.
func _reward_locomotion_v12(metrics: Dictionary, done: bool) -> float:
	if done:
		_last_sole = -1
		return -1.0
	var uprightness = float(metrics.get("uprightness", 0.0))
	var torso_tilt = float(metrics.get("torso_tilt", 0.0))
	var forward_velocity = float(metrics.get("forward_velocity", 0.0))
	var left_c = float(metrics.get("left_contact", 0.0))
	var right_c = float(metrics.get("right_contact", 0.0))
	var l_knee := float(metrics.get("j_l_knee", 0.0))
	var r_knee := float(metrics.get("j_r_knee", 0.0))
	var l_foot_fwd := float(metrics.get("l_foot_fwd", 0.0))
	var r_foot_fwd := float(metrics.get("r_foot_fwd", 0.0))
	var upright_gate = clampf(uprightness, 0.0, 1.0)

	var knee_alt := clampf(absf(l_knee - r_knee) / 0.4, 0.0, 1.0)
	var knee_flex_reward = 0.4 * knee_alt * upright_gate

	# Forward progress capped at a STABLE pace (0.30 m/s) — no rush bonus past it.
	var fwd_score = clampf(forward_velocity / 0.30, 0.0, 1.0)
	var forward_reward = 1.4 * upright_gate * fwd_score
	var backward_penalty = 0.6 * clampf(-forward_velocity / 0.3, 0.0, 1.0)

	var swing_advance := 0.0
	if left_c < 0.5:
		swing_advance = maxf(swing_advance, clampf(l_foot_fwd / 0.25, 0.0, 1.0))
	if right_c < 0.5:
		swing_advance = maxf(swing_advance, clampf(r_foot_fwd / 0.25, 0.0, 1.0))
	var step_advance_reward = 1.0 * swing_advance * upright_gate

	var single_support = 1.0 if absf((left_c + right_c) - 1.0) < 0.5 else 0.0
	var swing_reward = 0.3 * single_support * upright_gate
	var alternation_reward = 0.0
	if single_support > 0.5:
		var sole := 0 if left_c > 0.5 else 1
		if _last_sole >= 0 and sole != _last_sole:
			alternation_reward = 0.4 * upright_gate
		_last_sole = sole

	var alive_bonus = 0.3
	var tilt_penalty = 0.2 * torso_tilt

	var reward = alive_bonus + knee_flex_reward + forward_reward + step_advance_reward + swing_reward + alternation_reward - tilt_penalty - backward_penalty
	return clampf(reward, -1.0, 5.0)


# J_walk v11 — DIRECT the learned step FORWARD. v10 cracked the knee mechanic
# (knee_alternation 0.04->0.28) but MARCHED IN PLACE (fwd_vel ~0/negative) because
# the in-place gait rewards (knee+coord+swing+alternation ~3.4) dwarfed forward
# (0.6). v11 keeps a (reduced) knee reward so the mechanic survives, but makes
# FORWARD progress dominant (linear ramp from standstill) + a backward penalty, so
# the now-learned stepping is channelled into actual displacement.
func _reward_locomotion_v11(metrics: Dictionary, done: bool) -> float:
	if done:
		_last_sole = -1
		return -1.0
	var uprightness = float(metrics.get("uprightness", 0.0))
	var torso_tilt = float(metrics.get("torso_tilt", 0.0))
	var forward_velocity = float(metrics.get("forward_velocity", 0.0))
	var left_c = float(metrics.get("left_contact", 0.0))
	var right_c = float(metrics.get("right_contact", 0.0))
	var l_knee := float(metrics.get("j_l_knee", 0.0))
	var r_knee := float(metrics.get("j_r_knee", 0.0))
	var l_foot_fwd := float(metrics.get("l_foot_fwd", 0.0))
	var r_foot_fwd := float(metrics.get("r_foot_fwd", 0.0))
	var upright_gate = clampf(uprightness, 0.0, 1.0)

	# Keep the knee mechanic alive (reduced — it's learned).
	var knee_alt := clampf(absf(l_knee - r_knee) / 0.4, 0.0, 1.0)
	var knee_flex_reward = 0.4 * knee_alt * upright_gate

	# DOMINANT forward progress: linear ramp from standstill to target (gradient out
	# of in-place), plus an explicit penalty for moving BACKWARD.
	var fwd_score = clampf(forward_velocity / 0.6, 0.0, 1.0)
	var forward_reward = 1.6 * upright_gate * fwd_score
	var backward_penalty = 0.6 * clampf(-forward_velocity / 0.3, 0.0, 1.0)

	# STEP-FORWARD: swing foot planted ahead of the torso.
	var swing_advance := 0.0
	if left_c < 0.5:
		swing_advance = maxf(swing_advance, clampf(l_foot_fwd / 0.25, 0.0, 1.0))
	if right_c < 0.5:
		swing_advance = maxf(swing_advance, clampf(r_foot_fwd / 0.25, 0.0, 1.0))
	var step_advance_reward = 1.0 * swing_advance * upright_gate

	# stepping rhythm (reduced).
	var single_support = 1.0 if absf((left_c + right_c) - 1.0) < 0.5 else 0.0
	var swing_reward = 0.3 * single_support * upright_gate
	var alternation_reward = 0.0
	if single_support > 0.5:
		var sole := 0 if left_c > 0.5 else 1
		if _last_sole >= 0 and sole != _last_sole:
			alternation_reward = 0.4 * upright_gate
		_last_sole = sole

	var alive_bonus = 0.3
	var tilt_penalty = 0.2 * torso_tilt

	var reward = alive_bonus + knee_flex_reward + forward_reward + step_advance_reward + swing_reward + alternation_reward - tilt_penalty - backward_penalty
	return clampf(reward, -1.0, 5.0)


# J_walk v10 — KNEE FLEXION as the driver (user's visual diagnosis: v9 moves the
# legs but keeps them STRAIGHT, balancing on the ankles — so it can never lift a
# foot to swing it forward, and alternation is impossible). v7/v9 only rewarded a
# bent knee when the foot was ALREADY airborne (a chicken-and-egg the agent dodged),
# and hip anti-phase is gameable with straight legs. v10 adds a STRONG, ungated
# reward for ALTERNATING knee flexion (one knee bent, one straight = the stepping
# posture) — it now PAYS to bend a knee, which is the prerequisite to lift & step.
func _reward_locomotion_v10(metrics: Dictionary, done: bool) -> float:
	if done:
		_last_sole = -1
		return -1.0
	var uprightness = float(metrics.get("uprightness", 0.0))
	var torso_tilt = float(metrics.get("torso_tilt", 0.0))
	var forward_velocity = float(metrics.get("forward_velocity", 0.0))
	var left_c = float(metrics.get("left_contact", 0.0))
	var right_c = float(metrics.get("right_contact", 0.0))
	var l_hip := float(metrics.get("j_l_hip_x", 0.0))
	var r_hip := float(metrics.get("j_r_hip_x", 0.0))
	var l_knee := float(metrics.get("j_l_knee", 0.0))
	var r_knee := float(metrics.get("j_r_knee", 0.0))
	var l_foot_fwd := float(metrics.get("l_foot_fwd", 0.0))
	var r_foot_fwd := float(metrics.get("r_foot_fwd", 0.0))
	var upright_gate = clampf(uprightness, 0.0, 1.0)

	# ALTERNATING KNEE FLEXION (the missing prerequisite): one knee bent, one straight.
	# Ungated (does NOT require an airborne foot) so it breaks the chicken-and-egg.
	var knee_alt := clampf(absf(l_knee - r_knee) / 0.4, 0.0, 1.0)
	var knee_flex_reward = 0.9 * knee_alt * upright_gate

	# hip anti-phase (reduced — gameable straight-legged).
	var hip_anti := clampf(-l_hip * r_hip / (GAIT_HIP_AMP * GAIT_HIP_AMP), 0.0, 1.0)
	var coord_reward = 0.4 * hip_anti * upright_gate

	# STEP-FORWARD: swing foot planted ahead of the torso (from v9).
	var swing_advance := 0.0
	if left_c < 0.5:
		swing_advance = maxf(swing_advance, clampf(l_foot_fwd / 0.25, 0.0, 1.0))
	if right_c < 0.5:
		swing_advance = maxf(swing_advance, clampf(r_foot_fwd / 0.25, 0.0, 1.0))
	var step_advance_reward = 1.0 * swing_advance * upright_gate

	# foot lift (single-support) + stance alternation.
	var single_support = 1.0 if absf((left_c + right_c) - 1.0) < 0.5 else 0.0
	var swing_reward = 0.5 * single_support * upright_gate
	var alternation_reward = 0.0
	if single_support > 0.5:
		var sole := 0 if left_c > 0.5 else 1
		if _last_sole >= 0 and sole != _last_sole:
			alternation_reward = 0.6 * upright_gate
		_last_sole = sole

	var alive_bonus = 0.3
	var fwd_score = clampf(forward_velocity / 0.6, 0.0, 1.0)
	var forward_reward = 0.6 * upright_gate * fwd_score
	var tilt_penalty = 0.2 * torso_tilt

	var reward = alive_bonus + knee_flex_reward + coord_reward + step_advance_reward + swing_reward + alternation_reward + forward_reward - tilt_penalty
	return clampf(reward, -1.0, 5.0)


# J_walk v9 — v7 coordination + a STRONG "STEP FORWARD" reward (user's insight: with
# wider feet the agent is now STABLE but won't DISPLACE; reward the swing foot
# planting AHEAD of the torso = a real forward step, and pair with higher exploration
# noise to force leg movement so it DISCOVERS stepping now that it can catch itself).
func _reward_locomotion_v9(metrics: Dictionary, done: bool) -> float:
	if done:
		_last_sole = -1
		return -1.0
	var uprightness = float(metrics.get("uprightness", 0.0))
	var torso_tilt = float(metrics.get("torso_tilt", 0.0))
	var forward_velocity = float(metrics.get("forward_velocity", 0.0))
	var left_c = float(metrics.get("left_contact", 0.0))
	var right_c = float(metrics.get("right_contact", 0.0))
	var l_hip := float(metrics.get("j_l_hip_x", 0.0))
	var r_hip := float(metrics.get("j_r_hip_x", 0.0))
	var l_knee := float(metrics.get("j_l_knee", 0.0))
	var r_knee := float(metrics.get("j_r_knee", 0.0))
	var l_foot_fwd := float(metrics.get("l_foot_fwd", 0.0))
	var r_foot_fwd := float(metrics.get("r_foot_fwd", 0.0))
	var upright_gate = clampf(uprightness, 0.0, 1.0)

	# Coordination posture (phase-free, from v7).
	var hip_anti := clampf(-l_hip * r_hip / (GAIT_HIP_AMP * GAIT_HIP_AMP), 0.0, 1.0)
	var swing_knee := 0.0
	if left_c < 0.5:
		swing_knee += clampf(l_knee / GAIT_KNEE_AMP, 0.0, 1.0)
	if right_c < 0.5:
		swing_knee += clampf(r_knee / GAIT_KNEE_AMP, 0.0, 1.0)
	swing_knee = clampf(swing_knee, 0.0, 1.0)
	var coord := 0.5 * hip_anti + 0.5 * swing_knee

	# STEP-FORWARD: the SWING foot (airborne) planted AHEAD of the torso.
	var swing_advance := 0.0
	if left_c < 0.5:
		swing_advance = maxf(swing_advance, clampf(l_foot_fwd / 0.25, 0.0, 1.0))
	if right_c < 0.5:
		swing_advance = maxf(swing_advance, clampf(r_foot_fwd / 0.25, 0.0, 1.0))

	# Stepping rhythm: single-support + stance-foot alternation (from v7).
	var single_support = 1.0 if absf((left_c + right_c) - 1.0) < 0.5 else 0.0
	var alternation_reward = 0.0
	if single_support > 0.5:
		var sole := 0 if left_c > 0.5 else 1
		if _last_sole >= 0 and sole != _last_sole:
			alternation_reward = 0.6 * upright_gate
		_last_sole = sole

	var alive_bonus = 0.3
	var coord_reward = 0.7 * coord * upright_gate
	var step_advance_reward = 1.2 * swing_advance * upright_gate   # STRONG — advance a foot forward
	var swing_reward = 0.3 * single_support * upright_gate
	var fwd_score = clampf(forward_velocity / 0.6, 0.0, 1.0)
	var forward_reward = 0.6 * upright_gate * fwd_score
	var tilt_penalty = 0.2 * torso_tilt

	var reward = alive_bonus + coord_reward + step_advance_reward + swing_reward + alternation_reward + forward_reward - tilt_penalty
	return clampf(reward, -1.0, 4.0)


# J_walk v7 — PHASE-FREE GAIT-COORDINATION IMITATION. Strict DeepMimic needs a phase
# clock as a POLICY INPUT (the policy must know where it is in the cycle); adding that
# breaks the fixed 74-d proprio contract, and a phase-locked reference with no phase
# input is unlearnable (the policy converges to the phase-AVERAGE = neutral legs). So
# v7 imitates the walking COORDINATION PATTERN, computable from observable state with
# no clock: (a) HIPS in anti-phase (one leg fore, one aft) — a DENSE gradient toward a
# stepping stance even from standstill; (b) the SWING leg's knee FLEXED (the lifted
# foot bends its knee) — the core of a step; (c) foot-contact ALTERNATION + forward
# velocity for real stepping movement. Key dof: 0=L_hip_x,2=R_hip_x,4=L_knee,5=R_knee.
const GAIT_HIP_AMP := 0.35   # hip swing scale (rad)
const GAIT_KNEE_AMP := 0.5   # knee flexion scale (rad)

func _reward_locomotion_v7(metrics: Dictionary, done: bool) -> float:
	if done:
		_last_sole = -1
		return -1.0
	var uprightness = float(metrics.get("uprightness", 0.0))
	var torso_tilt = float(metrics.get("torso_tilt", 0.0))
	var forward_velocity = float(metrics.get("forward_velocity", 0.0))
	var left_c = float(metrics.get("left_contact", 0.0))
	var right_c = float(metrics.get("right_contact", 0.0))
	var l_hip := float(metrics.get("j_l_hip_x", 0.0))
	var r_hip := float(metrics.get("j_r_hip_x", 0.0))
	var l_knee := float(metrics.get("j_l_knee", 0.0))
	var r_knee := float(metrics.get("j_r_knee", 0.0))
	var upright_gate = clampf(uprightness, 0.0, 1.0)

	# (a)+(b) coordination posture (dense, phase-free).
	# anti-phase hips: product < 0 (opposite) is good.
	var hip_anti := clampf(-l_hip * r_hip / (GAIT_HIP_AMP * GAIT_HIP_AMP), 0.0, 1.0)
	# swing-leg knee flexed: the airborne foot (contact ~0) should bend its knee.
	var swing_knee := 0.0
	if left_c < 0.5:
		swing_knee += clampf(l_knee / GAIT_KNEE_AMP, 0.0, 1.0)
	if right_c < 0.5:
		swing_knee += clampf(r_knee / GAIT_KNEE_AMP, 0.0, 1.0)
	swing_knee = clampf(swing_knee, 0.0, 1.0)
	var coord := 0.5 * hip_anti + 0.5 * swing_knee

	# (c) stepping rhythm: single-support + stance-foot alternation.
	var single_support = 1.0 if absf((left_c + right_c) - 1.0) < 0.5 else 0.0
	var alternation_reward = 0.0
	if single_support > 0.5:
		var sole := 0 if left_c > 0.5 else 1
		if _last_sole >= 0 and sole != _last_sole:
			alternation_reward = 0.6 * upright_gate
		_last_sole = sole

	var alive_bonus = 0.3
	var coord_reward = 1.0 * coord * upright_gate          # dense walking-posture gradient
	var swing_reward = 0.4 * single_support * upright_gate
	var fwd_score = clampf(forward_velocity / 0.6, 0.0, 1.0)
	var forward_reward = 0.6 * upright_gate * fwd_score
	var tilt_penalty = 0.2 * torso_tilt

	var reward = alive_bonus + coord_reward + swing_reward + alternation_reward + forward_reward - tilt_penalty
	return clampf(reward, -1.0, 4.0)


# J_walk v6 — v5 (linear-to-target gait) but with the target velocity ENV-DRIVEN so
# a curriculum can ramp it. Identical reward shape to v5; only target_v changes.
func _reward_locomotion_v6(metrics: Dictionary, done: bool) -> float:
	if done:
		_last_sole = -1
		return -1.0
	var uprightness = float(metrics.get("uprightness", 0.0))
	var height = float(metrics.get("height", 0.0))
	var torso_tilt = float(metrics.get("torso_tilt", 0.0))
	var forward_velocity = float(metrics.get("forward_velocity", 0.0))
	var left_c = float(metrics.get("left_contact", 0.0))
	var right_c = float(metrics.get("right_contact", 0.0))

	var target_v = _target_velocity()
	var alive_bonus = 0.3
	var height_score = clampf((height - 0.35) / (0.85 - 0.35), 0.0, 1.0)
	var upright_gate = clampf(uprightness, 0.0, 1.0)
	var vel_score = clampf(forward_velocity / max(0.05, target_v), 0.0, 1.0)
	var forward_reward = 1.0 * upright_gate * vel_score

	var single_support = 1.0 if absf((left_c + right_c) - 1.0) < 0.5 else 0.0
	var swing_reward = 0.4 * single_support * upright_gate
	var alternation_reward = 0.0
	if single_support > 0.5:
		var sole := 0 if left_c > 0.5 else 1
		if _last_sole >= 0 and sole != _last_sole:
			alternation_reward = 0.6 * upright_gate
		_last_sole = sole

	var tilt_penalty = 0.2 * torso_tilt
	var reward = alive_bonus + (0.3 * height_score) + forward_reward + swing_reward + alternation_reward - tilt_penalty
	return clampf(reward, -1.0, 4.0)


# J_walk v5 — GAIT + a gradient OUT of standstill. v4 (target-velocity gait) got
# STUCK STANDING: warm-started from J0 it sat flat-footed (single_support ~9%,
# fwd_vel ~0) because NOTHING pulled it to move — the triangular target-velocity
# term is ~0 (and flat) at v=0, and the gait bonuses only fire once already
# stepping. v5 swaps the velocity term for a LINEAR ramp 0->1 as v goes 0->target,
# CAPPED at the target (so there is a positive gradient from standstill, but still
# no overshoot/dive bonus). Gait terms (single-support + alternation) unchanged.
func _reward_locomotion_v5(metrics: Dictionary, done: bool) -> float:
	if done:
		_last_sole = -1
		return -1.0
	var uprightness = float(metrics.get("uprightness", 0.0))
	var height = float(metrics.get("height", 0.0))
	var torso_tilt = float(metrics.get("torso_tilt", 0.0))
	var forward_velocity = float(metrics.get("forward_velocity", 0.0))
	var left_c = float(metrics.get("left_contact", 0.0))
	var right_c = float(metrics.get("right_contact", 0.0))

	var target_v = 0.6
	var alive_bonus = 0.3
	var height_score = clampf((height - 0.35) / (0.85 - 0.35), 0.0, 1.0)
	var upright_gate = clampf(uprightness, 0.0, 1.0)
	# Linear ramp from standstill to the target, capped (gradient out of v=0; no dive).
	var vel_score = clampf(forward_velocity / target_v, 0.0, 1.0)
	var forward_reward = 1.0 * upright_gate * vel_score

	var single_support = 1.0 if absf((left_c + right_c) - 1.0) < 0.5 else 0.0
	var swing_reward = 0.4 * single_support * upright_gate
	var alternation_reward = 0.0
	if single_support > 0.5:
		var sole := 0 if left_c > 0.5 else 1
		if _last_sole >= 0 and sole != _last_sole:
			alternation_reward = 0.6 * upright_gate
		_last_sole = sole

	var tilt_penalty = 0.2 * torso_tilt
	var reward = alive_bonus + (0.3 * height_score) + forward_reward + swing_reward + alternation_reward - tilt_penalty
	return clampf(reward, -1.0, 4.0)


# J_walk v4 — GAIT shaping. v1/v2/v3 proved that rewarding VELOCITY alone never
# induces a stepping gait (drift / dive / unstable lurch): you cannot just "add
# forward speed" to a stander — walking needs an alternating step cycle that stays
# dynamically stable WHILE moving. v4 rewards the gait directly, using the per-foot
# ground-contact signals (reward-only, already computed in Godot):
#   - keep the v3 target-velocity term (hold ~0.6 m/s, no dive overshoot bonus),
#   - SINGLE-SUPPORT bonus: credit having exactly one foot down (i.e. a foot in the
#     air mid-step) — rewards lifting feet instead of shuffling flat-footed,
#   - ALTERNATION bonus: credit each switch of the sole-contact foot (left->right->
#     left ...) — rewards a real walking rhythm, not hopping on one foot.
# All gait terms are gated by uprightness (a topple earns nothing) and done=-1.
# Expect to TUNE the weights (warned). _last_sole carries the alternation state
# across steps (reward_manager is a persistent singleton); reset on done.
func _reward_locomotion_v4(metrics: Dictionary, done: bool) -> float:
	if done:
		_last_sole = -1
		return -1.0
	var uprightness = float(metrics.get("uprightness", 0.0))
	var height = float(metrics.get("height", 0.0))
	var torso_tilt = float(metrics.get("torso_tilt", 0.0))
	var forward_velocity = float(metrics.get("forward_velocity", 0.0))
	var left_c = float(metrics.get("left_contact", 0.0))
	var right_c = float(metrics.get("right_contact", 0.0))

	var target_v = 0.6
	var alive_bonus = 0.3
	var height_score = clampf((height - 0.35) / (0.85 - 0.35), 0.0, 1.0)
	var upright_gate = clampf(uprightness, 0.0, 1.0)
	var vel_score = clampf(1.0 - absf(forward_velocity - target_v) / target_v, 0.0, 1.0)
	var forward_reward = 1.0 * upright_gate * vel_score

	# Gait: single-support + alternation of the stance foot.
	var single_support = 1.0 if absf((left_c + right_c) - 1.0) < 0.5 else 0.0
	var swing_reward = 0.4 * single_support * upright_gate
	var alternation_reward = 0.0
	if single_support > 0.5:
		var sole := 0 if left_c > 0.5 else 1
		if _last_sole >= 0 and sole != _last_sole:
			alternation_reward = 0.6 * upright_gate
		_last_sole = sole

	var tilt_penalty = 0.2 * torso_tilt
	var reward = alive_bonus + (0.3 * height_score) + forward_reward + swing_reward + alternation_reward - tilt_penalty
	return clampf(reward, -1.0, 4.0)


# J_walk v3 — TARGET-VELOCITY. v1 (forward too weak) drifted+survived; v2 (forward
# dominant, "faster=better") DIVED forward and fell at ~100 steps. Both are
# degenerate optima because the gait is never the unique best. v3 rewards being
# CLOSE to a moderate target speed: max at the target, 0 at standstill AND at >=2x
# target — so overshooting (the dive) earns nothing extra, and the ONLY way to keep
# scoring is to hold the target speed upright for the whole episode = walk. Gated by
# uprightness (a topple kills the credit) and done=-1.
func _reward_locomotion_v3(metrics: Dictionary, done: bool) -> float:
	if done:
		return -1.0
	var uprightness = float(metrics.get("uprightness", 0.0))
	var height = float(metrics.get("height", 0.0))
	var torso_tilt = float(metrics.get("torso_tilt", 0.0))
	var forward_velocity = float(metrics.get("forward_velocity", 0.0))

	var target_v = 0.6                                              # m/s walking pace
	var alive_bonus = 0.3
	var height_score = clampf((height - 0.35) / (0.85 - 0.35), 0.0, 1.0)
	var upright_gate = clampf(uprightness, 0.0, 1.0)
	# Triangular: 1 at target_v, linearly to 0 at v=0 and v=2*target_v (no overshoot bonus).
	var vel_score = clampf(1.0 - absf(forward_velocity - target_v) / target_v, 0.0, 1.0)
	var forward_reward = 1.5 * upright_gate * vel_score
	var tilt_penalty = 0.2 * torso_tilt

	var reward = alive_bonus + (0.3 * height_score) + forward_reward - tilt_penalty
	return clampf(reward, -1.0, 4.0)


# J_walk: reward forward displacement while staying upright. forward_velocity is the
# SIGNED torso-forward speed (linear_velocity · torso_forward). It is gated by
# uprightness so brief forward speed gained by toppling earns ~nothing, and capped
# so the agent cannot "win" by exploding/sprinting (cf. motor breakdance-jumping).
# alive_bonus + height keep "crawl / lie down and slide" from becoming the optimum;
# done = -1 makes a fall strictly worse than sustained upright walking.
#
# v2 REBALANCE: v1 plateaued — the agent drifted forward at ~0.3 m/s and TRADED
# speed for survival (len up, fwd_vel down) because the forward term (weight 1.0,
# ~0.3/step) was dwarfed by alive+height (~0.8/step), so "stand still and survive"
# beat "commit to walking". v2 makes forward DOMINANT: alive 0.5->0.3, forward
# weight 1.0->2.0, reward ceiling 3.0->4.0 (so the gradient survives up to the
# 1.5 m/s cap instead of saturating). Now standing ~0.57/step vs walking 1 m/s
# ~2.37/step — walking strictly dominates, while the uprightness gate + done=-1
# still forbid the dive.
func _reward_locomotion_v2(metrics: Dictionary, done: bool) -> float:
	if done:
		return -1.0
	var uprightness = float(metrics.get("uprightness", 0.0))                 # [0,1]
	var height = float(metrics.get("height", 0.0))                           # metres (standing ~0.9)
	var torso_tilt = float(metrics.get("torso_tilt", 0.0))                   # 1 - uprightness
	var forward_velocity = float(metrics.get("forward_velocity", 0.0))       # m/s, signed forward

	var alive_bonus = 0.3
	var height_score = clampf((height - 0.35) / (0.85 - 0.35), 0.0, 1.0)
	var upright_gate = clampf(uprightness, 0.0, 1.0)
	# Cap to a walking pace: reward progress up to ~1.5 m/s, no credit for sprinting
	# beyond it, mild floor so small backward slips are not catastrophic.
	var fwd = clampf(forward_velocity, -0.5, 1.5)
	var forward_reward = 2.0 * upright_gate * fwd          # the DOMINANT driver
	var tilt_penalty = 0.2 * torso_tilt

	var reward = alive_bonus + (0.3 * height_score) + forward_reward - tilt_penalty
	return clampf(reward, -1.0, 4.0)


# J0: STATIC active balance (kept verbatim — produced policy_best.pt, 401/401).
func _reward_active_balance_v3(metrics: Dictionary, done: bool) -> float:
	var uprightness = float(metrics.get("uprightness", 0.0))
	var height = float(metrics.get("height", 0.0))
	var torso_tilt = float(metrics.get("torso_tilt", 0.0))
	var horizontal_speed = float(metrics.get("horizontal_speed", 0.0))
	var com_offset = float(metrics.get("com_support_offset", 0.0))

	if done:
		return -1.0

	var alive_bonus = 0.5
	var height_score = clampf((height - 0.35) / (0.85 - 0.35), 0.0, 1.0)
	var tilt_penalty = 0.3 * torso_tilt
	var drift_penalty = 0.7 * clampf(com_offset / 0.12, 0.0, 3.0)
	var speed_penalty = 0.3 * clampf(horizontal_speed, 0.0, 1.0)

	var reward = alive_bonus + (0.6 * uprightness) + (0.5 * height_score) - tilt_penalty - drift_penalty - speed_penalty
	return clampf(reward, -1.0, 3.0)
