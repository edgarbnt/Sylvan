"""Shared constants for the Sylvan Phase 1 walking skeleton."""

from pathlib import Path

SCHEMA_VERSION = "phase1.v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REPLAY_BUFFER_DIR = DATA_DIR / "replay_buffer"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
REPORTS_DIR = DATA_DIR / "reports"
GODOT_PROJECT_DIR = PROJECT_ROOT / "godot"

DEFAULT_BATCH_SIZE = 8
DEFAULT_SEQUENCE_LENGTH = 16
DEFAULT_TRAIN_STEPS = 100
DEFAULT_EPOCHS = 5
DEFAULT_NUM_EPISODES = 40

# QUADRUPED REDESIGN (2026-06-08): body changed bipedal humanoid -> dog/lizard quadruped
# (trunk + 4 legs × 3 DOF). New contract. Layout in
# godot/scripts/agent/sylvan_agent.gd::_rebuild_proprioception:
#   7 (height + trunk lin/ang vel) + 9 bodies × 6 (=54) + 4 foot contacts + 3 COM
#   + 12 joint angles + 12 joint velocities + 2 gait-phase clock [sin 2πφ, cos 2πφ] = 94.
# action_dim = 12 (per leg: hip_x sagittal swing, hip_z lateral abduction, knee_x flexion).
# PERCEPTION: vision_shape (12,) = the egocentric food radar (12 angular sectors, proximity
# in [0,1]; godot/scripts/agent/perception.gd). The policy input is the CONCATENATION
# [proprio(94) ++ vision(12)] = 106; vision is appended LAST so warm-starts zero-pad those
# 12 columns. This change invalidates all bipedal checkpoints (walk/survival/WM) — fresh start.
# SALAMANDER morphology (2026-06-15): the single rigid torso is split into TWO segments
# (front="torso", rear="torso_back") joined by a lateral-bend SPINE joint (yaw). Turning is by
# bending the spine (research: breaks the skid yaw ceiling), not skid alone. Dims grow:
# proprio 94→102 (10 bodies×6 + 13 joint angles + 13 vels), action 12→13 (+spine). This
# invalidates ALL quadruped checkpoints (residual7/ft3/bc_init/wm) — fresh CPG→BC→RL base.
DEFAULT_PROPRIO_DIM = 132  # HEXAPOD: 7 + 13 bodies×6(=78) + 6 contacts + 3 COM + 18 angles + 18 vels + 2 gait
DEFAULT_ACTION_DIM = 18     # HEXAPOD: 6 legs×3 (hip_x,hip_z,knee), no spine
DEFAULT_VISION_SHAPE = (12,)
LOCOMOTION_METRIC_KEYS = (
    "uprightness",
    "forward_velocity",
    "torso_tilt",
    "height",
    "ground_contact",
    "effort",
    "pose_error",
)
DEFAULT_METRICS_DIM = len(LOCOMOTION_METRIC_KEYS)

# Homeostasis max_energy (godot/scripts/agent/homeostasis.gd). Used to NORMALISE energy
# into [0,1] before it enters the world-model observation, so the 1-d energy signal sits
# on the same scale as the radar [0,1] and proprio (~O(1)) instead of being a raw 0..100
# value that would dominate / be ignored. The WM obs is [proprio ++ vision ++ energy/100].
DEFAULT_MAX_ENERGY = 100.0

RUN_METADATA_FILENAME = "run_metadata.json"
EPISODE_FILE_SUFFIX = ".jsonl"
