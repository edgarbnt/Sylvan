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

# J2 full body (arms + neck/head): proprio 74->120, action 10->18. Layout in
# godot/scripts/agent/sylvan_agent.gd::_rebuild_proprioception (12 bodies, 18 DOF).
# +2 (120->122): gait phase clock [sin 2πφ, cos 2πφ] APPENDED last, the policy input
# for the periodic walk reward (Siekmann/Cassie). Warm-starts zero-pad these 2 columns.
# PERCEPTION (J2 tranche perception): vision_shape (0,) -> (12,) = the egocentric food
# radar (12 angular sectors, proximity in [0,1]; godot/scripts/agent/perception.gd). The
# policy input is the CONCATENATION [proprio(122) ++ vision(12)] = 134; vision is appended
# LAST so warm-starts zero-pad those 12 columns → the walk is preserved until it learns
# to use the radar (same trick as the gait-phase 120->122 above).
DEFAULT_PROPRIO_DIM = 122
DEFAULT_ACTION_DIM = 18
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

RUN_METADATA_FILENAME = "run_metadata.json"
EPISODE_FILE_SUFFIX = ".jsonl"
