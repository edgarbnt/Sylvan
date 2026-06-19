"""Configuration helpers for the Sylvan Phase 1 stack."""

from dataclasses import dataclass, field
from pathlib import Path

from .constants import (
    CHECKPOINTS_DIR,
    DEFAULT_ACTION_DIM,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_METRICS_DIM,
    DEFAULT_NUM_EPISODES,
    DEFAULT_PROPRIO_DIM,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_TRAIN_STEPS,
    DEFAULT_VISION_SHAPE,
    GODOT_PROJECT_DIR,
    REPORTS_DIR,
    REPLAY_BUFFER_DIR,
)


@dataclass(slots=True)
class PathsConfig:
    replay_buffer_dir: Path = REPLAY_BUFFER_DIR
    checkpoints_dir: Path = CHECKPOINTS_DIR
    reports_dir: Path = REPORTS_DIR


@dataclass(slots=True)
class EnvConfig:
    proprio_dim: int = DEFAULT_PROPRIO_DIM
    action_dim: int = DEFAULT_ACTION_DIM
    metrics_dim: int = DEFAULT_METRICS_DIM
    vision_shape: tuple[int, ...] = DEFAULT_VISION_SHAPE
    max_episode_steps: int = 400  # Long enough to see sustained standing, short enough to cycle fast
    seed: int = 42

    @property
    def vision_dim(self) -> int:
        d = 1
        for n in self.vision_shape:
            d *= n
        return d

    @property
    def policy_input_dim(self) -> int:
        # The network consumes [proprio ++ vision]; this is its true input width.
        return self.proprio_dim + self.vision_dim

    @property
    def wm_obs_dim(self) -> int:
        # The WORLD MODEL consumes [proprio ++ vision ++ energy(1, normalised to [0,1])].
        # Energy is what makes the latent "hungry" → the planner can minimise predicted
        # hunger (the LeCun intrinsic cost) and so navigate to food.
        return self.proprio_dim + self.vision_dim + 1


@dataclass(slots=True)
class TrainConfig:
    batch_size: int = DEFAULT_BATCH_SIZE
    # 16->32: the controller imagines `imagined_horizon` (30) steps, but the WM was
    # only trained on 16-step sequences, so it extrapolated past its training length
    # in imagination (part of why the dream had no drift). Train on >= the horizon.
    sequence_length: int = 32
    train_steps: int = DEFAULT_TRAIN_STEPS
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = 3e-4
    latent_dim: int = 128
    hidden_dim: int = 256
    replay_window_size: int = 8


@dataclass(slots=True)
class ControllerConfig:
    hidden_dim: int = 128
    learning_rate: float = 3e-4
    epochs: int = 50
    imagined_horizon: int = 30  # Longer dream horizon → anticipatory balance (catch slow drift early)
    discount: float = 0.97
    exploration_noise_initial: float = 0.5
    exploration_noise_final: float = 0.15
    replay_window_size: int = 8
    # Number of imagined-rollout batches per epoch. Was hard-coded to 20, which
    # starved the controller vs the World Model (200/epoch). Parity helps it learn.
    imagination_batches: int = 150
    # Exploration noise injected into imagined rollouts (kept config-driven instead
    # of a magic 0.1) to avoid deterministic policy saturation in the dreams.
    # Bumped 0.15->0.30 (active_balance): with a still imagined trajectory the
    # horizontal-COM-speed reward penalty never fires (imagined h_speed~0) and the
    # actor stays pinned at the upright-return cap. More noise knocks the imagined
    # agent off-balance so the penalty bites and a balance gradient appears.
    imagination_noise: float = 0.30


@dataclass(slots=True)
class DayConfig:
    num_episodes: int = DEFAULT_NUM_EPISODES
    run_name: str = "phase1_day"
    collector: str = "godot"


@dataclass(slots=True)
class GodotConfig:
    executable: str = ""
    project_dir: Path = GODOT_PROJECT_DIR
    headless: bool = True
    use_policy: bool = False
    policy_host: str = "127.0.0.1"
    policy_port: int = 0


@dataclass(slots=True)
class SylvanConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    day: DayConfig = field(default_factory=DayConfig)
    godot: GodotConfig = field(default_factory=GodotConfig)

    def __post_init__(self) -> None:
        import os
        # Apply overrides from environment variables (especially useful for subprocesses)
        if "SYLVAN_STEPS_PER_DAY" in os.environ:
            try:
                self.env.max_episode_steps = int(os.environ["SYLVAN_STEPS_PER_DAY"])
            except ValueError:
                pass
        if "SYLVAN_BATCH_SIZE" in os.environ:
            try:
                self.train.batch_size = int(os.environ["SYLVAN_BATCH_SIZE"])
            except ValueError:
                pass
        if "SYLVAN_EPOCHS_PER_NIGHT" in os.environ:
            try:
                epochs = int(os.environ["SYLVAN_EPOCHS_PER_NIGHT"])
                self.train.epochs = epochs
                self.controller.epochs = epochs
            except ValueError:
                pass
        if "SYLVAN_WM_EPOCHS" in os.environ:
            try:
                self.train.epochs = int(os.environ["SYLVAN_WM_EPOCHS"])
            except ValueError:
                pass
        if "SYLVAN_CONTROLLER_EPOCHS" in os.environ:
            try:
                self.controller.epochs = int(os.environ["SYLVAN_CONTROLLER_EPOCHS"])
            except ValueError:
                pass

    def ensure_directories(self) -> None:
        self.paths.replay_buffer_dir.mkdir(parents=True, exist_ok=True)
        self.paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.paths.reports_dir.mkdir(parents=True, exist_ok=True)
