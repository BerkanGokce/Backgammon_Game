"""Central training configuration."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class TrainingConfig:
    episodes: int = 10_000
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 128
    replay_buffer_size: int = 100_000
    warmup_steps: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 200_000
    target_update_frequency: int = 1_000
    checkpoint_frequency: int = 1_000
    evaluation_frequency: int = 1_000
    evaluation_games: int = 100
    gradient_clip: float = 10.0
    hidden_size: int = 256
    action_hidden_size: int = 128
    seed: int = 42
    device: str = "auto"
    max_turns: int = 10_000
    use_result_points: bool = True
    snapshot_frequency: int = 1_000
    snapshot_probability: float = 0.20
    snapshot_pool_size: int = 3
    log_dir: str = "logs"
    checkpoint_dir: str = "models/checkpoints"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainingConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})
