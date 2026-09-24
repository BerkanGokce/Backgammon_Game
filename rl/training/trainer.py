"""Training orchestration, checkpointing, evaluation, and TensorBoard logging."""

from __future__ import annotations

import logging
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from backgammon.constants import BLACK, WHITE
from rl.agents.heuristic_agent import HeuristicAgent
from rl.agents.random_agent import RandomAgent
from rl.agents.rl_agent import RLAgent
from rl.config import TrainingConfig
from rl.environment import BackgammonEnv
from rl.replay_buffer import ReplayBuffer

from .evaluation import evaluate_agents
from .self_play import play_self_play_episode

LOGGER = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(
        self, config: TrainingConfig, resume: str | Path | None = None
    ) -> None:
        self.config = config
        seed_everything(config.seed)
        self.start_episode = 0
        if resume:
            self.agent, metadata = RLAgent.from_checkpoint(
                resume, device=config.device, load_optimizer=True
            )
            self.start_episode = int(metadata.get("episode", 0))
            # Architecture comes from the checkpoint; run-control values come
            # from the new invocation.
            config.hidden_size = self.agent.config.hidden_size
            config.action_hidden_size = self.agent.config.action_hidden_size
            self.agent.config = config
            self.agent.replay = ReplayBuffer(config.replay_buffer_size, config.seed)
            for group in self.agent.optimizer.param_groups:
                group["lr"] = config.learning_rate
        else:
            self.agent = RLAgent(config)
        self.env = BackgammonEnv(
            seed=config.seed,
            use_result_points=config.use_result_points,
            max_turns=config.max_turns,
        )
        self.writer = SummaryWriter(log_dir=config.log_dir)
        self.recent_points: deque[float] = deque(maxlen=100)
        self.recent_turns: deque[int] = deque(maxlen=100)
        self.snapshot_pool: deque[RLAgent] = deque(maxlen=config.snapshot_pool_size)
        self.training_rng = random.Random(config.seed + 991)
        self.best_evaluation_score = float("-inf")

    def train(self) -> RLAgent:
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        final_episode = self.start_episode + self.config.episodes
        try:
            for episode in range(self.start_episode + 1, final_episode + 1):
                frozen = None
                learner_color = None
                if (
                    self.snapshot_pool
                    and self.training_rng.random() < self.config.snapshot_probability
                ):
                    frozen = self.training_rng.choice(list(self.snapshot_pool))
                    learner_color = WHITE if episode % 2 == 0 else BLACK
                result = play_self_play_episode(
                    self.env,
                    self.agent,
                    seed=self.config.seed + episode,
                    frozen_opponent=frozen,
                    learner_color=learner_color,
                )
                self.recent_points.append(result.points)
                self.recent_turns.append(result.turns)
                self.writer.add_scalar("train/result_points", result.points, episode)
                self.writer.add_scalar("train/game_length", result.turns, episode)
                self.writer.add_scalar("train/epsilon", self.agent.epsilon, episode)
                if result.average_loss is not None:
                    self.writer.add_scalar("train/loss", result.average_loss, episode)
                if episode % 100 == 0:
                    self.writer.add_scalar(
                        "train/average_points_100",
                        sum(self.recent_points) / len(self.recent_points),
                        episode,
                    )
                    LOGGER.info(
                        "episode=%d epsilon=%.3f turns=%.1f replay=%d",
                        episode,
                        self.agent.epsilon,
                        sum(self.recent_turns) / len(self.recent_turns),
                        len(self.agent.replay),
                    )
                if episode % self.config.evaluation_frequency == 0:
                    score = self._evaluate_and_log(episode)
                    if score > self.best_evaluation_score:
                        self.best_evaluation_score = score
                        self.agent.save_checkpoint(
                            checkpoint_dir / "best_model.pt",
                            episode,
                            {"evaluation_score": score},
                        )
                if episode % self.config.checkpoint_frequency == 0:
                    path = checkpoint_dir / f"checkpoint_{episode:06d}.pt"
                    self.agent.save_checkpoint(path, episode)
                if episode % self.config.snapshot_frequency == 0:
                    self._add_snapshot()
            self.agent.save_checkpoint(
                checkpoint_dir / "latest_model.pt", final_episode
            )
            if self.best_evaluation_score == float("-inf"):
                self.agent.save_checkpoint(
                    checkpoint_dir / "best_model.pt", final_episode
                )
            return self.agent
        finally:
            self.writer.flush()
            self.writer.close()

    def _evaluate_and_log(self, episode: int) -> float:
        games = self.config.evaluation_games
        random_result = evaluate_agents(
            self.agent, RandomAgent(self.config.seed), games, self.config.seed + episode
        )
        heuristic_result = evaluate_agents(
            self.agent,
            HeuristicAgent(self.config.seed),
            games,
            self.config.seed + episode,
        )
        self.writer.add_scalar(
            "evaluation/win_rate_random", random_result.win_rate, episode
        )
        self.writer.add_scalar(
            "evaluation/win_rate_heuristic", heuristic_result.win_rate, episode
        )
        self.writer.add_scalar(
            "evaluation/average_points_random", random_result.average_points, episode
        )
        return random_result.win_rate + heuristic_result.win_rate

    def _add_snapshot(self) -> None:
        snapshot = RLAgent(self.config)
        snapshot.online_network.load_state_dict(self.agent.online_network.state_dict())
        snapshot.target_network.load_state_dict(self.agent.online_network.state_dict())
        snapshot.online_network.eval()
        self.snapshot_pool.append(snapshot)
