"""DQN-style state-action value agent for a variable legal action set."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from backgammon.constants import BLACK, WHITE
from backgammon.game import Game
from backgammon.moves import TurnAction
from rl.config import TrainingConfig
from rl.encoding import (
    ACTION_DIM,
    OBSERVATION_DIM,
    encode_action,
    encode_actions,
    encode_state,
)
from rl.networks import StateActionQNetwork
from rl.replay_buffer import ReplayBuffer

from .base_agent import BaseAgent


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def alternating_player_targets(
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Zero-sum target when the next encoded state belongs to the opponent."""

    return rewards - gamma * next_values * (~done)


class RLAgent(BaseAgent):
    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self.device = resolve_device(self.config.device)
        self.online_network = StateActionQNetwork(
            OBSERVATION_DIM,
            ACTION_DIM,
            self.config.hidden_size,
            self.config.action_hidden_size,
        ).to(self.device)
        self.target_network = StateActionQNetwork(
            OBSERVATION_DIM,
            ACTION_DIM,
            self.config.hidden_size,
            self.config.action_hidden_size,
        ).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=self.config.learning_rate
        )
        self.replay = ReplayBuffer(self.config.replay_buffer_size, self.config.seed)
        self.rng = random.Random(self.config.seed)
        self.steps_done = 0
        self.optimization_steps = 0

    @property
    def epsilon(self) -> float:
        progress = min(1.0, self.steps_done / max(1, self.config.epsilon_decay_steps))
        return self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def q_values(self, state: np.ndarray, actions: list[TurnAction]) -> np.ndarray:
        if not actions:
            return np.empty(0, dtype=np.float32)
        states = (
            torch.as_tensor(state, device=self.device)
            .unsqueeze(0)
            .repeat(len(actions), 1)
        )
        player = WHITE if state[52] > 0 else BLACK
        encoded_actions = torch.as_tensor(
            encode_actions(actions, player), device=self.device
        )
        self.online_network.eval()
        with torch.inference_mode():
            values = self.online_network(states, encoded_actions)
        return values.cpu().numpy()

    def select_action(
        self, game: Game, legal_actions: list[TurnAction], explore: bool = False
    ) -> TurnAction:
        actions = legal_actions or [TurnAction()]
        if explore:
            self.steps_done += 1
            if self.rng.random() < self.epsilon:
                return self.rng.choice(actions)
        values = self.q_values(encode_state(game.state), actions)
        return actions[int(np.argmax(values))]

    def remember(
        self,
        state: np.ndarray,
        action: TurnAction,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_actions: list[TurnAction],
    ) -> None:
        player = WHITE if state[52] > 0 else BLACK
        next_player = WHITE if next_state[52] > 0 else BLACK
        self.replay.add(
            state,
            encode_action(action, player),
            reward,
            next_state,
            done,
            encode_actions(next_actions, next_player),
        )

    def optimize(self) -> float | None:
        minimum = max(self.config.batch_size, self.config.warmup_steps)
        if len(self.replay) < minimum:
            return None
        batch = self.replay.sample(self.config.batch_size)
        states = torch.as_tensor(
            np.stack([item.state for item in batch]), device=self.device
        )
        actions = torch.as_tensor(
            np.stack([item.action for item in batch]), device=self.device
        )
        rewards = torch.as_tensor(
            [item.reward for item in batch], dtype=torch.float32, device=self.device
        )
        done = torch.as_tensor(
            [item.done for item in batch], dtype=torch.bool, device=self.device
        )

        self.online_network.train()
        predictions = self.online_network(states, actions)
        next_values = torch.zeros(len(batch), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            for index, transition in enumerate(batch):
                if transition.done or len(transition.next_actions) == 0:
                    continue
                next_state = (
                    torch.as_tensor(transition.next_state, device=self.device)
                    .unsqueeze(0)
                    .repeat(len(transition.next_actions), 1)
                )
                candidate_actions = torch.as_tensor(
                    transition.next_actions, device=self.device
                )
                next_values[index] = self.target_network(
                    next_state, candidate_actions
                ).max()
            # Encodings are always from the player to move. The next state is
            # the opponent's perspective, hence the zero-sum sign inversion.
            targets = alternating_player_targets(
                rewards, next_values, done, self.config.gamma
            )

        loss = nn.functional.smooth_l1_loss(predictions, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss became non-finite")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.online_network.parameters(), self.config.gradient_clip
        )
        self.optimizer.step()
        self.optimization_steps += 1
        if self.optimization_steps % self.config.target_update_frequency == 0:
            self.update_target()
        return float(loss.detach().cpu())

    def update_target(self) -> None:
        self.target_network.load_state_dict(self.online_network.state_dict())

    def save_checkpoint(
        self,
        path: str | Path,
        episode: int,
        statistics: dict[str, Any] | None = None,
    ) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": 1,
                "episode": episode,
                "steps_done": self.steps_done,
                "optimization_steps": self.optimization_steps,
                "config": self.config.to_dict(),
                "online_network": self.online_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "statistics": statistics or {},
            },
            destination,
        )
        return destination

    def load_checkpoint(
        self, path: str | Path, load_optimizer: bool = True
    ) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online_network.load_state_dict(checkpoint["online_network"])
        self.target_network.load_state_dict(
            checkpoint.get("target_network", checkpoint["online_network"])
        )
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps_done = int(checkpoint.get("steps_done", 0))
        self.optimization_steps = int(checkpoint.get("optimization_steps", 0))
        return checkpoint

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, device: str = "auto", load_optimizer: bool = False
    ) -> tuple[RLAgent, dict[str, Any]]:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        config = TrainingConfig.from_dict(raw.get("config", {}))
        config.device = device
        agent = cls(config)
        metadata = agent.load_checkpoint(path, load_optimizer=load_optimizer)
        return agent, metadata
