"""PyTorch state-action value network."""

from __future__ import annotations

import torch
from torch import nn


class StateActionQNetwork(nn.Module):
    """Score Q(s,a) for arbitrary batches of engine-generated actions."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        action_hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, action_hidden_size),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size + action_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        state_features = self.state_encoder(states)
        action_features = self.action_encoder(actions)
        return self.value_head(
            torch.cat((state_features, action_features), dim=-1)
        ).squeeze(-1)
