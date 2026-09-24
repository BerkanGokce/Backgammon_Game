"""Copy-safe experience replay for variable legal action sets."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Transition:
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    next_actions: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int | None = None) -> None:
        self._items: deque[Transition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_actions: np.ndarray,
    ) -> None:
        self._items.append(
            Transition(
                state.copy(),
                action.copy(),
                float(reward),
                next_state.copy(),
                bool(done),
                next_actions.copy(),
            )
        )

    def sample(self, batch_size: int) -> list[Transition]:
        return self._rng.sample(list(self._items), batch_size)
