"""Reinforcement-learning integration for Backgammon."""

from .encoding import ACTION_DIM, OBSERVATION_DIM, encode_action, encode_state
from .environment import BackgammonEnv

__all__ = [
    "ACTION_DIM",
    "OBSERVATION_DIM",
    "BackgammonEnv",
    "encode_action",
    "encode_state",
]
