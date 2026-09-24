"""Helpers for batching the variable legal action set."""

from __future__ import annotations

import numpy as np

from backgammon.constants import WHITE, Player
from backgammon.moves import TurnAction

from .encoding import encode_actions


def legal_action_batch(actions: list[TurnAction], player: Player = WHITE) -> np.ndarray:
    """Return one action feature row per engine-generated legal turn."""

    return encode_actions(actions, player)
