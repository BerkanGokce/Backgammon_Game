"""Replaceable, documented neural-network encodings."""

from __future__ import annotations

import numpy as np

from backgammon.constants import BAR, OFF, WHITE, Player
from backgammon.game import GameState
from backgammon.moves import TurnAction

MAX_MOVES = 4
LOCATION_COUNT = 26  # points 0..23, BAR, OFF
MOVE_FEATURES = LOCATION_COUNT * 2 + 6 + 1
ACTION_DIM = MAX_MOVES * MOVE_FEATURES + 1  # 237
OBSERVATION_DIM = 24 * 2 + 2 + 2 + 1 + 6 + 1  # 60


def _relative_physical_point(relative_point: int, player: Player) -> int:
    """Both players see point 0 as their bearing-off edge."""

    return 23 - relative_point if player == WHITE else relative_point


def encode_state(state: GameState) -> np.ndarray:
    """Encode a 60-value decision state from the player-to-move perspective.

    Features are 24 pairs of own/opponent checker counts (divided by 15), own
    and opponent bar/off counts, physical player sign, six remaining-die count
    features (divided by four), and a doubles flag. Dice have already been
    rolled, so this is a post-chance-node decision state.
    """

    player = state.current_player
    opponent = Player(-int(player))
    values: list[float] = []
    for relative in range(24):
        physical = _relative_physical_point(relative, player)
        count = state.board.points[physical]
        values.append(max(0, count * int(player)) / 15.0)
        values.append(max(0, count * int(opponent)) / 15.0)
    values.extend(
        (
            state.board.bar[player] / 15.0,
            state.board.bar[opponent] / 15.0,
            state.board.off[player] / 15.0,
            state.board.off[opponent] / 15.0,
            float(player),
        )
    )
    values.extend(state.remaining_dice.count(face) / 4.0 for face in range(1, 7))
    values.append(float(state.dice[0] == state.dice[1]))
    encoded = np.asarray(values, dtype=np.float32)
    if encoded.shape != (OBSERVATION_DIM,):
        raise AssertionError(f"Unexpected state encoding shape: {encoded.shape}")
    return encoded


def _location_index(location: int, player: Player) -> int:
    if 0 <= location < 24:
        return 23 - location if player == WHITE else location
    if location == BAR:
        return 24
    if location == OFF:
        return 25
    raise ValueError(f"Unknown Backgammon location: {location}")


def encode_action(action: TurnAction, player: Player = WHITE) -> np.ndarray:
    """Encode up to four ordered checker movements as 237 float features."""

    if len(action.moves) > MAX_MOVES:
        raise ValueError("A Backgammon action cannot contain more than four moves")
    encoded = np.zeros(ACTION_DIM, dtype=np.float32)
    for slot, move in enumerate(action.moves):
        offset = slot * MOVE_FEATURES
        encoded[offset + _location_index(move.source, player)] = 1.0
        encoded[offset + LOCATION_COUNT + _location_index(move.destination, player)] = (
            1.0
        )
        encoded[offset + LOCATION_COUNT * 2 + move.die - 1] = 1.0
        encoded[offset + MOVE_FEATURES - 1] = 1.0
    encoded[-1] = len(action.moves) / MAX_MOVES
    return encoded


def encode_actions(actions: list[TurnAction], player: Player = WHITE) -> np.ndarray:
    if not actions:
        return np.empty((0, ACTION_DIM), dtype=np.float32)
    return np.stack([encode_action(action, player) for action in actions])
