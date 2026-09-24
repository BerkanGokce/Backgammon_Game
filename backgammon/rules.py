"""Authoritative Backgammon legality and terminal-scoring rules."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from .board import Board
from .constants import (
    BAR,
    BLACK,
    CHECKERS_PER_PLAYER,
    NUM_POINTS,
    OFF,
    WHITE,
    Player,
    direction,
    entry_point,
    home_points,
    opponent,
)
from .moves import Move, TurnAction


def all_in_home(board: Board, player: Player) -> bool:
    if board.bar[player]:
        return False
    home = set(home_points(player))
    return all(
        not (value * int(player) > 0) or index in home
        for index, value in enumerate(board.points)
    )


def _can_bear_off(board: Board, player: Player, source: int, die: int) -> bool:
    if not all_in_home(board, player):
        return False
    exact_distance = 24 - source if player == WHITE else source + 1
    if die == exact_distance:
        return True
    if die < exact_distance:
        return False
    # An oversized die may remove only the checker farthest from OFF.
    if player == WHITE:
        return not any(board.points[index] > 0 for index in range(18, source))
    return not any(board.points[index] < 0 for index in range(source + 1, 6))


def legal_moves_for_die(board: Board, player: Player, die: int) -> list[Move]:
    """Generate legal single moves for one die in the current intermediate state."""

    if not 1 <= die <= 6:
        raise ValueError("A die must be between 1 and 6")
    sign = int(player)
    if board.bar[player] > 0:
        destination = entry_point(player, die)
        if board.points[destination] * sign >= -1:
            return [Move(BAR, destination, die)]
        return []

    result: list[Move] = []
    for source, count in enumerate(board.points):
        if count * sign <= 0:
            continue
        destination = source + direction(player) * die
        if 0 <= destination < NUM_POINTS:
            if board.points[destination] * sign >= -1:
                result.append(Move(source, destination, die))
        elif _can_bear_off(board, player, source, die):
            result.append(Move(source, OFF, die))
    return result


def _search_sequences(
    board: Board,
    player: Player,
    remaining: Counter[int],
    prefix: tuple[Move, ...],
    output: list[TurnAction],
) -> None:
    extended = False
    for die in sorted(remaining, reverse=True):
        if remaining[die] <= 0:
            continue
        moves = legal_moves_for_die(board, player, die)
        for move in moves:
            extended = True
            next_board = board.copy()
            next_board.apply_move(move, player)
            next_remaining = remaining.copy()
            next_remaining[die] -= 1
            _search_sequences(
                next_board, player, next_remaining, prefix + (move,), output
            )
    if not extended:
        output.append(TurnAction(prefix))


def expanded_dice(dice: Iterable[int]) -> tuple[int, ...]:
    pair = tuple(dice)
    if len(pair) == 2 and pair[0] == pair[1]:
        return (pair[0],) * 4
    return pair


def generate_legal_turns(
    board: Board,
    dice: Iterable[int],
    player: Player,
    *,
    expand_roll_doubles: bool = True,
) -> list[TurnAction]:
    """Return all complete sequences allowed by mandatory dice-usage rules.

    The search explores dice in every order against intermediate board states,
    keeps sequences using the maximum possible number of dice, and applies the
    higher-die rule when exactly one of two distinct dice can be played.
    """

    original = tuple(dice)
    if not original:
        return []
    available = expanded_dice(original) if expand_roll_doubles else original
    sequences: list[TurnAction] = []
    _search_sequences(board, player, Counter(available), (), sequences)
    max_length = max((len(action) for action in sequences), default=0)
    if max_length == 0:
        return []
    sequences = [action for action in sequences if len(action) == max_length]

    if max_length == 1 and len(set(original)) > 1:
        high_die = max(original)
        if any(action.moves[0].die == high_die for action in sequences):
            sequences = [
                action for action in sequences if action.moves[0].die == high_die
            ]

    # Equal dice can reach the same tuple through indistinguishable search paths.
    return list(dict.fromkeys(sequences))


def winner(board: Board) -> Player | None:
    if board.off[WHITE] == CHECKERS_PER_PLAYER:
        return WHITE
    if board.off[BLACK] == CHECKERS_PER_PLAYER:
        return BLACK
    return None


def win_multiplier(board: Board, winning_player: Player) -> int:
    """Return 1 (single), 2 (gammon), or 3 (backgammon)."""

    loser = opponent(winning_player)
    if board.off[loser] > 0:
        return 1
    loser_on_bar = board.bar[loser] > 0
    loser_in_winner_home = any(
        board.points[index] * int(loser) > 0 for index in home_points(winning_player)
    )
    return 3 if loser_on_bar or loser_in_winner_home else 2
