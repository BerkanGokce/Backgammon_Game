"""Constants shared by the engine."""

from enum import IntEnum


class Player(IntEnum):
    """Checker owners. White moves toward point 23; Black toward point 0."""

    WHITE = 1
    BLACK = -1


WHITE = Player.WHITE
BLACK = Player.BLACK
NUM_POINTS = 24
CHECKERS_PER_PLAYER = 15
BAR = -1
OFF = 24


def opponent(player: Player) -> Player:
    return Player(-int(player))


def direction(player: Player) -> int:
    return 1 if player == WHITE else -1


def home_points(player: Player) -> range:
    return range(18, 24) if player == WHITE else range(6)


def entry_point(player: Player, die: int) -> int:
    """Return the physical point used to enter from the bar."""

    return die - 1 if player == WHITE else 24 - die
