"""Immutable move value objects."""

from dataclasses import dataclass

from .constants import BAR, OFF


def location_name(location: int) -> str:
    if location == BAR:
        return "BAR"
    if location == OFF:
        return "OFF"
    return str(location + 1)


@dataclass(frozen=True, slots=True)
class Move:
    source: int
    destination: int
    die: int

    def __str__(self) -> str:
        return f"{location_name(self.source)}->{location_name(self.destination)} ({self.die})"


@dataclass(frozen=True, slots=True)
class TurnAction:
    """A complete legal turn. An empty action represents a forced pass."""

    moves: tuple[Move, ...] = ()

    def __init__(self, moves: tuple[Move, ...] | list[Move] = ()) -> None:
        object.__setattr__(self, "moves", tuple(moves))

    def __len__(self) -> int:
        return len(self.moves)

    def __iter__(self):
        return iter(self.moves)

    def __str__(self) -> str:
        return ", ".join(map(str, self.moves)) if self.moves else "PASS"
