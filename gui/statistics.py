"""Persistent statistics for games completed in the desktop application."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from backgammon.constants import BLACK, WHITE, Player

LOGGER = logging.getLogger(__name__)
DEFAULT_STATISTICS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "statistics.json"
)


@dataclass(slots=True)
class GameStatistics:
    """Lifetime totals that can be serialized as a small, readable JSON file."""

    games_played: int = 0
    white_wins: int = 0
    black_wins: int = 0
    white_points: int = 0
    black_points: int = 0
    human_games: int = 0
    human_wins: int = 0
    human_losses: int = 0
    singles: int = 0
    gammons: int = 0
    backgammons: int = 0
    total_turns: int = 0

    @classmethod
    def load(cls, path: str | Path = DEFAULT_STATISTICS_PATH) -> GameStatistics:
        source = Path(path)
        if not source.exists():
            return cls()
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("statistics root must be an object")
            values: dict[str, int] = {}
            for field_name in cls.__dataclass_fields__:
                value = payload.get(field_name, 0)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"invalid value for {field_name}")
                values[field_name] = value
            return cls(**values)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            LOGGER.warning("Could not load statistics from %s: %s", source, error)
            return cls()

    def save(self, path: str | Path = DEFAULT_STATISTICS_PATH) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f"{destination.suffix}.tmp")
        temporary.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    def record_game(
        self,
        winner: Player,
        result_points: int,
        turns: int,
        human_players: set[Player] | None = None,
    ) -> None:
        if winner not in (WHITE, BLACK):
            raise ValueError("winner must be White or Black")
        if result_points not in (1, 2, 3):
            raise ValueError("result points must be 1, 2, or 3")
        if turns < 1:
            raise ValueError("turn count must be positive")

        self.games_played += 1
        self.total_turns += turns
        if winner == WHITE:
            self.white_wins += 1
            self.white_points += result_points
        else:
            self.black_wins += 1
            self.black_points += result_points

        if result_points == 1:
            self.singles += 1
        elif result_points == 2:
            self.gammons += 1
        else:
            self.backgammons += 1

        people = human_players or set()
        if len(people) == 1:
            self.human_games += 1
            if winner in people:
                self.human_wins += 1
            else:
                self.human_losses += 1

    @staticmethod
    def percentage(part: int, total: int) -> float:
        return part * 100.0 / total if total else 0.0

    @property
    def white_win_rate(self) -> float:
        return self.percentage(self.white_wins, self.games_played)

    @property
    def black_win_rate(self) -> float:
        return self.percentage(self.black_wins, self.games_played)

    @property
    def human_win_rate(self) -> float:
        return self.percentage(self.human_wins, self.human_games)

    @property
    def average_turns(self) -> float:
        return self.total_turns / self.games_played if self.games_played else 0.0
