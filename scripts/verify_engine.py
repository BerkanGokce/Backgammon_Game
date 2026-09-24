"""Dependency-free rule-engine smoke and stochastic invariant verification."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backgammon.board import Board
from backgammon.constants import BAR, BLACK, OFF, WHITE
from backgammon.game import Game
from backgammon.moves import Move, TurnAction
from backgammon.rules import generate_legal_turns


def focused_checks() -> None:
    board = Board.initial()
    board.validate()
    assert board.pip_count(WHITE) == board.pip_count(BLACK) == 167

    board = Board.empty()
    board.off[WHITE] = 14
    board.bar[WHITE] = 1
    board.off[BLACK] = 13
    board.points[3] = -2
    assert generate_legal_turns(board, (4,), WHITE) == []
    assert generate_legal_turns(board, (3,), WHITE)[0].moves == (Move(BAR, 2, 3),)

    board = Board.empty()
    board.off[WHITE] = 14
    board.bar[WHITE] = 1
    board.off[BLACK] = 13
    board.points[7] = -2
    actions = generate_legal_turns(board, (3, 5), WHITE)
    assert len(actions) == 1 and actions[0].moves == (Move(BAR, 4, 5),)

    board = Board.empty()
    board.off[WHITE] = 13
    board.points[19] = 1
    board.points[22] = 1
    board.off[BLACK] = 15
    oversized = generate_legal_turns(board, (6,), WHITE)
    assert len(oversized) == 1
    assert oversized[0].moves == (Move(19, OFF, 6),)


def random_simulation(games: int, seed: int) -> dict[str, float | int]:
    rng = random.Random(seed)
    outcomes = {1: 0, 2: 0, 3: 0}
    total_turns = 0
    for game_index in range(games):
        game = Game(seed=seed + game_index)
        while not game.is_over:
            legal = game.legal_actions()
            action = rng.choice(legal) if legal else TurnAction()
            assert not legal or action in legal
            game.play_turn(action)
            game.state.board.validate()
            assert all(abs(count) <= 15 for count in game.state.board.points)
            if game.state.turn_number > 10_000:
                raise AssertionError(f"Game {game_index} did not terminate")
        outcomes[game.state.result_points] += 1
        total_turns += game.state.turn_number
    return {
        "games": games,
        "average_turns": total_turns / games if games else 0.0,
        "singles": outcomes[1],
        "gammons": outcomes[2],
        "backgammons": outcomes[3],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    focused_checks()
    result = random_simulation(args.games, args.seed)
    print(result)
    print("All dependency-free engine checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
