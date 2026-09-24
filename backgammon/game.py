"""Game lifecycle built on the pure board and rules modules."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .board import Board
from .constants import WHITE, Player, opponent
from .moves import Move, TurnAction
from .rules import expanded_dice, generate_legal_turns, win_multiplier, winner


@dataclass(slots=True)
class GameState:
    board: Board = field(default_factory=Board.initial)
    current_player: Player = WHITE
    dice: tuple[int, int] = (1, 1)
    remaining_dice: tuple[int, ...] = (1, 1, 1, 1)
    winning_player: Player | None = None
    result_points: int = 0
    turn_number: int = 0

    def copy(self) -> GameState:
        return GameState(
            board=self.board.copy(),
            current_player=self.current_player,
            dice=self.dice,
            remaining_dice=self.remaining_dice,
            winning_player=self.winning_player,
            result_points=self.result_points,
            turn_number=self.turn_number,
        )


class Game:
    """Owns dice randomness and applies only complete legal turn actions."""

    def __init__(self, seed: int | None = None, state: GameState | None = None) -> None:
        self.rng = random.Random(seed)
        self.state = state.copy() if state else GameState()
        if state is None:
            self.roll_dice()

    def reset(self, seed: int | None = None) -> GameState:
        if seed is not None:
            self.rng.seed(seed)
        self.state = GameState()
        self.roll_dice()
        return self.state.copy()

    def set_dice(self, first: int, second: int) -> None:
        if not (1 <= first <= 6 and 1 <= second <= 6):
            raise ValueError("Dice must be between 1 and 6")
        self.state.dice = (first, second)
        self.state.remaining_dice = expanded_dice(self.state.dice)

    def roll_dice(self) -> tuple[int, int]:
        self.set_dice(self.rng.randint(1, 6), self.rng.randint(1, 6))
        return self.state.dice

    def legal_actions(self) -> list[TurnAction]:
        if self.state.winning_player is not None:
            return []
        return generate_legal_turns(
            self.state.board,
            self.state.remaining_dice,
            self.state.current_player,
            expand_roll_doubles=False,
        )

    def apply_single_move(self, move: Move) -> None:
        """Apply one move from an already validated complete GUI/AI action."""

        self.state.board.apply_move(move, self.state.current_player)
        dice = list(self.state.remaining_dice)
        try:
            dice.remove(move.die)
        except ValueError as error:
            raise ValueError("Move die is not available") from error
        self.state.remaining_dice = tuple(dice)

    def play_turn(self, action: TurnAction) -> None:
        legal = self.legal_actions()
        if legal:
            if action not in legal:
                raise ValueError(f"Illegal turn action: {action}")
        elif action.moves:
            raise ValueError("Only a pass is legal when no checker can move")
        for move in action:
            self.apply_single_move(move)
        self.finish_partial_turn()

    def finish_partial_turn(self) -> None:
        winning_player = winner(self.state.board)
        if winning_player is not None:
            self.state.winning_player = winning_player
            self.state.result_points = win_multiplier(self.state.board, winning_player)
            return
        self.state.current_player = opponent(self.state.current_player)
        self.state.turn_number += 1
        self.roll_dice()

    @property
    def is_over(self) -> bool:
        return self.state.winning_player is not None
