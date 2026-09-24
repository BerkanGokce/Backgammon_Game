"""A small Gymnasium-style environment around the authoritative engine."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from backgammon.game import Game
from backgammon.moves import TurnAction

from .encoding import OBSERVATION_DIM, encode_state

try:
    from gymnasium import spaces
except ImportError:  # Allows the rule engine to be imported before optional setup.
    spaces = None


class BackgammonEnv:
    """Decision states always contain dice already rolled for the current player."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": ["ansi"]}

    def __init__(
        self,
        seed: int | None = None,
        use_result_points: bool = True,
        max_turns: int = 10_000,
    ) -> None:
        self.game = Game(seed=seed)
        self.use_result_points = use_result_points
        self.max_turns = max_turns
        self.observation_space = (
            spaces.Box(-1.0, 1.0, shape=(OBSERVATION_DIM,), dtype=np.float32)
            if spaces is not None
            else None
        )
        # Legal actions are variable-sized and supplied by get_legal_actions().
        self.action_space = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        self.game.reset(seed)
        return encode_state(self.game.state), self._info()

    def get_legal_actions(self) -> list[TurnAction]:
        if self.game.is_over:
            return []
        actions = self.game.legal_actions()
        # Passing is an explicit environment action only when the engine proves
        # no checker move is legal.
        return actions if actions else [TurnAction()]

    def step(
        self, action: TurnAction | int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.game.is_over:
            raise RuntimeError("step() called after the game terminated; call reset()")
        legal = self.get_legal_actions()
        selected = legal[action] if isinstance(action, int) else action
        if selected not in legal:
            raise ValueError("Action is not in the current legal-action set")
        self.game.play_turn(selected)
        terminated = self.game.is_over
        truncated = not terminated and self.game.state.turn_number >= self.max_turns
        reward = 0.0
        if terminated:
            reward = float(
                self.game.state.result_points if self.use_result_points else 1
            )
        return (
            encode_state(self.game.state),
            reward,
            terminated,
            truncated,
            self._info(),
        )

    def _info(self) -> dict[str, Any]:
        state = self.game.state
        return {
            "current_player": int(state.current_player),
            "dice": state.dice,
            "remaining_dice": state.remaining_dice,
            "turn": state.turn_number,
            "winner": int(state.winning_player) if state.winning_player else None,
            "result_points": state.result_points,
            "legal_action_count": len(self.get_legal_actions()),
        }

    def render(self) -> str:
        state = self.game.state
        text = (
            f"turn={state.turn_number} player={state.current_player.name} "
            f"dice={state.dice} bar={state.board.bar} off={state.board.off}\n"
            f"points={state.board.points}"
        )
        return text
