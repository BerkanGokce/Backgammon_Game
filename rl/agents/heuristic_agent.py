"""Transparent rule-based evaluation baseline."""

from __future__ import annotations

import random

from backgammon.constants import OFF, home_points, opponent
from backgammon.game import Game
from backgammon.moves import TurnAction

from .base_agent import BaseAgent


class HeuristicAgent(BaseAgent):
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def _score(self, game: Game, action: TurnAction) -> float:
        player = game.state.current_player
        rival = opponent(player)
        before = game.state.board
        after = before.copy()
        hits = 0
        for move in action.moves:
            if (
                move.destination != OFF
                and after.points[move.destination] * int(player) == -1
            ):
                hits += 1
            after.apply_move(move, player)

        pip_gain = before.pip_count(player) - after.pip_count(player)
        made_points = sum(1 for count in after.points if count * int(player) >= 2)
        blots = sum(1 for count in after.points if count * int(player) == 1)
        home_points_made = sum(
            1 for point in home_points(player) if after.points[point] * int(player) >= 2
        )
        borne_off = after.off[player] - before.off[player]
        opponent_bar_gain = after.bar[rival] - before.bar[rival]
        return (
            pip_gain
            + hits * 18
            + opponent_bar_gain * 10
            + made_points * 2
            + home_points_made * 3
            + borne_off * 12
            - blots * 2.5
        )

    def select_action(
        self, game: Game, legal_actions: list[TurnAction], explore: bool = False
    ) -> TurnAction:
        del explore
        if not legal_actions:
            return TurnAction()
        scores = [self._score(game, action) for action in legal_actions]
        best = max(scores)
        candidates = [
            action for action, score in zip(legal_actions, scores) if score == best
        ]
        return self.rng.choice(candidates)
