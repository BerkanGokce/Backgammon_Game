"""Uniform random baseline."""

import random

from backgammon.game import Game
from backgammon.moves import TurnAction

from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def select_action(
        self, game: Game, legal_actions: list[TurnAction], explore: bool = False
    ) -> TurnAction:
        del game, explore
        if not legal_actions:
            return TurnAction()
        return self.rng.choice(legal_actions)
