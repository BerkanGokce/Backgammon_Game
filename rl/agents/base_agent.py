"""Agent interface shared by simulations, evaluation, training, and GUI."""

from __future__ import annotations

from abc import ABC, abstractmethod

from backgammon.game import Game
from backgammon.moves import TurnAction


class BaseAgent(ABC):
    @abstractmethod
    def select_action(
        self, game: Game, legal_actions: list[TurnAction], explore: bool = False
    ) -> TurnAction:
        raise NotImplementedError
