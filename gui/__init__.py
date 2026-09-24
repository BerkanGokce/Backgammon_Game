"""Pygame desktop interface with a lazy application import."""

from typing import Any

__all__ = ["BackgammonApp"]


def __getattr__(name: str) -> Any:
    if name == "BackgammonApp":
        from .app import BackgammonApp

        return BackgammonApp
    raise AttributeError(name)
