"""Self-play training and evaluation."""

from .evaluation import EvaluationResult, evaluate_agents
from .trainer import Trainer

__all__ = ["EvaluationResult", "Trainer", "evaluate_agents"]
