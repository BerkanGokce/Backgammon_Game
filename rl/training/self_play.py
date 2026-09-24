"""One-episode self-play runner."""

from __future__ import annotations

from dataclasses import dataclass

from backgammon.constants import Player
from rl.agents.base_agent import BaseAgent
from rl.agents.rl_agent import RLAgent
from rl.environment import BackgammonEnv


@dataclass(slots=True)
class EpisodeResult:
    points: int
    turns: int
    winner: int | None
    average_loss: float | None
    truncated: bool


def play_self_play_episode(
    env: BackgammonEnv,
    agent: RLAgent,
    seed: int | None = None,
    frozen_opponent: BaseAgent | None = None,
    learner_color: Player | None = None,
) -> EpisodeResult:
    observation, _ = env.reset(seed=seed)
    losses: list[float] = []
    terminated = truncated = False
    reward = 0.0

    while not (terminated or truncated):
        actions = env.get_legal_actions()
        actor: BaseAgent = agent
        explore = True
        if (
            frozen_opponent is not None
            and learner_color is not None
            and env.game.state.current_player != learner_color
        ):
            actor = frozen_opponent
            explore = False
        action = actor.select_action(env.game, actions, explore=explore)
        next_observation, reward, terminated, truncated, info = env.step(action)
        next_actions = [] if terminated or truncated else env.get_legal_actions()
        agent.remember(
            observation,
            action,
            reward,
            next_observation,
            terminated or truncated,
            next_actions,
        )
        loss = agent.optimize()
        if loss is not None:
            losses.append(loss)
        observation = next_observation

    return EpisodeResult(
        points=info["result_points"],
        turns=info["turn"],
        winner=info["winner"],
        average_loss=sum(losses) / len(losses) if losses else None,
        truncated=truncated,
    )
