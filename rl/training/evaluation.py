"""Seeded, color-swapped agent evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from backgammon.constants import BLACK, WHITE
from rl.agents.base_agent import BaseAgent
from rl.environment import BackgammonEnv


@dataclass(slots=True)
class EvaluationResult:
    games: int
    wins: int
    losses: int
    draws: int
    gammons: int
    backgammons: int
    average_points: float
    average_game_length: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    def to_dict(self) -> dict[str, float | int]:
        result = asdict(self)
        result["win_rate"] = self.win_rate
        return result


def evaluate_agents(
    agent: BaseAgent,
    opponent: BaseAgent,
    games: int = 100,
    seed: int = 42,
    max_turns: int = 10_000,
) -> EvaluationResult:
    wins = losses = draws = gammons = backgammons = total_turns = 0
    point_balance = 0
    for game_index in range(games):
        agent_color = WHITE if game_index % 2 == 0 else BLACK
        players = {
            agent_color: agent,
            BLACK if agent_color == WHITE else WHITE: opponent,
        }
        env = BackgammonEnv(seed=seed + game_index, max_turns=max_turns)
        env.reset(seed=seed + game_index)
        terminated = truncated = False
        info = env._info()
        while not (terminated or truncated):
            actor = players[env.game.state.current_player]
            legal = env.get_legal_actions()
            action = actor.select_action(env.game, legal, explore=False)
            _, _, terminated, truncated, info = env.step(action)
        total_turns += info["turn"]
        if truncated or info["winner"] is None:
            draws += 1
            continue
        points = int(info["result_points"])
        if info["winner"] == int(agent_color):
            wins += 1
            point_balance += points
            gammons += int(points == 2)
            backgammons += int(points == 3)
        else:
            losses += 1
            point_balance -= points
    return EvaluationResult(
        games=games,
        wins=wins,
        losses=losses,
        draws=draws,
        gammons=gammons,
        backgammons=backgammons,
        average_points=point_balance / games if games else 0.0,
        average_game_length=total_turns / games if games else 0.0,
    )
