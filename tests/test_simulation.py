import pytest

from backgammon.constants import BLACK, WHITE
from rl.agents.random_agent import RandomAgent
from rl.environment import BackgammonEnv


@pytest.mark.slow
def test_one_thousand_random_games_preserve_invariants_and_terminate():
    agent = RandomAgent(seed=927)
    for game_index in range(1_000):
        env = BackgammonEnv(seed=game_index, max_turns=10_000)
        env.reset(seed=game_index)
        terminated = truncated = False
        while not (terminated or truncated):
            legal = env.get_legal_actions()
            action = agent.select_action(env.game, legal)
            assert action in legal
            _, _, terminated, truncated, _ = env.step(action)
            board = env.game.state.board
            assert board.checker_count(WHITE) == 15
            assert board.checker_count(BLACK) == 15
            assert all(abs(count) <= 15 for count in board.points)
            assert all(count >= 0 for count in board.bar.values())
            assert all(count >= 0 for count in board.off.values())
        assert terminated and not truncated
