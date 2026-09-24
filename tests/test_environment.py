import numpy as np
import pytest

from backgammon.moves import TurnAction
from rl.encoding import ACTION_DIM, OBSERVATION_DIM, encode_action
from rl.environment import BackgammonEnv


def test_reset_returns_decision_state_and_legal_actions():
    env = BackgammonEnv(seed=42)
    observation, info = env.reset(seed=42)
    assert observation.shape == (OBSERVATION_DIM,)
    assert observation.dtype == np.float32
    assert info["legal_action_count"] == len(env.get_legal_actions())


def test_action_encoding_shape_and_pass():
    encoded = encode_action(TurnAction())
    assert encoded.shape == (ACTION_DIM,)
    assert np.count_nonzero(encoded) == 0


def test_step_rejects_action_outside_legal_set():
    env = BackgammonEnv(seed=3)
    env.reset(seed=3)
    with pytest.raises(ValueError):
        env.step(TurnAction())


def test_legal_step_switches_perspective():
    env = BackgammonEnv(seed=3)
    before, _ = env.reset(seed=3)
    after, reward, terminated, truncated, info = env.step(env.get_legal_actions()[0])
    assert after.shape == before.shape
    assert reward == 0
    assert not terminated
    assert not truncated
    assert info["current_player"] == -1
