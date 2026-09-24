import numpy as np
import pytest

torch = pytest.importorskip("torch")

from backgammon.constants import WHITE
from backgammon.game import Game
from backgammon.moves import Move, TurnAction
from rl.agents.rl_agent import RLAgent, alternating_player_targets
from rl.config import TrainingConfig
from rl.encoding import LOCATION_COUNT, encode_action, encode_state


def tiny_config() -> TrainingConfig:
    return TrainingConfig(
        batch_size=2,
        warmup_steps=2,
        replay_buffer_size=10,
        hidden_size=32,
        action_hidden_size=16,
        device="cpu",
        target_update_frequency=1,
    )


def test_checkpoint_round_trip_preserves_inference(tmp_path):
    game = Game(seed=4)
    actions = game.legal_actions()
    agent = RLAgent(tiny_config())
    before = agent.q_values(encode_state(game.state), actions)
    path = agent.save_checkpoint(tmp_path / "agent.pt", episode=7)
    loaded, metadata = RLAgent.from_checkpoint(path, device="cpu")
    after = loaded.q_values(encode_state(game.state), actions)
    np.testing.assert_allclose(before, after)
    assert metadata["episode"] == 7


def test_optimization_changes_parameters_and_loss_is_finite():
    config = tiny_config()
    agent = RLAgent(config)
    game = Game(seed=5)
    state = encode_state(game.state)
    action = game.legal_actions()[0]
    encoded_next_actions = game.legal_actions()
    for reward in (0.0, 1.0):
        agent.remember(state, action, reward, state, False, encoded_next_actions)
    before = [
        parameter.detach().clone() for parameter in agent.online_network.parameters()
    ]
    loss = agent.optimize()
    assert loss is not None and np.isfinite(loss)
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, agent.online_network.parameters())
    )


def test_alternating_player_bellman_target_inverts_next_value():
    rewards = torch.tensor([0.0, 3.0])
    next_values = torch.tensor([0.8, 99.0])
    done = torch.tensor([False, True])
    targets = alternating_player_targets(rewards, next_values, done, gamma=0.5)
    torch.testing.assert_close(targets, torch.tensor([-0.4, 3.0]))


def test_white_action_points_are_reflected_to_player_relative_coordinates():
    encoded = encode_action(TurnAction([Move(3, 7, 4)]), WHITE)
    assert encoded[20] == 1.0  # White physical point 3 becomes relative point 20.
    assert encoded[LOCATION_COUNT + 16] == 1.0
