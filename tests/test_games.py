from backgammon.board import Board
from backgammon.constants import BLACK, WHITE
from backgammon.game import Game, GameState


def test_game_applies_only_generated_action_and_switches_player():
    game = Game(seed=1)
    action = game.legal_actions()[0]
    game.play_turn(action)
    assert game.state.current_player == BLACK
    assert game.state.turn_number == 1


def test_terminal_game_records_backgammon():
    board = Board.empty()
    board.off[WHITE] = 14
    board.points[23] = 1
    board.points[21] = -1
    board.points[13] = -14
    state = GameState(
        board=board, current_player=WHITE, dice=(1, 2), remaining_dice=(1, 2)
    )
    game = Game(seed=1, state=state)
    winning_action = next(
        a for a in game.legal_actions() if a.moves[0].destination == 24
    )
    game.play_turn(winning_action)
    assert game.state.winning_player == WHITE
    assert game.state.result_points == 3


def test_partially_consumed_double_is_not_expanded_again():
    board = Board.empty()
    board.off[WHITE] = 14
    board.points[13] = 1
    board.off[BLACK] = 15
    state = GameState(
        board=board,
        current_player=WHITE,
        dice=(3, 3),
        remaining_dice=(3, 3),
    )
    game = Game(seed=2, state=state)
    assert game.legal_actions()
    assert all(len(action.moves) == 2 for action in game.legal_actions())
