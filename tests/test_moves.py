from backgammon.board import Board
from backgammon.constants import BAR, BLACK, WHITE
from backgammon.moves import Move
from backgammon.rules import generate_legal_turns


def board_with_off(white: int = 15, black: int = 15) -> Board:
    board = Board.empty()
    board.off[WHITE] = white
    board.off[BLACK] = black
    return board


def test_both_dice_are_used_when_possible():
    board = board_with_off(14, 15)
    board.points[13] = 1
    actions = generate_legal_turns(board, (2, 3), WHITE)
    assert actions
    assert all(len(action.moves) == 2 for action in actions)
    assert {tuple(move.die for move in action.moves) for action in actions} == {
        (2, 3),
        (3, 2),
    }


def test_higher_die_rule_when_only_one_can_be_used():
    board = board_with_off(14, 13)
    board.bar[WHITE] = 1
    board.points[7] = -2
    actions = generate_legal_turns(board, (3, 5), WHITE)
    assert actions == [type(actions[0])([Move(BAR, 4, 5)])]


def test_bar_entries_and_blocked_bar():
    board = board_with_off(14, 11)
    board.bar[WHITE] = 1
    board.points[0] = -2
    board.points[1] = -2
    actions = generate_legal_turns(board, (1, 2), WHITE)
    assert actions == []


def test_doubles_use_four_moves():
    board = board_with_off(14, 15)
    board.points[3] = 1
    actions = generate_legal_turns(board, (3, 3), WHITE)
    assert actions
    assert all(len(action.moves) == 4 for action in actions)
    assert actions[0].moves[-1].destination == 15


def test_hit_can_open_second_die_move():
    board = board_with_off(14, 14)
    board.points[15] = 1
    board.points[18] = -1
    actions = generate_legal_turns(board, (3, 2), WHITE)
    assert any(action.moves[0] == Move(15, 18, 3) for action in actions)
    assert all(len(action.moves) == 2 for action in actions)


def test_oversized_bear_off_farthest_checker_only():
    board = board_with_off(13, 15)
    board.points[19] = 1
    board.points[22] = 1
    actions = generate_legal_turns(board, (6,), WHITE)
    assert [action.moves[0].source for action in actions] == [19]


def test_black_orientation():
    board = board_with_off(15, 14)
    board.points[20] = -1
    actions = generate_legal_turns(board, (4,), BLACK)
    assert actions[0].moves == (Move(20, 16, 4),)
