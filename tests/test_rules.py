from backgammon.board import Board
from backgammon.constants import BAR, BLACK, OFF, WHITE
from backgammon.moves import Move
from backgammon.rules import all_in_home, legal_moves_for_die, win_multiplier


def valid_empty_board() -> Board:
    board = Board.empty()
    board.off[WHITE] = 15
    board.off[BLACK] = 15
    return board


def test_ordinary_move_and_blocked_point():
    board = valid_empty_board()
    board.off[WHITE] = 14
    board.points[13] = 1
    board.off[BLACK] = 13
    board.points[16] = -2
    assert Move(13, 15, 2) in legal_moves_for_die(board, WHITE, 2)
    assert Move(13, 16, 3) not in legal_moves_for_die(board, WHITE, 3)


def test_hit_moves_blot_to_bar():
    board = valid_empty_board()
    board.off[WHITE] = 14
    board.points[13] = 1
    board.off[BLACK] = 14
    board.points[16] = -1
    move = Move(13, 16, 3)
    assert move in legal_moves_for_die(board, WHITE, 3)
    board.apply_move(move, WHITE)
    assert board.points[16] == 1
    assert board.bar[BLACK] == 1


def test_bar_has_priority_and_can_be_blocked():
    board = valid_empty_board()
    board.off[WHITE] = 14
    board.bar[WHITE] = 1
    board.off[BLACK] = 13
    board.points[3] = -2
    assert legal_moves_for_die(board, WHITE, 4) == []
    assert legal_moves_for_die(board, WHITE, 3) == [Move(BAR, 2, 3)]


def test_bearing_off_exact_and_oversized():
    board = valid_empty_board()
    board.off[WHITE] = 13
    board.points[21] = 1
    board.points[19] = 1
    assert all_in_home(board, WHITE)
    assert Move(21, OFF, 3) in legal_moves_for_die(board, WHITE, 3)
    assert Move(21, OFF, 6) not in legal_moves_for_die(board, WHITE, 6)
    assert Move(19, OFF, 6) in legal_moves_for_die(board, WHITE, 6)


def test_win_multipliers():
    normal = valid_empty_board()
    normal.off[BLACK] = 1
    normal.points[13] = -14
    assert win_multiplier(normal, WHITE) == 1

    gammon = valid_empty_board()
    gammon.off[BLACK] = 0
    gammon.points[13] = -15
    assert win_multiplier(gammon, WHITE) == 2

    backgammon = valid_empty_board()
    backgammon.off[BLACK] = 0
    backgammon.points[21] = -1
    backgammon.points[13] = -14
    assert win_multiplier(backgammon, WHITE) == 3
