from backgammon.board import Board
from backgammon.constants import BLACK, WHITE, direction, entry_point, home_points


def test_starting_position_and_checker_counts():
    board = Board.initial()
    assert board.points[0] == 2
    assert board.points[11] == 5
    assert board.points[23] == -2
    assert board.points[12] == -5
    assert board.checker_count(WHITE) == 15
    assert board.checker_count(BLACK) == 15
    board.validate()


def test_board_copy_is_independent():
    board = Board.initial()
    clone = board.copy()
    clone.points[0] -= 1
    clone.bar[WHITE] += 1
    assert board.points[0] == 2
    assert board.bar[WHITE] == 0


def test_initial_pip_counts_are_symmetric():
    board = Board.initial()
    assert board.pip_count(WHITE) == 167
    assert board.pip_count(BLACK) == 167


def test_reversed_player_orientation():
    assert direction(WHITE) == 1
    assert direction(BLACK) == -1
    assert tuple(home_points(WHITE)) == tuple(range(18, 24))
    assert tuple(home_points(BLACK)) == tuple(range(6))
    assert entry_point(WHITE, 1) == 0
    assert entry_point(BLACK, 1) == 23
