import json

from backgammon.constants import BLACK, WHITE
from gui.statistics import GameStatistics


def test_statistics_round_trip_and_percentages(tmp_path):
    path = tmp_path / "nested" / "statistics.json"
    statistics = GameStatistics()
    statistics.record_game(WHITE, 1, 20, {WHITE})
    statistics.record_game(BLACK, 3, 30, {WHITE})
    statistics.record_game(BLACK, 2, 25, {WHITE, BLACK})
    statistics.save(path)

    restored = GameStatistics.load(path)
    assert restored.games_played == 3
    assert restored.white_wins == 1
    assert restored.black_wins == 2
    assert restored.human_games == 2
    assert restored.human_wins == 1
    assert restored.human_losses == 1
    assert restored.singles == 1
    assert restored.gammons == 1
    assert restored.backgammons == 1
    assert restored.white_win_rate == 100 / 3
    assert restored.black_win_rate == 200 / 3
    assert restored.human_win_rate == 50.0
    assert restored.average_turns == 25.0


def test_missing_or_invalid_statistics_start_empty(tmp_path):
    path = tmp_path / "statistics.json"
    assert GameStatistics.load(path) == GameStatistics()

    path.write_text(json.dumps({"games_played": -1}), encoding="utf-8")
    assert GameStatistics.load(path) == GameStatistics()
