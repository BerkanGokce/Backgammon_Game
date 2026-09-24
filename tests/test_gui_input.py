from backgammon.board import Board
from backgammon.constants import BLACK, WHITE
from backgammon.game import Game, GameState
from backgammon.moves import Move, TurnAction
from gui.input_handler import HumanTurnController
from rl.agents.random_agent import RandomAgent


def test_gui_human_prefix_controller_completes_game_against_random():
    """Drive Human-vs-Random without a display, using exactly the GUI click logic."""

    game = Game(seed=314)
    random_agent = RandomAgent(seed=271)
    safety = 0
    while not game.is_over:
        legal = game.legal_actions()
        if not legal:
            game.play_turn(TurnAction())
        elif game.state.current_player == BLACK:
            game.play_turn(random_agent.select_action(game, legal))
        else:
            controller = HumanTurnController(legal)
            while not controller.complete:
                candidate = controller.actions[0].moves[controller.step]
                assert candidate.source in controller.valid_sources
                assert controller.select_source(candidate.source)
                assert candidate.destination in controller.valid_destinations
                selected_moves = controller.select_destination(candidate.destination)
                assert selected_moves is not None
                for selected in selected_moves:
                    game.apply_single_move(selected)
            game.finish_partial_turn()
        game.state.board.validate()
        safety += 1
        assert safety < 10_000
    assert game.state.winning_player in (WHITE, BLACK)


def test_board_renderer_draws_headlessly(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame

    from gui.board_renderer import BoardRenderer

    pygame.display.quit()
    pygame.display.init()
    screen = pygame.display.set_mode((900, 640))
    game = Game(seed=12)
    renderer = BoardRenderer(screen)
    layout = renderer.layout()
    assert layout.off.right < layout.board.left

    # Regions are numbered 1/2 across the top and 3/4 across the bottom.
    # Black's home (0..5) is region 1; White's home (18..23) is region 3.
    for point in range(6):
        x, top = renderer._point_geometry(point, layout)
        assert top and x < layout.board.centerx
    for point in range(18, 24):
        x, top = renderer._point_geometry(point, layout)
        assert not top and x < layout.board.centerx

    # Hit-testing must be the inverse of rendering for every point.
    for point in range(24):
        x, top = renderer._point_geometry(point, layout)
        y = layout.board.top + 10 if top else layout.board.bottom - 10
        assert renderer.location_at((int(x), y)) == point

    renderer.draw(
        game.state,
        "Render smoke test",
        source_highlights={23, 12},
        destination_highlights={20, 7},
        debug_lines=["headless"],
    )
    action = game.legal_actions()[0]
    animated_move = action.moves[0]
    renderer.draw(
        game.state,
        "Animation smoke test",
        animated_move=animated_move,
        animated_player=game.state.current_player,
        animation_progress=0.5,
    )
    game.apply_single_move(animated_move)
    assert animated_move.die not in game.state.remaining_dice or (
        game.state.dice[0] == game.state.dice[1]
    )
    renderer.draw(game.state, "Consumed die smoke test")
    game.state.winning_player = WHITE
    game.state.result_points = 1
    renderer.draw(game.state, "White wins")
    pygame.display.flip()
    assert screen.get_size() == (900, 640)
    pygame.display.quit()


def test_compound_destinations_show_individual_and_total_dice():
    board = Board.empty()
    board.off[WHITE] = 14
    board.points[0] = 1
    board.off[BLACK] = 15
    state = GameState(
        board=board,
        current_player=WHITE,
        dice=(2, 3),
        remaining_dice=(2, 3),
    )
    game = Game(seed=1, state=state)
    actions = game.legal_actions()
    controller = HumanTurnController(actions)
    assert controller.select_source(0)
    assert controller.valid_destinations == {2, 3, 5}
    assert controller.destination_labels == {2: "2", 3: "3", 5: "5"}
    selected = controller.select_destination(5)
    assert selected == (
        Move(0, 2, 2),
        Move(2, 5, 3),
    )
    for move in selected:
        game.apply_single_move(move)
    assert game.state.board.points[5] == 1
    assert game.state.remaining_dice == ()
    assert controller.complete


def test_double_click_choice_uses_highest_available_die():
    actions = [
        TurnAction([Move(0, 2, 2), Move(2, 5, 3)]),
        TurnAction([Move(0, 3, 3), Move(3, 5, 2)]),
    ]
    controller = HumanTurnController(actions)
    assert controller.play_highest_die(0) == Move(0, 3, 3)
    assert controller.step == 1
    assert controller.actions == [actions[1]]


def test_ai_move_animates_before_board_mutation(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame

    from gui.app import BackgammonApp

    app = BackgammonApp(mode="human-ai", seed=19)
    try:
        app._start_game()
        # Make Black the actor and inject one engine-generated AI action so the
        # timing state machine can be tested without waiting on its worker.
        app.game.state.current_player = BLACK
        action = app.game.legal_actions()[0]
        app.ai_moves = list(action.moves)
        before = app.game.state.board.points.copy()

        app._update()
        assert app.animated_move == action.moves[0]
        assert app.game.state.board.points == before

        app.animation_started_at -= app.AI_MOVE_DURATION + 1.0
        app._update()
        assert app.game.state.board.points != before
    finally:
        app.executor.shutdown(wait=False, cancel_futures=True)
        pygame.quit()


def test_app_detects_manual_double_click_and_plays_high_die(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame

    from gui.app import BackgammonApp

    app = BackgammonApp(mode="human-human", seed=23)
    try:
        app._start_game()
        app._update()
        assert app.human_turn is not None
        source = next(iter(app.human_turn.valid_sources))
        layout = app.renderer.layout()
        x, y, _ = app.renderer._checker_position(
            layout, source, abs(app.game.state.board.points[source]) - 1
        )
        before = app.game.state.board.points.copy()
        app._handle_click((x, y), clicks=1)
        app._handle_click((x, y), clicks=1)
        assert app.game.state.board.points != before
    finally:
        app.executor.shutdown(wait=False, cancel_futures=True)
        pygame.quit()


def test_reverse_move_restores_partial_human_turn(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame

    from gui.app import BackgammonApp

    app = BackgammonApp(mode="human-human", seed=31)
    try:
        app._start_game()
        board = Board.empty()
        board.points[0] = 2
        board.off[WHITE] = 13
        board.points[23] = -15
        app.game = Game(
            seed=31,
            state=GameState(
                board=board,
                current_player=WHITE,
                dice=(2, 3),
                remaining_dice=(2, 3),
            ),
        )
        app._update()
        assert app.human_turn is not None
        assert not app.can_reverse_move

        layout = app.renderer.layout()
        source_x, source_y, _ = app.renderer._checker_position(layout, 0, 1)
        destination_x, _ = app.renderer._point_geometry(2, layout)
        app._handle_click((source_x, source_y))
        app._handle_click((int(destination_x), layout.board.top + 12))

        assert app.game.state.board.points[0] == 1
        assert app.game.state.board.points[2] == 1
        assert app.game.state.remaining_dice == (3,)
        assert app.can_reverse_move
        app._draw()

        app._handle_click(app.renderer.layout().undo.center)
        assert app.game.state.board.points[0] == 2
        assert app.game.state.board.points[2] == 0
        assert app.game.state.remaining_dice == (2, 3)
        assert app.human_turn is not None
        assert app.human_turn.selected_source == 0
        assert not app.can_reverse_move

        # Spending both dice completes the turn and permanently hides undo.
        destination_x, _ = app.renderer._point_geometry(5, layout)
        app._handle_click((int(destination_x), layout.board.top + 12))
        assert app.game.state.current_player == BLACK
        assert app.human_turn is None
        assert not app.can_reverse_move
        assert not app.human_undo_stack
    finally:
        app.executor.shutdown(wait=False, cancel_futures=True)
        pygame.quit()


def test_home_statistics_navigation_and_completed_game_memory(monkeypatch, tmp_path):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame

    from gui.app import BackgammonApp
    from gui.statistics import GameStatistics

    statistics_path = tmp_path / "statistics.json"
    app = BackgammonApp(
        mode="human-ai",
        seed=47,
        statistics_path=statistics_path,
    )
    try:
        assert app.page == "home"
        app._draw()

        home_layout = app.menu_renderer.home_layout()
        app._handle_click(home_layout.statistics.center)
        assert app.page == "statistics"
        app._draw()
        app._handle_click(app.menu_renderer.statistics_layout().back.center)
        assert app.page == "home"

        app._handle_click(app.menu_renderer.home_layout().start_game.center)
        assert app.page == "game"
        app.game.state.winning_player = WHITE
        app.game.state.result_points = 2
        app.game.state.turn_number = 41
        app._update()
        app._update()

        saved = GameStatistics.load(statistics_path)
        assert saved.games_played == 1
        assert saved.white_wins == 1
        assert saved.human_games == 1
        assert saved.human_wins == 1
        assert saved.gammons == 1
        assert saved.average_turns == 42.0

        app._handle_click(app.renderer.layout().home.center)
        assert app.page == "home"
        assert app.statistics.games_played == 1
    finally:
        app.executor.shutdown(wait=False, cancel_futures=True)
        pygame.quit()
