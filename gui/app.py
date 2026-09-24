"""Responsive Pygame application using the exact training engine."""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import ClassVar

import pygame

from backgammon.constants import BLACK, WHITE, Player
from backgammon.game import Game, GameState
from backgammon.moves import Move, TurnAction
from rl.agents.base_agent import BaseAgent
from rl.agents.heuristic_agent import HeuristicAgent
from rl.agents.random_agent import RandomAgent
from rl.agents.rl_agent import RLAgent

from .board_renderer import BoardRenderer
from .input_handler import HumanTurnController
from .menu_renderer import MenuRenderer
from .statistics import DEFAULT_STATISTICS_PATH, GameStatistics

LOGGER = logging.getLogger(__name__)


class BackgammonApp:
    MODES: ClassVar[set[str]] = {"human-ai", "human-human", "ai-ai", "random-ai"}
    PAGES: ClassVar[set[str]] = {"home", "game", "statistics"}
    AI_MOVE_DURATION: ClassVar[float] = 0.62
    DOUBLE_CLICK_SECONDS: ClassVar[float] = 0.34

    def __init__(
        self,
        model_path: str | Path | None = None,
        mode: str = "human-ai",
        seed: int = 42,
        debug: bool = False,
        statistics_path: str | Path | None = None,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unknown game mode: {mode}")
        pygame.init()
        self.screen = pygame.display.set_mode((1100, 760), pygame.RESIZABLE)
        pygame.display.set_caption("Backgammon RL")
        self.clock = pygame.time.Clock()
        self.renderer = BoardRenderer(self.screen)
        self.menu_renderer = MenuRenderer(self.screen)
        self.seed = seed
        self.debug = debug
        self.mode = mode
        self.page = "home"
        self.statistics_path = Path(statistics_path or DEFAULT_STATISTICS_PATH)
        self.statistics = GameStatistics.load(self.statistics_path)
        self.game = Game(seed=seed)
        self.game_result_recorded = False
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="backgammon-ai"
        )
        self.ai_future: Future[TurnAction] | None = None
        self.ai_moves: list[Move] = []
        self.animated_move: Move | None = None
        self.animated_player: Player | None = None
        self.animation_started_at = 0.0
        self.human_turn: HumanTurnController | None = None
        self.human_undo_stack: list[tuple[GameState, HumanTurnController]] = []
        self.last_click_at = 0.0
        self.last_click_location: int | None = None
        self.status_note = ""
        self.generation = 0

        ai_agent: BaseAgent
        if model_path and Path(model_path).is_file():
            ai_agent, metadata = RLAgent.from_checkpoint(
                model_path, load_optimizer=False
            )
            self.status_note = (
                f"Loaded checkpoint at episode {metadata.get('episode', '?')}"
            )
        else:
            ai_agent = HeuristicAgent(seed)
            self.status_note = "No trained checkpoint selected; using the heuristic AI."
        if mode == "human-ai":
            self.players: dict[Player, BaseAgent | None] = {
                WHITE: None,
                BLACK: ai_agent,
            }
        elif mode == "human-human":
            self.players = {WHITE: None, BLACK: None}
        elif mode == "ai-ai":
            self.players = {WHITE: ai_agent, BLACK: HeuristicAgent(seed + 1)}
        else:
            self.players = {WHITE: RandomAgent(seed + 1), BLACK: ai_agent}

    def run(self) -> None:
        running = True
        try:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        if self.page == "home":
                            running = False
                        else:
                            self._show_home()
                    elif event.type == pygame.VIDEORESIZE:
                        self.screen = pygame.display.set_mode(
                            (max(720, event.w), max(600, event.h)), pygame.RESIZABLE
                        )
                        self.renderer.screen = self.screen
                        self.menu_renderer.screen = self.screen
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self._handle_click(event.pos, getattr(event, "clicks", 1))
                self._update()
                self._draw()
                pygame.display.flip()
                self.clock.tick(60)
        finally:
            self.executor.shutdown(wait=False, cancel_futures=True)
            pygame.quit()

    def _start_game(self) -> None:
        self._new_game()
        self.page = "game"

    def _show_home(self) -> None:
        self._record_completed_game()
        if self.ai_future is not None:
            self.ai_future.cancel()
            self.ai_future = None
        self.page = "home"

    def _new_game(self) -> None:
        self._record_completed_game()
        self.generation += 1
        self.game = Game(seed=self.seed + self.generation)
        if self.ai_future is not None:
            self.ai_future.cancel()
        self.ai_future = None
        self.ai_moves.clear()
        self.animated_move = None
        self.animated_player = None
        self.human_turn = None
        self.human_undo_stack.clear()
        self.last_click_at = 0.0
        self.last_click_location = None
        self.game_result_recorded = False
        self.status_note = "New game started. Dice roll automatically each turn."

    def _handle_click(self, position: tuple[int, int], clicks: int = 1) -> None:
        if self.page == "home":
            layout = self.menu_renderer.home_layout()
            if layout.start_game.collidepoint(position):
                self._start_game()
            elif layout.statistics.collidepoint(position):
                self.page = "statistics"
            return
        if self.page == "statistics":
            if self.menu_renderer.statistics_layout().back.collidepoint(position):
                self.page = "home"
            return

        layout = self.renderer.layout()
        if layout.home.collidepoint(position):
            self._show_home()
            return
        if layout.new_game.collidepoint(position):
            self._new_game()
            return
        if self.can_reverse_move and layout.undo.collidepoint(position):
            self._reverse_human_move()
            return
        if (
            self.game.is_over
            or self.players[self.game.state.current_player] is not None
        ):
            return
        if self.human_turn is None:
            return
        location = self.renderer.location_at(position)
        if location is None:
            return
        now = time.monotonic()
        is_double_click = clicks >= 2 or (
            location == self.last_click_location
            and now - self.last_click_at <= self.DOUBLE_CLICK_SECONDS
        )
        self.last_click_at = now
        self.last_click_location = location
        if is_double_click:
            controller_before = self.human_turn.copy()
            move = self.human_turn.play_highest_die(location)
            if move is not None:
                self.human_undo_stack.append(
                    (self.game.state.copy(), controller_before)
                )
                self.game.apply_single_move(move)
                self._finish_human_turn_if_complete()
            return
        if self.human_turn.selected_source is None:
            self.human_turn.select_source(location)
            return
        controller_before = self.human_turn.copy()
        moves = self.human_turn.select_destination(location)
        if moves is None:
            self.human_turn.select_source(location)
            return
        for index, move in enumerate(moves):
            undo_controller = controller_before.copy()
            if index:
                prefix = moves[:index]
                undo_controller.actions = [
                    action
                    for action in controller_before.actions
                    if action.moves[
                        controller_before.step : controller_before.step + index
                    ]
                    == prefix
                ]
                undo_controller.step += index
                undo_controller.selected_source = None
            self.human_undo_stack.append((self.game.state.copy(), undo_controller))
            self.game.apply_single_move(move)
        self._finish_human_turn_if_complete()

    @property
    def can_reverse_move(self) -> bool:
        """A human may rewind moves only while the current turn is unfinished."""

        return (
            self.human_turn is not None
            and bool(self.human_undo_stack)
            and not self.human_turn.complete
            and not self.game.is_over
            and self.players[self.game.state.current_player] is None
        )

    def _reverse_human_move(self) -> None:
        if not self.can_reverse_move:
            return
        state, controller = self.human_undo_stack.pop()
        self.game.state = state.copy()
        self.human_turn = controller.copy()
        self.last_click_at = 0.0
        self.last_click_location = None
        self.status_note = "Last move reversed. Choose a move for the restored die."

    def _finish_human_turn_if_complete(self) -> None:
        if self.human_turn is None:
            return
        if self.human_turn.complete:
            self.human_undo_stack.clear()
            self.game.finish_partial_turn()
            self.human_turn = None

    def _update(self) -> None:
        if self.page != "game":
            return
        if self.game.is_over:
            self._record_completed_game()
            return
        actor = self.players[self.game.state.current_player]
        if actor is None:
            if self.human_turn is None:
                self.human_undo_stack.clear()
                actions = self.game.legal_actions()
                if not actions:
                    self.game.play_turn(TurnAction())
                else:
                    self.human_turn = HumanTurnController(actions)
            return

        self.human_turn = None
        self.human_undo_stack.clear()
        if self.animated_move is not None:
            elapsed = time.monotonic() - self.animation_started_at
            if elapsed < self.AI_MOVE_DURATION:
                return
            self.game.apply_single_move(self.animated_move)
            self.animated_move = None
            self.animated_player = None
            if not self.ai_moves:
                self.game.finish_partial_turn()
            return
        if self.ai_moves:
            self.animated_move = self.ai_moves.pop(0)
            self.animated_player = self.game.state.current_player
            self.animation_started_at = time.monotonic()
            return
        if self.ai_future is None:
            actions = self.game.legal_actions() or [TurnAction()]
            self.ai_future = self.executor.submit(
                actor.select_action, self.game, actions, False
            )
            return
        if self.ai_future.done():
            try:
                action = self.ai_future.result()
            except (
                Exception
            ) as error:  # Keep the window responsive and explain failure.
                LOGGER.exception("AI move failed")
                self.status_note = f"AI error: {error}"
                self.ai_future = None
                return
            self.ai_future = None
            if not action.moves:
                self.game.play_turn(action)
            else:
                self.ai_moves = list(action.moves)

    def _record_completed_game(self) -> None:
        if self.game_result_recorded or not self.game.is_over:
            return
        winner = self.game.state.winning_player
        if winner is None:
            return
        human_players = {
            player for player, agent in self.players.items() if agent is None
        }
        self.statistics.record_game(
            winner,
            self.game.state.result_points,
            self.game.state.turn_number + 1,
            human_players,
        )
        self.game_result_recorded = True
        try:
            self.statistics.save(self.statistics_path)
        except OSError as error:
            LOGGER.exception("Could not save game statistics")
            self.status_note = f"Statistics could not be saved: {error}"

    def _draw(self) -> None:
        if self.page == "home":
            self.menu_renderer.draw_home(self.statistics)
            return
        if self.page == "statistics":
            self.menu_renderer.draw_statistics(self.statistics)
            return

        state = self.game.state
        sources: set[int] = set()
        destinations: set[int] = set()
        destination_labels: dict[int, str] = {}
        selected = None
        if self.human_turn:
            sources = self.human_turn.valid_sources
            destinations = self.human_turn.valid_destinations
            destination_labels = self.human_turn.destination_labels
            selected = self.human_turn.selected_source

        if state.winning_player:
            winner = "White" if state.winning_player == WHITE else "Black"
            names = {1: "single", 2: "gammon", 3: "backgammon"}
            status = f"{winner} wins a {names[state.result_points]} ({state.result_points} point(s))."
        elif self.ai_future is not None:
            status = "AI is thinking…"
        elif self.ai_moves or self.animated_move is not None:
            status = "AI is moving…"
        else:
            status = (
                self.status_note
                or "Select a highlighted checker, then a green destination."
            )
        debug_lines: list[str] = []
        if self.debug:
            debug_lines = [
                f"dice={state.dice} remaining={state.remaining_dice} legal={len(self.game.legal_actions())}",
                f"pips white={state.board.pip_count(WHITE)} black={state.board.pip_count(BLACK)} turn={state.turn_number}",
            ]
        animation_progress = 0.0
        if self.animated_move is not None:
            animation_progress = min(
                1.0,
                (time.monotonic() - self.animation_started_at) / self.AI_MOVE_DURATION,
            )
        self.renderer.draw(
            state,
            status,
            sources,
            destinations,
            selected,
            debug_lines,
            destination_labels=destination_labels,
            animated_move=self.animated_move,
            animated_player=self.animated_player,
            animation_progress=animation_progress,
            show_undo=self.can_reverse_move,
        )
