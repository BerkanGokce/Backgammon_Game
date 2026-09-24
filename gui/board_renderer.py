"""Responsive, asset-free rendering of a standard Backgammon board."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pygame

from backgammon.constants import BAR, BLACK, OFF, WHITE, Player
from backgammon.game import GameState
from backgammon.moves import Move


@dataclass(slots=True)
class Layout:
    board: pygame.Rect
    off: pygame.Rect
    new_game: pygame.Rect
    home: pygame.Rect
    undo: pygame.Rect
    bar_width: int
    point_width: float


class BoardRenderer:
    BACKGROUND = (13, 20, 29)
    PANEL = (24, 34, 45)
    WOOD = (78, 101, 105)
    FRAME = (35, 48, 58)
    LIGHT_POINT = (215, 166, 99)
    DARK_POINT = (62, 120, 119)
    WHITE_CHECKER = (241, 236, 219)
    BLACK_CHECKER = (31, 39, 50)
    ACCENT = (245, 190, 71)
    LEGAL = (71, 215, 151)

    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        pygame.font.init()
        self.font = pygame.font.SysFont("segoeui", 20)
        self.small_font = pygame.font.SysFont("consolas", 15)
        self.title_font = pygame.font.SysFont("segoeui", 26, bold=True)

    def layout(self) -> Layout:
        width, height = self.screen.get_size()
        margin = max(18, int(min(width, height) * 0.035))
        status_height = 92
        off_width = max(48, int(width * 0.065))
        board_left = margin + off_width + margin // 2
        board = pygame.Rect(
            board_left,
            margin + 34,
            max(500, width - board_left - margin),
            max(390, height - margin * 2 - status_height),
        )
        bar_width = max(38, int(board.width * 0.065))
        point_width = (board.width - bar_width) / 12
        off = pygame.Rect(margin, board.top, off_width, board.height)
        button_width = 130
        new_game = pygame.Rect(
            width - margin - button_width, height - 48, button_width, 34
        )
        home = pygame.Rect(
            new_game.left - 12 - button_width,
            new_game.top,
            button_width,
            new_game.height,
        )
        undo_width = 148
        undo = pygame.Rect(
            home.left - 12 - undo_width, new_game.top, undo_width, new_game.height
        )
        return Layout(board, off, new_game, home, undo, bar_width, point_width)

    def _point_geometry(self, point: int, layout: Layout) -> tuple[float, bool]:
        """Map logical points to a board rotated 180 degrees for play."""

        board = layout.board
        half = (board.width - layout.bar_width) / 2
        if 0 <= point <= 5:
            x = board.left + (point + 0.5) * layout.point_width
            return x, True
        if 6 <= point <= 11:
            x = (
                board.left
                + half
                + layout.bar_width
                + (point - 6 + 0.5) * layout.point_width
            )
            return x, True
        if 18 <= point <= 23:
            x = board.left + (23 - point + 0.5) * layout.point_width
            return x, False
        x = (
            board.left
            + half
            + layout.bar_width
            + (17 - point + 0.5) * layout.point_width
        )
        return x, False

    def location_at(self, position: tuple[int, int]) -> int | None:
        layout = self.layout()
        x, y = position
        if layout.off.collidepoint(position):
            return OFF
        if not layout.board.collidepoint(position):
            return None
        half = (layout.board.width - layout.bar_width) / 2
        bar_left = layout.board.left + half
        if bar_left <= x <= bar_left + layout.bar_width:
            return BAR
        top = y < layout.board.centery
        if x < bar_left:
            column = min(5, int((x - layout.board.left) / layout.point_width))
            return column if top else 23 - column
        column = min(
            5,
            int((x - (bar_left + layout.bar_width)) / layout.point_width),
        )
        return 6 + column if top else 17 - column

    def draw(
        self,
        state: GameState,
        status: str,
        source_highlights: set[int] | None = None,
        destination_highlights: set[int] | None = None,
        selected_source: int | None = None,
        debug_lines: list[str] | None = None,
        *,
        destination_labels: dict[int, str] | None = None,
        animated_move: Move | None = None,
        animated_player: Player | None = None,
        animation_progress: float = 0.0,
        show_undo: bool = False,
    ) -> None:
        self._draw_background()
        layout = self.layout()
        shadow = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        pygame.draw.rect(
            shadow,
            (0, 0, 0, 105),
            layout.board.move(0, 9),
            border_radius=14,
        )
        self.screen.blit(shadow, (0, 0))
        pygame.draw.rect(self.screen, self.FRAME, layout.board, border_radius=8)
        inner = layout.board.inflate(-12, -12)
        pygame.draw.rect(self.screen, self.WOOD, inner, border_radius=5)
        pygame.draw.rect(self.screen, (104, 128, 129), inner, 2, border_radius=5)
        self._draw_points(layout)
        self._draw_bar(layout)
        self._draw_off_tray(layout, state)
        hidden_source = (
            (animated_player, animated_move.source)
            if animated_move is not None and animated_player is not None
            else None
        )
        self._draw_checkers(layout, state, hidden_source)
        self._draw_highlights(
            layout,
            state,
            source_highlights or set(),
            destination_highlights or set(),
            destination_labels or {},
        )
        if animated_move is not None and animated_player is not None:
            self._draw_animated_checker(
                layout,
                state,
                animated_move,
                animated_player,
                animation_progress,
            )
        self._draw_dice(layout, state)
        self._draw_header(state)
        self._draw_footer(layout, status, debug_lines or [], show_undo)
        if selected_source is not None:
            self._draw_selected(layout, selected_source)
        if state.winning_player is not None:
            self._draw_game_over_overlay(layout, state)

    def _draw_background(self) -> None:
        self.screen.fill(self.BACKGROUND)
        width, height = self.screen.get_size()
        for y in range(0, height, 4):
            blend = y / max(1, height)
            color = (
                int(13 + 8 * blend),
                int(20 + 10 * blend),
                int(29 + 13 * blend),
            )
            pygame.draw.rect(self.screen, color, (0, y, width, 4))

    def _draw_points(self, layout: Layout) -> None:
        height = layout.board.height * 0.43
        for point in range(24):
            x, top = self._point_geometry(point, layout)
            half_width = layout.point_width * 0.46
            base_y = layout.board.top + 7 if top else layout.board.bottom - 7
            tip_y = base_y + height if top else base_y - height
            vertices = [(x - half_width, base_y), (x + half_width, base_y), (x, tip_y)]
            color = self.LIGHT_POINT if point % 2 == 0 else self.DARK_POINT
            pygame.draw.polygon(self.screen, color, vertices)

    def _draw_bar(self, layout: Layout) -> None:
        half = (layout.board.width - layout.bar_width) / 2
        bar = pygame.Rect(
            layout.board.left + half,
            layout.board.top + 6,
            layout.bar_width,
            layout.board.height - 12,
        )
        pygame.draw.rect(self.screen, self.FRAME, bar)

    def _draw_off_tray(self, layout: Layout, state: GameState) -> None:
        pygame.draw.rect(self.screen, self.FRAME, layout.off, border_radius=5)
        mid = layout.off.centery
        pygame.draw.line(
            self.screen, self.WOOD, (layout.off.left, mid), (layout.off.right, mid), 3
        )
        # Black exits by region 1 (upper-left); White exits by region 3
        # (lower-left), so their borne-off checkers stay beside their homes.
        for player, y in ((BLACK, layout.off.top + 12), (WHITE, mid + 12)):
            count = state.board.off[player]
            available = layout.off.height / 2 - 25
            step = max(3, min(10, available / max(1, count)))
            for index in range(count):
                checker = pygame.Rect(
                    layout.off.left + 9,
                    int(y + index * step),
                    layout.off.width - 18,
                    7,
                )
                pygame.draw.rect(
                    self.screen,
                    self.WHITE_CHECKER if player == WHITE else self.BLACK_CHECKER,
                    checker,
                    border_radius=3,
                )

    def _checker_position(
        self, layout: Layout, point: int, level: int
    ) -> tuple[int, int, int]:
        x, top = self._point_geometry(point, layout)
        radius = max(
            10, int(min(layout.point_width * 0.38, layout.board.height * 0.037))
        )
        spacing = radius * 1.72
        y = (
            layout.board.top + 10 + radius + level * spacing
            if top
            else layout.board.bottom - 10 - radius - level * spacing
        )
        return int(x), int(y), radius

    def _draw_checker_disc(
        self,
        position: tuple[int, int],
        radius: int,
        player: Player,
        *,
        glow: bool = False,
    ) -> None:
        x, y = position
        color = self.WHITE_CHECKER if player == WHITE else self.BLACK_CHECKER
        outline = (126, 116, 96) if player == WHITE else (9, 14, 20)
        pygame.draw.circle(self.screen, (0, 0, 0, 80), (x + 2, y + 4), radius + 2)
        if glow:
            pygame.draw.circle(self.screen, self.ACCENT, (x, y), radius + 7, 3)
        pygame.draw.circle(self.screen, outline, (x, y), radius + 2)
        pygame.draw.circle(self.screen, color, (x, y), radius)
        shine = (255, 253, 242) if player == WHITE else (70, 83, 96)
        pygame.draw.arc(
            self.screen,
            shine,
            pygame.Rect(x - radius + 4, y - radius + 4, radius * 2 - 8, radius * 2 - 8),
            math.pi * 0.9,
            math.pi * 1.8,
            2,
        )

    def _draw_checkers(
        self,
        layout: Layout,
        state: GameState,
        hidden_source: tuple[Player, int] | None = None,
    ) -> None:
        for point, signed_count in enumerate(state.board.points):
            count = abs(signed_count)
            owner = WHITE if signed_count > 0 else BLACK
            if hidden_source == (owner, point):
                count -= 1
            if not count:
                continue
            visible = min(count, 5)
            for level in range(visible):
                x, y, radius = self._checker_position(layout, point, level)
                self._draw_checker_disc((x, y), radius, owner)
            if count > 5:
                x, y, _ = self._checker_position(layout, point, 4)
                text = self.small_font.render(
                    str(count),
                    True,
                    (220, 70, 60) if signed_count > 0 else (245, 230, 190),
                )
                self.screen.blit(text, text.get_rect(center=(x, y)))
        self._draw_bar_checkers(layout, state, hidden_source)

    def _bar_checker_position(
        self, layout: Layout, player: Player, level: int
    ) -> tuple[int, int, int]:
        x = layout.board.centerx
        radius = max(10, int(layout.bar_width * 0.34))
        offset = radius + level * radius * 1.6
        y = (
            layout.board.centery - offset
            if player == WHITE
            else layout.board.centery + offset
        )
        return x, int(y), radius

    def _draw_bar_checkers(
        self,
        layout: Layout,
        state: GameState,
        hidden_source: tuple[Player, int] | None = None,
    ) -> None:
        for player in (BLACK, WHITE):
            count = state.board.bar[player]
            if hidden_source == (player, BAR):
                count -= 1
            for index in range(min(count, 5)):
                x, y, radius = self._bar_checker_position(layout, player, index)
                self._draw_checker_disc((x, y), radius, player)
            if count > 5:
                x = layout.board.centerx
                radius = max(10, int(layout.bar_width * 0.34))
                text = self.small_font.render(str(count), True, self.ACCENT)
                y = layout.board.centery + (1 if player == BLACK else -1) * radius * 4
                self.screen.blit(text, text.get_rect(center=(x, int(y))))

    def _draw_highlights(
        self,
        layout: Layout,
        state: GameState,
        sources: set[int],
        destinations: set[int],
        labels: dict[int, str],
    ) -> None:
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        player = state.current_player
        for source in sources:
            if 0 <= source < 24:
                count = abs(state.board.points[source])
                x, y, radius = self._checker_position(
                    layout, source, max(0, min(count, 5) - 1)
                )
                pygame.draw.circle(overlay, (*self.ACCENT, 85), (x, y), radius + 8)
                pygame.draw.circle(overlay, (*self.ACCENT, 230), (x, y), radius + 6, 3)
            elif source == BAR:
                count = state.board.bar[player]
                x, y, radius = self._bar_checker_position(
                    layout, player, max(0, min(count, 5) - 1)
                )
                pygame.draw.circle(overlay, (*self.ACCENT, 210), (x, y), radius + 6, 3)

        point_height = layout.board.height * 0.43
        for destination in destinations:
            if 0 <= destination < 24:
                x, top = self._point_geometry(destination, layout)
                y = (
                    layout.board.top + point_height * 0.82
                    if top
                    else layout.board.bottom - point_height * 0.82
                )
            else:
                x, y = layout.off.center
                if player == BLACK:
                    y = layout.off.top + layout.off.height // 4
                else:
                    y = layout.off.top + layout.off.height * 3 // 4
            pygame.draw.circle(overlay, (*self.LEGAL, 45), (int(x), int(y)), 24)
            pygame.draw.circle(overlay, (*self.LEGAL, 235), (int(x), int(y)), 17, 3)
        self.screen.blit(overlay, (0, 0))

        for destination in destinations:
            if 0 <= destination < 24:
                x, top = self._point_geometry(destination, layout)
                y = (
                    layout.board.top + point_height * 0.82
                    if top
                    else layout.board.bottom - point_height * 0.82
                )
            else:
                x = layout.off.centerx
                y = (
                    layout.off.top + layout.off.height // 4
                    if player == BLACK
                    else layout.off.top + layout.off.height * 3 // 4
                )
            label = self.small_font.render(
                labels.get(destination, ""), True, (231, 255, 244)
            )
            self.screen.blit(label, label.get_rect(center=(int(x), int(y))))

    def _off_checker_position(
        self, layout: Layout, player: Player, index: int
    ) -> tuple[int, int, int]:
        radius = max(9, min(17, layout.off.width // 3))
        half_top = layout.off.top if player == BLACK else layout.off.centery
        available = layout.off.height / 2 - radius * 2
        step = max(3, min(radius * 0.65, available / max(1, index + 1)))
        return layout.off.centerx, int(half_top + radius + 8 + index * step), radius

    def _animation_position(
        self,
        layout: Layout,
        state: GameState,
        location: int,
        player: Player,
        *,
        destination: bool,
    ) -> tuple[int, int, int]:
        if location == BAR:
            level = max(0, state.board.bar[player] - 1)
            return self._bar_checker_position(layout, player, min(level, 4))
        if location == OFF:
            return self._off_checker_position(layout, player, state.board.off[player])
        count = abs(state.board.points[location])
        if destination:
            owns_destination = state.board.points[location] * int(player) > 0
            level = count if owns_destination else 0
        else:
            level = max(0, count - 1)
        return self._checker_position(layout, location, min(level, 4))

    def _draw_animated_checker(
        self,
        layout: Layout,
        state: GameState,
        move: Move,
        player: Player,
        progress: float,
    ) -> None:
        start_x, start_y, start_radius = self._animation_position(
            layout, state, move.source, player, destination=False
        )
        end_x, end_y, end_radius = self._animation_position(
            layout, state, move.destination, player, destination=True
        )
        progress = max(0.0, min(1.0, progress))
        eased = progress * progress * (3.0 - 2.0 * progress)
        x = start_x + (end_x - start_x) * eased
        y = start_y + (end_y - start_y) * eased
        arc = math.sin(math.pi * eased) * min(72.0, abs(end_x - start_x) * 0.18 + 24)
        y -= arc
        radius = int(start_radius + (end_radius - start_radius) * eased)
        self._draw_checker_disc((int(x), int(y)), radius, player, glow=True)

    def _draw_selected(self, layout: Layout, source: int) -> None:
        if 0 <= source < 24:
            x, top = self._point_geometry(source, layout)
            y = layout.board.top + 20 if top else layout.board.bottom - 20
        else:
            x, y = layout.board.center
        pygame.draw.circle(self.screen, self.ACCENT, (int(x), int(y)), 10, 3)

    def _draw_dice(self, layout: Layout, state: GameState) -> None:
        size = max(36, int(layout.board.height * 0.072))
        start_x = layout.board.centerx - size - 5
        y = layout.board.centery - size // 2
        for index, value in enumerate(state.dice):
            rect = pygame.Rect(start_x + index * (size + 10), y, size, size)
            used = state.dice[0] != state.dice[1] and value not in state.remaining_dice
            pygame.draw.rect(self.screen, (7, 12, 17), rect.move(0, 4), border_radius=9)
            fill = (61, 69, 75) if used else (245, 241, 224)
            border = (88, 96, 102) if used else (255, 255, 248)
            pip = (31, 37, 43) if not used else (38, 43, 47)
            pygame.draw.rect(self.screen, fill, rect, border_radius=9)
            pygame.draw.rect(self.screen, border, rect, 2, border_radius=9)
            self._draw_die_pips(rect, value, pip)
            if used:
                veil = pygame.Surface((size, size), pygame.SRCALPHA)
                veil.fill((5, 10, 15, 105))
                self.screen.blit(veil, rect.topleft)

        if state.dice[0] == state.dice[1]:
            used_moves = 4 - len(state.remaining_dice)
            segment_width = max(8, size // 3)
            total_width = segment_width * 4 + 9
            segment_x = layout.board.centerx - total_width // 2
            segment_y = y + size + 9
            for index in range(4):
                color = (58, 68, 75) if index < used_moves else self.LEGAL
                pygame.draw.rect(
                    self.screen,
                    color,
                    (
                        segment_x + index * (segment_width + 3),
                        segment_y,
                        segment_width,
                        4,
                    ),
                    border_radius=2,
                )

    def _draw_die_pips(
        self, rect: pygame.Rect, value: int, color: tuple[int, int, int]
    ) -> None:
        positions = {
            "tl": (0.27, 0.27),
            "tr": (0.73, 0.27),
            "ml": (0.27, 0.50),
            "c": (0.50, 0.50),
            "mr": (0.73, 0.50),
            "bl": (0.27, 0.73),
            "br": (0.73, 0.73),
        }
        patterns = {
            1: ("c",),
            2: ("tl", "br"),
            3: ("tl", "c", "br"),
            4: ("tl", "tr", "bl", "br"),
            5: ("tl", "tr", "c", "bl", "br"),
            6: ("tl", "tr", "ml", "mr", "bl", "br"),
        }
        radius = max(3, rect.width // 13)
        for key in patterns[value]:
            px, py = positions[key]
            pygame.draw.circle(
                self.screen,
                color,
                (int(rect.left + rect.width * px), int(rect.top + rect.height * py)),
                radius,
            )

    def _draw_header(self, state: GameState) -> None:
        player = "White" if state.current_player == WHITE else "Black"
        layout = self.layout()
        title = self.title_font.render("BACKGAMMON", True, (239, 244, 244))
        self.screen.blit(title, (layout.board.left, 8))
        color = self.WHITE_CHECKER if state.current_player == WHITE else (70, 84, 99)
        label = self.font.render(f"{player} to move", True, (218, 228, 229))
        label_rect = label.get_rect(right=layout.board.right, centery=24)
        pygame.draw.circle(self.screen, color, (label_rect.left - 15, 24), 6)
        self.screen.blit(label, label_rect)

    def _draw_footer(
        self,
        layout: Layout,
        status: str,
        debug_lines: list[str],
        show_undo: bool,
    ) -> None:
        controls_left = layout.undo.left if show_undo else layout.home.left
        status_panel = pygame.Rect(
            layout.board.left,
            layout.board.bottom + 8,
            max(180, controls_left - layout.board.left - 12),
            38,
        )
        pygame.draw.rect(self.screen, self.PANEL, status_panel, border_radius=9)
        text = self.font.render(status, True, (220, 232, 232))
        if text.get_width() > status_panel.width - 26:
            text = pygame.transform.smoothscale(
                text,
                (status_panel.width - 26, text.get_height()),
            )
        self.screen.blit(text, (status_panel.left + 13, status_panel.top + 7))
        if show_undo:
            self._draw_undo_button(layout)
        self._draw_home_button(layout)
        self._draw_new_game_button(layout)
        for index, line in enumerate(debug_lines[:2]):
            rendered = self.small_font.render(line, True, (139, 167, 172))
            self.screen.blit(
                rendered, (layout.board.left, layout.board.bottom + 50 + index * 18)
            )

    def _draw_new_game_button(self, layout: Layout) -> None:
        pygame.draw.rect(
            self.screen, (4, 11, 17), layout.new_game.move(0, 3), border_radius=8
        )
        pygame.draw.rect(self.screen, (45, 142, 122), layout.new_game, border_radius=8)
        pygame.draw.rect(
            self.screen, (76, 181, 152), layout.new_game, 2, border_radius=8
        )
        label = self.font.render("New Game", True, (255, 255, 255))
        self.screen.blit(label, label.get_rect(center=layout.new_game.center))

    def _draw_home_button(self, layout: Layout) -> None:
        pygame.draw.rect(
            self.screen, (4, 11, 17), layout.home.move(0, 3), border_radius=8
        )
        pygame.draw.rect(self.screen, (52, 69, 84), layout.home, border_radius=8)
        pygame.draw.rect(self.screen, (92, 116, 133), layout.home, 2, border_radius=8)
        label = self.font.render("Main Menu", True, (255, 255, 255))
        self.screen.blit(label, label.get_rect(center=layout.home.center))

    def _draw_undo_button(self, layout: Layout) -> None:
        pygame.draw.rect(
            self.screen, (4, 11, 17), layout.undo.move(0, 3), border_radius=8
        )
        pygame.draw.rect(self.screen, (88, 70, 132), layout.undo, border_radius=8)
        pygame.draw.rect(self.screen, (144, 119, 196), layout.undo, 2, border_radius=8)
        label = self.font.render("Reverse Move", True, (255, 255, 255))
        self.screen.blit(label, label.get_rect(center=layout.undo.center))

    def _draw_game_over_overlay(self, layout: Layout, state: GameState) -> None:
        veil = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        veil.fill((5, 10, 16, 175))
        self.screen.blit(veil, (0, 0))
        card = pygame.Rect(0, 0, min(480, layout.board.width - 60), 205)
        card.center = layout.board.center
        pygame.draw.rect(self.screen, (8, 14, 20), card.move(0, 8), border_radius=18)
        pygame.draw.rect(self.screen, self.PANEL, card, border_radius=18)
        pygame.draw.rect(self.screen, (62, 88, 94), card, 2, border_radius=18)
        winner = "WHITE" if state.winning_player == WHITE else "BLACK"
        title = self.title_font.render(f"{winner} WINS", True, self.ACCENT)
        self.screen.blit(title, title.get_rect(center=(card.centerx, card.top + 54)))
        result_names = {1: "Single game", 2: "Gammon", 3: "Backgammon"}
        result = self.font.render(
            f"{result_names[state.result_points]} · {state.result_points} point(s)",
            True,
            (222, 232, 233),
        )
        self.screen.blit(result, result.get_rect(center=(card.centerx, card.top + 101)))
        hint = self.small_font.render(
            "Choose New Game or return to the Main Menu", True, (148, 175, 180)
        )
        self.screen.blit(hint, hint.get_rect(center=(card.centerx, card.top + 145)))
        self._draw_home_button(layout)
        self._draw_new_game_button(layout)
