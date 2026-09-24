"""Modern home and lifetime-statistics screens for the desktop application."""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from .statistics import GameStatistics


@dataclass(slots=True)
class HomeLayout:
    start_game: pygame.Rect
    statistics: pygame.Rect


@dataclass(slots=True)
class StatisticsLayout:
    back: pygame.Rect


class MenuRenderer:
    BACKGROUND = (13, 20, 29)
    PANEL = (24, 34, 45)
    PANEL_BORDER = (55, 76, 87)
    TEXT = (235, 241, 241)
    MUTED = (145, 166, 171)
    ACCENT = (245, 190, 71)
    GREEN = (45, 142, 122)
    PURPLE = (88, 70, 132)

    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        pygame.font.init()
        self.hero_font = pygame.font.SysFont("segoeui", 52, bold=True)
        self.title_font = pygame.font.SysFont("segoeui", 32, bold=True)
        self.value_font = pygame.font.SysFont("segoeui", 30, bold=True)
        self.font = pygame.font.SysFont("segoeui", 20)
        self.small_font = pygame.font.SysFont("segoeui", 16)

    def home_layout(self) -> HomeLayout:
        width, height = self.screen.get_size()
        button_width = min(340, width - 100)
        button_height = 56
        left = (width - button_width) // 2
        start_y = min(height - 180, height // 2 + 65)
        return HomeLayout(
            pygame.Rect(left, start_y, button_width, button_height),
            pygame.Rect(left, start_y + 72, button_width, button_height),
        )

    def statistics_layout(self) -> StatisticsLayout:
        width, height = self.screen.get_size()
        margin = max(24, int(min(width, height) * 0.05))
        return StatisticsLayout(pygame.Rect(margin, height - margin - 44, 138, 44))

    def draw_home(self, statistics: GameStatistics) -> None:
        self._draw_background()
        width, height = self.screen.get_size()
        center_x = width // 2

        self._draw_checker_pair(center_x, max(78, height // 7))
        title = self.hero_font.render("BACKGAMMON", True, self.TEXT)
        self.screen.blit(
            title, title.get_rect(center=(center_x, max(160, height // 4)))
        )
        subtitle = self.font.render(
            "Classic strategy, intelligent competition", True, self.MUTED
        )
        self.screen.blit(
            subtitle,
            subtitle.get_rect(center=(center_x, max(207, height // 4 + 47))),
        )

        summary_y = max(260, height // 2 - 40)
        summary_width = min(460, width - 80)
        summary = pygame.Rect(
            center_x - summary_width // 2, summary_y, summary_width, 72
        )
        pygame.draw.rect(self.screen, self.PANEL, summary, border_radius=14)
        pygame.draw.rect(self.screen, self.PANEL_BORDER, summary, 1, border_radius=14)
        self._summary_value(
            summary.left + summary.width // 4,
            summary.centery,
            str(statistics.games_played),
            "GAMES PLAYED",
        )
        rate = f"{statistics.human_win_rate:.1f}%" if statistics.human_games else "—"
        self._summary_value(
            summary.left + summary.width * 3 // 4,
            summary.centery,
            rate,
            "HUMAN WIN RATE",
        )
        pygame.draw.line(
            self.screen,
            self.PANEL_BORDER,
            (summary.centerx, summary.top + 14),
            (summary.centerx, summary.bottom - 14),
        )

        layout = self.home_layout()
        self._draw_button(layout.start_game, "Start Game", self.GREEN)
        self._draw_button(layout.statistics, "Game Statistics", self.PURPLE)
        hint = self.small_font.render(
            "Your completed-game history is saved automatically.", True, self.MUTED
        )
        self.screen.blit(
            hint,
            hint.get_rect(
                center=(center_x, min(height - 24, layout.statistics.bottom + 38))
            ),
        )

    def draw_statistics(self, statistics: GameStatistics) -> None:
        self._draw_background()
        width, height = self.screen.get_size()
        margin = max(24, int(min(width, height) * 0.05))
        title = self.title_font.render("GAME STATISTICS", True, self.TEXT)
        self.screen.blit(title, (margin, margin))
        subtitle = self.small_font.render(
            "Lifetime results from completed desktop games", True, self.MUTED
        )
        self.screen.blit(subtitle, (margin, margin + 43))

        content_top = margin + 90
        gap = 14
        columns = 3
        card_width = (width - 2 * margin - gap * (columns - 1)) // columns
        card_height = 112 if height >= 680 else 98
        cards = [
            ("Games played", str(statistics.games_played), "All completed games"),
            (
                "Average length",
                f"{statistics.average_turns:.1f}",
                "Turns per game",
            ),
            (
                "Points won",
                str(statistics.white_points + statistics.black_points),
                "All result points",
            ),
            (
                "Your wins",
                str(statistics.human_wins),
                f"{statistics.human_win_rate:.1f}% win rate vs AI",
            ),
            (
                "Your losses",
                str(statistics.human_losses),
                f"{statistics.human_games} Human vs AI games",
            ),
            (
                "White wins",
                str(statistics.white_wins),
                f"{statistics.white_win_rate:.1f}% of all games",
            ),
            (
                "Black wins",
                str(statistics.black_wins),
                f"{statistics.black_win_rate:.1f}% of all games",
            ),
            ("Single games", str(statistics.singles), "Worth 1 point"),
            (
                "2 / 3 point games",
                f"{statistics.gammons} / {statistics.backgammons}",
                "Gammons / backgammons",
            ),
        ]
        for index, (label, value, detail) in enumerate(cards):
            row, column = divmod(index, columns)
            rect = pygame.Rect(
                margin + column * (card_width + gap),
                content_top + row * (card_height + gap),
                card_width,
                card_height,
            )
            self._draw_stat_card(rect, label, value, detail)

        layout = self.statistics_layout()
        self._draw_button(layout.back, "Back", self.PURPLE)

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

    def _draw_checker_pair(self, center_x: int, center_y: int) -> None:
        pygame.draw.circle(self.screen, (5, 10, 15), (center_x - 25, center_y + 5), 31)
        pygame.draw.circle(self.screen, (239, 234, 216), (center_x - 25, center_y), 29)
        pygame.draw.circle(
            self.screen, (207, 196, 168), (center_x - 25, center_y), 29, 2
        )
        pygame.draw.circle(self.screen, (5, 10, 15), (center_x + 25, center_y + 5), 31)
        pygame.draw.circle(self.screen, (31, 39, 50), (center_x + 25, center_y), 29)
        pygame.draw.circle(
            self.screen, (91, 110, 126), (center_x + 25, center_y), 29, 2
        )

    def _summary_value(self, x: int, y: int, value: str, label: str) -> None:
        rendered_value = self.title_font.render(value, True, self.ACCENT)
        rendered_label = self.small_font.render(label, True, self.MUTED)
        self.screen.blit(rendered_value, rendered_value.get_rect(center=(x, y - 9)))
        self.screen.blit(rendered_label, rendered_label.get_rect(center=(x, y + 22)))

    def _draw_button(
        self, rect: pygame.Rect, label: str, color: tuple[int, int, int]
    ) -> None:
        pygame.draw.rect(self.screen, (4, 10, 16), rect.move(0, 4), border_radius=12)
        pygame.draw.rect(self.screen, color, rect, border_radius=12)
        border = tuple(min(255, channel + 38) for channel in color)
        pygame.draw.rect(self.screen, border, rect, 2, border_radius=12)
        rendered = self.font.render(label, True, (255, 255, 255))
        self.screen.blit(rendered, rendered.get_rect(center=rect.center))

    def _draw_stat_card(
        self, rect: pygame.Rect, label: str, value: str, detail: str
    ) -> None:
        pygame.draw.rect(self.screen, (5, 10, 16), rect.move(0, 4), border_radius=12)
        pygame.draw.rect(self.screen, self.PANEL, rect, border_radius=12)
        pygame.draw.rect(self.screen, self.PANEL_BORDER, rect, 1, border_radius=12)
        label_surface = self.small_font.render(label.upper(), True, self.MUTED)
        value_surface = self.value_font.render(value, True, self.ACCENT)
        detail_surface = self.small_font.render(detail, True, self.TEXT)
        self.screen.blit(label_surface, (rect.left + 16, rect.top + 12))
        self.screen.blit(value_surface, (rect.left + 16, rect.top + 34))
        self.screen.blit(detail_surface, (rect.left + 16, rect.bottom - 27))
