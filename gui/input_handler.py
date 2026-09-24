"""Human partial-turn selection constrained by complete legal actions."""

from __future__ import annotations

from backgammon.moves import Move, TurnAction


class HumanTurnController:
    """Filter complete actions after each click so no dead-end prefix is possible."""

    def __init__(self, actions: list[TurnAction]) -> None:
        self.actions = actions
        self.step = 0
        self.selected_source: int | None = None

    def copy(self) -> HumanTurnController:
        clone = HumanTurnController(self.actions.copy())
        clone.step = self.step
        clone.selected_source = self.selected_source
        return clone

    @property
    def valid_sources(self) -> set[int]:
        return {
            action.moves[self.step].source
            for action in self.actions
            if self.step < len(action.moves)
        }

    @property
    def valid_destinations(self) -> set[int]:
        return set(self.destination_options)

    @property
    def destination_options(self) -> dict[int, list[tuple[Move, ...]]]:
        """Immediate and same-checker compound moves keyed by destination.

        If a checker can legally use 2 and then 3, both intermediate points and
        the total-distance point are exposed. Every option is still a prefix of
        a complete engine-generated legal turn.
        """

        if self.selected_source is None:
            return {}
        options: dict[int, list[tuple[Move, ...]]] = {}
        for action in self.actions:
            if (
                self.step >= len(action.moves)
                or action.moves[self.step].source != self.selected_source
            ):
                continue
            chain: list[Move] = []
            expected_source = self.selected_source
            for move in action.moves[self.step :]:
                if move.source != expected_source:
                    break
                chain.append(move)
                sequence = tuple(chain)
                bucket = options.setdefault(move.destination, [])
                if sequence not in bucket:
                    bucket.append(sequence)
                expected_source = move.destination
        return options

    @property
    def destination_labels(self) -> dict[int, str]:
        labels: dict[int, str] = {}
        for destination, sequences in self.destination_options.items():
            totals = sorted(
                {sum(move.die for move in sequence) for sequence in sequences}
            )
            labels[destination] = "/".join(str(total) for total in totals)
        return labels

    def select_source(self, source: int) -> bool:
        if source not in self.valid_sources:
            return False
        self.selected_source = source
        return True

    def select_destination(self, destination: int) -> tuple[Move, ...] | None:
        if self.selected_source is None:
            return None
        sequences = self.destination_options.get(destination, [])
        if not sequences:
            self.selected_source = None
            return None
        # A normal click prefers the shortest interpretation. This avoids using
        # two dice when a single move reaches the same destination. Remaining
        # ambiguity is resolved with the lower die, preserving prior behavior.
        sequence = min(sequences, key=lambda moves: (len(moves), moves[0].die))
        self.actions = [
            action
            for action in self.actions
            if action.moves[self.step : self.step + len(sequence)] == sequence
        ]
        self.step += len(sequence)
        self.selected_source = None
        return sequence

    def play_highest_die(self, source: int) -> Move | None:
        """Choose the legal first move with the highest die for double-click."""

        candidates = [
            action.moves[self.step]
            for action in self.actions
            if self.step < len(action.moves)
            and action.moves[self.step].source == source
        ]
        if not candidates:
            return None
        move = max(candidates, key=lambda candidate: candidate.die)
        self.actions = [
            action for action in self.actions if action.moves[self.step] == move
        ]
        self.step += 1
        self.selected_source = None
        return move

    @property
    def complete(self) -> bool:
        return bool(self.actions) and all(
            self.step == len(action.moves) for action in self.actions
        )
