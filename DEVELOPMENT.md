# Development log

## Phase 1 — engine

- Chose signed point counts with explicit per-player bar/off dictionaries. This
  keeps copying cheap while leaving state readable in tests.
- Mirrored the standard position at the user's request: White moves toward point
  23 and Black toward point 0; rules, encodings, tests, and GUI share that direction.
- Rotated the GUI point layout 180 degrees so Black's home is visual region 1
  (upper-left) and White's home is region 3 (lower-left), with the off tray on
  that same side.
- Reworked interaction and presentation with compound-die destinations,
  highest-die double-click play, consumed-die states, interpolated AI checker
  animation, partial-turn move reversal, a modernized visual system, and an
  explicit winner overlay.
- Added a home screen and a responsive statistics dashboard backed by an atomic
  JSON lifetime record. Only completed GUI games are counted, and each result is
  guarded against being recorded twice.
- Made `Move` and `TurnAction` immutable value objects.
- Implemented complete-turn depth-first generation over the remaining dice.
  Mandatory maximum usage and the higher-die rule are post-search constraints,
  which is simpler and safer than embedding special cases in single moves.
- Kept primitive board mutation separate from legality. Only generated complete
  actions are accepted by `Game.play_turn`.

## Phase 2 — simulation and baselines

- Added uniform Random and deterministic-seeded Heuristic agents behind one
  interface used by simulation, evaluation, training, and the GUI.
- Added a 1,000-game invariant test and a standalone validation command.

## Phase 3 — environment and encoding

- Defined post-roll decision states and a player-relative 60-value observation.
- Represented complete turn actions in 237 values rather than using an arbitrary
  fixed global action catalogue.
- Exposed a Gymnasium-style five-value `step` result while retaining variable
  legal actions through `get_legal_actions()`.

## Phases 4–5 — evaluation and learning

- Used a state-action Q network so all current candidates can be scored in a
  single batch.
- Used the alternating-player zero-sum target `r - gamma * max Q(next)`.
- Added replay, target updates, clipping, checkpoints, resume, seeded color-swap
  evaluation, TensorBoard, and a bounded frozen-opponent snapshot pool.

## Phase 6 — GUI

- Rendered the board procedurally for reliable resizing and no asset dependency.
- Kept partial mouse moves legal by filtering complete engine-generated actions.
- Moved inference to a worker thread and kept Pygame events/rendering on the main
  thread; selected moves animate individually.

## Verification note

The authoring machine exposed the Windows `py` launcher but no installed runtime,
so verification used a temporary workspace-local Python 3.11.9 runtime with the
declared requirements. The final checks completed successfully:

- `pytest -q`: **38 passed**, including menu/statistics persistence,
  move-reversal coverage, and 1,000 full random games;
- dependency-free engine verifier: **1,000 games**, all invariants passed;
- two-episode CPU training smoke run: checkpoints and TensorBoard event written;
- saved checkpoint evaluation: load and seeded color-swapped games completed;
- headless GUI render and complete Human-vs-Random input-controller game passed;
- Ruff check/format and Python bytecode compilation passed.
