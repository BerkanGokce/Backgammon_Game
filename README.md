# Backgammon RL

A complete, modular Backgammon application with an authoritative rules engine,
variable-action-space reinforcement learning, seeded evaluation, checkpointing,
TensorBoard metrics, and a resizable Pygame desktop interface.

The implementation targets Python 3.11+ and Windows. CUDA is selected
automatically when PyTorch detects it; every feature also works on CPU.

## Installation

From PowerShell in this directory:

```powershell
conda create --name backgammon-rl python=3.11 -y
conda activate backgammon-rl
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `conda` is not recognized, install Miniconda or Anaconda, reopen PowerShell,
and repeat the commands above.

## Verify the engine

```powershell
pytest
python main.py simulate --games 1000 --seed 42
```

The test suite includes focused movement, hitting, bar, dice-usage, bearing-off,
terminal scoring, encoding, environment, learning, checkpoint, and 1,000-game
random simulation checks. The standalone simulation command prints outcome and
average-length statistics after validating checker-count invariants on every
turn.

## Play

```powershell
python main.py play
python main.py play --model models/checkpoints/best_model.pt
python main.py play --mode human-human
python main.py play --mode ai-ai --model models/checkpoints/best_model.pt
python main.py play --mode random-ai --model models/checkpoints/best_model.pt
python main.py play --debug --model models/checkpoints/best_model.pt
```

White is the human in Human vs AI mode. Dice are rolled automatically. Click a
gold-highlighted checker and then a green destination. A destination badge shows
the die distance, including compound same-checker moves such as `2 + 3 = 5`.
Double-clicking a checker immediately plays its highest currently legal die.
Used dice darken as moves are completed. After at least one die is used, a
**Reverse Move** button remains available until the rest of the legal turn is
finished; it restores the checker, captured blot (if any), die, and legal move
choices from immediately before the move. The input controller filters the
original complete legal turns after every partial move, so it cannot accept a
locally legal move that would violate mandatory dice usage later in the turn.
AI inference runs in a worker thread and every AI checker move is smoothly
animated. A winner card appears over the board when the game ends.
On screen, Black gathers in region 1 (upper-left) and White gathers in region 3
(lower-left); their borne-off trays are immediately to the left of those homes.

The app opens on a home screen. **Start Game** opens the board and
**Game Statistics** opens the lifetime record. Completed GUI games are saved
automatically in `data/statistics.json`, so the totals survive closing and
restarting the program. Statistics include games played, personal wins/losses
and win rate for Human vs AI games, White/Black win rates, average turns,
points, singles, gammons, and backgammons. Human-vs-Human and AI-vs-AI results
still count in the overall and color totals but not in the personal record.
Use **Main Menu** on the board, or press Escape, to return home.

If no checkpoint is supplied (or the path does not exist), the app displays a
clear status message and uses the heuristic opponent, keeping the game playable.

## Train

Quick smoke run:

```powershell
python main.py train --episodes 10 --batch-size 32 --warmup-steps 100
```

Longer self-play run:

```powershell
python main.py train --episodes 100000 --batch-size 128 --learning-rate 0.0001 --seed 42
```

Resume a checkpoint (the argument's episode count is the number of *additional*
episodes):

```powershell
python main.py train --episodes 50000 --resume models/checkpoints/checkpoint_010000.pt
```

Useful options are visible with `python main.py train --help`. Configuration is
centralized in `TrainingConfig`; CLI values are not scattered through the
training code. Checkpoints contain both networks, optimizer state, architecture
configuration, episode, exploration/optimization counters, and statistics.
Periodic files, `latest_model.pt`, and the best evaluation model are written to
`models/checkpoints/`.

Training is self-play against the live network. A small bounded pool of frozen
network snapshots is also sampled on some games, reducing opponent
non-stationarity while preserving ordinary self-play as the main data source.

## Evaluate

```powershell
python main.py evaluate --model models/checkpoints/best_model.pt --opponent random --games 1000 --seed 42
python main.py evaluate --model models/checkpoints/best_model.pt --opponent heuristic --games 1000 --seed 42
```

The evaluator alternates the model's color and uses deterministic per-game
seeds. It reports wins, losses, draws, win rate, gammons, backgammons, signed
average points, and average game length. Dice still make results noisy; use
hundreds or thousands of games for comparisons.

## TensorBoard

```powershell
tensorboard --logdir logs
```

Metrics include loss, exploration rate, result points, game length, rolling
training averages, and periodic win rates and average points against Random and
Heuristic agents.

## Architecture

```text
backgammon/           authoritative board, moves, rules, and game lifecycle
rl/
  encoding.py         replaceable state/action encodings
  environment.py      Gymnasium-style reset/step interface
  networks.py         PyTorch Q(s,a) model
  replay_buffer.py    copy-safe encoded transitions
  agents/             random, heuristic, and RL agents
  training/           self-play, evaluation, and trainer orchestration
gui/                   board/menu rendering, input, and persistent statistics
tests/                 rule, environment, RL, and simulation verification
main.py                play/train/evaluate/test/simulate commands
```

Dependencies point inward: GUI and RL both use the game engine, while the engine
knows nothing about either. The GUI and agents never decide move legality.

## Board and rules representation

The 24 points are indexed `0..23`. Positive counts belong to White and negative
counts to Black. White moves toward point 23; Black moves toward point 0. Each
player has separate bar and off counts. `BAR = -1` and `OFF = 24` appear only in
move values; they are not point indices.

`generate_legal_turns(board, dice, player)` recursively tries every remaining
die against each intermediate board. It then keeps sequences using the maximum
number of dice and applies the higher-die rule when only one of two distinct
dice can be played. Bar priority, hits, blocks, doubles, exact bearing off, and
oversized bearing off are handled by the same generator. Equal-die duplicate
sequences are removed. A turn action is immutable and contains zero to four
`Move` values; an empty action is a forced pass only when no checker move exists.

Terminal results are one point for a single, two for a gammon, and three for a
backgammon. Match play and a doubling cube can be added above this result model
without changing checker movement.

## Neural encodings and variable action space

The observation has **60 float32 values**:

- 48 point features: own and opponent checker counts on each of 24 points;
- own/opponent bar and off counts (4);
- physical player sign (1);
- remaining counts for die faces 1..6 (6);
- doubles flag (1).

Points are reflected for White, so both players see relative point 0 as their
bearing-off edge. Counts are normalized. The dice are already rolled: observations are
decision states after the chance node, never dice choices controlled by an
agent.

An action has **237 float32 values**. Each of four ordered move slots has one-hot
source (26 locations), one-hot destination (26), one-hot die (6), and a used
flag; one final feature holds normalized move count. Point locations in actions
are reflected for White in the same way as observation points, keeping the
state/action geometry aligned from either player's perspective.

There is deliberately no huge fixed action catalogue. The engine first returns
all legal complete turn actions. The network encodes the state once conceptually,
batches every candidate action, and predicts `Q(s,a)` for each. Greedy play takes
the highest score; epsilon-greedy training samples only from legal candidates.

## RL convention

The implementation is a compact DQN-style learner with online and target
state-action networks, replay, Huber loss, gradient clipping, and epsilon decay.
Terminal reward is +1/+2/+3 to the player that made the winning turn (or +1 when
result-point rewards are disabled). There is no handcrafted intermediate
reward.

Every stored state is normalized to its current player. After a nonterminal
action, the next encoded state belongs to the opponent. Therefore the zero-sum
Bellman target is:

```text
target = reward - gamma * max_a' Q_target(next_state, a')
```

The minus sign is intentional and tested in the learning path. The next roll is
sampled by the environment, so replay learns an expectation over dice outcomes
rather than treating chance as an action.

## Reproducibility

`--seed` controls Python, NumPy, PyTorch, agent exploration, and environment dice
RNGs. Evaluations use deterministic derived game seeds and swap colors. Exact
bitwise reproducibility is not guaranteed across PyTorch/CUDA versions or GPU
kernels; CPU runs are generally more repeatable.

## Extension points

The current layers allow alternative encoders, networks, agents, and trainers to
be substituted independently. A doubling cube, match/Crawford state, stronger
baselines, TD-style value learning, policy/value methods, MCTS, parallel
self-play, tournaments, and network play can be added without moving rule logic
into the GUI or learners.

See [DEVELOPMENT.md](DEVELOPMENT.md) for the short architectural decision log.
