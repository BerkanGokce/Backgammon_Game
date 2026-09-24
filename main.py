"""Command-line entry point for playing, training, evaluation, and tests."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backgammon with self-play RL")
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    play = subparsers.add_parser("play", help="launch the Pygame desktop app")
    play.add_argument("--model", type=Path, help="trained .pt checkpoint")
    play.add_argument(
        "--mode",
        choices=("human-ai", "human-human", "ai-ai", "random-ai"),
        default="human-ai",
    )
    play.add_argument("--seed", type=int, default=42)
    play.add_argument("--debug", action="store_true")

    train = subparsers.add_parser("train", help="train by self-play")
    train.add_argument("--episodes", type=int, default=10_000)
    train.add_argument("--resume", type=Path)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    train.add_argument("--gamma", type=float, default=0.99)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--replay-buffer-size", type=int, default=100_000)
    train.add_argument("--warmup-steps", type=int, default=1_000)
    train.add_argument("--epsilon-start", type=float, default=1.0)
    train.add_argument("--epsilon-end", type=float, default=0.05)
    train.add_argument("--epsilon-decay-steps", type=int, default=200_000)
    train.add_argument("--target-update-frequency", type=int, default=1_000)
    train.add_argument("--checkpoint-frequency", type=int, default=1_000)
    train.add_argument("--evaluation-frequency", type=int, default=1_000)
    train.add_argument("--evaluation-games", type=int, default=100)
    train.add_argument("--log-dir", default="logs")
    train.add_argument("--checkpoint-dir", default="models/checkpoints")

    evaluate = subparsers.add_parser("evaluate", help="evaluate an agent")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument(
        "--opponent", choices=("random", "heuristic"), default="heuristic"
    )
    evaluate.add_argument("--games", type=int, default=1_000)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--device", default="auto")

    test = subparsers.add_parser("test", help="run pytest")
    test.add_argument("pytest_args", nargs="*")

    simulate = subparsers.add_parser(
        "simulate", help="run random-vs-random engine validation"
    )
    simulate.add_argument("--games", type=int, default=1_000)
    simulate.add_argument("--seed", type=int, default=42)
    return parser


def command_train(args: argparse.Namespace) -> int:
    from rl.config import TrainingConfig
    from rl.training.trainer import Trainer

    config = TrainingConfig(
        episodes=args.episodes,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        replay_buffer_size=args.replay_buffer_size,
        warmup_steps=args.warmup_steps,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        target_update_frequency=args.target_update_frequency,
        checkpoint_frequency=args.checkpoint_frequency,
        evaluation_frequency=args.evaluation_frequency,
        evaluation_games=args.evaluation_games,
        seed=args.seed,
        device=args.device,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
    )
    trainer = Trainer(config, resume=args.resume)
    trainer.train()
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    if not args.model.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.model}")
    from rl.agents.heuristic_agent import HeuristicAgent
    from rl.agents.random_agent import RandomAgent
    from rl.agents.rl_agent import RLAgent
    from rl.training.evaluation import evaluate_agents

    agent, metadata = RLAgent.from_checkpoint(args.model, device=args.device)
    opponent = (
        RandomAgent(args.seed + 1)
        if args.opponent == "random"
        else HeuristicAgent(args.seed + 1)
    )
    result = evaluate_agents(agent, opponent, args.games, args.seed)
    output = result.to_dict()
    output["checkpoint_episode"] = int(metadata.get("episode", 0))
    print(json.dumps(output, indent=2))
    return 0


def command_simulate(args: argparse.Namespace) -> int:
    from rl.agents.random_agent import RandomAgent
    from rl.environment import BackgammonEnv

    agent = RandomAgent(args.seed)
    total_turns = 0
    outcomes = {1: 0, 2: 0, 3: 0}
    for index in range(args.games):
        env = BackgammonEnv(seed=args.seed + index)
        env.reset(seed=args.seed + index)
        terminated = truncated = False
        while not (terminated or truncated):
            legal = env.get_legal_actions()
            action = agent.select_action(env.game, legal)
            if action not in legal:
                raise AssertionError("Agent selected an action outside the legal set")
            _, _, terminated, truncated, info = env.step(action)
            board = env.game.state.board
            board.validate()
            if any(abs(point) > 15 for point in board.points):
                raise AssertionError("Impossible point count")
        if truncated:
            raise RuntimeError(f"Game {index} exceeded the turn limit")
        outcomes[info["result_points"]] += 1
        total_turns += info["turn"]
    print(
        json.dumps(
            {
                "games": args.games,
                "average_turns": total_turns / args.games if args.games else 0,
                "singles": outcomes[1],
                "gammons": outcomes[2],
                "backgammons": outcomes[3],
                "status": "all invariants passed",
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "train":
        return command_train(args)
    if args.command == "evaluate":
        return command_evaluate(args)
    if args.command == "test":
        return subprocess.call([sys.executable, "-m", "pytest", *args.pytest_args])
    if args.command == "simulate":
        return command_simulate(args)
    if args.command == "play":
        from gui.app import BackgammonApp

        BackgammonApp(args.model, args.mode, args.seed, args.debug).run()
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
