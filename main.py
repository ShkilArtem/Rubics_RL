"""One entry point for every lesson in the project.

Change the values in the SETTINGS block, then run:

    python main.py
"""

from __future__ import annotations

from pathlib import Path
from random import Random

from rubiks_rl import (
    ACTIONS,
    RandomAgent,
    RubiksCube,
    TabularTrainingConfig,
    evaluate_agent,
    train_tabular_agent,
)


# =============================================================================
# SETTINGS: this is the main section you are expected to edit.
# =============================================================================

# Available modes:
# demo, random, train_tabular, train_dqn, evaluate, train_value, solve
RUN_MODE = "train_tabular"


SEED = 45
SCRAMBLE_DEPTH = 1
EVALUATION_EPISODES = 200

# Tabular Q-learning settings
TABULAR_DEPTHS = (1, 2)
TABULAR_EPISODES_PER_DEPTH = 8_000

# DQN curriculum settings
DQN_MAX_DEPTH = 5
DQN_MAX_STEPS_PER_DEPTH = 100_000
DQN_CHECKPOINT = Path("checkpoints/dqn_depth_1.pt")

# Value-learning and search settings
VALUE_MAX_DEPTH = 10
VALUE_CHECKPOINT = Path("checkpoints/value_depth_10.pt")
SEARCH_WEIGHT = 1.5
SEARCH_STATE_LIMIT = 100_000
SEARCH_TIME_LIMIT_SECONDS = 30.0


def run_demo() -> None:
    moves = ["R", "U", "R'", "U'"]
    cube = RubiksCube()

    print("Initial cube:")
    print(cube)

    cube = cube.apply(moves)
    print(f"\nAfter moves: {' '.join(moves)}")
    print(cube)
    print(f"\nSolved: {cube.is_solved()}")


def run_random_baseline() -> None:
    rng = Random()
    for i in range(100):
        agent = RandomAgent(action_count=len(ACTIONS), seed= rng.randint(1,10000))
        result = evaluate_agent(
            agent,
            episodes=EVALUATION_EPISODES,
            scramble_depth=SCRAMBLE_DEPTH,
            seed=SEED + 10_000,
        )
        
        print(f"Random policy, depth {SCRAMBLE_DEPTH}: {result}")


def run_tabular_training() -> None:
    def show_progress(progress: object) -> None:
        print(progress)

    config = TabularTrainingConfig(
        scramble_depths=TABULAR_DEPTHS,
        episodes_per_depth=TABULAR_EPISODES_PER_DEPTH,
        seed=SEED,
    )
    agent = train_tabular_agent(config, progress_callback=show_progress)

    print("\nEvaluation without exploration:")
    for depth in TABULAR_DEPTHS:
        result = evaluate_agent(
            agent,
            episodes=EVALUATION_EPISODES,
            scramble_depth=depth,
            seed=SEED + 10_000 + depth,
        )
        print(f"depth {depth}: {result}")


def run_dqn_training() -> None:
    from rubiks_rl.dqn import DQNTrainingConfig, train_dqn_curriculum
    from rubiks_rl.metrics import save_dqn_metrics

    config = DQNTrainingConfig(
        max_depth=DQN_MAX_DEPTH,
        max_steps_per_depth=DQN_MAX_STEPS_PER_DEPTH,
        evaluation_episodes=EVALUATION_EPISODES,
        seed=SEED,
    )
    result = train_dqn_curriculum(config, progress_callback=print)
    csv_path, plot_path = save_dqn_metrics(result.progress)

    print(f"\nDevice: {result.agent.device}")
    print(f"Reached depth: {result.reached_depth}")
    print(f"Completed curriculum: {result.completed_curriculum}")
    print(f"Metrics: {csv_path}")
    if plot_path is not None:
        print(f"Plot: {plot_path}")


def run_dqn_evaluation() -> None:
    from rubiks_rl.dqn import DQNAgent

    if not DQN_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint {DQN_CHECKPOINT} does not exist. Run train_dqn first."
        )

    agent = DQNAgent.load(DQN_CHECKPOINT, seed=SEED)
    result = evaluate_agent(
        agent,
        episodes=EVALUATION_EPISODES,
        scramble_depth=SCRAMBLE_DEPTH,
        seed=SEED + 20_000,
        encode_one_hot=True,
    )
    print(f"Device: {agent.device}")
    print(f"DQN checkpoint: {DQN_CHECKPOINT}")
    print(f"depth {SCRAMBLE_DEPTH}: {result}")


def run_value_training() -> None:
    from rubiks_rl.value_learning import ValueTrainingConfig, train_value_network

    config = ValueTrainingConfig(
        max_scramble_depth=VALUE_MAX_DEPTH,
        checkpoint_path=str(VALUE_CHECKPOINT),
        seed=SEED,
    )
    estimator = train_value_network(config, progress_callback=print)
    print(f"\nDevice: {estimator.device}")
    print(f"Checkpoint: {VALUE_CHECKPOINT}")


def run_solver() -> None:
    from rubiks_rl.search import weighted_a_star
    from rubiks_rl.value_learning import NeuralValueEstimator

    if not VALUE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint {VALUE_CHECKPOINT} does not exist. Run train_value first."
        )

    rng = Random(SEED)
    scrambled_cube, scramble_actions = RubiksCube().scramble(
        SCRAMBLE_DEPTH, rng
    )
    estimator = NeuralValueEstimator.load(VALUE_CHECKPOINT)

    print(f"Device: {estimator.device}")
    print(f"Scramble shown for debugging only: {' '.join(scramble_actions)}")
    print("\nBefore solving:")
    print(scrambled_cube)

    result = weighted_a_star(
        scrambled_cube,
        estimator,
        weight=SEARCH_WEIGHT,
        max_expanded_states=SEARCH_STATE_LIMIT,
        time_limit_seconds=SEARCH_TIME_LIMIT_SECONDS,
    )
    print(f"\nSearch result: {result}")

    if result.solved:
        solved_cube = scrambled_cube.apply(result.actions)
        print(f"Solution: {' '.join(result.actions)}")
        print("\nAfter solving:")
        print(solved_cube)


def main() -> None:
    modes = {
        "demo": run_demo,
        "random": run_random_baseline,
        "train_tabular": run_tabular_training,
        "train_dqn": run_dqn_training,
        "evaluate": run_dqn_evaluation,
        "train_value": run_value_training,
        "solve": run_solver,
    }

    if RUN_MODE not in modes:
        available = ", ".join(modes)
        raise ValueError(f"Unknown RUN_MODE {RUN_MODE!r}. Choose one of: {available}")

    try:
        modes[RUN_MODE]()
    except ImportError as error:
        print(error)
        print("Setup instructions are in README.md.")


if __name__ == "__main__":
    main()
