"""Training loop for the beginner-friendly tabular Q-learning agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .agents import TabularQAgent
from .env import RubiksCubeEnv


@dataclass(frozen=True)
class TabularTrainingConfig:
    scramble_depths: tuple[int, ...] = (1, 2)
    episodes_per_depth: int = 8_000
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    seed: int = 42
    progress_interval: int = 1_000


@dataclass(frozen=True)
class TabularProgress:
    depth: int
    episode: int
    epsilon: float
    recent_success_rate: float
    q_table_size: int


def train_tabular_agent(
    config: TabularTrainingConfig,
    progress_callback: Callable[[TabularProgress], None] | None = None,
) -> TabularQAgent:
    agent = TabularQAgent(
        action_count=12,
        learning_rate=config.learning_rate,
        discount_factor=config.discount_factor,
        epsilon=config.epsilon_start,
        seed=config.seed,
    )

    for depth_index, depth in enumerate(config.scramble_depths):
        env = RubiksCubeEnv(
            scramble_depth=depth,
            max_steps=2 * depth + 2,
            solved_reward=1.0,
            step_penalty=-0.01,
            timeout_penalty=-1.0,
            seed=config.seed + depth_index,
            encode_one_hot=False,
        )
        recent_results: list[bool] = []

        for episode in range(1, config.episodes_per_depth + 1):
            fraction = episode / config.episodes_per_depth
            agent.epsilon = (
                config.epsilon_start
                + fraction * (config.epsilon_end - config.epsilon_start)
            )

            observation, _info = env.reset()
            done = False

            while not done:
                action = agent.select_action(observation, explore=True)
                next_observation, reward, terminated, truncated, _info = env.step(
                    action
                )
                done = terminated or truncated
                agent.learn(observation, action, reward, next_observation, done)
                observation = next_observation

            recent_results.append(terminated)
            recent_results = recent_results[-config.progress_interval :]

            if (
                progress_callback is not None
                and episode % config.progress_interval == 0
            ):
                progress_callback(
                    TabularProgress(
                        depth=depth,
                        episode=episode,
                        epsilon=agent.epsilon,
                        recent_success_rate=sum(recent_results)
                        / len(recent_results),
                        q_table_size=len(agent.q_table),
                    )
                )

    agent.epsilon = config.epsilon_end
    return agent
