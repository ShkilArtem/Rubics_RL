"""Shared evaluation code for random, tabular and neural agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .agents import Observation
from .env import RubiksCubeEnv


class Agent(Protocol):
    def select_action(self, observation: Observation, explore: bool = True) -> int:
        ...


@dataclass(frozen=True)
class EvaluationResult:
    episodes: int
    solved: int
    success_rate: float
    average_reward: float
    average_steps: float

    def __str__(self) -> str:
        return (
            f"success={self.success_rate:.1%} "
            f"({self.solved}/{self.episodes}), "
            f"avg_reward={self.average_reward:.3f}, "
            f"avg_steps={self.average_steps:.2f}"
        )


def evaluate_agent(
    agent: Agent,
    episodes: int,
    scramble_depth: int,
    seed: int = 10_000,
    max_steps: int | None = None,
    encode_one_hot: bool = False,
) -> EvaluationResult:
    """Evaluate without exploration on a separate deterministic seed stream."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")

    episode_limit = max_steps if max_steps is not None else 2 * scramble_depth + 2
    env = RubiksCubeEnv(
        scramble_depth=scramble_depth,
        max_steps=episode_limit,
        solved_reward=1.0,
        step_penalty=-0.01,
        timeout_penalty=-1.0,
        seed=seed,
        encode_one_hot=encode_one_hot,
    )

    solved_count = 0
    total_reward = 0.0
    total_steps = 0

    for _ in range(episodes):
        observation, info = env.reset()
        terminated = bool(info["is_solved"])
        episode_done = terminated

        while not episode_done:
            action = agent.select_action(observation, explore=False)
            observation, reward, terminated, truncated, _info = env.step(action)
            total_reward += reward
            episode_done = terminated or truncated

        solved_count += int(terminated)
        total_steps += env.steps

    return EvaluationResult(
        episodes=episodes,
        solved=solved_count,
        success_rate=solved_count / episodes,
        average_reward=total_reward / episodes,
        average_steps=total_steps / episodes,
    )
