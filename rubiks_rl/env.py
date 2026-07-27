from __future__ import annotations

from random import Random
from typing import Any

from .cube import ACTIONS, RubiksCube


class RubiksCubeEnv:
    def __init__(
        self,
        scramble_depth: int = 1,
        max_steps: int = 50,
        solved_reward: float = 1.0,
        step_penalty: float = -0.01,
        timeout_penalty: float = -1.0,
        seed: int | None = None,
        encode_one_hot: bool = True,
    ) -> None:
        self.scramble_depth = scramble_depth
        self.max_steps = max_steps
        self.solved_reward = solved_reward
        self.step_penalty = step_penalty
        self.timeout_penalty = timeout_penalty
        self.encode_one_hot = encode_one_hot

        self.action_space = ACTIONS
        self.rng = Random(seed)

        self.cube = RubiksCube()
        self.steps = 0
        self.scramble_actions: list[str] = []

    def reset(
        self, scramble_depth: int | None = None, seed: int | None = None
    ) -> tuple[list[float] | tuple[int, ...], dict[str, Any]]:
        if seed is not None:
            self.rng.seed(seed)

        if scramble_depth is None:
            depth = self.scramble_depth
        else:
            depth = scramble_depth

        solved_cube = RubiksCube()
        self.cube, self.scramble_actions = solved_cube.scramble(depth, self.rng)
        self.steps = 0

        observation = self._make_observation()
        info = self._make_info()
        return observation, info

    def step(
        self, action: int | str
    ) -> tuple[list[float] | tuple[int, ...], float, bool, bool, dict[str, Any]]:
        action_name = self._convert_action_to_name(action)

        self.cube = self.cube.move(action_name)
        self.steps += 1

        terminated = self.cube.is_solved()
        truncated = self.steps >= self.max_steps and not terminated

        reward = self.step_penalty
        if terminated:
            reward += self.solved_reward
        if truncated:
            reward += self.timeout_penalty

        observation = self._make_observation()
        info = self._make_info()
        return observation, reward, terminated, truncated, info

    def render(self) -> str:
        return self.cube.as_cross()

    def _make_observation(self) -> list[float] | tuple[int, ...]:
        if self.encode_one_hot:
            return self.cube.one_hot()
        return self.cube.state

    def _make_info(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "is_solved": self.cube.is_solved(),
            "scramble_actions": list(self.scramble_actions),
        }

    def _convert_action_to_name(self, action: int | str) -> str:
        if isinstance(action, int):
            if action < 0 or action >= len(self.action_space):
                raise ValueError(f"Action index out of range: {action}")
            return self.action_space[action]

        if action not in self.action_space:
            raise ValueError(f"Unknown action: {action!r}")

        return action
