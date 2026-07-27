"""Simple agents used before we introduce neural networks."""

from __future__ import annotations

from collections.abc import Sequence
from random import Random


Observation = Sequence[int | float]


class RandomAgent:
    """A baseline that chooses every action with equal probability."""

    def __init__(self, action_count: int, seed: int = 0) -> None:
        self.action_count = action_count
        self.rng = Random(seed)

    def select_action(self, observation: Observation, explore: bool = True) -> int:
        # A random policy does not use the observation.  Keeping the argument
        # gives all agents the same interface.
        del observation, explore
        return self.rng.randrange(self.action_count)


class TabularQAgent:
    """Q-learning agent whose Q-values live in a Python dictionary."""

    def __init__(
        self,
        action_count: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.action_count = action_count
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.rng = Random(seed)

        # One state maps to one Q-value for each possible action.
        self.q_table: dict[tuple[int, ...], list[float]] = {}

    @staticmethod
    def state_key(observation: Observation) -> tuple[int, ...]:
        return tuple(int(value) for value in observation)

    def q_values(self, observation: Observation) -> list[float]:
        key = self.state_key(observation)
        if key not in self.q_table:
            self.q_table[key] = [0.0] * self.action_count
        return self.q_table[key]

    def select_action(self, observation: Observation, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.action_count)

        values = self.q_values(observation)
        best_value = max(values)
        best_actions = [
            index for index, value in enumerate(values) if value == best_value
        ]
        return self.rng.choice(best_actions)

    def learn(
        self,
        observation: Observation,
        action: int,
        reward: float,
        next_observation: Observation,
        done: bool,
    ) -> float:
        """Apply one Q-learning update and return the temporal-difference error."""

        values = self.q_values(observation)
        old_value = values[action]

        if done:
            target = reward
        else:
            next_best_value = max(self.q_values(next_observation))
            target = reward + self.discount_factor * next_best_value

        td_error = target - old_value
        values[action] = old_value + self.learning_rate * td_error
        return td_error

