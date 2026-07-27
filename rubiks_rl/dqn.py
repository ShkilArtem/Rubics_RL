"""Deep Q-learning for the Rubik's Cube curriculum.

This module intentionally imports PyTorch only when a neural-network mode is
used.  The cube simulator and tabular lessons therefore still work without ML
dependencies installed.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Any, NamedTuple

try:
    import numpy as np
    import torch
    from torch import Tensor, nn
except ImportError as error:  # pragma: no cover - tested through main's message
    raise ImportError(
        "DQN mode requires NumPy and PyTorch. Activate the Python 3.12 "
        "virtual environment and install requirements.txt."
    ) from error

from .cube import ACTIONS
from .env import RubiksCubeEnv
from .evaluation import EvaluationResult, evaluate_agent


class Transition(NamedTuple):
    state: tuple[float, ...]
    action: int
    reward: float
    next_state: tuple[float, ...]
    done: bool


class ReplayBuffer:
    """A fixed-size memory of transitions sampled in random mini-batches."""

    def __init__(self, capacity: int, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.memory: deque[Transition] = deque(maxlen=capacity)
        self.rng = Random(seed)

    def add(
        self,
        state: Sequence[float],
        action: int,
        reward: float,
        next_state: Sequence[float],
        done: bool,
    ) -> None:
        self.memory.append(
            Transition(
                tuple(state),
                action,
                reward,
                tuple(next_state),
                done,
            )
        )

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size > len(self.memory):
            raise ValueError("Cannot sample more transitions than the buffer contains.")
        return self.rng.sample(list(self.memory), batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class QNetwork(nn.Module):
    """MLP that estimates one Q-value for every possible cube action."""

    def __init__(
        self, input_size: int = 324, hidden_size: int = 256, action_count: int = 12
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, observations: Tensor) -> Tensor:
        return self.layers(observations)


def choose_device() -> "torch.device":
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNAgent:
    def __init__(
        self,
        network: QNetwork,
        device: "torch.device",
        epsilon: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.network = network
        self.device = device
        self.epsilon = epsilon
        self.rng = Random(seed)

    def select_action(
        self, observation: Sequence[int | float], explore: bool = True
    ) -> int:
        if explore and self.rng.random() < self.epsilon:
            return self.rng.randrange(len(ACTIONS))

        observation_tensor = torch.as_tensor(
            np.asarray(observation, dtype=np.float32),
            device=self.device,
        ).unsqueeze(0)

        self.network.eval()
        with torch.no_grad():
            q_values = self.network(observation_tensor)
        return int(q_values.argmax(dim=1).item())

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.network.state_dict(),
                "metadata": metadata or {},
            },
            destination,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: "torch.device | None" = None,
        seed: int = 0,
    ) -> "DQNAgent":
        selected_device = device or choose_device()
        checkpoint = torch.load(
            Path(path), map_location=selected_device, weights_only=False
        )
        network = QNetwork().to(selected_device)
        network.load_state_dict(checkpoint["model_state_dict"])
        return cls(network, selected_device, epsilon=0.0, seed=seed)


@dataclass(frozen=True)
class DQNTrainingConfig:
    max_depth: int = 5
    max_steps_per_depth: int = 100_000
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    batch_size: int = 128
    replay_capacity: int = 100_000
    warmup_steps: int = 2_000
    target_update_interval: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    evaluation_interval: int = 5_000
    evaluation_episodes: int = 200
    promotion_success_rate: float = 0.90
    promotion_streak: int = 3
    seed: int = 42
    checkpoint_directory: str = "checkpoints"


@dataclass(frozen=True)
class DQNProgress:
    total_steps: int
    depth: int
    epsilon: float
    mean_loss: float
    evaluation: EvaluationResult


@dataclass(frozen=True)
class DQNTrainingResult:
    agent: DQNAgent
    progress: tuple[DQNProgress, ...]
    reached_depth: int
    completed_curriculum: bool


def linear_epsilon(step: int, config: DQNTrainingConfig) -> float:
    fraction = min(step / config.epsilon_decay_steps, 1.0)
    return config.epsilon_start + fraction * (
        config.epsilon_end - config.epsilon_start
    )


def hard_update_target(target_network: QNetwork, online_network: QNetwork) -> None:
    """Copy all learned weights from the online network to the target network."""

    target_network.load_state_dict(online_network.state_dict())


def optimize_dqn(
    online_network: QNetwork,
    target_network: QNetwork,
    optimizer: "torch.optim.Optimizer",
    replay_buffer: ReplayBuffer,
    batch_size: int,
    discount_factor: float,
    device: "torch.device",
) -> float:
    transitions = replay_buffer.sample(batch_size)

    states = torch.as_tensor(
        np.asarray([item.state for item in transitions], dtype=np.float32),
        device=device,
    )
    actions = torch.as_tensor(
        [item.action for item in transitions],
        dtype=torch.int64,
        device=device,
    ).unsqueeze(1)
    rewards = torch.as_tensor(
        [item.reward for item in transitions],
        dtype=torch.float32,
        device=device,
    )
    next_states = torch.as_tensor(
        np.asarray([item.next_state for item in transitions], dtype=np.float32),
        device=device,
    )
    dones = torch.as_tensor(
        [item.done for item in transitions],
        dtype=torch.float32,
        device=device,
    )

    predicted_q_values = online_network(states).gather(1, actions).squeeze(1)

    with torch.no_grad():
        next_q_values = target_network(next_states).max(dim=1).values
        targets = rewards + discount_factor * (1.0 - dones) * next_q_values

    loss = nn.functional.smooth_l1_loss(predicted_q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_network.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())


def train_dqn_curriculum(
    config: DQNTrainingConfig,
    progress_callback: Callable[[DQNProgress], None] | None = None,
) -> DQNTrainingResult:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = choose_device()
    online_network = QNetwork().to(device)
    target_network = QNetwork().to(device)
    hard_update_target(target_network, online_network)
    target_network.eval()

    agent = DQNAgent(
        online_network,
        device=device,
        epsilon=config.epsilon_start,
        seed=config.seed,
    )
    optimizer = torch.optim.Adam(
        online_network.parameters(), lr=config.learning_rate
    )
    replay_buffer = ReplayBuffer(config.replay_capacity, seed=config.seed)

    total_steps = 0
    progress_records: list[DQNProgress] = []
    reached_depth = 1
    completed_curriculum = False

    for depth in range(1, config.max_depth + 1):
        reached_depth = depth
        env = RubiksCubeEnv(
            scramble_depth=depth,
            max_steps=2 * depth + 2,
            solved_reward=1.0,
            step_penalty=-0.01,
            timeout_penalty=-1.0,
            seed=config.seed + depth,
            encode_one_hot=True,
        )

        depth_steps = 0
        success_streak = 0
        losses_since_evaluation: list[float] = []
        best_success_rate = -1.0
        promote = False

        while depth_steps < config.max_steps_per_depth and not promote:
            observation, _info = env.reset()
            done = False

            while not done and depth_steps < config.max_steps_per_depth:
                agent.epsilon = linear_epsilon(total_steps, config)
                action = agent.select_action(observation, explore=True)
                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    _info,
                ) = env.step(action)
                done = terminated or truncated

                replay_buffer.add(
                    observation, action, reward, next_observation, done
                )
                observation = next_observation
                total_steps += 1
                depth_steps += 1

                if (
                    len(replay_buffer) >= config.warmup_steps
                    and len(replay_buffer) >= config.batch_size
                ):
                    online_network.train()
                    loss = optimize_dqn(
                        online_network,
                        target_network,
                        optimizer,
                        replay_buffer,
                        config.batch_size,
                        config.discount_factor,
                        device,
                    )
                    losses_since_evaluation.append(loss)

                if total_steps % config.target_update_interval == 0:
                    hard_update_target(target_network, online_network)

                if total_steps % config.evaluation_interval == 0:
                    evaluation = evaluate_agent(
                        agent,
                        episodes=config.evaluation_episodes,
                        scramble_depth=depth,
                        seed=100_000 + depth,
                        encode_one_hot=True,
                    )
                    mean_loss = (
                        sum(losses_since_evaluation)
                        / len(losses_since_evaluation)
                        if losses_since_evaluation
                        else 0.0
                    )
                    record = DQNProgress(
                        total_steps=total_steps,
                        depth=depth,
                        epsilon=agent.epsilon,
                        mean_loss=mean_loss,
                        evaluation=evaluation,
                    )
                    progress_records.append(record)
                    losses_since_evaluation.clear()

                    if progress_callback is not None:
                        progress_callback(record)

                    if evaluation.success_rate > best_success_rate:
                        best_success_rate = evaluation.success_rate
                        checkpoint_path = (
                            Path(config.checkpoint_directory)
                            / f"dqn_depth_{depth}.pt"
                        )
                        agent.save(
                            checkpoint_path,
                            metadata={
                                "depth": depth,
                                "total_steps": total_steps,
                                "success_rate": evaluation.success_rate,
                                "config": asdict(config),
                            },
                        )

                    if (
                        evaluation.success_rate
                        >= config.promotion_success_rate
                    ):
                        success_streak += 1
                    else:
                        success_streak = 0

                    promote = success_streak >= config.promotion_streak

        if not promote:
            break
        if depth == config.max_depth:
            completed_curriculum = True

    agent.epsilon = config.epsilon_end
    return DQNTrainingResult(
        agent=agent,
        progress=tuple(progress_records),
        reached_depth=reached_depth,
        completed_curriculum=completed_curriculum,
    )
