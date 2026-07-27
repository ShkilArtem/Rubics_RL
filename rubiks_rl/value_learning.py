"""Value learning used as a heuristic for deeper cube searches."""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Any

try:
    import numpy as np
    import torch
    from torch import Tensor, nn
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "Value-learning mode requires NumPy and PyTorch. Activate the Python "
        "3.12 virtual environment and install requirements.txt."
    ) from error

from .cube import ACTIONS, RubiksCube
from .dqn import choose_device


class ValueNetwork(nn.Module):
    """Estimate the remaining number of moves from a one-hot cube state."""

    def __init__(self, input_size: int = 324) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, observations: Tensor) -> Tensor:
        return self.layers(observations).squeeze(-1)


class NeuralValueEstimator:
    """Small adapter used by the search algorithm."""

    def __init__(self, network: ValueNetwork, device: "torch.device") -> None:
        self.network = network
        self.device = device

    def estimate(self, cubes: Sequence[RubiksCube]) -> list[float]:
        if not cubes:
            return []

        observations = np.asarray(
            [cube.one_hot() for cube in cubes], dtype=np.float32
        )
        tensor = torch.as_tensor(observations, device=self.device)
        self.network.eval()
        with torch.no_grad():
            values = self.network(tensor).clamp(min=0.0)
        return [float(value) for value in values.cpu().tolist()]

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
        cls, path: str | Path, device: "torch.device | None" = None
    ) -> "NeuralValueEstimator":
        selected_device = device or choose_device()
        checkpoint = torch.load(
            Path(path), map_location=selected_device, weights_only=False
        )
        network = ValueNetwork().to(selected_device)
        network.load_state_dict(checkpoint["model_state_dict"])
        return cls(network, selected_device)


@dataclass(frozen=True)
class ValueTrainingConfig:
    max_scramble_depth: int = 10
    bootstrap_steps: int = 20_000
    bellman_steps: int = 20_000
    batch_size: int = 256
    learning_rate: float = 0.001
    target_update_interval: int = 1_000
    report_interval: int = 500
    seed: int = 84
    checkpoint_path: str = "checkpoints/value_depth_10.pt"
    output_path: str = "outputs/value_metrics.csv"


@dataclass(frozen=True)
class ValueProgress:
    phase: str
    step: int
    loss: float


def _random_training_batch(
    rng: Random, batch_size: int, max_depth: int
) -> tuple[list[RubiksCube], list[float]]:
    cubes: list[RubiksCube] = []
    approximate_distances: list[float] = []

    for _ in range(batch_size):
        depth = rng.randint(1, max_depth)
        cube, _actions = RubiksCube().scramble(depth, rng)
        cubes.append(cube)
        approximate_distances.append(float(depth))

    return cubes, approximate_distances


def _observations_tensor(
    cubes: Sequence[RubiksCube], device: "torch.device"
) -> Tensor:
    observations = np.asarray(
        [cube.one_hot() for cube in cubes], dtype=np.float32
    )
    return torch.as_tensor(observations, device=device)


def train_value_network(
    config: ValueTrainingConfig,
    progress_callback: Callable[[ValueProgress], None] | None = None,
) -> NeuralValueEstimator:
    """Train in two phases: path-length bootstrap, then Bellman refinement."""

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = Random(config.seed)
    device = choose_device()

    network = ValueNetwork().to(device)
    target_network = ValueNetwork().to(device)
    target_network.load_state_dict(network.state_dict())
    target_network.eval()
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    records: list[ValueProgress] = []

    for step in range(1, config.bootstrap_steps + 1):
        cubes, distances = _random_training_batch(
            rng, config.batch_size, config.max_scramble_depth
        )
        observations = _observations_tensor(cubes, device)
        targets = torch.as_tensor(
            distances, dtype=torch.float32, device=device
        )

        predictions = network(observations)
        loss = nn.functional.smooth_l1_loss(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), max_norm=10.0)
        optimizer.step()

        if step % config.report_interval == 0:
            record = ValueProgress("bootstrap", step, float(loss.item()))
            records.append(record)
            if progress_callback is not None:
                progress_callback(record)

    target_network.load_state_dict(network.state_dict())

    for step in range(1, config.bellman_steps + 1):
        cubes, _distances = _random_training_batch(
            rng, config.batch_size, config.max_scramble_depth
        )
        observations = _observations_tensor(cubes, device)

        all_children: list[RubiksCube] = []
        for cube in cubes:
            all_children.extend(cube.move(action) for action in ACTIONS)

        child_observations = _observations_tensor(all_children, device)
        with torch.no_grad():
            child_values = target_network(child_observations)
            child_values = child_values.reshape(config.batch_size, len(ACTIONS))

            solved_mask = torch.as_tensor(
                [child.is_solved() for child in all_children],
                dtype=torch.bool,
                device=device,
            ).reshape(config.batch_size, len(ACTIONS))
            child_values = torch.where(
                solved_mask, torch.zeros_like(child_values), child_values
            )
            targets = 1.0 + child_values.clamp(min=0.0).min(dim=1).values

        predictions = network(observations)
        loss = nn.functional.smooth_l1_loss(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), max_norm=10.0)
        optimizer.step()

        if step % config.target_update_interval == 0:
            target_network.load_state_dict(network.state_dict())

        if step % config.report_interval == 0:
            record = ValueProgress("bellman", step, float(loss.item()))
            records.append(record)
            if progress_callback is not None:
                progress_callback(record)

    estimator = NeuralValueEstimator(network, device)
    estimator.save(
        config.checkpoint_path,
        metadata={
            "max_scramble_depth": config.max_scramble_depth,
            "config": asdict(config),
        },
    )

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["phase", "step", "loss"])
        for record in records:
            writer.writerow([record.phase, record.step, record.loss])

    return estimator

