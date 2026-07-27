"""Saving numeric training history and optional learning-curve plots."""

from __future__ import annotations

import csv
from pathlib import Path

from .dqn import DQNProgress


def save_dqn_metrics(
    progress: tuple[DQNProgress, ...] | list[DQNProgress],
    output_directory: str | Path = "outputs",
) -> tuple[Path, Path | None]:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "dqn_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "total_steps",
                "depth",
                "epsilon",
                "mean_loss",
                "success_rate",
                "average_reward",
                "average_steps",
            ]
        )
        for record in progress:
            writer.writerow(
                [
                    record.total_steps,
                    record.depth,
                    record.epsilon,
                    record.mean_loss,
                    record.evaluation.success_rate,
                    record.evaluation.average_reward,
                    record.evaluation.average_steps,
                ]
            )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return csv_path, None

    steps = [record.total_steps for record in progress]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(steps, [item.mean_loss for item in progress])
    axes[0, 0].set_title("DQN loss")
    axes[0, 1].plot(
        steps, [item.evaluation.success_rate for item in progress]
    )
    axes[0, 1].set_title("Success rate")
    axes[1, 0].plot(steps, [item.epsilon for item in progress])
    axes[1, 0].set_title("Epsilon")
    axes[1, 1].plot(
        steps, [item.evaluation.average_steps for item in progress]
    )
    axes[1, 1].set_title("Average episode length")

    for axis in axes.flat:
        axis.set_xlabel("Environment steps")
        axis.grid(alpha=0.3)

    figure.tight_layout()
    plot_path = output_path / "dqn_training.png"
    figure.savefig(plot_path, dpi=140)
    plt.close(figure)
    return csv_path, plot_path
