import importlib.util
import tempfile
import unittest
from pathlib import Path


ML_AVAILABLE = (
    importlib.util.find_spec("numpy") is not None
    and importlib.util.find_spec("torch") is not None
)


@unittest.skipUnless(ML_AVAILABLE, "NumPy and PyTorch are not installed")
class NeuralRLTests(unittest.TestCase):
    def test_network_output_and_replay_buffer(self) -> None:
        import torch

        from rubiks_rl.dqn import QNetwork, ReplayBuffer

        network = QNetwork()
        output = network(torch.zeros((4, 324)))
        self.assertEqual(tuple(output.shape), (4, 12))

        buffer = ReplayBuffer(capacity=2, seed=1)
        state = [0.0] * 324
        buffer.add(state, 3, 1.0, state, True)
        self.assertEqual(len(buffer), 1)
        self.assertEqual(buffer.sample(1)[0].action, 3)

    def test_target_network_update_copies_weights(self) -> None:
        import torch

        from rubiks_rl.dqn import QNetwork, hard_update_target

        online = QNetwork()
        target = QNetwork()
        with torch.no_grad():
            for parameter in online.parameters():
                parameter.fill_(0.25)

        hard_update_target(target, online)
        for online_parameter, target_parameter in zip(
            online.parameters(), target.parameters()
        ):
            self.assertTrue(torch.equal(online_parameter, target_parameter))

    def test_checkpoint_round_trip(self) -> None:
        import torch

        from rubiks_rl.dqn import DQNAgent, QNetwork

        device = torch.device("cpu")
        agent = DQNAgent(QNetwork().to(device), device)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.pt"
            agent.save(path, metadata={"depth": 1})
            loaded = DQNAgent.load(path, device=device)

        observation = [0.0] * 324
        self.assertEqual(
            agent.select_action(observation, explore=False),
            loaded.select_action(observation, explore=False),
        )

    def test_short_dqn_training_run(self) -> None:
        from rubiks_rl.dqn import DQNTrainingConfig, train_dqn_curriculum

        with tempfile.TemporaryDirectory() as directory:
            result = train_dqn_curriculum(
                DQNTrainingConfig(
                    max_depth=1,
                    max_steps_per_depth=80,
                    batch_size=8,
                    replay_capacity=100,
                    warmup_steps=8,
                    target_update_interval=20,
                    evaluation_interval=40,
                    evaluation_episodes=5,
                    promotion_success_rate=0.0,
                    promotion_streak=1,
                    epsilon_decay_steps=50,
                    checkpoint_directory=directory,
                    seed=7,
                )
            )

        self.assertTrue(result.completed_curriculum)
        self.assertGreaterEqual(len(result.progress), 1)

    def test_value_network_output_shape(self) -> None:
        import torch

        from rubiks_rl.value_learning import ValueNetwork

        network = ValueNetwork()
        output = network(torch.zeros((3, 324)))
        self.assertEqual(tuple(output.shape), (3,))

    def test_short_value_training_run_and_checkpoint(self) -> None:
        from rubiks_rl import RubiksCube
        from rubiks_rl.value_learning import (
            NeuralValueEstimator,
            ValueTrainingConfig,
            train_value_network,
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "value.pt"
            metrics_path = Path(directory) / "metrics.csv"
            estimator = train_value_network(
                ValueTrainingConfig(
                    max_scramble_depth=2,
                    bootstrap_steps=2,
                    bellman_steps=2,
                    batch_size=4,
                    target_update_interval=1,
                    report_interval=1,
                    checkpoint_path=str(checkpoint_path),
                    output_path=str(metrics_path),
                    seed=3,
                )
            )
            loaded = NeuralValueEstimator.load(
                checkpoint_path, device=estimator.device
            )

            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(metrics_path.exists())
            self.assertEqual(len(loaded.estimate([RubiksCube()])), 1)


if __name__ == "__main__":
    unittest.main()
