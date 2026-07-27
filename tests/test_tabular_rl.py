import unittest

from rubiks_rl import (
    RandomAgent,
    TabularQAgent,
    TabularTrainingConfig,
    evaluate_agent,
    train_tabular_agent,
)


class TabularRLTests(unittest.TestCase):
    def test_evaluation_handles_an_already_solved_reset(self) -> None:
        result = evaluate_agent(
            RandomAgent(action_count=12, seed=1),
            episodes=5,
            scramble_depth=0,
            seed=2,
        )

        self.assertEqual(result.success_rate, 1.0)
        self.assertEqual(result.average_steps, 0.0)

    def test_terminal_q_learning_update_matches_the_formula(self) -> None:
        agent = TabularQAgent(
            action_count=2,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.0,
        )

        td_error = agent.learn(
            observation=(0, 0),
            action=1,
            reward=1.0,
            next_observation=(1, 1),
            done=True,
        )

        self.assertAlmostEqual(td_error, 1.0)
        self.assertAlmostEqual(agent.q_values((0, 0))[1], 0.1)

    def test_non_terminal_update_bootstraps_from_next_state(self) -> None:
        agent = TabularQAgent(
            action_count=2,
            learning_rate=0.5,
            discount_factor=0.9,
            epsilon=0.0,
        )
        agent.q_values((1, 1))[0] = 2.0

        agent.learn(
            observation=(0, 0),
            action=1,
            reward=0.5,
            next_observation=(1, 1),
            done=False,
        )

        # target = 0.5 + 0.9 * 2.0 = 2.3
        # new Q = 0.0 + 0.5 * (2.3 - 0.0) = 1.15
        self.assertAlmostEqual(agent.q_values((0, 0))[1], 1.15)

    def test_depth_one_training_smoke_test(self) -> None:
        agent = train_tabular_agent(
            TabularTrainingConfig(
                scramble_depths=(1,),
                episodes_per_depth=5_000,
                seed=42,
                progress_interval=5_000,
            )
        )
        result = evaluate_agent(
            agent,
            episodes=200,
            scramble_depth=1,
            seed=10_042,
        )

        self.assertGreaterEqual(result.success_rate, 0.95)


if __name__ == "__main__":
    unittest.main()
