import unittest
from collections import Counter
from random import Random

from rubiks_rl import ACTIONS, INVERSE_ACTIONS, RubiksCube, RubiksCubeEnv
from rubiks_rl.geometry import CLOCKWISE_PERMUTATIONS, piece_color_signatures


class CubeEnvTests(unittest.TestCase):
    def test_each_move_is_reversed_by_inverse(self) -> None:
        solved = RubiksCube()

        for action in ACTIONS:
            moved = solved.move(action).move(INVERSE_ACTIONS[action])
            self.assertTrue(moved.is_solved(), action)

    def test_four_same_quarter_turns_return_to_solved(self) -> None:
        solved = RubiksCube()

        for action in ACTIONS:
            cube = solved.apply([action] * 4)
            self.assertTrue(cube.is_solved(), action)

    def test_scramble_can_be_solved_by_reversed_actions(self) -> None:
        cube, scramble = RubiksCube().scramble(depth=20, rng=Random(7))
        solution = [INVERSE_ACTIONS[action] for action in reversed(scramble)]

        self.assertTrue(cube.apply(solution).is_solved())

    def test_scramble_never_turns_the_same_face_twice_in_a_row(self) -> None:
        _cube, scramble = RubiksCube().scramble(depth=100, rng=Random(9))

        for first, second in zip(scramble, scramble[1:]):
            self.assertNotEqual(first[0], second[0])

    def test_each_move_is_a_bijection_of_all_stickers(self) -> None:
        expected_indices = list(range(54))

        for face, permutation in CLOCKWISE_PERMUTATIONS.items():
            self.assertEqual(sorted(permutation), expected_indices, face)

    def test_moves_preserve_complete_physical_pieces(self) -> None:
        solved_signatures = piece_color_signatures(RubiksCube().state)

        for action in ACTIONS:
            moved = RubiksCube().move(action)
            self.assertEqual(
                piece_color_signatures(moved.state),
                solved_signatures,
                action,
            )

    def test_front_turn_has_the_expected_adjacent_strips(self) -> None:
        faces = RubiksCube().move("F").as_faces()

        self.assertEqual(faces["U"][6:9], (4, 4, 4))
        self.assertEqual(faces["D"][0:3], (1, 1, 1))
        self.assertEqual(faces["L"][2::3], (3, 3, 3))
        self.assertEqual(faces["R"][0::3], (0, 0, 0))

    def test_cube_rejects_an_invalid_sticker_inventory(self) -> None:
        with self.assertRaises(ValueError):
            RubiksCube((0,) * 54)

    def test_env_reset_and_step_api(self) -> None:
        env = RubiksCubeEnv(scramble_depth=1, max_steps=5, seed=3, encode_one_hot=False)
        observation, info = env.reset()

        self.assertEqual(len(observation), 54)
        self.assertEqual(len(info["scramble_actions"]), 1)

        inverse = INVERSE_ACTIONS[info["scramble_actions"][0]]
        _, reward, terminated, truncated, step_info = env.step(inverse)

        self.assertGreater(reward, 0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(step_info["is_solved"])

    def test_reset_is_deterministic_when_seed_is_given(self) -> None:
        env = RubiksCubeEnv(scramble_depth=8, encode_one_hot=False)
        first_observation, first_info = env.reset(seed=123)
        second_observation, second_info = env.reset(seed=123)

        self.assertEqual(first_observation, second_observation)
        self.assertEqual(
            first_info["scramble_actions"],
            second_info["scramble_actions"],
        )

    def test_timeout_reward_and_flags(self) -> None:
        env = RubiksCubeEnv(
            scramble_depth=1,
            max_steps=1,
            solved_reward=1.0,
            step_penalty=-0.01,
            timeout_penalty=-1.0,
            seed=1,
            encode_one_hot=False,
        )
        _observation, info = env.reset()
        scramble_action = info["scramble_actions"][0]
        wrong_action = next(
            action
            for action in ACTIONS
            if action != INVERSE_ACTIONS[scramble_action]
        )
        _observation, reward, terminated, truncated, _info = env.step(
            wrong_action
        )

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertAlmostEqual(reward, -1.01)

    def test_one_hot_observation_shape_and_values(self) -> None:
        observation, _info = RubiksCubeEnv().reset(seed=4)

        self.assertEqual(len(observation), 324)
        self.assertEqual(Counter(observation), Counter({0.0: 270, 1.0: 54}))

    def test_cube_is_rendered_as_an_unfolded_cross(self) -> None:
        expected = "\n".join(
            (
                "      0 0 0",
                "      0 0 0",
                "      0 0 0",
                "4 4 4  2 2 2  1 1 1  5 5 5",
                "4 4 4  2 2 2  1 1 1  5 5 5",
                "4 4 4  2 2 2  1 1 1  5 5 5",
                "      3 3 3",
                "      3 3 3",
                "      3 3 3",
            )
        )

        self.assertEqual(RubiksCube().as_cross(), expected)
        self.assertEqual(str(RubiksCube()), expected)


if __name__ == "__main__":
    unittest.main()
