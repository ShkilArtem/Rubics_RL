import unittest

from rubiks_rl import RubiksCube
from rubiks_rl.search import weighted_a_star


class ZeroEstimator:
    def estimate(self, cubes: object) -> list[float]:
        return [0.0] * len(cubes)  # type: ignore[arg-type]


class SearchTests(unittest.TestCase):
    def test_search_finds_and_verifies_a_shallow_solution(self) -> None:
        cube = RubiksCube().apply(["R", "U"])
        result = weighted_a_star(
            cube,
            ZeroEstimator(),
            weight=1.0,
            max_expanded_states=5_000,
            time_limit_seconds=5.0,
        )

        self.assertTrue(result.solved, result.reason)
        self.assertTrue(cube.apply(result.actions).is_solved())

    def test_search_stops_cleanly_at_the_state_limit(self) -> None:
        cube = RubiksCube().move("R")
        result = weighted_a_star(
            cube,
            ZeroEstimator(),
            max_expanded_states=0,
            time_limit_seconds=5.0,
        )

        self.assertFalse(result.solved)
        self.assertEqual(result.reason, "state limit reached")


if __name__ == "__main__":
    unittest.main()
