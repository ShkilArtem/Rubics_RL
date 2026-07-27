from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from random import Random
from typing import ClassVar

from .geometry import CLOCKWISE_PERMUTATIONS, apply_permutation


# The cube has 6 faces. Each face has 9 stickers:
#
#   0 1 2
#   3 4 5
#   6 7 8
#
# We store the full cube as one long tuple with 54 numbers:
# U face, then R, then F, then D, then L, then B.
FACE_NAMES: tuple[str, ...] = ("U", "R", "F", "D", "L", "B")
FACE_COLORS: dict[str, int] = {
    "U": 0,
    "R": 1,
    "F": 2,
    "D": 3,
    "L": 4,
    "B": 5,
}

ACTIONS: tuple[str, ...] = (
    "U",
    "U'",
    "D",
    "D'",
    "L",
    "L'",
    "R",
    "R'",
    "F",
    "F'",
    "B",
    "B'",
)

INVERSE_ACTIONS: dict[str, str] = {
    "U": "U'",
    "U'": "U",
    "D": "D'",
    "D'": "D",
    "L": "L'",
    "L'": "L",
    "R": "R'",
    "R'": "R",
    "F": "F'",
    "F'": "F",
    "B": "B'",
    "B'": "B",
}


def solved_state() -> tuple[int, ...]:
    stickers: list[int] = []
    for face_name in FACE_NAMES:
        color = FACE_COLORS[face_name]
        stickers.extend([color] * 9)
    return tuple(stickers)


def split_state_into_faces(state: tuple[int, ...]) -> dict[str, list[int]]:
    faces: dict[str, list[int]] = {}
    for face_index, face_name in enumerate(FACE_NAMES):
        start = face_index * 9
        end = start + 9
        faces[face_name] = list(state[start:end])
    return faces


@dataclass(frozen=True)
class RubiksCube:
    state: tuple[int, ...] = solved_state()

    size: ClassVar[int] = 54

    def __post_init__(self) -> None:
        if len(self.state) != self.size:
            raise ValueError(f"A cube state must contain {self.size} stickers.")

        expected_counts = Counter({color: 9 for color in FACE_COLORS.values()})
        if Counter(self.state) != expected_counts:
            raise ValueError("A cube state must contain exactly 9 stickers of each colour.")

    def move(self, action: str) -> "RubiksCube":
        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action!r}")

        permutation = CLOCKWISE_PERMUTATIONS[action[0]]
        number_of_turns = 3 if action.endswith("'") else 1

        state = self.state
        for _ in range(number_of_turns):
            state = apply_permutation(state, permutation)
        return RubiksCube(state)

    def apply(self, actions: list[str] | tuple[str, ...]) -> "RubiksCube":
        cube = self
        for action in actions:
            cube = cube.move(action)
        return cube

    def scramble(self, depth: int, rng: Random | None = None) -> tuple["RubiksCube", list[str]]:
        if depth < 0:
            raise ValueError("Scramble depth must be non-negative.")

        rng = rng or Random()
        cube = self
        actions: list[str] = []
        previous_face: str | None = None

        for _ in range(depth):
            possible_actions = [
                action for action in ACTIONS if action[0] != previous_face
            ]

            action = rng.choice(possible_actions)
            cube = cube.move(action)
            actions.append(action)
            previous_face = action[0]

        return cube, actions

    def is_solved(self) -> bool:
        return self.state == solved_state()

    def one_hot(self) -> list[float]:
        encoded_state = [0.0] * (self.size * len(FACE_NAMES))

        for sticker_index, color in enumerate(self.state):
            position = sticker_index * len(FACE_NAMES) + color
            encoded_state[position] = 1.0

        return encoded_state

    def as_faces(self) -> dict[str, tuple[int, ...]]:
        faces = split_state_into_faces(self.state)
        return {face_name: tuple(faces[face_name]) for face_name in FACE_NAMES}

    def as_cross(self) -> str:
        """Return the cube stickers as a conventional unfolded cross."""
        faces = self.as_faces()

        def face_rows(face_name: str) -> list[str]:
            face = faces[face_name]
            return [" ".join(map(str, face[index : index + 3])) for index in range(0, 9, 3)]

        up = face_rows("U")
        right = face_rows("R")
        front = face_rows("F")
        down = face_rows("D")
        left = face_rows("L")
        back = face_rows("B")
        indent = " " * 6

        lines = [f"{indent}{line}" for line in up]
        lines.extend(
            f"{left[index]}  {front[index]}  {right[index]}  {back[index]}"
            for index in range(3)
        )
        lines.extend(f"{indent}{line}" for line in down)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.as_cross()
