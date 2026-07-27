"""Physical 3D geometry used by the Rubik's Cube simulator.

The public cube class stores only 54 colour numbers.  This module maps every
sticker to a position and an outward-facing direction in a small 3D coordinate
system.  Rotating a face then becomes a real 90-degree rotation instead of a
collection of hard-to-check row reversals.
"""

from __future__ import annotations

from collections import Counter


FACE_NAMES: tuple[str, ...] = ("U", "R", "F", "D", "L", "B")

Vector = tuple[int, int, int]
StickerGeometry = tuple[Vector, Vector]

FACE_NORMALS: dict[str, Vector] = {
    "U": (0, 1, 0),
    "R": (1, 0, 0),
    "F": (0, 0, 1),
    "D": (0, -1, 0),
    "L": (-1, 0, 0),
    "B": (0, 0, -1),
}


def _face_cell_geometry(face: str, row: int, column: int) -> StickerGeometry:
    """Return the 3D position and outward normal of one sticker."""

    # Coordinates are x=right, y=up and z=front.
    if face == "F":
        position = (column - 1, 1 - row, 1)
    elif face == "B":
        position = (1 - column, 1 - row, -1)
    elif face == "U":
        position = (column - 1, 1, row - 1)
    elif face == "D":
        position = (column - 1, -1, 1 - row)
    elif face == "R":
        position = (1, 1 - row, 1 - column)
    elif face == "L":
        position = (-1, 1 - row, column - 1)
    else:
        raise ValueError(f"Unknown face: {face!r}")

    return position, FACE_NORMALS[face]


def _build_sticker_geometry() -> tuple[StickerGeometry, ...]:
    result: list[StickerGeometry] = []
    for face in FACE_NAMES:
        for row in range(3):
            for column in range(3):
                result.append(_face_cell_geometry(face, row, column))
    return tuple(result)


STICKER_GEOMETRY = _build_sticker_geometry()
GEOMETRY_TO_INDEX = {
    geometry: index for index, geometry in enumerate(STICKER_GEOMETRY)
}


def _rotate_vector(vector: Vector, axis: int, direction: int) -> Vector:
    """Rotate a vector by 90 degrees around a positive coordinate axis."""

    x, y, z = vector

    if axis == 0:
        return (x, -z, y) if direction == 1 else (x, z, -y)
    if axis == 1:
        return (z, y, -x) if direction == 1 else (-z, y, x)
    if axis == 2:
        return (-y, x, z) if direction == 1 else (y, -x, z)
    raise ValueError(f"Axis must be 0, 1 or 2, got {axis}")


def _build_clockwise_permutation(face: str) -> tuple[int, ...]:
    """Map every old sticker index to its index after one clockwise turn."""

    normal = FACE_NORMALS[face]
    axis = next(index for index, value in enumerate(normal) if value != 0)
    layer_sign = normal[axis]

    # Clockwise as seen from outside is -90 degrees around the outward normal.
    direction = -layer_sign
    permutation: list[int] = []

    for position, sticker_normal in STICKER_GEOMETRY:
        if position[axis] == layer_sign:
            new_position = _rotate_vector(position, axis, direction)
            new_normal = _rotate_vector(sticker_normal, axis, direction)
            new_index = GEOMETRY_TO_INDEX[(new_position, new_normal)]
        else:
            new_index = GEOMETRY_TO_INDEX[(position, sticker_normal)]
        permutation.append(new_index)

    return tuple(permutation)


CLOCKWISE_PERMUTATIONS = {
    face: _build_clockwise_permutation(face) for face in FACE_NAMES
}


def apply_permutation(
    state: tuple[int, ...], permutation: tuple[int, ...]
) -> tuple[int, ...]:
    moved = [0] * len(state)
    for old_index, new_index in enumerate(permutation):
        moved[new_index] = state[old_index]
    return tuple(moved)


def piece_color_signatures(state: tuple[int, ...]) -> Counter[tuple[int, ...]]:
    """Return the colour groups belonging to the cube's physical pieces."""

    colors_by_position: dict[Vector, list[int]] = {}
    for color, (position, _normal) in zip(state, STICKER_GEOMETRY):
        colors_by_position.setdefault(position, []).append(color)

    signatures = [tuple(sorted(colors)) for colors in colors_by_position.values()]
    return Counter(signatures)
