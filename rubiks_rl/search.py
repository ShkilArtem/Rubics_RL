"""Weighted A* search guided by a learned value estimate."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from random import Random
from time import perf_counter
from typing import Protocol, Sequence

from .cube import ACTIONS, INVERSE_ACTIONS, RubiksCube


class ValueEstimator(Protocol):
    def estimate(self, cubes: Sequence[RubiksCube]) -> list[float]:
        ...


@dataclass(frozen=True)
class SearchResult:
    solved: bool
    actions: tuple[str, ...]
    expanded_states: int
    generated_states: int
    elapsed_seconds: float
    reason: str


@dataclass(frozen=True)
class SolverEvaluation:
    scrambles: int
    solved: int
    success_rate: float
    average_solution_length: float
    average_expanded_states: float


def weighted_a_star(
    start: RubiksCube,
    estimator: ValueEstimator,
    weight: float = 1.5,
    max_expanded_states: int = 100_000,
    time_limit_seconds: float = 30.0,
) -> SearchResult:
    """Find a solution while using the network as a non-admissible heuristic."""

    started_at = perf_counter()
    if start.is_solved():
        return SearchResult(True, (), 0, 0, 0.0, "already solved")

    start_h = estimator.estimate([start])[0]
    counter = 0
    # Heap item: priority, insertion order, path cost, cube, actions.
    frontier: list[
        tuple[float, int, int, RubiksCube, tuple[str, ...]]
    ] = [(weight * start_h, counter, 0, start, ())]
    best_cost: dict[tuple[int, ...], int] = {start.state: 0}
    expanded = 0
    generated = 0

    while frontier:
        elapsed = perf_counter() - started_at
        if elapsed >= time_limit_seconds:
            return SearchResult(
                False, (), expanded, generated, elapsed, "time limit reached"
            )
        if expanded >= max_expanded_states:
            return SearchResult(
                False, (), expanded, generated, elapsed, "state limit reached"
            )

        _priority, _order, path_cost, cube, actions = heapq.heappop(frontier)
        if path_cost != best_cost.get(cube.state):
            continue

        expanded += 1
        previous_action = actions[-1] if actions else None
        candidates: list[tuple[RubiksCube, tuple[str, ...], int]] = []

        for action in ACTIONS:
            if (
                previous_action is not None
                and action == INVERSE_ACTIONS[previous_action]
            ):
                continue

            child = cube.move(action)
            child_actions = actions + (action,)
            child_cost = path_cost + 1
            generated += 1

            if child.is_solved():
                elapsed = perf_counter() - started_at
                return SearchResult(
                    True,
                    child_actions,
                    expanded,
                    generated,
                    elapsed,
                    "solution found",
                )

            if child_cost >= best_cost.get(child.state, 10**9):
                continue
            best_cost[child.state] = child_cost
            candidates.append((child, child_actions, child_cost))

        heuristic_values = estimator.estimate(
            [candidate[0] for candidate in candidates]
        )
        for candidate, heuristic in zip(candidates, heuristic_values):
            child, child_actions, child_cost = candidate
            counter += 1
            priority = child_cost + weight * heuristic
            heapq.heappush(
                frontier,
                (priority, counter, child_cost, child, child_actions),
            )

    elapsed = perf_counter() - started_at
    return SearchResult(False, (), expanded, generated, elapsed, "frontier exhausted")


def evaluate_solver(
    estimator: ValueEstimator,
    scramble_depth: int,
    scrambles: int = 100,
    seed: int = 123_456,
    weight: float = 1.5,
    max_expanded_states: int = 100_000,
    time_limit_seconds: float = 30.0,
) -> SolverEvaluation:
    if scrambles <= 0:
        raise ValueError("scrambles must be positive")

    rng = Random(seed)
    solved_count = 0
    total_solution_length = 0
    total_expanded_states = 0

    for _ in range(scrambles):
        cube, _actions = RubiksCube().scramble(scramble_depth, rng)
        result = weighted_a_star(
            cube,
            estimator,
            weight=weight,
            max_expanded_states=max_expanded_states,
            time_limit_seconds=time_limit_seconds,
        )
        total_expanded_states += result.expanded_states

        if result.solved and cube.apply(result.actions).is_solved():
            solved_count += 1
            total_solution_length += len(result.actions)

    return SolverEvaluation(
        scrambles=scrambles,
        solved=solved_count,
        success_rate=solved_count / scrambles,
        average_solution_length=(
            total_solution_length / solved_count if solved_count else 0.0
        ),
        average_expanded_states=total_expanded_states / scrambles,
    )
