"""Tests for the corner pattern database.

The full database takes minutes to build, so these tests build a truncated
one (BFS stopped early) and check the properties that matter. A truncated
database still has exact distances for every state it did reach.
"""

import numpy as np
import pytest

from rubiks.corners import (
    NUM_CORNER_STATES,
    NUM_ORIENTATIONS,
    build_move_tables,
    encode,
    extract,
)
from rubiks.cube import ALL_MOVES, apply_move, apply_sequence, solved_state
from rubiks.pattern_db import UNVISITED, _apply_move_to_indices


@pytest.fixture(scope="module")
def shallow_db():
    """Distances for every corner state within 5 moves of solved."""
    permutation_table, orientation_table = build_move_tables()
    distances = np.full(NUM_CORNER_STATES, UNVISITED, dtype=np.uint8)
    solved = encode(np.arange(8), np.zeros(8, dtype=np.int8))
    distances[solved] = 0
    frontier = np.array([solved], dtype=np.int32)

    for depth in range(1, 6):
        discovered = []
        for move_index in range(len(ALL_MOVES)):
            candidates = _apply_move_to_indices(
                frontier, move_index, permutation_table, orientation_table
            )
            fresh = np.unique(candidates[distances[candidates] == UNVISITED])
            if fresh.size:
                distances[fresh] = depth
                discovered.append(fresh)
        frontier = np.concatenate(discovered)

    return distances


def test_solved_state_has_distance_zero(shallow_db):
    assert shallow_db[encode(*extract(solved_state()))] == 0


def test_one_move_from_solved_has_distance_one(shallow_db):
    """Every single move must leave the corners exactly one move from solved."""
    for move in ALL_MOVES:
        state = apply_move(solved_state(), move)
        assert shallow_db[encode(*extract(state))] == 1


def test_distance_changes_by_at_most_one_per_move(shallow_db):
    """Neighbouring states differ in distance by at most 1 -- the property
    that makes the heuristic usable for pruning."""
    rng = np.random.default_rng(0)
    for _ in range(100):
        length = int(rng.integers(1, 5))
        scramble = [ALL_MOVES[rng.integers(len(ALL_MOVES))] for _ in range(length)]
        state = apply_sequence(solved_state(), scramble)
        here = shallow_db[encode(*extract(state))]
        if here == UNVISITED:
            continue
        for move in ALL_MOVES:
            there = shallow_db[encode(*extract(apply_move(state, move)))]
            if there != UNVISITED:
                assert abs(int(here) - int(there)) <= 1


def test_scrambles_are_never_further_than_their_scramble_length(shallow_db):
    """A state reached in k moves cannot be more than k moves from solved.

    This is the admissibility direction that matters: the database must not
    overestimate, or search built on it returns wrong answers.
    """
    rng = np.random.default_rng(1)
    for _ in range(200):
        length = int(rng.integers(1, 6))
        scramble = [ALL_MOVES[rng.integers(len(ALL_MOVES))] for _ in range(length)]
        state = apply_sequence(solved_state(), scramble)
        distance = shallow_db[encode(*extract(state))]
        if distance != UNVISITED:
            assert distance <= length


def test_every_state_at_depth_d_has_a_neighbour_at_depth_d_minus_one(shallow_db):
    """A BFS distance is only correct if a shorter path out actually exists."""
    permutation_table, orientation_table = build_move_tables()
    rng = np.random.default_rng(2)
    reached = np.nonzero((shallow_db != UNVISITED) & (shallow_db > 0))[0]
    sample = rng.choice(reached, size=300, replace=False).astype(np.int32)

    best = np.full(sample.shape, 255, dtype=np.int64)
    for move_index in range(len(ALL_MOVES)):
        neighbours = _apply_move_to_indices(
            sample, move_index, permutation_table, orientation_table
        )
        best = np.minimum(best, shallow_db[neighbours].astype(np.int64))

    assert np.array_equal(best, shallow_db[sample].astype(np.int64) - 1)


def test_index_arithmetic_stays_in_range():
    rng = np.random.default_rng(3)
    permutation_table, orientation_table = build_move_tables()
    indices = rng.integers(0, NUM_CORNER_STATES, size=1000).astype(np.int32)
    for move_index in range(len(ALL_MOVES)):
        moved = _apply_move_to_indices(
            indices, move_index, permutation_table, orientation_table
        )
        assert moved.min() >= 0
        assert moved.max() < NUM_CORNER_STATES
        assert (moved % NUM_ORIENTATIONS < NUM_ORIENTATIONS).all()
