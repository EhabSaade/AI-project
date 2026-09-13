"""Tests for the corner pattern database.

The full database takes minutes to build, so these tests use the truncated
`shallow_db` fixture from conftest.py. A truncated database still has exact
distances for every state it did reach.
"""

import numpy as np

from rubiks.corners import (
    NUM_CORNER_STATES,
    NUM_ORIENTATIONS,
    build_move_tables,
    encode,
    encode_batch,
    extract,
)
from rubiks.cube import ALL_MOVES, apply_move, apply_sequence, solved_state
from rubiks.encoding import extract_corners
from rubiks.pattern_db import UNVISITED, _apply_move_to_indices, lookup, lookup_batch


def random_states(rng, count, max_length):
    states = []
    for _ in range(count):
        length = int(rng.integers(1, max_length + 1))
        scramble = [ALL_MOVES[rng.integers(len(ALL_MOVES))] for _ in range(length)]
        states.append(apply_sequence(solved_state(), scramble))
    return np.stack(states)


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


def test_batch_encoding_matches_scalar_encoding():
    """The vectorized rank must agree with the validated scalar one, row by row."""
    rng = np.random.default_rng(4)
    states = random_states(rng, 300, max_length=25)
    expected = np.array([encode(*extract(state)) for state in states])
    assert np.array_equal(encode_batch(*extract_corners(states)), expected)


def test_batch_encoding_covers_extreme_ranks():
    identity = np.arange(8)[None, :]
    reversed_order = np.arange(7, -1, -1)[None, :]
    all_twisted = np.array([[2, 2, 2, 2, 2, 2, 2, 1]])
    untwisted = np.zeros((1, 8), dtype=np.int64)
    assert encode_batch(identity, untwisted).tolist() == [0]
    assert encode_batch(reversed_order, all_twisted).tolist() == [
        encode(reversed_order[0], all_twisted[0])
    ]
    assert encode_batch(reversed_order, all_twisted)[0] == NUM_CORNER_STATES - 1


def test_lookup_batch_matches_scalar_lookup(shallow_db):
    rng = np.random.default_rng(5)
    states = random_states(rng, 300, max_length=5)
    expected = np.array([lookup(shallow_db, state) for state in states])
    assert np.array_equal(lookup_batch(shallow_db, states), expected)
