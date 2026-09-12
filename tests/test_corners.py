import numpy as np
import pytest

from rubiks import corners
from rubiks.cube import ALL_MOVES, apply_move, apply_sequence, solved_state


def random_scramble(rng, length=25):
    return [ALL_MOVES[rng.integers(len(ALL_MOVES))] for _ in range(length)]


def test_solved_state_extracts_to_identity():
    permutation, orientation = corners.extract(solved_state())
    assert np.array_equal(permutation, np.arange(8))
    assert np.array_equal(orientation, np.zeros(8))


def test_permutation_is_always_a_permutation():
    rng = np.random.default_rng(0)
    for _ in range(50):
        state = apply_sequence(solved_state(), random_scramble(rng))
        permutation, _ = corners.extract(state)
        assert sorted(permutation.tolist()) == list(range(8))


def test_orientation_sum_is_invariant_mod_three():
    """The defining constraint on corner twists; a wrong convention breaks it."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        state = apply_sequence(solved_state(), random_scramble(rng))
        _, orientation = corners.extract(state)
        assert orientation.sum() % 3 == 0


@pytest.mark.parametrize("move", ALL_MOVES)
def test_single_move_preserves_orientation_invariant(move):
    state = apply_move(solved_state(), move)
    _, orientation = corners.extract(state)
    assert orientation.sum() % 3 == 0


def test_permutation_rank_round_trips():
    rng = np.random.default_rng(2)
    for _ in range(500):
        rank = int(rng.integers(corners.NUM_PERMUTATIONS))
        assert corners.permutation_rank(corners.permutation_unrank(rank)) == rank


def test_permutation_rank_is_a_bijection_on_a_sample():
    ranks = {
        corners.permutation_rank(corners.permutation_unrank(r))
        for r in range(1000)
    }
    assert len(ranks) == 1000


def test_orientation_rank_round_trips():
    for rank in range(0, corners.NUM_ORIENTATIONS, 7):
        orientation = corners.orientation_unrank(rank)
        assert orientation.sum() % 3 == 0
        assert corners.orientation_rank(orientation) == rank


def test_encode_decode_round_trips_on_real_states():
    rng = np.random.default_rng(3)
    for _ in range(100):
        state = apply_sequence(solved_state(), random_scramble(rng))
        permutation, orientation = corners.extract(state)
        index = corners.encode(permutation, orientation)
        assert 0 <= index < corners.NUM_CORNER_STATES
        decoded_permutation, decoded_orientation = corners.decode(index)
        assert np.array_equal(decoded_permutation, permutation)
        assert np.array_equal(decoded_orientation, orientation)


@pytest.mark.parametrize("move", ALL_MOVES)
def test_corner_level_move_matches_facelet_engine(move):
    """Applying a move on the corner coordinates must agree with the engine."""
    rng = np.random.default_rng(4)
    state = apply_sequence(solved_state(), random_scramble(rng))
    permutation, orientation = corners.extract(state)

    expected = corners.extract(apply_move(state, move))
    actual = corners.apply_to_corners(permutation, orientation, move)

    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])


def test_move_tables_agree_with_direct_application():
    permutation_table, orientation_table = corners.build_move_tables()
    rng = np.random.default_rng(5)
    for _ in range(50):
        state = apply_sequence(solved_state(), random_scramble(rng))
        permutation, orientation = corners.extract(state)
        p_rank = corners.permutation_rank(permutation)
        o_rank = corners.orientation_rank(orientation)

        for m, move in enumerate(ALL_MOVES):
            moved_p, moved_o = corners.apply_to_corners(
                permutation, orientation, move
            )
            assert permutation_table[p_rank, m] == corners.permutation_rank(moved_p)
            assert orientation_table[o_rank, m] == corners.orientation_rank(moved_o)
