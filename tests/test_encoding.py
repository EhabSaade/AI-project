import numpy as np
import pytest

from rubiks import corners, encoding
from rubiks.cube import ALL_MOVES, apply_move, apply_sequence, solved_state


def random_scramble(rng, length=25):
    return [ALL_MOVES[rng.integers(len(ALL_MOVES))] for _ in range(length)]


def random_states(rng, count, length=25):
    return np.stack(
        [apply_sequence(solved_state(), random_scramble(rng, length)) for _ in range(count)]
    )


def test_edge_slots_and_stickers_are_well_formed():
    assert len(encoding.EDGE_POSITIONS) == 12
    assert encoding.EDGE_STICKERS.shape == (12, 2)
    assert len(set(encoding.EDGE_STICKERS.flatten().tolist())) == 24


def test_lookup_tables_cover_every_state_encountered():
    """A -1 would mean a colour combination the tables do not know about."""
    rng = np.random.default_rng(0)
    states = random_states(rng, 100)
    corner_cubelet, corner_orientation = encoding.extract_corners(states)
    edge_cubelet, edge_orientation = encoding.extract_edges(states)
    assert corner_cubelet.min() >= 0 and corner_orientation.min() >= 0
    assert edge_cubelet.min() >= 0 and edge_orientation.min() >= 0


def test_vectorized_corner_extraction_matches_corners_module():
    """Cross-check the fast path against the already-validated scalar one."""
    rng = np.random.default_rng(1)
    states = random_states(rng, 50)
    cubelet, orientation = encoding.extract_corners(states)
    for i, state in enumerate(states):
        expected_permutation, expected_orientation = corners.extract(state)
        assert np.array_equal(cubelet[i], expected_permutation)
        assert np.array_equal(orientation[i], expected_orientation)


def test_edge_permutation_is_always_a_permutation():
    rng = np.random.default_rng(2)
    states = random_states(rng, 100)
    cubelet, _ = encoding.extract_edges(states)
    for row in cubelet:
        assert sorted(row.tolist()) == list(range(12))


def test_edge_flip_parity_is_invariant():
    """Total edge flips stay even -- the defining constraint on edge orientation."""
    rng = np.random.default_rng(3)
    states = random_states(rng, 200)
    _, orientation = encoding.extract_edges(states)
    assert (orientation.sum(axis=1) % 2 == 0).all()


@pytest.mark.parametrize("move", ALL_MOVES)
def test_single_move_preserves_edge_flip_parity(move):
    state = apply_move(solved_state(), move)
    _, orientation = encoding.extract_edges(state[None, :])
    assert orientation.sum() % 2 == 0


def test_solved_state_encoding_is_identity_placement():
    """In a solved cube every cubelet is in its own slot, untwisted."""
    encoded = encoding.encode(solved_state()).reshape(20, 24)
    for cubelet in range(8):
        assert encoded[cubelet, cubelet * 3] == 1.0
    for edge in range(12):
        assert encoded[8 + edge, edge * 2] == 1.0


def test_encoding_is_one_hot_per_cubelet():
    rng = np.random.default_rng(4)
    states = random_states(rng, 100)
    encoded = encoding.encode(states).reshape(-1, 20, 24)
    assert np.array_equal(encoded.sum(axis=2), np.ones((100, 20), dtype=np.float32))
    assert set(np.unique(encoded).tolist()) <= {0.0, 1.0}


def test_encoding_shape_and_dtype():
    rng = np.random.default_rng(5)
    states = random_states(rng, 7)
    encoded = encoding.encode(states)
    assert encoded.shape == (7, encoding.ENCODED_SIZE) == (7, 480)
    assert encoded.dtype == np.float32
    assert encoding.encode(solved_state()).shape == (1, 480)


def test_distinct_states_get_distinct_encodings():
    rng = np.random.default_rng(6)
    states = random_states(rng, 200)
    unique_states = {s.tobytes() for s in states}
    unique_encodings = {e.tobytes() for e in encoding.encode(states)}
    assert len(unique_encodings) == len(unique_states)


def test_equal_states_get_equal_encodings():
    """A scramble and its undo-then-redo must encode identically."""
    rng = np.random.default_rng(7)
    scramble = random_scramble(rng, 12)
    once = apply_sequence(solved_state(), scramble)
    twice = apply_sequence(apply_sequence(once, ["U", "U'"]), [])
    assert np.array_equal(encoding.encode(once), encoding.encode(twice))


def test_batch_encoding_matches_single_encoding():
    rng = np.random.default_rng(8)
    states = random_states(rng, 20)
    batched = encoding.encode(states)
    for i, state in enumerate(states):
        assert np.array_equal(batched[i], encoding.encode(state)[0])
