import numpy as np
import pytest

from rubiks.cube2x2 import apply_move, apply_sequence, is_solved, solved_state

BASE_FACES = list("URFDLB")
OPPOSITE_FACE = {"U": "D", "D": "U", "R": "L", "L": "R", "F": "B", "B": "F"}


def test_solved_state_has_four_of_each_color():
    state = solved_state()
    for color in range(6):
        assert np.sum(state == color) == 4


def test_solved_state_is_solved():
    assert is_solved(solved_state())


@pytest.mark.parametrize("face", BASE_FACES)
def test_move_applied_four_times_is_identity(face):
    state = solved_state()
    for _ in range(4):
        state = apply_move(state, face)
    assert np.array_equal(state, solved_state())


@pytest.mark.parametrize("face", BASE_FACES)
def test_move_and_its_inverse_cancel(face):
    state = apply_move(solved_state(), face)
    state = apply_move(state, face + "'")
    assert np.array_equal(state, solved_state())


@pytest.mark.parametrize("face", BASE_FACES)
def test_double_move_equals_move_applied_twice(face):
    twice = apply_sequence(solved_state(), [face, face])
    double = apply_move(solved_state(), face + "2")
    assert np.array_equal(twice, double)


@pytest.mark.parametrize("face", BASE_FACES)
def test_single_move_actually_changes_state(face):
    state = apply_move(solved_state(), face)
    assert not np.array_equal(state, solved_state())


@pytest.mark.parametrize("face", BASE_FACES)
def test_opposite_face_untouched_by_move(face):
    opp = OPPOSITE_FACE[face]
    face_idx = "URFDLB".index(opp)
    before = solved_state()
    after = apply_move(before, face)
    before_slice = before[face_idx * 4 : face_idx * 4 + 4]
    after_slice = after[face_idx * 4 : face_idx * 4 + 4]
    assert np.array_equal(before_slice, after_slice)


def test_scramble_then_exact_inverse_returns_to_solved():
    scramble = ["U", "R'", "F2", "D", "L", "B'", "U2", "R", "F", "D'"]
    inverse = []
    for m in reversed(scramble):
        if m.endswith("'"):
            inverse.append(m[0])
        elif m.endswith("2"):
            inverse.append(m)
        else:
            inverse.append(m + "'")
    state = apply_sequence(solved_state(), scramble)
    state = apply_sequence(state, inverse)
    assert np.array_equal(state, solved_state())


def test_random_scramble_is_generally_not_solved():
    rng = np.random.default_rng(0)
    moves = [f + s for f in "URFDLB" for s in ("", "'", "2")]
    state = solved_state()
    for _ in range(20):
        state = apply_move(state, moves[rng.integers(len(moves))])
    assert not is_solved(state)
