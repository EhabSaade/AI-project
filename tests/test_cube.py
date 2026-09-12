import numpy as np
import pytest

from rubiks.cube import (
    apply_move,
    apply_sequence,
    is_solved,
    solved_state,
)

BASE_FACES = list("URFDLB")
OPPOSITE_FACE = {"U": "D", "D": "U", "R": "L", "L": "R", "F": "B", "B": "F"}


def test_solved_state_has_nine_of_each_color():
    state = solved_state()
    for color in range(6):
        assert np.sum(state == color) == 9


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
    before_slice = before[face_idx * 9 : face_idx * 9 + 9]
    after_slice = after[face_idx * 9 : face_idx * 9 + 9]
    assert np.array_equal(before_slice, after_slice)


@pytest.mark.parametrize("face", BASE_FACES)
def test_turned_face_center_unchanged(face):
    face_idx = "URFDLB".index(face)
    center_index = face_idx * 9 + 4
    before = solved_state()
    after = apply_move(before, face)
    assert before[center_index] == after[center_index]


def test_sexy_move_has_order_six():
    """(R U R' U') is a well-known cube algorithm with order 6."""
    state = solved_state()
    for i in range(1, 7):
        state = apply_sequence(state, ["R", "U", "R'", "U'"])
        if i < 6:
            assert not is_solved(state), f"returned to solved too early, at repetition {i}"
    assert is_solved(state)


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
