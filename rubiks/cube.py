"""Facelet-level (54-sticker) Rubik's Cube engine.

Faces are stored as a flat 54-element array, 9 stickers per face, in
row-major order (row 0 = top row of that face as drawn in the net below),
in this face order and index range:

        U U U
        U U U
        U U U
L L L   F F F   R R R   B B B
L L L   F F F   R R R   B B B
L L L   F F F   R R R   B B B
        D D D
        D D D
        D D D

    U:  0- 8      R:  9-17      F: 18-26
    D: 27-35      L: 36-44      B: 45-53

Each sticker's color is represented by the index (0-5) of the face it
started on when solved.

Move permutations are not hand-typed from a cycle table (an easy place to
introduce a subtle, hard-to-notice sign/direction error). Instead, every
sticker is assigned an explicit 3D position + outward-facing normal on a
3x3x3 cube centered at the origin, a real 90-degree rotation matrix is
applied to the stickers in the turning layer, and the resulting permutation
is derived by matching rotated (position, normal) pairs back to sticker
indices. This is then checked against known group-theoretic facts about
the cube in tests/test_cube.py (e.g. every move has order 4, opposite
faces are untouched by a move, (R U R' U')^6 == solved).
"""

from __future__ import annotations

import numpy as np

from rubiks._geometry import MOVE_DEF, build_cw_permutation

FACE_NAMES = "URFDLB"
U, R, F, D, L, B = range(6)

_CW_PERM = {face: build_cw_permutation(face, n=3) for face in MOVE_DEF}


def solved_state() -> np.ndarray:
    return np.array([f for f in range(6) for _ in range(9)], dtype=np.int8)


def apply_move(state: np.ndarray, move: str) -> np.ndarray:
    """Apply a single move (e.g. "U", "U'", "U2") and return the new state."""
    face = move[0]
    suffix = move[1:]
    perm = _CW_PERM[face]
    if suffix == "":
        return state[perm]
    if suffix == "2":
        return state[perm][perm]
    if suffix == "'":
        return state[perm][perm][perm]
    raise ValueError(f"Unknown move: {move!r}")


ALL_MOVES = [f + s for f in "URFDLB" for s in ("", "'", "2")]


def apply_sequence(state: np.ndarray, moves) -> np.ndarray:
    for m in moves:
        state = apply_move(state, m)
    return state


def is_solved(state: np.ndarray) -> bool:
    return all(np.all(state[f * 9 : f * 9 + 9] == state[f * 9]) for f in range(6))

