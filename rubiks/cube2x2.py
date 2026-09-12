"""2x2x2 (pocket cube) engine.

Same facelet approach as cube.py (see its module docstring), but with a
2x2 grid per face (4 stickers/face, 24 total) instead of 3x3. Reuses the
identical rotation-derived permutation logic from _geometry.py so the two
cube sizes share one validated implementation of the tricky part.

Face layout, 4 stickers per face, row-major:

    U:  0- 3      R:  4- 7      F:  8-11
    D: 12-15      L: 16-19      B: 20-23
"""

from __future__ import annotations

import numpy as np

from rubiks._geometry import MOVE_DEF, build_cw_permutation

FACE_NAMES = "URFDLB"
U, R, F, D, L, B = range(6)

_CW_PERM = {face: build_cw_permutation(face, n=2) for face in MOVE_DEF}

NUM_STICKERS = 24


def solved_state() -> np.ndarray:
    return np.array([f for f in range(6) for _ in range(4)], dtype=np.int8)


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
    return all(np.all(state[f * 4 : f * 4 + 4] == state[f * 4]) for f in range(6))
