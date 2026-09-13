"""Corner-only view of the 3x3x3 cube, and its coordinate encoding.

A pattern database over the eight corner cubelets needs the corner state as
a compact integer, and needs to apply moves to it quickly. This module
provides both.

The corner state is (permutation, orientation): which cubelet sits in each
of the eight corner slots, and how each is twisted (0, 1 or 2). Orientations
satisfy sum(orientation) == 0 (mod 3), so the eighth is determined by the
other seven, giving

    8! * 3^7 = 40,320 * 2,187 = 88,179,840

distinct states, encoded as

    index = permutation_rank * 2187 + orientation_rank

Crucially these two coordinates move independently: a face turn's effect on
the orientation vector depends only on the orientation vector (not on which
cubelet is where), and likewise for the permutation. So instead of one
88-million-entry transition table we can keep two small ones -- 40,320 x 18
and 2,187 x 18 -- and combine them. That is what makes a vectorized BFS over
the whole space practical.

Nothing here is hand-tabulated. The per-move slot permutation and twist are
read off the validated facelet engine by applying each move to a solved cube
and inspecting the result.
"""

from __future__ import annotations

from math import factorial

import numpy as np

from rubiks._geometry import build_index_geometry
from rubiks.cube import ALL_MOVES, apply_move, solved_state

NUM_CORNERS = 8
NUM_PERMUTATIONS = 40320  # 8!
NUM_ORIENTATIONS = 2187  # 3^7
NUM_CORNER_STATES = NUM_PERMUTATIONS * NUM_ORIENTATIONS

# The eight corner positions on a cube centred at the origin. The order is
# arbitrary but fixed; everything downstream is derived from it.
CORNER_POSITIONS = [
    (x, y, z) for y in (1, -1) for z in (1, -1) for x in (1, -1)
]

_GEOM_TO_INDEX = build_index_geometry(3)[1]

def _sticker_order(pos):
    """The slot's three sticker indices, ordered so twist counts consistently.

    The y-facing sticker comes first, so orientation 0 means "this cubelet's
    U/D-coloured sticker is on the U or D face" -- the usual convention.

    The remaining two must be ordered by the corner's chirality. Going
    around a corner in a fixed rotational direction visits the axes in one
    order at corners where x*y*z == 1 and the opposite order at the others.
    Using a single order everywhere measures twist backwards at half the
    corners: an R turn then reports +1 at all four affected corners instead
    of two at +1 and two at +2, and sum(orientation) % 3 == 0 fails.
    """
    x, y, z = pos
    facing_y = _GEOM_TO_INDEX[(pos, (0, y, 0))]
    facing_z = _GEOM_TO_INDEX[(pos, (0, 0, z))]
    facing_x = _GEOM_TO_INDEX[(pos, (x, 0, 0))]
    if x * y * z == 1:
        return [facing_y, facing_z, facing_x]
    return [facing_y, facing_x, facing_z]


CORNER_STICKERS = np.array(
    [_sticker_order(pos) for pos in CORNER_POSITIONS], dtype=np.int64
)

U_COLOR, D_COLOR = 0, 3

# Which cubelet is which, identified by its set of three sticker colours.
_SOLVED = solved_state()
_COLORS_TO_CUBELET = {
    frozenset(_SOLVED[CORNER_STICKERS[slot]].tolist()): slot
    for slot in range(NUM_CORNERS)
}


def extract(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Read (permutation, orientation) off a 54-sticker cube state.

    permutation[slot] is the id of the cubelet sitting in that slot, and
    orientation[slot] is which of the slot's three sticker positions holds
    that cubelet's U- or D-coloured sticker.
    """
    permutation = np.empty(NUM_CORNERS, dtype=np.int8)
    orientation = np.empty(NUM_CORNERS, dtype=np.int8)
    for slot in range(NUM_CORNERS):
        colors = state[CORNER_STICKERS[slot]]
        permutation[slot] = _COLORS_TO_CUBELET[frozenset(colors.tolist())]
        twist = np.nonzero((colors == U_COLOR) | (colors == D_COLOR))[0]
        orientation[slot] = twist[0]
    return permutation, orientation


def _move_effect(move: str) -> tuple[np.ndarray, np.ndarray]:
    """Slot permutation and added twist for one move.

    Applying a move to a solved cube gives exactly this: the cubelet found
    in slot i came from slot source[i], and picked up twist[i] doing so.
    """
    return extract(apply_move(solved_state(), move))


MOVE_EFFECT = {move: _move_effect(move) for move in ALL_MOVES}


def apply_to_corners(
    permutation: np.ndarray, orientation: np.ndarray, move: str
) -> tuple[np.ndarray, np.ndarray]:
    source, twist = MOVE_EFFECT[move]
    return permutation[source], (orientation[source] + twist) % 3


def permutation_rank(permutation) -> int:
    """Lehmer code: map a permutation of 0..7 to an integer in [0, 8!)."""
    rank = 0
    remaining = list(permutation)
    for i in range(NUM_CORNERS - 1):
        value = remaining[0]
        smaller = sum(1 for other in remaining[1:] if other < value)
        rank = rank * (NUM_CORNERS - i) + smaller
        remaining.pop(0)
    return rank


def permutation_unrank(rank: int) -> np.ndarray:
    available = list(range(NUM_CORNERS))
    permutation = np.empty(NUM_CORNERS, dtype=np.int8)
    for i in range(NUM_CORNERS):
        weight = factorial(NUM_CORNERS - 1 - i)
        choice, rank = divmod(rank, weight)
        permutation[i] = available.pop(choice)
    return permutation


def orientation_rank(orientation) -> int:
    """Base-3 encoding of the first seven twists; the eighth is implied."""
    rank = 0
    for value in orientation[: NUM_CORNERS - 1]:
        rank = rank * 3 + int(value)
    return rank


def orientation_unrank(rank: int) -> np.ndarray:
    orientation = np.empty(NUM_CORNERS, dtype=np.int8)
    for i in range(NUM_CORNERS - 2, -1, -1):
        rank, orientation[i] = divmod(rank, 3)
    orientation[NUM_CORNERS - 1] = (-int(orientation[:-1].sum())) % 3
    return orientation


def encode(permutation, orientation) -> int:
    return permutation_rank(permutation) * NUM_ORIENTATIONS + orientation_rank(
        orientation
    )


_PERMUTATION_WEIGHTS = np.array(
    [factorial(NUM_CORNERS - 1 - i) for i in range(NUM_CORNERS - 1)], dtype=np.int64
)
_ORIENTATION_WEIGHTS = 3 ** np.arange(NUM_CORNERS - 2, -1, -1, dtype=np.int64)
_LATER_SLOT = np.triu(np.ones((NUM_CORNERS, NUM_CORNERS), dtype=bool), k=1)


def encode_batch(permutations: np.ndarray, orientations: np.ndarray) -> np.ndarray:
    """Vectorized `encode` over rows: (batch, 8) arrays -> (batch,) indices."""
    values = permutations.astype(np.int64)
    # [row, i, j] is True when slot j comes after slot i and holds a smaller cubelet.
    smaller_later = (values[:, None, :] < values[:, :, None]) & _LATER_SLOT
    lehmer = smaller_later.sum(axis=2)[:, : NUM_CORNERS - 1]
    permutation_ranks = lehmer @ _PERMUTATION_WEIGHTS
    orientation_ranks = (
        orientations[:, : NUM_CORNERS - 1].astype(np.int64) @ _ORIENTATION_WEIGHTS
    )
    return permutation_ranks * NUM_ORIENTATIONS + orientation_ranks


def decode(index: int) -> tuple[np.ndarray, np.ndarray]:
    perm_rank, ori_rank = divmod(index, NUM_ORIENTATIONS)
    return permutation_unrank(perm_rank), orientation_unrank(ori_rank)


def build_move_tables() -> tuple[np.ndarray, np.ndarray]:
    """Transition tables for the two coordinates, shaped (states, moves).

    Returns (permutation_table, orientation_table). Both are derived from
    MOVE_EFFECT, so both inherit the facelet engine's validated geometry.
    """
    permutation_table = np.empty(
        (NUM_PERMUTATIONS, len(ALL_MOVES)), dtype=np.uint16
    )
    for rank in range(NUM_PERMUTATIONS):
        permutation = permutation_unrank(rank)
        for m, move in enumerate(ALL_MOVES):
            source = MOVE_EFFECT[move][0]
            permutation_table[rank, m] = permutation_rank(permutation[source])

    orientation_table = np.empty(
        (NUM_ORIENTATIONS, len(ALL_MOVES)), dtype=np.uint16
    )
    for rank in range(NUM_ORIENTATIONS):
        orientation = orientation_unrank(rank)
        for m, move in enumerate(ALL_MOVES):
            source, twist = MOVE_EFFECT[move]
            moved = (orientation[source] + twist) % 3
            orientation_table[rank, m] = orientation_rank(moved)

    return permutation_table, orientation_table
