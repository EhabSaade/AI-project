"""Network input encoding: a cube state as a 20 x 24 one-hot vector.

Following McAleer et al., a cube is described by its 20 movable cubelets --
8 corners and 12 edges -- rather than its 54 stickers. Each cubelet is
one-hot over the 24 places it could be: a corner over 8 slots x 3 twists, an
edge over 12 slots x 2 flips. Centres are omitted; they never move. The
result is 20 * 24 = 480 inputs.

Speed matters here. Autodidactic Iteration evaluates all 18 children of
every training state, so encoding sits in the inner loop and cannot be a
Python loop over cubelets. Instead the three (or two) sticker colours at a
slot are packed into a small integer key, and a precomputed table maps that
key straight to (which cubelet, which orientation). Everything then runs as
numpy operations over the whole batch at once.
"""

from __future__ import annotations

import numpy as np

from rubiks._geometry import build_index_geometry
from rubiks.corners import CORNER_STICKERS, NUM_CORNERS
from rubiks.cube import solved_state

NUM_EDGES = 12
NUM_CUBELETS = NUM_CORNERS + NUM_EDGES
PLACES_PER_CUBELET = 24
ENCODED_SIZE = NUM_CUBELETS * PLACES_PER_CUBELET

_GEOM_TO_INDEX = build_index_geometry(3)[1]

# Edge slots: exactly one coordinate is zero.
EDGE_POSITIONS = [
    pos
    for pos in (
        (x, y, z)
        for x in (-1, 0, 1)
        for y in (-1, 0, 1)
        for z in (-1, 0, 1)
    )
    if list(pos).count(0) == 1
]

# Axis priority for ordering a slot's stickers: y, then z, then x. Any fixed
# rule works, as long as it is the same rule at every slot.
_AXIS_PRIORITY = (1, 2, 0)


def _edge_sticker_order(pos):
    stickers = []
    for axis in _AXIS_PRIORITY:
        if pos[axis] != 0:
            normal = [0, 0, 0]
            normal[axis] = pos[axis]
            stickers.append(_GEOM_TO_INDEX[(pos, tuple(normal))])
    return stickers


EDGE_STICKERS = np.array(
    [_edge_sticker_order(pos) for pos in EDGE_POSITIONS], dtype=np.int64
)

_SOLVED = solved_state()


def _build_lookup(sticker_table, num_slots, num_orientations):
    """Map packed sticker colours -> (cubelet id, orientation).

    A cubelet's colours, read in its home slot's canonical order, are its
    signature. Placed elsewhere with orientation o, the same colours appear
    rotated by o, so every (cubelet, orientation) pair gives one key.
    """
    width = sticker_table.shape[1]
    cubelet_of = np.full(6 ** width, -1, dtype=np.int8)
    orientation_of = np.full(6 ** width, -1, dtype=np.int8)

    for cubelet in range(num_slots):
        home = _SOLVED[sticker_table[cubelet]].tolist()
        for orientation in range(num_orientations):
            colors = [home[(j - orientation) % width] for j in range(width)]
            key = 0
            for color in colors:
                key = key * 6 + int(color)
            cubelet_of[key] = cubelet
            orientation_of[key] = orientation

    return cubelet_of, orientation_of


_CORNER_CUBELET, _CORNER_ORIENTATION = _build_lookup(CORNER_STICKERS, NUM_CORNERS, 3)
_EDGE_CUBELET, _EDGE_ORIENTATION = _build_lookup(EDGE_STICKERS, NUM_EDGES, 2)


def _pack(colors):
    """Pack the last axis of colour values into a single base-6 integer."""
    weights = 6 ** np.arange(colors.shape[-1] - 1, -1, -1)
    return (colors * weights).sum(axis=-1)


def extract_corners(states: np.ndarray):
    """Vectorized (cubelet, orientation) per corner slot for a batch."""
    keys = _pack(states[:, CORNER_STICKERS].astype(np.int64))
    return _CORNER_CUBELET[keys], _CORNER_ORIENTATION[keys]


def extract_edges(states: np.ndarray):
    """Vectorized (cubelet, orientation) per edge slot for a batch."""
    keys = _pack(states[:, EDGE_STICKERS].astype(np.int64))
    return _EDGE_CUBELET[keys], _EDGE_ORIENTATION[keys]


def encode(states: np.ndarray) -> np.ndarray:
    """Encode a batch of cube states as (batch, 480) float32.

    Accepts a single (54,) state or a (batch, 54) array.
    """
    states = np.atleast_2d(states)
    batch = states.shape[0]
    out = np.zeros((batch, NUM_CUBELETS, PLACES_PER_CUBELET), dtype=np.float32)
    rows = np.arange(batch)[:, None]

    corner_cubelet, corner_orientation = extract_corners(states)
    corner_slots = np.arange(NUM_CORNERS)[None, :]
    out[rows, corner_cubelet, corner_slots * 3 + corner_orientation] = 1.0

    edge_cubelet, edge_orientation = extract_edges(states)
    edge_slots = np.arange(NUM_EDGES)[None, :]
    out[rows, NUM_CORNERS + edge_cubelet, edge_slots * 2 + edge_orientation] = 1.0

    return out.reshape(batch, ENCODED_SIZE)
