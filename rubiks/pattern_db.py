"""Corner pattern database for the 3x3x3 cube.

An exact, admissible heuristic: for any cube state, the true number of moves
needed to solve its eight corners is a lower bound on the number needed to
solve the whole cube, since every move that helps the rest of the cube also
has to leave the corners solved at the end.

The database is one `uint8` per corner state, indexed by the rank from
`corners.py` -- 88,179,840 entries, about 88 MB. It deliberately is not a
dictionary: an earlier BFS in this project exhausted memory at ~27 million
dictionary entries keyed on state bytes (see FINDINGS.md), and this table is
three times larger again.

Filling it is a breadth-first search over the whole space. That is roughly
1.6 billion transitions, far too many to step through one at a time in
Python, so each level applies one move to the entire frontier at once using
the two small coordinate tables from `corners.py`.
"""

from __future__ import annotations

import numpy as np

from rubiks.corners import (
    NUM_CORNER_STATES,
    NUM_ORIENTATIONS,
    build_move_tables,
    encode,
    extract,
)
from rubiks.cube import ALL_MOVES

UNVISITED = 255


def _apply_move_to_indices(indices, move_index, permutation_table, orientation_table):
    permutation_coord, orientation_coord = np.divmod(indices, NUM_ORIENTATIONS)
    moved_permutation = permutation_table[permutation_coord, move_index].astype(
        np.int32
    )
    moved_orientation = orientation_table[orientation_coord, move_index].astype(
        np.int32
    )
    return moved_permutation * NUM_ORIENTATIONS + moved_orientation


def build(progress=None) -> tuple[np.ndarray, list[int]]:
    """BFS over every corner state. Returns (distances, states_per_depth)."""
    permutation_table, orientation_table = build_move_tables()

    distances = np.full(NUM_CORNER_STATES, UNVISITED, dtype=np.uint8)
    solved = encode(np.arange(8), np.zeros(8, dtype=np.int8))
    distances[solved] = 0

    frontier = np.array([solved], dtype=np.int32)
    depth_counts = [1]
    depth = 0

    while frontier.size:
        depth += 1
        discovered = []
        for move_index in range(len(ALL_MOVES)):
            candidates = _apply_move_to_indices(
                frontier, move_index, permutation_table, orientation_table
            )
            fresh = np.unique(candidates[distances[candidates] == UNVISITED])
            if fresh.size:
                # Marked immediately so later moves this level skip them.
                distances[fresh] = depth
                discovered.append(fresh)

        if not discovered:
            break

        frontier = np.concatenate(discovered)
        depth_counts.append(int(frontier.size))
        if progress is not None:
            progress(depth, frontier.size, int((distances != UNVISITED).sum()))

    return distances, depth_counts


def lookup(distances: np.ndarray, state: np.ndarray) -> int:
    """Corner distance for a full 54-sticker cube state."""
    return int(distances[encode(*extract(state))])
