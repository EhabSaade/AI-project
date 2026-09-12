"""Breadth-first search over the 2x2x2 state space.

Only R, U and F are turned. The 2x2x2 has no centre pieces, so it has no
absolute reference frame: with all six faces available, every genuine
configuration shows up 24 times in a facelet representation like ours
(once per whole-cube orientation), giving 8! * 3^7 = 88,179,840 states.
Pinning the DBL corner -- i.e. never turning the three faces that touch
it -- removes that redundancy while still reaching every configuration,
leaving 88,179,840 / 24 = 3,674,160.

Move generation is vectorized across the whole frontier (one numpy
fancy-index per move per level) rather than looping per-state.
"""

from __future__ import annotations

import numpy as np

from rubiks.cube2x2 import apply_move, solved_state

# The three faces that leave the DBL corner untouched; see module docstring.
MOVES = [f + s for f in "RUF" for s in ("", "'", "2")]

_IDENTITY = np.arange(24, dtype=np.int8)
MOVE_PERM = {m: apply_move(_IDENTITY.copy(), m) for m in MOVES}


def inverse_move(move: str) -> str:
    if move.endswith("'"):
        return move[0]
    if move.endswith("2"):
        return move
    return move + "'"


INVERSE = {m: inverse_move(m) for m in MOVES}


def build_table(max_depth: int | None = None, progress=None):
    """BFS outward from solved.

    Returns (table, depth_counts) where table maps a state's raw bytes to
    (depth, optimal move to play from that state toward solved) and
    depth_counts[d] is the number of states at distance d.

    The stored move is the inverse of the one that discovered the state,
    since BFS explores outward from solved: undoing that move steps back
    toward the solved state. Where several optimal moves exist, whichever
    is found first is kept -- any of them is a valid optimal label.
    """
    start = solved_state()
    table = {start.tobytes(): (0, None)}
    frontier = start.reshape(1, 24)
    depth_counts = [1]
    depth = 0

    while max_depth is None or depth < max_depth:
        depth += 1
        new_rows = []
        for move in MOVES:
            neighbors = frontier[:, MOVE_PERM[move]]
            inverse = INVERSE[move]
            for row in neighbors:
                key = row.tobytes()
                if key not in table:
                    table[key] = (depth, inverse)
                    new_rows.append(row)

        if not new_rows:
            break

        depth_counts.append(len(new_rows))
        frontier = np.stack(new_rows)
        if progress is not None:
            progress(depth, len(new_rows), len(table))

    return table, depth_counts
