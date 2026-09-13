import numpy as np
import pytest

from rubiks.corners import NUM_CORNER_STATES, build_move_tables, encode
from rubiks.cube import ALL_MOVES
from rubiks.pattern_db import UNVISITED, _apply_move_to_indices


@pytest.fixture(scope="session")
def shallow_db():
    """Exact corner distances for every state within 5 moves of solved; 255 beyond.

    The full database takes minutes to build, so tests use this truncated one.
    A search bounded to at most 5 moves prunes exactly as it would with the full
    database: any state stored as 255 is at least 6 moves away, which exceeds
    such a bound either way.
    """
    permutation_table, orientation_table = build_move_tables()
    distances = np.full(NUM_CORNER_STATES, UNVISITED, dtype=np.uint8)
    solved = encode(np.arange(8), np.zeros(8, dtype=np.int8))
    distances[solved] = 0
    frontier = np.array([solved], dtype=np.int32)

    for depth in range(1, 6):
        discovered = []
        for move_index in range(len(ALL_MOVES)):
            candidates = _apply_move_to_indices(
                frontier, move_index, permutation_table, orientation_table
            )
            fresh = np.unique(candidates[distances[candidates] == UNVISITED])
            if fresh.size:
                distances[fresh] = depth
                discovered.append(fresh)
        frontier = np.concatenate(discovered)

    return distances
