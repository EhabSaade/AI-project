import numpy as np

from rubiks.bfs2x2 import build_table
from rubiks.cube2x2 import apply_move, is_solved, solved_state

# Published distance distribution for the 2x2x2 cube in the face-turn
# metric with the DBL corner pinned. Matching these exactly is strong
# evidence the engine's move geometry is correct; a full BFS runs to
# depth 11 (3,674,160 states, ~30s) so the test stops early.
KNOWN_DEPTH_COUNTS = [1, 9, 54, 321, 1847, 9992, 50136, 227536]


def test_depth_distribution_matches_published_values():
    _, depth_counts = build_table(max_depth=len(KNOWN_DEPTH_COUNTS) - 1)
    assert depth_counts == KNOWN_DEPTH_COUNTS


def test_stored_moves_step_toward_solved():
    table, _ = build_table(max_depth=5)
    rng = np.random.default_rng(0)
    keys = list(table)
    for key in rng.choice(len(keys), size=200, replace=False):
        state_bytes = keys[key]
        depth, move = table[state_bytes]
        if depth == 0:
            continue
        state = np.frombuffer(state_bytes, dtype=np.int8)
        nearer = apply_move(state, move)
        assert table[nearer.tobytes()][0] == depth - 1


def test_following_optimal_moves_solves_the_cube():
    table, _ = build_table(max_depth=6)
    rng = np.random.default_rng(1)
    keys = list(table)
    for key in rng.choice(len(keys), size=50, replace=False):
        state = np.frombuffer(keys[key], dtype=np.int8).copy()
        depth = table[state.tobytes()][0]
        for _ in range(depth):
            state = apply_move(state, table[state.tobytes()][1])
        assert is_solved(state)
