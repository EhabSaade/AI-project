import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rubiks.cube import ALL_MOVES, apply_move, apply_sequence, is_solved, solved_state
from rubiks.network import CubeNet
from rubiks.solvers import (
    CountingPolicy,
    beam_search,
    greedy,
    hybrid_search,
    ida_star,
    network_policy,
    pdb_heuristic,
)

AXIS = {"U": 1, "D": 1, "R": 0, "L": 0, "F": 2, "B": 2}


def inverse(move):
    if move.endswith("'"):
        return move[0]
    if move.endswith("2"):
        return move
    return move + "'"


def axis_alternating_scramble(rng, depth):
    """No two consecutive moves share an axis, so short scrambles cannot cancel."""
    moves = []
    for _ in range(depth):
        choices = [
            m for m in ALL_MOVES if not moves or AXIS[m[0]] != AXIS[moves[-1][0]]
        ]
        moves.append(choices[rng.integers(len(choices))])
    return moves


def path_oracle(moves, distractor=False):
    """A policy that knows the way back along one scramble.

    On the scramble's path it scores the undoing move 0.0 and every other move
    -10; off the path every move scores -10. With `distractor`, one move that
    leads off the path scores 0.1 at every path state, so the top-ranked move
    is always wrong and the second-ranked is always right.
    """
    path = [solved_state()]
    for move in moves:
        path.append(apply_move(path[-1], move))
    on_path = {state.tobytes() for state in path}
    assert len(on_path) == len(path), "scramble revisited a state"

    preferred = {}
    for state, move in zip(path[1:], moves):
        correct = ALL_MOVES.index(inverse(move))
        wrong = None
        if distractor:
            wrong = next(
                i
                for i in range(len(ALL_MOVES))
                if i != correct
                and apply_move(state, ALL_MOVES[i]).tobytes() not in on_path
            )
        preferred[state.tobytes()] = (correct, wrong)

    def policy(states):
        scores = np.full((len(states), len(ALL_MOVES)), -10.0)
        for row, state in enumerate(states):
            entry = preferred.get(state.tobytes())
            if entry is not None:
                correct, wrong = entry
                scores[row, correct] = 0.0
                if wrong is not None:
                    scores[row, wrong] = 0.1
        return scores

    return policy


def path_heuristic(moves):
    """Exact distance for states on the scramble's path; 99 everywhere else."""
    distance = {}
    state = solved_state()
    distance[state.tobytes()] = 0
    for steps, move in enumerate(moves, start=1):
        state = apply_move(state, move)
        distance[state.tobytes()] = steps

    def heuristic(states):
        return np.array([distance.get(s.tobytes(), 99) for s in states], dtype=np.int64)

    return heuristic


def uniform_policy(states):
    return np.zeros((len(states), len(ALL_MOVES)))


def zero_heuristic(states):
    return np.zeros(len(states), dtype=np.int64)


def scrambled(rng, depth):
    moves = axis_alternating_scramble(rng, depth)
    return moves, apply_sequence(solved_state(), moves)


# --- greedy -------------------------------------------------------------------


def test_greedy_follows_an_honest_policy_home():
    rng = np.random.default_rng(0)
    for depth in range(1, 7):
        moves, state = scrambled(rng, depth)
        assert greedy(path_oracle(moves), state[None, :], budget=20).tolist() == [depth]


def test_greedy_reports_zero_moves_for_a_solved_cube():
    assert greedy(uniform_policy, solved_state()[None, :], budget=5).tolist() == [0]


def test_greedy_fails_when_the_top_move_is_always_wrong():
    rng = np.random.default_rng(1)
    for depth in range(1, 7):
        moves, state = scrambled(rng, depth)
        policy = path_oracle(moves, distractor=True)
        assert greedy(policy, state[None, :], budget=20).tolist() == [-1]


def test_greedy_solves_a_batch_independently():
    rng = np.random.default_rng(2)
    scrambles = [scrambled(rng, depth) for depth in (1, 3, 5)]
    oracles = [path_oracle(moves) for moves, _ in scrambles]

    def combined(states):
        return np.maximum.reduce([oracle(states) for oracle in oracles])

    states = np.stack([state for _, state in scrambles])
    assert greedy(combined, states, budget=20).tolist() == [1, 3, 5]


# --- beam search --------------------------------------------------------------


def test_beam_width_one_is_exactly_greedy():
    """The controlled comparison rests on this: width is the only difference."""
    rng = np.random.default_rng(3)
    for distractor in (False, True):
        for depth in range(1, 7):
            moves, state = scrambled(rng, depth)
            policy = path_oracle(moves, distractor)
            greedy_length = int(greedy(policy, state[None, :], budget=15)[0])
            path = beam_search(policy, state, width=1, budget=15)
            beam_length = -1 if path is None else len(path)
            assert beam_length == greedy_length


def test_a_wider_beam_recovers_what_greedy_misses():
    rng = np.random.default_rng(4)
    for depth in range(1, 7):
        moves, state = scrambled(rng, depth)
        policy = path_oracle(moves, distractor=True)

        assert greedy(policy, state[None, :], budget=20).tolist() == [-1]

        path = beam_search(policy, state, width=2, budget=20)
        assert path is not None
        assert len(path) == depth
        assert is_solved(apply_sequence(state, path))


def test_beam_path_actually_solves_the_cube():
    rng = np.random.default_rng(5)
    for depth in range(1, 7):
        moves, state = scrambled(rng, depth)
        path = beam_search(path_oracle(moves), state, width=5, budget=20)
        assert path is not None
        assert len(path) == depth
        assert is_solved(apply_sequence(state, path))


def test_beam_on_a_solved_cube_returns_an_empty_path():
    assert beam_search(uniform_policy, solved_state(), width=3, budget=5) == []


def test_beam_returns_none_when_the_budget_runs_out():
    rng = np.random.default_rng(6)
    _, state = scrambled(rng, 10)
    assert beam_search(uniform_policy, state, width=4, budget=2) is None


# --- hybrid -------------------------------------------------------------------


def test_hybrid_with_a_blind_heuristic_is_exactly_beam_search():
    """The heuristic comparison rests on this: with a heuristic that knows
    nothing, any difference from beam search would be a bug, not a result."""
    rng = np.random.default_rng(8)
    for distractor in (False, True):
        for width in (1, 2, 3):
            for depth in range(1, 6):
                moves, state = scrambled(rng, depth)
                policy = path_oracle(moves, distractor)
                assert hybrid_search(
                    policy, zero_heuristic, state, width, budget=12
                ) == beam_search(policy, state, width, budget=12)


def test_hybrid_prunes_the_branch_the_policy_wrongly_prefers():
    """Same misleading policy that defeats width-1 beam search; the heuristic
    rules out the wrong favourite, so width 1 is enough."""
    rng = np.random.default_rng(9)
    for depth in range(1, 7):
        moves, state = scrambled(rng, depth)
        policy = path_oracle(moves, distractor=True)

        assert beam_search(policy, state, width=1, budget=20) is None

        path = hybrid_search(policy, path_heuristic(moves), state, width=1, budget=20)
        assert path is not None
        assert len(path) == depth
        assert is_solved(apply_sequence(state, path))


def test_hybrid_is_the_first_single_pass_that_succeeds():
    """The pass-by-pass analysis rests on this: the hybrid is nothing more than
    independent pruned beam searches tried in order of increasing bound."""
    rng = np.random.default_rng(13)
    for width in (1, 2):
        for depth in range(2, 7):
            moves, state = scrambled(rng, depth)
            policy = path_oracle(moves, distractor=True)
            exact = path_heuristic(moves)

            def loose(states):
                # Underestimates by two on the path, so the first passes fail.
                return np.maximum(exact(states) - 2, 0)

            start = max(int(loose(state[None, :])[0]), 1)
            first_success = None
            for bound in range(start, 16):
                path = beam_search(policy, state, width, bound, loose, bound)
                if path is not None:
                    first_success = path
                    break

            assert hybrid_search(policy, loose, state, width, budget=15) == first_success


def test_hybrid_on_a_solved_cube_returns_an_empty_path():
    assert hybrid_search(uniform_policy, zero_heuristic, solved_state(), width=2, budget=5) == []


# --- nodes expanded -----------------------------------------------------------


def test_greedy_expands_one_node_per_move_played():
    rng = np.random.default_rng(14)
    for depth in range(1, 7):
        moves, state = scrambled(rng, depth)
        counted = CountingPolicy(path_oracle(moves))
        assert greedy(counted, state[None, :], budget=20).tolist() == [depth]
        assert counted.expanded == depth


def test_beam_expands_every_state_in_the_beam_at_each_step():
    """Width 4, budget 2, no solution: the root, then a full beam of 4."""
    _, state = scrambled(np.random.default_rng(6), 10)
    counted = CountingPolicy(uniform_policy)
    assert beam_search(counted, state, width=4, budget=2) is None
    assert counted.expanded == 1 + 4


def test_nothing_is_expanded_for_a_solved_cube():
    counted = CountingPolicy(uniform_policy)
    beam_search(counted, solved_state(), width=3, budget=5)
    hybrid_search(counted, zero_heuristic, solved_state(), width=3, budget=5)
    assert counted.expanded == 0


def test_hybrid_nodes_are_the_sum_over_its_passes():
    """Failed passes are real work, so they count."""
    rng = np.random.default_rng(15)
    for depth in range(2, 7):
        moves, state = scrambled(rng, depth)
        policy = path_oracle(moves, distractor=True)
        exact = path_heuristic(moves)

        def loose(states):
            return np.maximum(exact(states) - 2, 0)

        per_pass = 0
        start = max(int(loose(state[None, :])[0]), 1)
        for bound in range(start, 16):
            counted = CountingPolicy(policy)
            path = beam_search(counted, state, 2, bound, loose, bound)
            per_pass += counted.expanded
            if path is not None:
                break

        counted = CountingPolicy(policy)
        hybrid_search(counted, loose, state, width=2, budget=15)
        assert counted.expanded == per_pass
        assert counted.expanded > depth  # the early passes failed and still cost


# --- IDA* ---------------------------------------------------------------------


def test_ida_star_on_a_solved_cube():
    assert ida_star(zero_heuristic, solved_state(), budget=5, node_limit=10) == ([], 0)


def test_ida_star_finds_shortest_solutions_by_brute_force():
    rng = np.random.default_rng(10)
    for depth in (1, 2):
        for _ in range(3):
            _, state = scrambled(rng, depth)
            path, _ = ida_star(zero_heuristic, state, budget=depth, node_limit=10**6)
            assert path is not None
            assert len(path) == depth
            assert is_solved(apply_sequence(state, path))


def test_ida_star_move_pruning_keeps_commuting_faces_solvable_in_either_order():
    for moves in (["U", "D"], ["D", "U"], ["R2", "L'"], ["L'", "R2"]):
        state = apply_sequence(solved_state(), moves)
        path, _ = ida_star(zero_heuristic, state, budget=4, node_limit=10**6)
        assert path is not None
        assert len(path) == 2
        assert is_solved(apply_sequence(state, path))


def test_pattern_database_preserves_optimality_and_cuts_work(shallow_db):
    """An admissible heuristic must change the work done, never the answer."""
    heuristic = pdb_heuristic(shallow_db)
    rng = np.random.default_rng(11)
    blind_total = informed_total = 0
    for _ in range(6):
        length = int(rng.integers(1, 5))
        moves = [ALL_MOVES[rng.integers(len(ALL_MOVES))] for _ in range(length)]
        state = apply_sequence(solved_state(), moves)

        blind, blind_nodes = ida_star(zero_heuristic, state, budget=6, node_limit=10**6)
        informed, informed_nodes = ida_star(heuristic, state, budget=6, node_limit=10**6)

        assert len(informed) == len(blind)
        assert is_solved(apply_sequence(state, informed))
        blind_total += blind_nodes
        informed_total += informed_nodes

    assert informed_total < blind_total


def test_ida_star_gives_up_at_the_node_limit():
    rng = np.random.default_rng(12)
    _, state = scrambled(rng, 8)
    path, nodes = ida_star(zero_heuristic, state, budget=20, node_limit=50)
    assert path is None
    assert nodes > 50


# --- network policy -----------------------------------------------------------


def test_network_policy_returns_normalised_log_probabilities():
    model = CubeNet(trunk=(64,), head=32)
    policy = network_policy(model, torch.device("cpu"))
    rng = np.random.default_rng(7)
    states = np.stack([scrambled(rng, depth)[1] for depth in range(1, 9)])

    scores = policy(states)

    assert scores.shape == (8, len(ALL_MOVES))
    assert np.allclose(np.logaddexp.reduce(scores, axis=1), 0.0, atol=1e-4)
