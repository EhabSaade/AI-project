"""Solvers.

Network-driven solvers take a *policy*: a function from a batch of cube states
to a score for each of the 18 moves (for the trained network, its
log-probabilities). Heuristic-driven solvers take a *heuristic*: a function
from a batch of states to a lower bound on the moves remaining (the corner
pattern database). Both are plain functions, so a search never knows where its
numbers come from, and tests can substitute ones whose answers are known.

The four solvers form the project's comparison:

- `greedy`: play the policy's top move.
- `beam_search`: keep the `width` best partial solutions by cumulative score.
  At width 1 this is exactly greedy.
- `hybrid_search`: beam search that discards anything the heuristic proves
  cannot be finished within a move bound, raising the bound until it
  succeeds. With a heuristic that always returns 0 this is exactly beam search.
- `ida_star`: the classical optimal solver; heuristic only, no network.

Both "exactly" equalities are enforced by tests. They are what let each
comparison isolate a single change: greedy to beam adds search width, and beam
to hybrid adds the heuristic.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from rubiks.adi import MOVE_PERMUTATIONS, is_solved_batch
from rubiks.cube import ALL_MOVES
from rubiks.encoding import encode
from rubiks.pattern_db import lookup_batch

Policy = Callable[[np.ndarray], np.ndarray]
Heuristic = Callable[[np.ndarray], np.ndarray]

NUM_MOVES = len(ALL_MOVES)
FACE_OF_MOVE = np.arange(NUM_MOVES) // 3  # ALL_MOVES is ordered URFDLB x ("", "'", "2")
_NO_FACE = 6


def network_policy(model: torch.nn.Module, device: torch.device, batch_size: int = 4096) -> Policy:
    model.eval()

    @torch.no_grad()
    def policy(states: np.ndarray) -> np.ndarray:
        scores = np.empty((states.shape[0], NUM_MOVES), dtype=np.float32)
        for start in range(0, states.shape[0], batch_size):
            chunk = torch.from_numpy(encode(states[start : start + batch_size])).to(device)
            logits = model(chunk)[1]
            scores[start : start + chunk.shape[0]] = (
                torch.log_softmax(logits, dim=1).cpu().numpy()
            )
        return scores

    return policy


class CountingPolicy:
    """A policy that counts the states it scores.

    Every network-driven solver runs its policy exactly once on each state it
    expands, so `expanded` is the number of nodes expanded: states whose
    children were generated, the same quantity `ida_star` reports. The solvers
    themselves are unchanged by counting.
    """

    def __init__(self, policy: Policy):
        self.policy = policy
        self.expanded = 0

    def __call__(self, states: np.ndarray) -> np.ndarray:
        self.expanded += states.shape[0]
        return self.policy(states)


def pdb_heuristic(distances: np.ndarray) -> Heuristic:
    def heuristic(states: np.ndarray) -> np.ndarray:
        return lookup_batch(distances, states).astype(np.int64)

    return heuristic


def greedy(policy: Policy, states: np.ndarray, budget: int) -> np.ndarray:
    """Moves taken to solve each cube by always playing the top move; -1 if unsolved."""
    current = states.copy()
    lengths = np.full(current.shape[0], -1, dtype=np.int64)
    lengths[is_solved_batch(current)] = 0
    active = np.nonzero(lengths < 0)[0]

    for step in range(1, budget + 1):
        if active.size == 0:
            break
        moves = policy(current[active]).argmax(axis=1)
        current[active] = np.take_along_axis(
            current[active], MOVE_PERMUTATIONS[moves], axis=1
        )
        finished = is_solved_batch(current[active])
        lengths[active[finished]] = step
        active = active[~finished]

    return lengths


def beam_search(
    policy: Policy,
    state: np.ndarray,
    width: int,
    budget: int,
    heuristic: Heuristic | None = None,
    bound: int | None = None,
) -> list[str] | None:
    """A move sequence solving one cube, or None if none is found within budget.

    Children are scored from their parent's policy output, so each step runs
    the policy on at most `width` states rather than `width * 18`.

    Only children that survive into the beam are checked for being solved.
    Checking all 18 would hand beam search a free one-move lookahead that
    greedy does not get, and width 1 would no longer equal greedy.

    With a heuristic, children that provably cannot be solved within `bound`
    moves (moves so far + heuristic > bound) are discarded before ranking. For
    an admissible heuristic this never discards a child on a solution of at
    most `bound` moves.
    """
    if is_solved_batch(state[None, :])[0]:
        return []

    limit = budget if bound is None else bound
    beam = state[None, :]
    scores = np.zeros(1)
    history = []

    for depth in range(1, budget + 1):
        children = beam[:, MOVE_PERMUTATIONS].reshape(-1, state.shape[0])
        child_scores = (scores[:, None] + policy(beam)).ravel()

        candidates = np.arange(children.shape[0])
        if heuristic is not None:
            candidates = np.nonzero(depth + heuristic(children) <= limit)[0]
            if candidates.size == 0:
                return None

        order = candidates[np.argsort(-child_scores[candidates], kind="stable")]
        keep = order[_first_distinct(children[order])[:width]]

        history.append((keep // NUM_MOVES, keep % NUM_MOVES))
        beam = children[keep]
        scores = child_scores[keep]

        solved = is_solved_batch(beam)
        if solved.any():
            return _trace_back(history, int(np.argmax(solved)))

    return None


def hybrid_search(
    policy: Policy, heuristic: Heuristic, state: np.ndarray, width: int, budget: int
) -> list[str] | None:
    """Beam search under a move bound that starts at the heuristic's lower bound.

    Each pass allows `bound` moves and discards children the heuristic proves
    cannot be finished in time; a failed pass raises the bound by one. The
    policy decides which surviving children to keep, the heuristic decides
    which children may be considered at all.
    """
    if is_solved_batch(state[None, :])[0]:
        return []

    start = max(int(heuristic(state[None, :])[0]), 1)
    for bound in range(start, budget + 1):
        path = beam_search(policy, state, width, bound, heuristic, bound)
        if path is not None:
            return path
    return None


class _NodeLimitReached(Exception):
    pass


def _moves_after(last_face: int) -> np.ndarray:
    """Moves worth trying after a turn of `last_face`.

    A second turn of the same face merges with the first, and two opposite
    faces commute, so only one order of them is generated. Neither rule
    removes any shortest solution.
    """
    allowed = []
    for move in range(NUM_MOVES):
        face = FACE_OF_MOVE[move]
        same_face = face == last_face
        opposite_out_of_order = face == (last_face + 3) % 6 and face < last_face
        if last_face == _NO_FACE or not (same_face or opposite_out_of_order):
            allowed.append(move)
    return np.array(allowed)


_MOVES_AFTER_FACE = [_moves_after(face) for face in range(_NO_FACE + 1)]


def ida_star(
    heuristic: Heuristic, state: np.ndarray, budget: int, node_limit: int
) -> tuple[list[str] | None, int]:
    """Iterative-deepening A*: (shortest solution or None, nodes expanded).

    The solution is optimal whenever the heuristic is admissible. Gives up,
    returning None, once the bound passes `budget` or more than `node_limit`
    nodes have been expanded.
    """
    if is_solved_batch(state[None, :])[0]:
        return [], 0

    unbounded = 1 << 30
    path: list[str] = []
    nodes = 0
    bound = int(heuristic(state[None, :])[0])
    next_bound = unbounded

    def search(current: np.ndarray, depth: int, last_face: int) -> bool:
        nonlocal nodes, next_bound
        nodes += 1
        if nodes > node_limit:
            raise _NodeLimitReached

        moves = _MOVES_AFTER_FACE[last_face]
        children = current[MOVE_PERMUTATIONS[moves]]
        solved = is_solved_batch(children)

        if depth + 1 == bound:
            # Last move allowed: only an already-solved child can succeed.
            if solved.any():
                path.append(ALL_MOVES[moves[np.argmax(solved)]])
                return True
            next_bound = min(next_bound, bound + 1)
            return False

        costs = depth + 1 + heuristic(children)
        for i in np.argsort(costs, kind="stable"):
            if costs[i] > bound:
                next_bound = min(next_bound, int(costs[i]))
                continue
            path.append(ALL_MOVES[moves[i]])
            if solved[i] or search(children[i], depth + 1, FACE_OF_MOVE[moves[i]]):
                return True
            path.pop()
        return False

    while bound <= budget:
        next_bound = unbounded
        try:
            if search(state, 0, _NO_FACE):
                return path, nodes
        except _NodeLimitReached:
            return None, nodes
        if next_bound == unbounded:
            return None, nodes
        bound = next_bound

    return None, nodes


def _first_distinct(states: np.ndarray) -> np.ndarray:
    """Row indices of each distinct state's first occurrence, in original order."""
    rows = np.ascontiguousarray(states)
    keys = rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1])))
    _, first = np.unique(keys, return_index=True)
    return np.sort(first)


def _trace_back(history, index: int) -> list[str]:
    moves = []
    for parents, moves_taken in reversed(history):
        moves.append(ALL_MOVES[moves_taken[index]])
        index = parents[index]
    return moves[::-1]
