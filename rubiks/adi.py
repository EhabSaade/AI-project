"""Autodidactic Iteration: training the network without a solver or human data.

The difficulty with the cube is that only one state in 4.3e19 carries a
reward, so an agent acting randomly never sees one. ADI's answer is to
generate training states by scrambling *outward from solved*, so that some
are close enough for a one-move lookahead to find the goal, and to bootstrap
everything else off the network's own current estimates.

For each training state, all 18 children are scored with the current network.
The value target is the best child's reward-plus-value, and the policy target
is the move that achieved it. Nothing external labels the data: early on the
estimates are noise, but states next to the goal get a true signal from the
reward, and that signal propagates outward as training proceeds.

Samples are weighted by 1/depth. McAleer et al. report that without this the
training either diverges or collapses to a degenerate solution -- deep states,
whose bootstrapped targets are mostly noise early on, otherwise drown out the
shallow ones that carry real signal.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from rubiks.cube import ALL_MOVES, apply_move, solved_state
from rubiks.encoding import encode

MOVE_PERMUTATIONS = np.stack(
    [apply_move(np.arange(54, dtype=np.int8), move) for move in ALL_MOVES]
).astype(np.int64)

SOLVED_REWARD = 1.0
STEP_REWARD = -1.0


def is_solved_batch(states: np.ndarray) -> np.ndarray:
    """Boolean mask over a batch: is each cube solved?"""
    faces = states.reshape(*states.shape[:-1], 6, 9)
    return (faces == faces[..., :1]).all(axis=-1).all(axis=-1)


def generate_scrambles(
    sequences: int,
    max_depth: int,
    rng: np.random.Generator,
    prefix_sharing: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Scramble outward from solved, keeping every intermediate state.

    Returns (states, depths) with sequences*max_depth entries. Keeping the
    intermediate states is the point: it yields a spread of difficulties in
    one pass, with the shallow end supplying states near enough to the goal
    to be labelled from the reward rather than from a guess.

    `prefix_sharing` is the probability, at each step, that a sequence drops
    its own history and continues from another sequence's current state, so
    that sequences share prefixes and the batch contains repeated states. At 0
    this is exactly independent scrambling, drawing the same random numbers as
    before the option existed. Each depth contributes `sequences` states either
    way.
    """
    states = np.tile(solved_state(), (sequences, 1))
    all_states = np.empty((sequences * max_depth, 54), dtype=np.int8)
    all_depths = np.empty(sequences * max_depth, dtype=np.int64)

    for depth in range(max_depth):
        if prefix_sharing > 0:
            forked = rng.random(sequences) < prefix_sharing
            donors = rng.integers(sequences, size=int(forked.sum()))
            states[forked] = states[donors]
        moves = rng.integers(len(ALL_MOVES), size=sequences)
        states = np.take_along_axis(states, MOVE_PERMUTATIONS[moves], axis=1)
        start = depth * sequences
        all_states[start : start + sequences] = states
        all_depths[start : start + sequences] = depth + 1

    return all_states, all_depths


def children_of(states: np.ndarray) -> np.ndarray:
    """Every state's 18 successors, shaped (batch, 18, 54)."""
    return np.stack(
        [states[:, permutation] for permutation in MOVE_PERMUTATIONS], axis=1
    )


@torch.no_grad()
def compute_targets(
    model: nn.Module, states: np.ndarray, device: torch.device, batch_size: int = 8192
) -> tuple[torch.Tensor, torch.Tensor]:
    """Value and policy targets from a one-move lookahead.

    Solved children are terminal and contribute reward alone; their value is
    not estimated, which keeps the network's own drift out of the one signal
    in the problem that is actually grounded.
    """
    was_training = model.training
    model.eval()

    children = children_of(states)
    batch, num_moves = children.shape[0], children.shape[1]
    flat = children.reshape(batch * num_moves, 54)

    solved = is_solved_batch(flat)
    encoded = torch.from_numpy(encode(flat))

    values = torch.empty(batch * num_moves, dtype=torch.float32, device=device)
    for start in range(0, encoded.shape[0], batch_size):
        chunk = encoded[start : start + batch_size].to(device, non_blocking=True)
        values[start : start + chunk.shape[0]] = model(chunk)[0]

    solved_mask = torch.from_numpy(solved).to(device)
    values = torch.where(solved_mask, torch.zeros_like(values), values)
    rewards = torch.where(
        solved_mask,
        torch.full_like(values, SOLVED_REWARD),
        torch.full_like(values, STEP_REWARD),
    )

    scores = (rewards + values).reshape(batch, num_moves)
    target_values, target_policies = scores.max(dim=1)

    if was_training:
        model.train()
    return target_values, target_policies


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    states: np.ndarray,
    depths: np.ndarray,
    device: torch.device,
) -> dict:
    target_values, target_policies = compute_targets(model, states, device)

    model.train()
    inputs = torch.from_numpy(encode(states)).to(device)
    weights = torch.from_numpy(1.0 / depths.astype(np.float32)).to(device)

    values, policy_logits = model(inputs)
    value_loss = (weights * (values - target_values) ** 2).mean()
    policy_loss = (
        weights
        * nn.functional.cross_entropy(
            policy_logits, target_policies, reduction="none"
        )
    ).mean()
    loss = value_loss + policy_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        accuracy = (policy_logits.argmax(dim=1) == target_policies).float().mean()

    return {
        "loss": loss.item(),
        "value_loss": value_loss.item(),
        "policy_loss": policy_loss.item(),
        "target_accuracy": accuracy.item(),
    }
