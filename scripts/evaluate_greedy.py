"""Ground-truth check: can the trained network actually solve cubes?

The training log's `target_accuracy` only says how often the policy head
agrees with the value head -- both can be confidently wrong together. This
script measures against answers the network had no part in producing.

Two measurements, per scramble depth:

1. **Greedy solve rate.** Follow the network's top move repeatedly and see
   whether the cube reaches the solved state within a step budget. This is
   unambiguous: solved or not.

2. **Corner progress.** Whether the network's chosen move reduces the exact
   corner distance from the pattern database. Note this is a one-sided
   signal: corners are a relaxation of the full cube, so a move that is
   optimal overall need not reduce the corner distance. Reducing it is
   evidence of progress; not reducing it is not proof of a mistake.

Scramble depth is an upper bound on true distance, not the distance itself --
random scrambles can cancel (see FINDINGS.md, Step 5).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rubiks.adi import MOVE_PERMUTATIONS, is_solved_batch
from rubiks.cube import ALL_MOVES, solved_state
from rubiks.encoding import encode
from rubiks.network import CubeNet, default_device
from rubiks.pattern_db import lookup

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="adi_d12")
    parser.add_argument("--cubes", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_model(run_name, device):
    checkpoint = torch.load(
        ROOT / "runs" / run_name / "checkpoint.pt", map_location=device
    )
    saved = checkpoint["args"]
    model = CubeNet(trunk=tuple(saved["trunk"]), head=saved["head"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint["iteration"]


def scramble(count, depth, rng):
    states = np.tile(solved_state(), (count, 1))
    for _ in range(depth):
        moves = rng.integers(len(ALL_MOVES), size=count)
        states = np.take_along_axis(states, MOVE_PERMUTATIONS[moves], axis=1)
    return states


@torch.no_grad()
def top_moves(model, states, device):
    logits = model(torch.from_numpy(encode(states)).to(device))[1]
    return logits.argmax(dim=1).cpu().numpy()


@torch.no_grad()
def greedy_rollout(model, states, device, budget):
    """Steps taken to solve each cube, or -1 if the budget ran out."""
    current = states.copy()
    solved_at = np.full(current.shape[0], -1, dtype=np.int64)
    active = np.arange(current.shape[0])

    for step in range(1, budget + 1):
        if active.size == 0:
            break
        moves = top_moves(model, current[active], device)
        current[active] = np.take_along_axis(
            current[active], MOVE_PERMUTATIONS[moves], axis=1
        )
        finished = is_solved_batch(current[active])
        solved_at[active[finished]] = step
        active = active[~finished]

    return solved_at


def main():
    args = parse_args()
    device = default_device()
    model, iteration = load_model(args.run, device)
    pdb = np.load(ROOT / "data" / "corner_pdb.npy")
    rng = np.random.default_rng(args.seed)

    print(f"run={args.run} (iteration {iteration})  device={device}")
    print(f"{args.cubes} cubes per depth, greedy budget {args.budget} moves\n")
    print(f"{'depth':>5} {'solved':>8} {'median len':>11} {'corner progress':>16}")
    print("-" * 44)

    for depth in range(1, args.max_depth + 1):
        states = scramble(args.cubes, depth, rng)

        before = np.array([lookup(pdb, s) for s in states])
        moves = top_moves(model, states, device)
        stepped = np.take_along_axis(states, MOVE_PERMUTATIONS[moves], axis=1)
        after = np.array([lookup(pdb, s) for s in stepped])
        progress = float((after < before).mean())

        solved_at = greedy_rollout(model, states, device, args.budget)
        solve_rate = float((solved_at > 0).mean())
        lengths = solved_at[solved_at > 0]
        median = f"{np.median(lengths):.0f}" if lengths.size else "-"

        print(
            f"{depth:>5} {solve_rate:>7.0%} {median:>11} {progress:>15.0%}"
        )


if __name__ == "__main__":
    main()
