"""Why does the hybrid beat beam search at equal width? Run every pass on its own.

The hybrid tries move bounds h0, h0+1, ..., budget, where h0 is the pattern
database's lower bound for the cube, and stops at the first pass that solves it.
Each pass is an independent pruned beam search (a test asserts the hybrid is
exactly the first successful pass), so running every pass separately records,
for each cube, precisely which bounds solve it.

That separates two explanations of the hybrid's gain over plain beam search:

- better pruning: a single bound chosen in advance already solves the cubes beam
  search misses;
- several attempts: no single bound does, but different bounds solve different
  cubes, and trying them in turn is what helps.

It also measures what a one-pass hybrid would cost, which bears on the
equal-time comparison.

Usage:
    python scripts/hybrid_passes.py --width 100

Writes runs/<run>/hybrid_passes_w<width>.csv with one row per (cube, bound):
depth, cube, start (the first bound tried), bound, solved, length, ms. A cube
that is already solved gets a single row with bound 0.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent))
sys.path.insert(0, str(SCRIPTS))

from evaluate import load_model, scramble
from rubiks.adi import is_solved_batch
from rubiks.network import default_device
from rubiks.solvers import beam_search, network_policy, pdb_heuristic

ROOT = SCRIPTS.parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="adi_d12")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--cubes", type=int, default=50)
    parser.add_argument("--depths", type=int, nargs="+", default=list(range(1, 21)) + [50])
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = default_device()
    model, iteration = load_model(args.run, args.checkpoint, device)
    policy = network_policy(model, device)
    heuristic = pdb_heuristic(np.load(ROOT / "data" / "corner_pdb.npy"))

    output = ROOT / "runs" / args.run / f"hybrid_passes_w{args.width}.csv"
    print(f"checkpoint={args.checkpoint} (iteration {iteration})  width={args.width}  device={device}")
    print(f"{args.cubes} cubes per depth, budget {args.budget}, seed {args.seed} -> {output}\n")
    print(f"{'depth':>5} {'solved by hybrid':>17} {'passes run':>11} {'seconds':>8}")

    with open(output, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["depth", "cube", "start", "bound", "solved", "length", "ms"])

        for depth in args.depths:
            states = scramble(args.cubes, depth, np.random.default_rng([args.seed, depth]))
            depth_start = time.perf_counter()
            hybrid_solved = passes = 0

            for cube, state in enumerate(states):
                if is_solved_batch(state[None, :])[0]:
                    writer.writerow([depth, cube, 0, 0, 1, 0, 0.0])
                    hybrid_solved += 1
                    continue

                start = max(int(heuristic(state[None, :])[0]), 1)
                solved_any = False
                for bound in range(start, args.budget + 1):
                    began = time.perf_counter()
                    path = beam_search(policy, state, args.width, bound, heuristic, bound)
                    ms = 1000 * (time.perf_counter() - began)
                    passes += 1
                    solved = path is not None
                    solved_any |= solved
                    writer.writerow([depth, cube, start, bound, int(solved),
                                     len(path) if solved else -1, round(ms, 2)])
                hybrid_solved += solved_any

            handle.flush()
            print(f"{depth:>5} {hybrid_solved:>13}/{args.cubes} {passes:>11} "
                  f"{time.perf_counter() - depth_start:>8.1f}", flush=True)

    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
