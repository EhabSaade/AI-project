"""Measure a solver against ground truth: does it actually solve cubes?

Usage:
    python scripts/evaluate.py --solver greedy
    python scripts/evaluate.py --solver beam --width 100
    python scripts/evaluate.py --solver hybrid --width 100
    python scripts/evaluate.py --solver ida --node-limit 200000 --cubes 10
    python scripts/evaluate.py --solver greedy --checkpoint model_020000.pt
    python scripts/evaluate.py --solver ida --min-depth 50 --max-depth 50

The training log's `target_accuracy` only says how often the policy head
agrees with the value head -- both can be confidently wrong together. This
script measures against answers the network had no part in producing: whether
the cube reaches the solved state.

Each depth's cubes are seeded from (--seed, depth), so every solver and every
checkpoint evaluated with the same --seed and --cubes sees exactly the same
cubes. (Changing --cubes changes the cubes.) Two files are written to
runs/<run>/, named for the solver, checkpoint and depth range:

    eval_<solver>[_<checkpoint>]_d<min>-<max>.csv        one row per depth
    eval_<solver>[_<checkpoint>]_d<min>-<max>_cubes.csv  one row per cube

The per-cube file is keyed by (depth, cube), so different solvers' results on
the same cube can be joined -- e.g. to compare a solution with IDA*'s optimal
one.

Reading the output:
- Scramble depth is an upper bound on true distance; random scrambles cancel.
- Median length is over solved cubes only, so read it with the solve rate.
- Corner progress (greedy only) is whether the top move reduces the exact
  corner distance. It is one-sided: corners are a relaxation of the whole
  cube, so a move that is optimal overall need not reduce it.
- Nodes is the mean number of nodes expanded per cube: states whose children
  were generated. For IDA* that is its search calls; for the network solvers,
  the states the network scored, including every failed hybrid pass.
  (Files written before node counting was added leave the column empty for
  the network solvers; --output-subdir reruns them without overwriting.)
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rubiks.adi import MOVE_PERMUTATIONS
from rubiks.cube import ALL_MOVES, solved_state
from rubiks.network import CubeNet, default_device
from rubiks.pattern_db import lookup_batch
from rubiks.solvers import (
    CountingPolicy,
    beam_search,
    greedy,
    hybrid_search,
    ida_star,
    network_policy,
    pdb_heuristic,
)

ROOT = Path(__file__).resolve().parent.parent
NETWORK_SOLVERS = {"greedy", "beam", "hybrid"}
PDB_SOLVERS = {"greedy", "hybrid", "ida"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="adi_d12")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    parser.add_argument("--solver", choices=["greedy", "beam", "hybrid", "ida"], default="greedy")
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--node-limit", type=int, default=200000)
    parser.add_argument("--cubes", type=int, default=50)
    parser.add_argument("--min-depth", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-subdir", type=str, default="",
                        help="write under runs/<run>/<subdir> instead of runs/<run>")
    return parser.parse_args()


def load_model(run_name, checkpoint_name, device):
    checkpoint = torch.load(
        ROOT / "runs" / run_name / checkpoint_name, map_location=device
    )
    saved = checkpoint["args"]
    model = CubeNet(trunk=tuple(saved["trunk"]), head=saved["head"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint["iteration"]


def scramble(count, depth, rng):
    states = np.tile(solved_state(), (count, 1))
    for _ in range(depth):
        moves = rng.integers(len(ALL_MOVES), size=count)
        states = np.take_along_axis(states, MOVE_PERMUTATIONS[moves], axis=1)
    return states


def solver_label(args):
    if args.solver == "greedy":
        return "greedy"
    if args.solver == "ida":
        return f"ida_n{args.node_limit}"
    return f"{args.solver}_w{args.width}"


def solve_all(args, policy, heuristic, states):
    """Solution length (-1 if unsolved) and nodes expanded, per cube."""
    if args.solver == "greedy":
        counted = CountingPolicy(policy)
        lengths = greedy(counted, states, args.budget)
        # Greedy is batched across cubes, so its count is a total: one node per
        # move played, and an unsolved cube plays the whole budget.
        nodes = np.where(lengths >= 0, lengths, args.budget)
        assert nodes.sum() == counted.expanded, "greedy node count does not add up"
        return lengths, nodes

    lengths, nodes = [], []
    for state in states:
        if args.solver == "ida":
            path, expanded = ida_star(heuristic, state, args.budget, args.node_limit)
        else:
            counted = CountingPolicy(policy)
            if args.solver == "beam":
                path = beam_search(counted, state, args.width, args.budget)
            else:
                path = hybrid_search(counted, heuristic, state, args.width, args.budget)
            expanded = counted.expanded
        nodes.append(expanded)
        lengths.append(-1 if path is None else len(path))
    return np.array(lengths), np.array(nodes)


def corner_progress(policy, states, pdb):
    moves = policy(states).argmax(axis=1)
    stepped = np.take_along_axis(states, MOVE_PERMUTATIONS[moves], axis=1)
    return float((lookup_batch(pdb, stepped) < lookup_batch(pdb, states)).mean())


def text(value, fmt):
    return "-" if np.isnan(value) else format(value, fmt)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    args = parse_args()
    device = default_device()

    policy, iteration = None, None
    if args.solver in NETWORK_SOLVERS:
        model, iteration = load_model(args.run, args.checkpoint, device)
        policy = network_policy(model, device)

    pdb = np.load(ROOT / "data" / "corner_pdb.npy") if args.solver in PDB_SOLVERS else None
    heuristic = pdb_heuristic(pdb) if pdb is not None else None

    label = solver_label(args)
    suffix = f"_{Path(args.checkpoint).stem}" if policy is not None else ""
    stem = f"eval_{label}{suffix}_d{args.min_depth}-{args.max_depth}"
    output_dir = ROOT / "runs" / args.run / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    if policy is not None:
        print(f"run={args.run}  checkpoint={args.checkpoint} (iteration {iteration})")
    print(f"solver={label}  device={device}")
    print(f"{args.cubes} cubes per depth, budget {args.budget} moves, seed {args.seed}\n")
    print(
        f"{'depth':>5} {'solved':>8} {'median len':>11} {'corner progress':>16} "
        f"{'nodes':>10} {'ms/cube':>9}"
    )
    print("-" * 65)

    summary_rows, cube_rows = [], []
    for depth in range(args.min_depth, args.max_depth + 1):
        states = scramble(args.cubes, depth, np.random.default_rng([args.seed, depth]))

        start = time.perf_counter()
        lengths, nodes = solve_all(args, policy, heuristic, states)
        ms_per_cube = 1000 * (time.perf_counter() - start) / args.cubes

        solved = lengths >= 0
        solve_rate = float(solved.mean())
        median = float(np.median(lengths[solved])) if solved.any() else float("nan")
        progress = (
            corner_progress(policy, states, pdb) if args.solver == "greedy" else float("nan")
        )
        mean_nodes = float(nodes.mean())

        print(
            f"{depth:>5} {solve_rate:>7.0%} {text(median, '.0f'):>11} "
            f"{text(progress, '.0%'):>16} {text(mean_nodes, ',.0f'):>10} {ms_per_cube:>9.1f}",
            flush=True,
        )
        summary_rows.append([depth, solve_rate, median, progress, mean_nodes, ms_per_cube])
        for cube, length in enumerate(lengths):
            cube_rows.append([depth, cube, int(length), int(nodes[cube])])

    summary_path = output_dir / f"{stem}.csv"
    cubes_path = output_dir / f"{stem}_cubes.csv"
    write_csv(
        summary_path,
        ["depth", "solve_rate", "median_length", "corner_progress", "mean_nodes", "ms_per_cube"],
        summary_rows,
    )
    write_csv(cubes_path, ["depth", "cube", "length", "nodes"], cube_rows)
    print(f"\nwrote {summary_path}\nwrote {cubes_path}")


if __name__ == "__main__":
    main()
