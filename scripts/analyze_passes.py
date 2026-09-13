"""Is the hybrid's gain over beam search better pruning, or repeated attempts?

Reads runs/<run>/hybrid_passes_w<width>.csv (from hybrid_passes.py), which records
for every cube whether each individual pass -- one pruned beam search at one move
bound -- solves it, and compares against beam search and the full hybrid on the
same cubes.

A "one-pass hybrid" commits to a single bound in advance, chosen either relative
to the pattern database's lower bound for the cube (start + k) or as a fixed
number of moves (B). If some one-pass hybrid matches the full hybrid, the gain is
pruning within a pass. If every one-pass hybrid falls back towards beam search,
the gain comes from trying several differently pruned passes.

A pass whose bound is below the cube's lower bound prunes every child, so it
fails; those passes were not run and count as unsolved.

Usage:
    python scripts/analyze_passes.py --width 100
"""

import argparse
import csv
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="adi_d12")
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--budget", type=int, default=30)
    return parser.parse_args()


def mcnemar_p(only_first, only_second):
    discordant = only_first + only_second
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, k) for k in range(min(only_first, only_second) + 1))
    return min(1.0, 2 * tail / 2 ** discordant)


def load_eval_lengths(run_dir, solver):
    lengths = {}
    for path in sorted(run_dir.glob(f"eval_{solver}_checkpoint_d*_cubes.csv")):
        with open(path) as handle:
            for row in csv.DictReader(handle):
                lengths[(int(row["depth"]), int(row["cube"]))] = int(row["length"])
    return lengths


def load_eval_ms(run_dir, solver):
    """Mean ms per cube for each depth, from the per-depth summary files."""
    ms = {}
    for path in sorted(run_dir.glob(f"eval_{solver}_checkpoint_d*.csv")):
        if path.name.endswith("_cubes.csv"):
            continue
        with open(path) as handle:
            for row in csv.DictReader(handle):
                ms[int(row["depth"])] = float(row["ms_per_cube"])
    return ms


def load_passes(path):
    starts = {}
    passes = defaultdict(dict)  # key -> {bound: (solved, length, ms)}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            key = (int(row["depth"]), int(row["cube"]))
            starts[key] = int(row["start"])
            passes[key][int(row["bound"])] = (row["solved"] == "1", int(row["length"]), float(row["ms"]))
    return starts, passes


def main():
    args = parse_args()
    run_dir = ROOT / "runs" / args.run
    starts, passes = load_passes(run_dir / f"hybrid_passes_w{args.width}.csv")
    beam = load_eval_lengths(run_dir, f"beam_w{args.width}")
    hybrid = load_eval_lengths(run_dir, f"hybrid_w{args.width}")
    beam_depth_ms = load_eval_ms(run_dir, f"beam_w{args.width}")

    cubes = sorted(starts)
    trivial = {key for key in cubes if starts[key] == 0}
    beam_solved = {key: beam[key] >= 0 for key in cubes}

    # --- the full hybrid, rebuilt from its passes -------------------------------
    full_solved, full_ms, mismatches = {}, {}, 0
    for key in cubes:
        if key in trivial:
            full_solved[key], full_ms[key], length = True, 0.0, 0
        else:
            spent, length = 0.0, -1
            for bound in sorted(passes[key]):
                solved, pass_length, ms = passes[key][bound]
                spent += ms
                if solved:
                    length = pass_length
                    break
            full_solved[key], full_ms[key] = length >= 0, spent
        mismatches += length != hybrid[key]

    print(f"{len(cubes)} cubes, width {args.width}")
    print(f"first successful pass reproduces the evaluated hybrid on "
          f"{len(cubes) - mismatches} of {len(cubes)} cubes\n")

    def one_pass(bound_for):
        solved, ms = {}, {}
        for key in cubes:
            if key in trivial:
                solved[key], ms[key] = True, 0.0
                continue
            entry = passes[key].get(bound_for(starts[key]))
            solved[key], ms[key] = (entry[0], entry[2]) if entry else (False, 0.0)
        return solved, ms

    def summary(solved):
        gained = sum(solved[k] and not beam_solved[k] for k in cubes)
        lost = sum(beam_solved[k] and not solved[k] for k in cubes)
        return sum(solved.values()), gained, lost

    relative = {k: one_pass(lambda s, k=k: min(s + k, args.budget)) for k in range(args.budget)}
    absolute = {b: one_pass(lambda s, b=b: b) for b in range(1, args.budget + 1)}
    best_k = max(relative, key=lambda k: sum(relative[k][0].values()))
    best_b = max(absolute, key=lambda b: sum(absolute[b][0].values()))

    beam_mean_ms = np.mean([beam_depth_ms[depth] for depth, _ in cubes])
    rows = [
        ("beam search (no database)", beam_solved, None),
        ("hybrid, all passes", full_solved, full_ms),
        (f"one pass, bound = start (tightest)", *relative[0]),
        (f"one pass, bound = start + {best_k} (best k)", *relative[best_k]),
        (f"one pass, bound = {best_b} (best fixed)", *absolute[best_b]),
        (f"one pass, bound = 20", *absolute[20]),
        (f"one pass, bound = {args.budget} (loosest)", *absolute[args.budget]),
    ]
    print(f"{'solver':<42} {'solved':>6} {'gained':>7} {'lost':>5} {'McNemar p':>10} {'ms/cube':>8}")
    for name, solved, ms in rows:
        total, gained, lost = summary(solved)
        mean_ms = beam_mean_ms if ms is None else np.mean([ms[k] for k in cubes])
        print(f"{name:<42} {total:>6} {gained:>7} {lost:>5} {mcnemar_p(gained, lost):>10.2e} {mean_ms:>8.1f}")
    print("(gained / lost are relative to beam search, on the same cubes)\n")

    print("one pass at bound = start + k, every k:")
    print("   k " + " ".join(f"{k:>4}" for k in range(0, args.budget, 2)))
    print("solv " + " ".join(f"{sum(relative[k][0].values()):>4}" for k in range(0, args.budget, 2)))
    print("\none pass at a fixed bound B, every B from 8:")
    print("   B " + " ".join(f"{b:>4}" for b in range(8, args.budget + 1, 2)))
    print("solv " + " ".join(f"{sum(absolute[b][0].values()):>4}" for b in range(8, args.budget + 1, 2)))

    gained_cubes = [k for k in cubes if full_solved[k] and not beam_solved[k]]
    print(f"\nthe {len(gained_cubes)} cubes the full hybrid solved and beam search did not:")
    solving_bounds = [sum(entry[0] for entry in passes[k].values()) for k in gained_cubes]
    tried = [len(passes[k]) for k in gained_cubes]
    print(f"  solved by the best one-pass hybrid (start + {best_k}): "
          f"{sum(relative[best_k][0][k] for k in gained_cubes)}")
    print(f"  solved by the loosest pass (bound {args.budget}): "
          f"{sum(passes[k].get(args.budget, (False,))[0] for k in gained_cubes)}")
    print(f"  bounds tried per cube: median {np.median(tried):.0f}")
    print(f"  bounds that solve each cube: median {np.median(solving_bounds):.0f}, "
          f"distribution {dict(sorted(Counter(solving_bounds).items()))}")
    print(f"  share of tried bounds that solve: median "
          f"{np.median([s / t for s, t in zip(solving_bounds, tried)]):.0%}")


if __name__ == "__main__":
    main()
