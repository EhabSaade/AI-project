"""Does sharing scramble prefixes make a cache worthwhile?

The proposal asks whether correlated (shared-prefix) scramble generation makes
memoization useful. One kind of cache would be stale: ADI's training targets
depend on the network's weights, which change every iteration, so a target
remembered from an earlier iteration is out of date.

Another kind is exact. Within one iteration the weights are fixed, so a child
state that appears several times in a batch only needs evaluating once. Because
skipping a duplicate evaluation cannot change any target, its saving can be
measured without training at all -- which is what this script does.

For each prefix-sharing rate, averaged over batches generated with the same
settings as training:

- distinct children: the share of target evaluations a within-iteration cache
  would still have to run; the rest are saved
- distinct states: the cost side -- a repeated training state carries no new
  information, so sharing makes each batch less varied
- dedupe time: what finding the duplicates itself costs per batch
- distinct states by depth: where in the batch the repeats occur

Not measured: whether less varied batches make the network learn worse. That
needs paired training runs.

Usage:
    python scripts/cache_study.py
    python scripts/cache_study.py --rates 0 0.5 0.9 --batches 50
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rubiks.adi import children_of, generate_scrambles

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 0.9])
    parser.add_argument("--sequences", type=int, default=256)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def distinct_rows(rows: np.ndarray) -> int:
    contiguous = np.ascontiguousarray(rows)
    keys = contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1])))
    return int(np.unique(keys).size)


def main():
    args = parse_args()
    depths_range = range(1, args.max_depth + 1)
    output_dir = ROOT / "runs" / "cache_study"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"{args.batches} batches per rate, {args.sequences} sequences x depth "
        f"{args.max_depth} = {args.sequences * args.max_depth} states, "
        f"{args.sequences * args.max_depth * 18} children per batch\n"
    )
    print(f"{'sharing':>8} {'distinct children':>18} {'evals saved':>12} "
          f"{'distinct states':>16} {'dedupe ms':>10}")
    print("-" * 68)

    summary_rows, depth_rows = [], []
    for rate in args.rates:
        rng = np.random.default_rng(args.seed)
        child_fractions, state_fractions, dedupe_seconds = [], [], []
        per_depth = np.zeros(args.max_depth)

        for _ in range(args.batches):
            states, depths = generate_scrambles(args.sequences, args.max_depth, rng, rate)
            children = children_of(states).reshape(-1, states.shape[1])

            start = time.perf_counter()
            distinct_children = distinct_rows(children)
            dedupe_seconds.append(time.perf_counter() - start)

            child_fractions.append(distinct_children / children.shape[0])
            state_fractions.append(distinct_rows(states) / states.shape[0])
            for index, depth in enumerate(depths_range):
                per_depth[index] += distinct_rows(states[depths == depth]) / args.sequences

        child_fraction = float(np.mean(child_fractions))
        state_fraction = float(np.mean(state_fractions))
        dedupe_ms = 1000 * float(np.mean(dedupe_seconds))
        per_depth /= args.batches

        print(f"{rate:>8.2f} {child_fraction:>17.1%} {1 - child_fraction:>11.1%} "
              f"{state_fraction:>15.1%} {dedupe_ms:>10.1f}")
        summary_rows.append([rate, child_fraction, 1 - child_fraction, state_fraction, dedupe_ms])
        depth_rows.append([rate, *per_depth.tolist()])

    print("\ndistinct states at each depth (share of sequences)")
    print(f"{'sharing':>8} " + " ".join(f"{d:>5}" for d in depths_range))
    for rate, *fractions in depth_rows:
        print(f"{rate:>8.2f} " + " ".join(f"{f:>5.0%}" for f in fractions))

    with open(output_dir / "summary.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["prefix_sharing", "distinct_children", "evaluations_saved",
                         "distinct_states", "dedupe_ms"])
        writer.writerows(summary_rows)
    with open(output_dir / "by_depth.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["prefix_sharing", *[f"depth_{d}" for d in depths_range]])
        writer.writerows(depth_rows)
    print(f"\nwrote {output_dir / 'summary.csv'}\nwrote {output_dir / 'by_depth.csv'}")


if __name__ == "__main__":
    main()
