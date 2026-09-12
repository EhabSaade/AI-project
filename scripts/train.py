"""Train the value/policy network with Autodidactic Iteration.

Usage:
    python scripts/train.py --iterations 2000 --max-depth 8

Checkpoints and a CSV log are written to runs/<name>/. The CSV is the raw
material for the training curves in the report, so it is written every
iteration rather than only at the end.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rubiks.adi import generate_scrambles, train_step
from rubiks.network import CubeNet, default_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--sequences", type=int, default=128)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--trunk", type=int, nargs="+", default=[2048, 1024])
    parser.add_argument("--head", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="adi")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    return parser.parse_args()


def main():
    args = parse_args()
    device = default_device()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    run_dir = Path(__file__).resolve().parent.parent / "runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = CubeNet(trunk=tuple(args.trunk), head=args.head).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    parameters = sum(p.numel() for p in model.parameters())

    print(f"device={device}  parameters={parameters:,}")
    print(f"states per iteration = {args.sequences * args.max_depth}")
    print(f"logging to {run_dir}")

    log_path = run_dir / "log.csv"
    with open(log_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["iteration", "loss", "value_loss", "policy_loss", "target_accuracy", "seconds"]
        )

        start = time.time()
        for iteration in range(1, args.iterations + 1):
            states, depths = generate_scrambles(args.sequences, args.max_depth, rng)
            stats = train_step(model, optimizer, states, depths, device)

            writer.writerow(
                [
                    iteration,
                    stats["loss"],
                    stats["value_loss"],
                    stats["policy_loss"],
                    stats["target_accuracy"],
                    round(time.time() - start, 2),
                ]
            )
            handle.flush()

            if iteration % args.log_every == 0 or iteration == 1:
                print(
                    f"iter {iteration:6d} | loss {stats['loss']:8.4f} | "
                    f"value {stats['value_loss']:8.4f} | "
                    f"policy {stats['policy_loss']:7.4f} | "
                    f"agree {stats['target_accuracy']:.3f} | "
                    f"{time.time() - start:6.1f}s",
                    flush=True,
                )

            if iteration % args.checkpoint_every == 0:
                torch.save(
                    {
                        "iteration": iteration,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "args": vars(args),
                    },
                    run_dir / "checkpoint.pt",
                )

    torch.save(
        {
            "iteration": args.iterations,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        run_dir / "checkpoint.pt",
    )
    print(f"done in {time.time() - start:.1f}s -> {run_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
