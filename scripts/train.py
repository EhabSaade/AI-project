"""Train the value/policy network with Autodidactic Iteration.

Usage:
    python scripts/train.py --iterations 20000 --max-depth 12 --name adi_d12
    python scripts/train.py --resume --iterations 60000 --name adi_d12

Writes to runs/<name>/:
    log.csv          one row per iteration (appended to on resume)
    checkpoint.pt    latest model + optimizer, for resuming
    model_NNNNNN.pt  model-only snapshots, for comparing training lengths

--iterations is the total to train to, not the number to add. A resumed run
takes its experimental settings (network size, learning rate, batch, depth,
seed) from the checkpoint, so it continues the same experiment; only the
operational options on the command line apply.
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

EXPERIMENT_KEYS = ("trunk", "head", "lr", "sequences", "max_depth", "seed")
LOG_HEADER = ["iteration", "loss", "value_loss", "policy_loss", "target_accuracy", "seconds"]


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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--snapshot-every", type=int, default=10000)
    return parser.parse_args()


def last_logged_seconds(log_path):
    with open(log_path) as handle:
        rows = list(csv.DictReader(handle))
    return float(rows[-1]["seconds"]) if rows else 0.0


def save_snapshot(model, iteration, settings, run_dir):
    torch.save(
        {"iteration": iteration, "model": model.state_dict(), "args": settings},
        run_dir / f"model_{iteration:06d}.pt",
    )


def main():
    args = parse_args()
    device = default_device()
    run_dir = Path(__file__).resolve().parent.parent / "runs" / args.name
    checkpoint_path = run_dir / "checkpoint.pt"
    log_path = run_dir / "log.csv"

    settings = {key: getattr(args, key) for key in EXPERIMENT_KEYS}
    start_iteration = 0
    elapsed_before = 0.0
    checkpoint = None

    if args.resume:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        settings = {key: checkpoint["args"][key] for key in EXPERIMENT_KEYS}
        start_iteration = checkpoint["iteration"]
        elapsed_before = last_logged_seconds(log_path)
        if start_iteration >= args.iterations:
            print(f"already at iteration {start_iteration}; nothing to do")
            return
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

    model = CubeNet(trunk=tuple(settings["trunk"]), head=settings["head"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # checkpoint.pt is about to be overwritten; keep the model we started from.
        if not (run_dir / f"model_{start_iteration:06d}.pt").exists():
            save_snapshot(model, start_iteration, settings, run_dir)

    # Offset by the start point so a resumed run does not replay its opening scrambles.
    torch.manual_seed(settings["seed"] + start_iteration)
    rng = np.random.default_rng(settings["seed"] + start_iteration)

    parameters = sum(p.numel() for p in model.parameters())
    print(f"device={device}  parameters={parameters:,}")
    print(f"settings: {settings}")
    print(f"iterations {start_iteration + 1} -> {args.iterations}")
    print(f"states per iteration = {settings['sequences'] * settings['max_depth']}")
    print(f"logging to {run_dir}")

    def save_checkpoint(iteration):
        torch.save(
            {
                "iteration": iteration,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": settings,
            },
            checkpoint_path,
        )

    with open(log_path, "a" if args.resume else "w", newline="") as handle:
        writer = csv.writer(handle)
        if not args.resume:
            writer.writerow(LOG_HEADER)

        start = time.time()
        for iteration in range(start_iteration + 1, args.iterations + 1):
            states, depths = generate_scrambles(
                settings["sequences"], settings["max_depth"], rng
            )
            stats = train_step(model, optimizer, states, depths, device)
            elapsed = elapsed_before + time.time() - start

            writer.writerow(
                [
                    iteration,
                    stats["loss"],
                    stats["value_loss"],
                    stats["policy_loss"],
                    stats["target_accuracy"],
                    round(elapsed, 2),
                ]
            )
            handle.flush()

            if iteration % args.log_every == 0 or iteration == start_iteration + 1:
                print(
                    f"iter {iteration:6d} | loss {stats['loss']:8.4f} | "
                    f"value {stats['value_loss']:8.4f} | "
                    f"policy {stats['policy_loss']:7.4f} | "
                    f"agree {stats['target_accuracy']:.3f} | "
                    f"{elapsed:7.1f}s",
                    flush=True,
                )

            if iteration % args.checkpoint_every == 0:
                save_checkpoint(iteration)
            if iteration % args.snapshot_every == 0:
                save_snapshot(model, iteration, settings, run_dir)

    save_checkpoint(args.iterations)
    print(f"done in {time.time() - start:.1f}s -> {checkpoint_path}")


if __name__ == "__main__":
    main()
