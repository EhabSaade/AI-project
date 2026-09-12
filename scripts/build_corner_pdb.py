"""Build and save the 3x3x3 corner pattern database.

Exhaustive BFS over all 88,179,840 corner states (see rubiks/pattern_db.py),
saved as a flat uint8 numpy array of about 88 MB.

The documented maximum corner distance in the face-turn metric is 11, so the
search should terminate there; the script reports what it actually finds.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rubiks.pattern_db import build

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "corner_pdb.npy"


def main():
    start = time.time()

    def report(depth, new_states, total):
        print(
            f"depth {depth:2d}: {new_states:10d} new states | "
            f"total {total:10d} | elapsed {time.time() - start:6.1f}s",
            flush=True,
        )

    distances, depth_counts = build(progress=report)

    print()
    print(f"BFS complete in {time.time() - start:.1f}s")
    print(f"States reached: {int((distances != 255).sum())}")
    print(f"Max depth: {len(depth_counts) - 1}")
    print(f"States per depth: {depth_counts}")

    OUTPUT.parent.mkdir(exist_ok=True)
    np.save(OUTPUT, distances)
    print(f"Saved to {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
