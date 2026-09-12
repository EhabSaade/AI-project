"""Build and save the complete 2x2x2 optimal-move table.

Runs an exhaustive BFS outward from solved (see rubiks/bfs2x2.py) and
pickles the resulting state -> (distance, optimal move) table.

Expected output, matching the published results for the pocket cube:
3,674,160 reachable states with a maximum distance of 11 face turns.
"""

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rubiks.bfs2x2 import build_table

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "table_2x2.pkl"


def main():
    start = time.time()

    def report(depth, new_states, total):
        print(
            f"depth {depth:2d}: {new_states:9d} new states | "
            f"total {total:9d} | elapsed {time.time() - start:6.1f}s",
            flush=True,
        )

    table, depth_counts = build_table(progress=report)

    print()
    print(f"BFS complete in {time.time() - start:.1f}s")
    print(f"Total reachable states: {len(table)}")
    print(f"Max depth: {len(depth_counts) - 1}")
    print(f"States per depth: {depth_counts}")

    OUTPUT.parent.mkdir(exist_ok=True)
    with open(OUTPUT, "wb") as f:
        pickle.dump(table, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
