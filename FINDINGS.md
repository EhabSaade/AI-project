# Running Findings Log

Working notes recorded as each step completes, to be drawn on when writing
the final report. Each entry records what was built, what was measured, and
what it implies for later steps. Failures and surprises are recorded too --
they are often the most useful material for the report.

---

## Step 1: Cube engine (3x3x3)

**Built:** `rubiks/cube.py` -- a facelet-level engine. The cube is a flat
54-element array (6 faces x 9 stickers), each sticker holding the index of
the face it started on when solved. All 18 moves are supported (U, U', U2
and so on for each of the six faces).

**Design decision -- deriving moves instead of tabulating them.** The usual
way to implement cube moves is to hand-write, for each face turn, which
sticker index moves to which other index. That table is long, and a single
transposed or sign-flipped entry produces a cube that behaves *almost*
correctly -- correct enough to pass casual inspection, wrong enough to
invalidate every result built on top of it.

Instead, each sticker is assigned a 3D position and an outward-facing normal
vector on a cube centred at the origin. A face turn applies a real 90-degree
rotation matrix to every sticker in the turning layer, and the permutation is
recovered by matching each rotated (position, normal) pair back to a sticker
index. The geometry is stated once; the permutations are derived. Adding the
2x2x2 cube later required no new move logic at all (see Step 2), which is
itself evidence the abstraction was the right one.

**Validation:** 41 tests, all passing. The tests deliberately check
properties that hold for a real Rubik's cube rather than merely checking the
code against itself:

- every face turn has order 4 (four applications return to solved)
- a move followed by its inverse is the identity
- `X2` equals `X` applied twice
- a move never disturbs the opposite face
- a move never moves the turned face's centre sticker
- an arbitrary scramble followed by its exact inverse returns to solved
- `(R U R' U')` returns to solved after exactly 6 repetitions, and not before

The last of these is the strongest single check: the "sexy move" having
order 6 is a well-known property of the cube group, and an engine with a
geometry error would be very unlikely to reproduce it.

---

## Step 2: Cube engine (2x2x2)

**Built:** `rubiks/cube2x2.py` (24 stickers, 6 faces x 4) and
`rubiks/_geometry.py`, which now holds the shared rotation math for both cube
sizes, parameterised by n.

**Note for the report:** the 2x2x2 engine required no new move logic -- only
a change of grid size. Both cubes share one implementation of the part most
likely to contain a subtle bug, so validating one validates the other.

---

## Step 3: Exhaustive BFS over the 2x2x2 state space

**Built:** `rubiks/bfs2x2.py` (reusable BFS) and `scripts/build_2x2_table.py`
(CLI wrapper). Breadth-first search outward from the solved state, recording
for every reachable state its exact distance to solved and an optimal move to
play from it. Move generation is vectorised across the whole frontier (one
numpy fancy-index per move per level) rather than looping state by state.

### Finding 3a: the 2x2x2 has no absolute reference frame

The first run used all 18 moves and did not terminate -- it passed 27.5
million states at depth 8 and exhausted memory. The reachable count was
heading for 8! x 3^7 = 88,179,840, not the expected 3,674,160.

The cause is that the 2x2x2 has no centre pieces, so it has no fixed frame of
reference. Our representation keeps the six face *labels* fixed in space,
which means every genuine configuration appears 24 times in the state space,
once per whole-cube orientation:

    88,179,840 / 24 = 3,674,160

The standard remedy, adopted here, is to pin one corner (DBL) by never
turning the three faces that touch it, leaving only R, U and F. Those three
faces still reach every configuration of the puzzle, so nothing is lost.

**Why this belongs in the report:** it is a concrete example of the
difference between a puzzle's abstract state space and a particular chosen
encoding of it. The published figure of 3.67 million counts configurations up
to whole-cube rotation; a naive fixed-frame implementation counts something
24 times larger, and only fails at run time.

### Finding 3b: dict-based state storage does not scale, and this constrains Step 4

The failed run died with a `MemoryError` at roughly 27.5 million entries, in
a Python dictionary keyed on raw state bytes. Each entry costs on the order
of 150+ bytes once key, value tuple and dictionary overhead are counted.

**Direct consequence for the next step:** the planned corner pattern database
for the 3x3x3 cube has 88,179,840 entries -- almost exactly the scale that
just failed. It therefore cannot be a dictionary. It must be a flat numpy
`uint8` array addressed by a computed rank (permutation and orientation
mapped to an integer index), which is about 88 MB rather than many gigabytes.
This was already the intended design; the failed run turned it from a
preference into a demonstrated requirement, which is worth stating in the
report as a measured result rather than an assumption.

### Finding 3c: the engine reproduces published results exactly

With the R/U/F move set the search completed in 32 seconds:

| Quantity | This implementation | Published value for the pocket cube |
|---|---|---|
| Reachable states | 3,674,160 | 3,674,160 |
| Maximum distance (face-turn metric) | 11 | 11 |

Distance distribution found:

| Depth | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| States | 1 | 9 | 54 | 321 | 1,847 | 9,992 | 50,136 | 227,536 | 870,072 | 1,887,748 | 623,800 | 2,644 |

These sum to exactly 3,674,160. The distribution has the characteristic
shape reported for this puzzle: a peak at depth 9 holding over half of all
states, then a steep fall to only 2,644 states at the maximum distance.

**Why this matters.** The Step 1 tests check the engine against itself: they
would still pass if the engine simulated a subtly different puzzle. This
check compares it against an external, independently established result. A
geometry error -- a face turning the wrong way, a sticker mapped to the wrong
slot -- would produce a different state count and a different distribution.
Reproducing all of these figures simultaneously is therefore strong evidence
that the engine is geometrically correct, and by extension so is the shared
rotation code the 3x3x3 engine uses.

*(For the final report: cite a source for the published pocket-cube figures
rather than relying on this log. The 3,674,160 state count and God's number
of 11 are both widely documented; confirm the per-depth table against the
same source.)*

**Artifacts:** `data/table_2x2.pkl`, 116 MB, gitignored (regenerate with
`python scripts/build_2x2_table.py`, ~32s).

**Regression tests added:** the published depth distribution is now asserted
up to depth 7 (a full run is too slow for a test suite); every stored move is
checked to step exactly one closer to solved; and greedily following the
stored moves is checked to solve the cube. Suite now at 78 tests.

---

## Step 4: Corner pattern database (3x3x3)

**Built:** `rubiks/corners.py` (corner coordinates), `rubiks/pattern_db.py`
(the search), `scripts/build_corner_pdb.py` (CLI wrapper). The database gives
the exact number of moves needed to solve the eight corner cubelets of any
cube state. Because solving the whole cube requires solving its corners, that
number is a lower bound on the full solution length -- an admissible
heuristic, suitable for pruning a search without losing optimality.

### Design: two coordinates instead of one table

A corner state is (permutation, orientation): which cubelet sits in each of
the eight slots, and how each is twisted. Twists satisfy sum == 0 (mod 3), so
the eighth is implied by the other seven:

    8! * 3^7 = 40,320 * 2,187 = 88,179,840 states

The two coordinates move independently -- a face turn's effect on the
orientation vector depends only on the orientation vector, not on which
cubelet is where, and likewise for permutation. So rather than one
88-million-entry transition table, there are two small ones (40,320 x 18 and
2,187 x 18) that combine. This is what makes a vectorised BFS possible: each
level applies one move to the entire frontier in a single numpy operation
instead of stepping through ~1.6 billion transitions in Python.

As in Step 1, nothing is hand-tabulated. Each move's slot permutation and
twist are read off the validated facelet engine by applying that move to a
solved cube and inspecting the result.

### Finding 4a: two bugs the tests caught before anything was built on them

Writing the tests before the database was worthwhile: both bugs would have
produced a plausible-looking but wrong heuristic rather than an error.

1. **Twist convention.** Corner twist must be measured in a consistent
   rotational direction, and the two chiralities of corner (x*y*z = +1 and
   -1) require opposite cyclic orderings of their three stickers. Using one
   ordering everywhere measured twist backwards at half the corners: an R
   turn reported +1 at all four affected corners (sum 4) instead of two at
   +1 and two at +2 (sum 6). The `sum(orientation) % 3 == 0` test caught
   this immediately, on eight of the eighteen moves.
2. **Rank/unrank mismatch.** The permutation unranking extracted
   factorial-base digits from the wrong end, so it did not invert the
   ranking. Caught by a round-trip test.

**For the report:** the first bug is the more interesting one. It does not
break any self-consistency property -- the engine still simulates a valid
puzzle -- it breaks agreement with the *mathematical* invariant that defines
corner orientation. It was only detectable by testing against that invariant,
not by testing the code against itself.

### Finding 4b: the flat-array design, measured against the dict it replaced

The same search that exhausted memory as a dictionary in Step 3 now runs to
completion. Directly comparable, since the failed Step 3 run was exploring
this same state space (its per-depth counts are identical to this one's):

| | Dict of state bytes (Step 3) | Flat uint8 array (Step 4) |
|---|---|---|
| Reaching 27.5M states | 204.6 s, then `MemoryError` | 14.9 s |
| Completing all 88.2M states | did not complete | 73.6 s |
| Memory | many GB (exhausted) | 88 MB |

### Finding 4c: results, and why completeness is itself a check

| Quantity | This implementation | Expected |
|---|---|---|
| States reached | 88,179,840 | 8! * 3^7 = 88,179,840 |
| Maximum corner distance (face-turn metric) | 11 | 11 |
| Build time | 73.6 s | -- |
| Size on disk | 88 MB | -- |

Distance distribution:

| Depth | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| States | 1 | 18 | 243 | 2,874 | 28,000 | 205,416 | 1,168,516 | 5,402,628 | 20,776,176 | 45,391,616 | 15,139,616 | 64,736 |

Reaching *every* state is a meaningful check in itself, not just a
completion message. It shows the rank/unrank encoding is a bijection onto the
whole index range -- had it collided, fewer distinct indices would have been
reached; had it left gaps, they could never have been filled -- and that the
derived move tables generate the full corner group.

**Artifacts:** `data/corner_pdb.npy`, 88 MB, gitignored (regenerate with
`python scripts/build_corner_pdb.py`, ~74s).

**Tests added:** 6 tests over a truncated database (a full build is too slow
for the suite), covering the properties the search will depend on: the solved
state is at distance 0; every single move from solved gives distance 1;
neighbouring states differ by at most 1; a state reached in k moves is never
recorded as further than k away (the admissibility direction -- overestimating
would make search built on it return wrong answers); and every state at depth
d has a neighbour at depth d-1. Suite now at 128 tests.

---

## Step 5: State encoding, network, and ADI training

**Built:** `rubiks/encoding.py` (network input), `rubiks/network.py` (model),
`rubiks/adi.py` (the training algorithm), `scripts/train.py` (CLI).

**Encoding.** Following McAleer et al., a cube is described by its 20 movable
cubelets rather than its 54 stickers: each is one-hot over the 24 places it
could be (a corner over 8 slots x 3 twists, an edge over 12 slots x 2 flips),
giving 480 inputs. Centres are omitted as they never move. Because ADI scores
all 18 children of every training state, encoding sits in the inner loop, so
it is fully vectorised -- sticker colours at a slot are packed into a small
integer key and a precomputed table maps that key directly to (cubelet,
orientation).

**Network.** A shared trunk feeding a value head and an 18-way policy head.
The default trunk (2048 -> 1024) is deliberately smaller than the paper's
(4096 -> 2048): with one laptop GPU rather than three server GPUs, a smaller
network seeing more states is the better trade. 4,142,611 parameters.

**Training.** Value target is the best child's reward-plus-value; policy
target is the move that achieved it; samples weighted by 1/depth as the paper
requires. Solved children are treated as terminal, contributing their reward
alone rather than an estimated value -- this keeps the network's own drift
out of the only grounded signal in the problem.

### Finding 5a: edge orientation needed no chirality correction

Corners required opposite sticker orderings for the two chiralities (Finding
4a). Edges did not: a single axis-priority ordering satisfied the edge
parity invariant (sum of flips even) on the first attempt. The reason is
structural -- an edge has only two stickers, so there is no cyclic order to
get backwards, only a binary flip. Worth stating in the report as the
explanation for why one case was subtle and the other was not.

### Finding 5b: a failing test that was wrong, not the code

The first version of the terminal-handling test pinned the value head to
+50 and expected the target to be +1. It failed at 49. The code was correct:
with all non-terminal children valued at 50, they legitimately score 49 and
the max rightly prefers them. The test could not distinguish correct from
incorrect behaviour. Rewritten with the value head pinned to -50, the two
outcomes separate cleanly -- +1 if solved children are treated as terminal,
-49 if their estimate is wrongly used -- and it passes.

**For the report:** worth one line in a methodology section. A test that
fails is not automatically evidence of a bug, and a test that passes is only
worth what its ability to discriminate is.

### Finding 5c: throughput, and what compute is actually available

| Measure | Value |
|---|---|
| Iterations per second | ~10 |
| Training states per second | ~10,000 (each scoring 18 children) |
| Time for 150 iterations | 16 s |
| Hardware | one RTX 3060 Laptop GPU, 6 GB |

McAleer et al. report roughly 8 billion cube visits over 44 hours across
three GPUs. Direct comparison needs care, since it is not clear whether a
"visit" counts a training state or a child evaluation, but the order of
magnitude is the point: this project operates at a small fraction of that
budget, which is exactly the compute-efficiency question the proposal asks.

### Finding 5d: a live example of why scramble-inverse labels are wrong

Inspecting a trained network on a random 5-move scramble, `B D2 D R' R2`:

- `D2` then `D` compose to `D'`, and `R'` then `R2` compose to `R`, so the
  scramble is really `B D' R` -- three moves, not five.
- The pattern database independently reports an exact corner distance of 3,
  confirming the cancellation.
- The optimal first move is therefore `R'`, and the network -- after only 150
  iterations, 16 seconds of training -- assigns it probability 0.906.

This is a concrete instance of the labelling problem raised when this project
was first scoped: a naive supervised approach that labels each scrambled
state with the inverse of the move used to generate it would have labelled
this state `R2`, which is simply wrong. It is also why ADI bootstraps targets
from a lookahead instead of from the scramble that produced the state. Good
material for the report, since it demonstrates the argument rather than
asserting it.

### Project structure decision: notebooks as the analysis layer

Considered moving the project into Jupyter notebooks. Decided against moving
the *core*, and added notebooks alongside it instead:

| Layer | Location | Reason |
|---|---|---|
| Engine, coordinates, PDB, encoding, ADI | `rubiks/` | needs the test suite; it has caught five real bugs so far |
| Long jobs (PDB build, training) | `scripts/` | run for minutes to hours; must survive kernel restarts |
| Plots, tables, model inspection, report figures | `notebooks/` | genuinely better suited |

The deciding argument is Finding 4a: the corner-twist bug produced a valid
puzzle that failed no self-consistency check, and was caught only by a test
asserting a mathematical invariant. Notebook cells cannot be tested that way,
and a silently wrong pattern database would corrupt every downstream result
without ever raising an error. No restructuring was needed for this, since
the code was already an importable package; `notebooks/01_results.ipynb`
simply imports it. The package is now also pip-installable (`pip install -e
.`), which permanently fixes the import errors hit earlier when running files
directly.

### Finding 5e: the completed training run

20,000 iterations, 3,072 states each, max scramble depth 12, in 2h 2m on one
RTX 3060 Laptop GPU. Loss fell from 0.997 to 0.088; policy/value agreement
rose from 0.063 (chance for 18 moves is 0.056) to 0.803.

Training never diverged or collapsed, which was the main risk identified for
this step -- McAleer et al. report ADI doing exactly that without 1/depth
sample weighting. Most progress came in the first ~500 iterations, after
which the curve flattened into a slow grind.

**The metric's limitation.** `target_accuracy` compares the policy head's
choice against the target the value head produced moments earlier, so it
measures the two heads agreeing with each other, not correctness. Both can
be wrong together. It is reported in the training curve as a stability
indicator, and should be described that way rather than as a solve rate.

---

## Step 6a: Ground-truth evaluation of the greedy policy

**Built:** `scripts/evaluate_greedy.py`. Follows the network's top move
repeatedly and asks whether the cube actually reaches the solved state --
a measurement the network had no hand in producing. 50 cubes per depth,
30-move budget.

| Scramble depth | Solved | Median length | Corner progress |
|---|---|---|---|
| 1 | 100% | 1 | 100% |
| 2 | 100% | 2 | 94% |
| 3 | 100% | 3 | 100% |
| 4 | 100% | 4 | 88% |
| 5 | 100% | 4 | 78% |
| 6 | 96% | 5 | 82% |
| 7 | 90% | 6 | 62% |
| 8 | 90% | 6 | 64% |
| 9 | 80% | 7 | 58% |
| 10 | 62% | 7 | 46% |
| 11 | 56% | 8 | 48% |
| 12 | 24% | 8 | 38% |

### Finding 6a-i: the policy is optimal at shallow depths

At depths 1-4 the median solution length equals the scramble depth exactly.
The network is not merely reaching a solution, it is taking a shortest path.

At depth 5 the median is 4 -- shorter than the scramble -- which is the move
cancellation of Finding 5d showing up in aggregate rather than in a single
hand-checked example.

### Finding 6a-ii: greedy degrades with depth, as the paper predicts

Solve rate falls from 100% to 24% between depth 5 and depth 12. This matches
the shape McAleer et al. report for their own greedy baseline, which falls
away sharply while their full MCTS solver stays near 100%.

**This is the central justification for the project's hybrid search, now
measured rather than cited.** The argument for adding search on top of the
network is no longer "the paper says greedy is worse"; it is this table, from
this network, on this hardware. Steps 6b and 7 have a concrete baseline to
beat, and the report can present the improvement as a controlled comparison
against it.

### Caveats to carry into the report

- **Median length is conditional on solving.** At depth 12 the median of 8
  describes the 24% of cubes that were solved, not a typical cube. Solve rate
  and solution length must be read together.
- **Scramble depth is an upper bound on true distance**, not the distance
  itself, because random scrambles cancel.
- **Corner progress is a one-sided signal.** Corners are a relaxation of the
  full cube, so a move that is optimal overall need not reduce the corner
  distance. A reduction is evidence of progress; the absence of one is not
  proof of a mistake. It tracks solve rate closely here (100% down to 38%),
  which is a useful consistency check between two independent measures.
- **50 cubes per depth** is a small sample; the final report should rerun
  this at a larger sample size once the fast vectorised lookup exists.

---

## Status

| Step | State |
|---|---|
| 1. 3x3x3 engine | Done, validated |
| 2. 2x2x2 engine | Done, validated |
| 3. 2x2x2 exhaustive BFS + ground-truth table | Done, matches published results |
| 4. Corner pattern database (3x3x3) | Done, complete and validated |
| 5. Encoding, network, ADI training loop | Done; 20k iterations trained |
| 6a. Greedy baseline, measured against ground truth | Done |
| 6b. Beam search baseline | Next |
| 7. Hybrid PDB + network search | Not started |
| 8. Evaluation and write-up | Not started |

Test suite: 166 tests.
