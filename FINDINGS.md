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

**Built:** `scripts/evaluate_greedy.py`, since folded into
`scripts/evaluate.py --solver greedy`. Follows the network's top move
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

## Training continuation: making the comparison valid

The 20k-iteration network plateaued early, so training was extended to 60k
iterations. Every solver in the final comparison must then be evaluated on
the same network, otherwise differences between solvers could come from
differences between networks.

**Built:** `scripts/train.py --resume`.

- A resumed run takes its *experimental* settings (network size, learning
  rate, batch, depth, seed) from the checkpoint, so it continues the same
  experiment. Only operational options (logging, checkpoint frequency, total
  iterations) come from the command line.
- The log is appended to, so the training curve stays one continuous series.
- Random seeds are offset by the starting iteration, so the resumed run does
  not replay the scrambles it began with.
- Model-only snapshots are saved every 10k iterations. The starting model is
  saved before `checkpoint.pt` is overwritten, so `model_020000.pt` is exactly
  the network Step 6a measured.

**Verified on the real run, not assumed:** loss resumed at 0.0915 after 0.0882
at iteration 20k (a failed weight load would restart near 1.0), and the
printed settings were the checkpoint's `sequences=256, max_depth=12` rather
than the command-line defaults of 128 and 8.

**Bonus for the report:** the 10k snapshots allow a solve-rate-versus-training
-length figure (20k, 30k, ... 60k) using the same evaluation script and cubes.

### Finding T1 (preliminary, pending 60k): longer training improves real solving, not just loss

Greedy solve rate on the identical 50 cubes per depth (seed 0), 20k versus 40k
iterations:

| Scramble depth | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20k | 96% | 98% | 80% | 72% | 48% | 38% | 30% | 26% | 22% | 14% | 8% | 2% | 4% | 2% | 0% |
| 40k | 98% | 100% | 86% | 84% | 58% | 40% | 42% | 38% | 26% | 24% | 8% | 2% | 4% | 4% | 0% |

Depths 1-5 were 100% for both.

- **Gains concentrate at depths 8-15**, by up to 12 points. Shallow depths were
  already saturated; depths 16 and beyond remain near zero for both.
- **The 40k network is never worse at any depth.** With 50 cubes, a 2-point
  difference is a single cube and could be noise, but the larger gains (depths 9,
  10, 12, 13, 15) all point the same way, on paired cubes.
- **The network solves some cubes beyond its training distribution:** 24-38% at
  depths 13-15, although it was only trained on scrambles of at most 12 moves.

This is also why the training-loss curve alone would have been misleading: loss
moved only from about 0.088 to 0.08 over the same 20k iterations, which looks
like a plateau, while the solve rate at depths 9-15 rose by 4-12 points.

**Note on the Step 6a table:** those numbers used a different cube set (before
per-depth seeding) and are not comparable with this one; for example depth 10
read 62% there and 48% here for the same 20k network. This table supersedes it.

### Finding T1 (final): the gains stop by about 40k iterations

Training finished at 60,000 iterations: 16,484 s of training in total (about 4.6
hours on one RTX 3060 Laptop GPU), final loss 0.078, policy/value agreement about
0.80. The 20k-to-40k half of the extension took about 1.3 hours, as did the
40k-to-60k half.

Greedy solve rate on the identical 50 cubes per depth, for every 10k snapshot
(60k is the final `checkpoint.pt`):

| Depth | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20k | 96% | 98% | 80% | 72% | 48% | 38% | 30% | 26% | 22% | 14% | 8% | 2% | 4% | 2% | 0% |
| 30k | 94% | 98% | 82% | 80% | 58% | 40% | 36% | 32% | 26% | 22% | 10% | 2% | 6% | 4% | 4% |
| 40k | 98% | 100% | 86% | 84% | 58% | 40% | 42% | 38% | 26% | 24% | 8% | 2% | 4% | 4% | 0% |
| 50k | 98% | 98% | 84% | 88% | 54% | 46% | 40% | 32% | 30% | 26% | 12% | 4% | 12% | 4% | 2% |
| 60k | 96% | 98% | 90% | 90% | 56% | 44% | 38% | 40% | 22% | 22% | 16% | 2% | 4% | 2% | 0% |

Depths 1-5 are 100% at every checkpoint; the 60k network solves 0% at depth 50.

| Iterations | 20k | 30k | 40k | 50k | 60k |
|---|---|---|---|---|---|
| Mean solve rate, depths 8-15 | 41.25% | 47.00% | 49.75% | 50.00% | 50.25% |
| Gain over previous snapshot | - | +5.75 | +2.75 | +0.25 | +0.25 |
| Mean solve rate, depths 1-20 | 52.0% | 54.7% | 55.7% | 56.5% | 56.0% |

- **The preliminary conclusion holds only up to 40k.** From 20k to 40k the mean over
  depths 8-15 rose 8.5 points; from 40k to 60k it rose 0.5, well within noise (each
  band mean pools 400 cubes, a standard error of roughly 2.5 points).
- **Between 40k and 60k the per-depth numbers move up and down by 2-8 points** (1-4
  cubes) with no consistent direction -- noise, not a trend.
- **The band choice does not drive the result.** Depths 8-15 were picked, after
  seeing the data, as the range where no checkpoint sits at 100% or near 0%. The
  mean over all depths 1-20 shows the same flattening.
- **The second 1.3 hours of training bought nothing measurable.** Whatever now limits
  the network is not the iteration count. Plausible limits -- untested here -- are
  the maximum training scramble depth of 12 and the network's size.
- The final comparison uses the 60k checkpoint as planned; because 40k-60k are
  indistinguishable, that choice does not affect its conclusions.

---

## Step 6b: Beam search (implemented, tested, and evaluated -- results in Finding 6b-ii)

**Built:** `rubiks/solvers.py` (greedy, beam search, network policy) and
`scripts/evaluate.py`, which replaces `evaluate_greedy.py` and runs any solver
on any checkpoint.

### Design decisions

- **Policies are plain functions** from a batch of states to 18 move scores.
  The search does not know where scores come from, so tests can substitute a
  policy whose correct answers are known.
- **Beam ranks by cumulative log-probability**, so at width 1 it is exactly
  greedy. That makes width the only variable between the two solvers, and it
  is enforced by a test asserting equal results, not merely claimed.
- **Only children that survive into the beam are checked for being solved.**
  Checking all 18 would give beam search a free one-move lookahead that greedy
  does not get, and width 1 would stop equalling greedy.
- **Children are scored from their parent's policy output**, so each step runs
  the network on at most `width` states rather than `width x 18`.
- **Duplicate states within a step are collapsed**, keeping the best-scoring
  copy, so beam slots are not wasted on the same position.

### Finding 6b-i: a test that isolates why width matters

A test policy knows the way back along a scramble but always ranks one wrong
move just above the correct one. Greedy fails on every scramble from depth 1
to 6; beam search at width 2 solves every one, in exactly the scramble length.
This is the mechanism the real comparison measures -- a correct move ranked
second rather than first -- demonstrated in isolation with a known answer.

### Evaluation changes that affect comparability with Step 6a

- **Each depth's cubes are seeded from (seed, depth)**, so every solver and
  every checkpoint sees identical cubes. As a consequence the cube sets differ
  from the run in Step 6a, whose table is superseded by the rerun on the final
  checkpoint rather than being directly comparable with it.
- **Greedy now reports 0 moves for a scramble that cancels back to solved**
  (for example R then R', about a 1-in-18 chance at depth 2). Previously it
  would first move away from the solved state.
- **Time per cube is recorded**, because search cost is part of what is being
  compared, not only solve rate.

### Batched pattern-database lookup

`corners.encode_batch` and `pattern_db.lookup_batch` compute the corner rank
for a whole batch with a vectorised Lehmer code, instead of a Python loop per
cube. Step 7's search needs this, since it looks up the heuristic for every
node it expands. Tested row-by-row against the validated scalar version and at
the extreme ranks, 0 and 88,179,839.

Measured throughput on a batch of 360,000 child states, on CPU while training
was sharing the machine (so if anything pessimistic):

| Operation | States per second |
|---|---|
| Generating all 18 children | ~11,000,000 |
| `lookup_batch` (vectorised) | ~1,060,000 |
| `lookup` (scalar, one cube at a time) | ~19,600 |

The vectorised lookup is about 54x faster, and agrees with the scalar version
on the real 88M-entry database. Lookup, not move generation, is now the
per-node bottleneck for any search that consults the heuristic.

---

### Finding 6b-ii: beam search on the final network

Final 60k checkpoint, the Step 8 protocol's 50 cubes per depth, budget 30 moves.
Greedy and IDA\* (200,000-node limit) are on the identical cubes.

| Scramble depth | Greedy | Beam, width 100 | Beam, width 1000 | IDA\* |
|---|---|---|---|---|
| 1-5 | 100% | 100% | 100% | 100% |
| 6 | 96% | 100% | 100% | 100% |
| 7 | 98% | 100% | 100% | 100% |
| 8 | 90% | 100% | 100% | 100% |
| 9 | 90% | 100% | 100% | 100% |
| 10 | 56% | 98% | 100% | 98% |
| 11 | 44% | 88% | 96% | 86% |
| 12 | 38% | 86% | 94% | 80% |
| 13 | 40% | 80% | 88% | 60% |
| 14 | 22% | 68% | 80% | 50% |
| 15 | 22% | 48% | 64% | 38% |
| 16 | 16% | 50% | 68% | 36% |
| 17 | 2% | 24% | 34% | 12% |
| 18 | 4% | 30% | 44% | 12% |
| 19 | 2% | 16% | 36% | 8% |
| 20 | 0% | 18% | 32% | 6% |
| 50 | 0% | 0% | 6% | 0% |

Time per cube:

| Scramble depth | Greedy | Beam, width 100 | Beam, width 1000 | IDA\* |
|---|---|---|---|---|
| 10 | 0.4 ms | 10.9 ms | 68 ms | 932 ms |
| 15 | 0.5 ms | 29.7 ms | 204 ms | 7,550 ms |
| 20 | 0.4 ms | 40.9 ms | 299 ms | 9,592 ms |
| 50 | 3.2 ms | 43.8 ms | 340 ms | 9,788 ms |

- **Search width recovers most of what greedy loses.** At depth 12, greedy's 38%
  becomes 86% at width 100 and 94% at width 1000; at depth 20, 0% becomes 18% and
  32%. The mechanism is the one Finding 6b-i isolated with a test policy: the right
  move is often ranked second or lower rather than first.
- **Beam search overtakes IDA\* from about depth 11, far faster.** At depth 15, width
  100 solves 48% in 30 ms per cube against IDA\*'s 38% in 7.6 s -- roughly 250 times
  faster. This holds for IDA\* at its 200,000-node limit with a corners-only database;
  a larger budget or a stronger database would move it.
- **On fully scrambled cubes (depth 50), only width 1000 solves any**: 3 of 50, with
  median solution length 27. That is longer than God's number of 20, so these
  solutions are clearly not optimal -- beam search trades solution quality for reach.
- **Where optimality can be checked, beam search is almost always optimal.** Compared
  with IDA\*'s optimal solution on the same cube, on cubes both finished
  (`notebooks/02_comparison.ipynb`):

  | Solver | Cubes compared | Matched the optimum | Mean extra moves | Worst |
  |---|---|---|---|---|
  | Greedy | 556 | 97% | 0.04 | 3 |
  | Beam, width 100 | 674 | 98% | 0.04 | 8 |
  | Beam, width 1000 | 686 | 99% | 0.05 | 19 |

  No solution was ever shorter than IDA\*'s, which the notebook asserts as a
  consistency check on the whole pipeline. The limit of this measure is that it only
  sees cubes IDA\* could finish -- the easier ones at deep scramble depths. Beam
  search's long solutions (median 27 moves at depth 50) lie outside it. So the fair
  summary is: near-optimal wherever optimality can be verified, unverified beyond.
- **Beam search finishes many cubes IDA\* could not.** Of the cubes IDA\* left
  unsolved at depths 13, 14, 15 and 16 (20, 25, 31 and 32 cubes), width 1000 solved
  14, 17, 13 and 16, and width 100 solved 11, 13, 6 and 9.

**Caveats:** 50 cubes per depth, so differences of a few points are one or two
cubes (depth 17 reading below depth 18 is an example of that noise). IDA\*'s times
were measured while training shared the machine and beam search's afterwards, which
flatters beam search somewhat -- but by nowhere near the two orders of magnitude
separating them.

## Step 7: Hybrid search and PDB-only IDA* (implemented and tested)

**Built:** `hybrid_search`, `ida_star` and `pdb_heuristic` in
`rubiks/solvers.py`; `--solver hybrid` and `--solver ida` in
`scripts/evaluate.py`. The truncated test database moved to
`tests/conftest.py` so it is built once per test run instead of twice.

### Hybrid design

The hybrid is beam search with one addition: any child the heuristic proves
cannot be solved within the current move bound (moves so far + heuristic >
bound) is discarded before ranking. The bound starts at the heuristic's lower
bound for the starting cube and rises by one after each failed pass. The
policy still decides *which* surviving children to keep; the heuristic decides
which children may be considered at all.

- **With a heuristic that always returns 0, the hybrid is exactly beam
  search** (same width and budget). This is enforced by a test comparing the
  returned paths, so any beam-versus-hybrid difference in the results can be
  attributed to the pattern database alone.
- **A test isolates the mechanism.** The misleading policy from Step 6b, which
  defeats width-1 beam search on every scramble, is paired with a heuristic
  that knows the true distance along the scramble. Width-1 hybrid then solves
  every scramble in exactly its length: the heuristic rules out the move the
  policy wrongly prefers, so the second-ranked (correct) move survives.
- **The cost of multiple passes is not hidden:** time per cube covers all
  passes, so it is part of the comparison.

### IDA* design

Iterative-deepening A* with the corner database as heuristic. Optimal whenever
the heuristic is admissible, which Step 4 established.

- **Move pruning:** the same face is never turned twice in a row (the turns
  merge), and of two opposite faces, which commute, only one order is
  generated. Neither rule removes any shortest solution; a test checks that
  `U D`, `D U`, `R2 L'` and `L' R2` are all still solved in two moves.
- **At the last move allowed by the bound, only an already-solved child can
  succeed**, so the heuristic is not consulted there. That level holds most of
  the nodes.
- **A node limit** stops a search that has become impractical, so the
  evaluation can report where it breaks down instead of hanging.
- **A test checks that the database changes the work, never the answer:** on
  random scrambles, IDA* with the database and IDA* with a zero heuristic
  (plain iterative deepening) return solutions of identical length, with far
  fewer nodes in total using the database.

### Finding 7a: IDA* is far stronger at these depths than predicted

> **Superseded in part by Finding 7d.** This 5-cube probe showed 100% through
> depth 12; the 50-cube evaluation shows misses from depth 10 onward. The
> conclusion that scramble depth overstates distance still stands.

The prediction, going in, was that IDA* would become impractical by around
scramble depth 12, based on Korf reporting days per cube for his optimal
solver. Probe: 5 cubes per depth, 200,000 node limit, same cubes as every other
solver.

| Scramble depth | Solved | Median optimal length | Mean nodes | ms / cube |
|---|---|---|---|---|
| 1 | 100% | 1 | 1 | 0.1 |
| 2 | 100% | 2 | 2 | 0.2 |
| 3 | 100% | 2 | 2 | 0.2 |
| 4 | 100% | 3 | 3 | 0.3 |
| 5 | 100% | 5 | 45 | 2.2 |
| 6 | 100% | 6 | 31 | 1.5 |
| 7 | 100% | 5 | 152 | 7.7 |
| 8 | 100% | 7 | 260 | 15.2 |
| 9 | 100% | 9 | 10,343 | 560.2 |
| 10 | 100% | 8 | 4,583 | 256.2 |
| 11 | 100% | 7 | 10,518 | 582.6 |
| 12 | 100% | 8 | 3,404 | 182.0 |

**The prediction was wrong, and the reason matters for the whole evaluation:
scramble depth greatly overstates distance.** A 12-move random scramble is a
median of only 8 moves from solved. Korf's figure is for fully random cubes,
roughly 18 moves out, and search effort grows by roughly the branching factor
(about 13 after move pruning) for every additional move of distance. At 8-9
moves the corner database is an informative enough bound for search to finish
in well under a second.

**Consequences for Step 8:**

1. **Report against true distance, not scramble depth.** IDA* returns optimal
   lengths cheaply in this range, and seeding by (seed, depth) means every
   solver faces the same cubes, so each solver's solution length can be
   compared directly with the optimum on the same cube.
2. **IDA* is not a slow straw man in this range**: it is fast, complete and
   optimal. Any case for the network has to be made on deeper cubes, where
   IDA* stops finishing, or on speed. A deeper probe is running to locate that
   point.
3. **The network was trained on scrambles of at most 12 moves**, which is only
   about 8 moves of true distance. Evaluating on deeper cubes also tests
   whether it generalises beyond the distances it was trained on.

**Caveats:** 5 cubes per depth is a small sample; node counts are heavy-tailed
(depth 9 averaged more nodes than depths 10 and 12), so they will not rise
smoothly with depth at this sample size; times were measured while training
was sharing the CPU.

### Finding 7b: where IDA* breaks down

> **Superseded in part by Finding 7d.** The 50-cube evaluation shows the breakdown
> is gradual from depth 10, not a start at depth 13, and that search effort grows
> about 3.9x per extra move over these distances, not about 13x.

A second probe extended the same settings to scramble depth 20.

| Scramble depth | Solved | Median optimal length (solved only) | Mean nodes | ms / cube |
|---|---|---|---|---|
| 13 | 60% | 8 | 100,451 | 5,501 |
| 14 | 80% | 10 | 85,693 | 4,645 |
| 15 | 40% | 10 | 160,185 | 9,161 |
| 16 | 20% | 9 | 160,279 | 8,330 |
| 17 | 40% | 8 | 124,051 | 6,305 |
| 18 | 20% | 8 | 160,212 | 7,971 |
| 19 | 40% | 10 | 131,715 | 6,638 |
| 20 | 0% | - | 200,001 | 9,869 |

**Reproducibility check, for free.** Depths 1-12 of this second probe matched
the first probe exactly -- every solve rate, median and node count. Seeding
each depth separately means rerunning a depth really does rerun the same cubes.

**The breakdown starts at scramble depth 13**, one move beyond the deepest
scrambles the network was trained on, and reaches 0% by depth 20.

**What IDA* does solve there is survivorship.** The cubes finished at depths
13-19 have median optimal lengths of 8-10 moves: these are the scrambles that
happened to land close to solved. Under this node limit, IDA* with the corner
database reaches roughly 10 moves of true distance, and the unsolved cubes are
presumably further than that.

**A larger node limit would buy little.** Search effort grows by roughly the
branching factor, about 13 after move pruning, for every additional move of
distance, so ten times the nodes buys less than one extra move of reach. Step 8
should measure this growth from per-cube node counts against optimal length,
rather than leave it as an estimate.

**Why this is the interesting region.** Depth 13 and beyond is both where the
classical optimal solver stops finishing and where the network is being tested
outside the distances it trained on. It is the region in which the question
"does the network add anything a pattern database cannot?" has a non-trivial
answer.

### Finding 7c: twenty random moves do not produce a random cube

God's number -- 20 moves in the face-turn metric -- bounds *distance*: no
position is more than 20 moves from solved. It does not follow that a 20-move
random scramble produces a typical position, because random moves cancel and
repeat. Most positions of the cube lie around 17-18 moves from solved, while
Finding 7a showed a 12-move scramble lands a median of only 8 away.

To check how many random moves a scramble needs, corners were used as a
measurable proxy. The corner database gives the exact distribution of corner
distances over all 88,179,840 corner positions (mean 8.76). For 2,000 scrambles
at each depth, the corner distances were compared with that exact distribution
using total variation (TV) distance: 0 means indistinguishable, 1 means entirely
different. As a noise floor, 2,000 genuinely random corner positions score about
0.014 against the exact distribution.

| Scramble depth | 5 | 10 | 12 | 15 | 18 | 20 | 25 | 30 | 40 | 50 | 75 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mean corner distance | 3.74 | 5.94 | 6.66 | 7.48 | 7.98 | 8.22 | 8.55 | 8.67 | 8.72 | 8.79 | 8.78 | 8.79 |
| TV from random | 0.997 | 0.767 | 0.597 | 0.400 | 0.253 | 0.198 | 0.082 | 0.040 | 0.019 | 0.025 | 0.029 | 0.021 |

- **At depth 20 scrambles are still clearly not random** (TV 0.198, about 14
  times the noise floor).
- **From about depth 40, corners are indistinguishable from random** within
  sampling noise; depths 50, 75 and 100 are no closer.
- **Caveat:** this measures corners only. Random corners are necessary but not
  sufficient for a random cube -- edges could mix more slowly -- so 40 is a lower
  bound on the depth needed, not the depth itself.

This is why the evaluation includes depth 50, and it is the same concern that
led McAleer et al. to scramble their test cubes 1,000 times.

### Finding 7d: the 50-cube evaluation corrects Findings 7a and 7b

The final IDA* evaluation, run to the Step 8 protocol (50 cubes per depth,
200,000-node limit), supersedes the 5-cube probes.

| Scramble depth | Solved | Median optimal length (solved) | Mean nodes | ms / cube |
|---|---|---|---|---|
| 1 | 100% | 1 | 1 | 0.1 |
| 2 | 100% | 2 | 2 | 0.2 |
| 3 | 100% | 3 | 3 | 0.2 |
| 4 | 100% | 4 | 8 | 0.5 |
| 5 | 100% | 4 | 12 | 1.1 |
| 6 | 100% | 5 | 64 | 3.7 |
| 7 | 100% | 6 | 133 | 7.6 |
| 8 | 100% | 6 | 589 | 32.6 |
| 9 | 100% | 7 | 3,667 | 193.2 |
| 10 | 98% | 8 | 18,574 | 931.9 |
| 11 | 86% | 8 | 47,341 | 2,318.6 |
| 12 | 80% | 9 | 63,203 | 3,142.8 |
| 13 | 60% | 8 | 90,416 | 4,433.3 |
| 14 | 50% | 9 | 109,979 | 5,704.2 |
| 15 | 38% | 9 | 141,748 | 7,550.1 |
| 16 | 36% | 10 | 143,919 | 8,425.4 |
| 17 | 12% | 10 | 178,332 | 9,831.1 |
| 18 | 12% | 9 | 180,480 | 9,242.7 |
| 19 | 8% | 8 | 186,273 | 9,370.8 |
| 20 | 6% | 10 | 191,616 | 9,592.2 |
| 50 | 0% | - | 200,001 | 9,787.6 |

Times were measured while training shared the machine; solve rates, lengths and
node counts are unaffected by that.

**Three corrections:**

1. **IDA\* is not 100% through depth 12** (claimed in 7a). With 50 cubes it first
   misses at depth 10 (98%), then 86% at depth 11 and 80% at depth 12. The 5-cube
   probe happened to draw only cubes it could finish.
2. **The breakdown does not start at depth 13** (claimed in 7b). It is gradual from
   depth 10, falls through 50% at depth 14, and reaches 6% at depth 20 and 0% on
   fully scrambled cubes at depth 50.
3. **Search effort does not grow about 13x per extra move** (estimated, not
   measured, in 7b). Measured from the per-cube results, median nodes by optimal
   solution length over solved cubes:

   | Optimal length | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
   |---|---|---|---|---|---|---|---|---|---|---|---|
   | Cubes | 75 | 56 | 65 | 67 | 84 | 74 | 62 | 74 | 53 | 48 | 31 |
   | Median nodes | 1 | 2 | 3 | 4 | 16 | 50 | 202 | 532 | 5,319 | 33,218 | 82,868 |
   | Ratio to previous | - | 2.0x | 1.5x | 1.3x | 4.0x | 3.2x | 4.0x | 2.6x | 10.0x | 6.2x | 2.5x |

   A log-linear fit over optimal lengths 4-9 gives **about 3.9x per extra move**, so
   ten times the nodes buys about 1.7 moves of reach rather than less than one. The
   13x figure is the brute-force branching factor after move pruning; at these
   distances the pattern database prunes enough that effort grows far more slowly.
   The ratio does climb with distance (10.0x from 8 to 9 moves, 6.2x from 9 to 10),
   and the 11-move ratio is biased low by survivorship -- only the 11-move cubes that
   happened to be cheap finished within the limit -- so growth at larger distances is
   likely steeper than the fit suggests.

**Consistent with Finding 7c:** IDA* still finishes 6% of depth-20 scrambles but 0%
at depth 50, as expected if depth-20 scrambles are closer to solved than random
cubes are.

**Worth stating in the report as a methodological point:** the 5-cube probes
supported two confident conclusions that the 50-cube run overturned, and an estimate
made by reasoning rather than measurement was off by more than a factor of three.
The same care about sample size applies to the classical baseline as to the network.

### Step 8 protocol, fixed from these probes

- **Cubes:** 50 per depth, seed 0, identical for every solver and checkpoint.
- **Depths:** 1-20, plus depth 50 as a stand-in for a fully scrambled cube (see
  Finding 7c: corners are indistinguishable from random from about depth 40,
  but clearly not at depth 20).
- **Network solvers:** evaluated on the final 60k-iteration checkpoint only.
- **Search widths:** beam search and hybrid at widths 100 and 1000. Chosen from a
  cost pilot on the 40k snapshot with 3 of the deepest cubes (outputs discarded,
  since 3 cubes is not the protocol's cube set):

  | Solver | Depth 20 | Depth 50 |
  |---|---|---|
  | beam, width 100 | - | 0%, 0.12 s/cube |
  | beam, width 1000 | - | 0%, 0.5 s/cube |
  | hybrid, width 100 | 0%, 1.2 s/cube | 0%, 1.2 s/cube |
  | hybrid, width 1000 | 2 of 3 solved, 3.9 s/cube | 0%, 9.0 s/cube |

  The hybrid is the expensive one on cubes it cannot solve, because each failed
  pass is followed by another with a longer move bound. Even so, all four settings
  fit in roughly 1.5 hours for the full protocol, so no width had to be dropped.
  The 2-of-3 at depth 20 is anecdotal (IDA\* finishes 6% there) and is only a
  reason to look closely, not a result.
- **IDA\*:** 200,000 node limit (about 10 s per cube at the limit). It does not
  use the network, so it runs while training finishes; its times were measured
  with training sharing the machine, while its solve rates, node counts and
  optimal lengths are unaffected by that.
- **Reporting:** results against IDA\*'s optimal length on the same cube wherever
  IDA\* finished, not only against scramble depth.

---

## Step 8 results: does the pattern database help?

All results on the final 60k network, with the protocol's 50 cubes per depth at
depths 1-20 and 50 (1,050 cubes), budget 30 moves. Every solver faced the same
cubes, so outcomes are compared cube by cube. The tables are printed by
`notebooks/02_comparison.ipynb`; the figures are `figures/search_width.png`,
`figures/pattern_database_effect.png` and `figures/cost_versus_reach.png`.

### Finding 8a: at equal width, the pattern database strictly helps

Beam search and the hybrid at width 100, paired on the same cubes (depths where
they differ; at depths 1-9 both solved every cube):

| Depth | Both solved | Hybrid only | Beam only | Neither |
|---|---|---|---|---|
| 10 | 49 | 1 | 0 | 0 |
| 11 | 44 | 1 | 0 | 5 |
| 12 | 43 | 4 | 0 | 3 |
| 13 | 40 | 4 | 0 | 6 |
| 14 | 34 | 3 | 0 | 13 |
| 15 | 24 | 4 | 0 | 22 |
| 16 | 25 | 6 | 0 | 19 |
| 17 | 12 | 2 | 0 | 36 |
| 18 | 15 | 6 | 0 | 29 |
| 19 | 8 | 3 | 0 | 39 |
| 20 | 9 | 3 | 0 | 38 |
| 50 | 0 | 1 | 0 | 49 |

- **Across all 1,050 cubes the hybrid solved 38 that beam search did not, and beam
  search solved none that the hybrid did not.** On those 38 discordant cubes the
  exact two-sided McNemar test gives p = 2 x 0.5^38, about 7 x 10^-12.
- **It never made a solution worse.** On the 753 cubes both solved, the hybrid's
  solution was shorter on 2% and longer on none (mean difference -0.04 moves).
- **Zero losses is a result, not a guarantee.** Pruning changes which children fill
  the beam, and the hybrid runs several passes under a changing bound, so it could in
  principle miss a solution beam search finds. It did not, on any of 1,050 cubes.
- This is the comparison the whole design was built to make interpretable: a test
  asserts the hybrid equals beam search when the heuristic knows nothing, so this
  difference can be attributed to the pattern database alone.

### Finding 8b: but at equal time, a wider beam does better

Time per cube, beam search versus hybrid at width 100 (both measured in the same
run after training had finished):

| Depth | 5 | 10 | 11 | 14 | 17 | 20 | 50 |
|---|---|---|---|---|---|---|---|
| Beam, width 100 | 4.8 ms | 10.9 ms | 14.2 ms | 21.6 ms | 38.8 ms | 40.9 ms | 43.8 ms |
| Hybrid, width 100 | 3.5 ms | 27.5 ms | 101.4 ms | 261.1 ms | 682.3 ms | 662.8 ms | 756.8 ms |
| Ratio | 0.7x | 2.5x | 7.1x | 12.1x | 17.6x | 16.2x | 17.3x |

Summed over all 21 depths the hybrid costs **12.3 times** as much. The overhead
appears exactly where the benefit does, from depth 10 on: each failed pass restarts
beam search with a bound one move longer, and every child's corner distance is
looked up.

That raises the question the fixed-width comparison cannot answer: does the database
beat spending the same time on search width? Beam search at width 1000 costs about
as much per cube as hybrid width 100 on deep cubes, and less overall (2.63 s against
4.78 s, summing per-cube means over the 21 depths). Paired on the same cubes:

| | Beam, width 1000 | Hybrid, width 100 |
|---|---|---|
| Cubes only this solver solved | **40** | 10 |
| Total time (sum of per-cube means) | **2.63 s** | 4.78 s |
| Shorter solution, on the 781 cubes both solved | **18** | 2 |

- **For less time, the ten-times-wider beam solves significantly more cubes** (exact
  McNemar p = 2.4 x 10^-5) with slightly shorter solutions (mean -0.08 moves). In
  this implementation, compute spent on search width buys more than compute spent on
  the corner pattern database.
- **The two are partly complementary:** the 10 cubes only the hybrid solved show the
  database finds some solutions that width alone does not.

**Conditions on this conclusion, all worth stating in the report:**

1. **The hybrid's cost is mostly its repeated passes**, which restart from the
   heuristic's lower bound and grow the bound one move at a time. A variant that
   starts nearer the final bound or reuses work between passes could be much
   cheaper, and the equal-time comparison could change with it.
2. **The database covers corners only**, the weakest standard pattern database. An
   edge database would prune far more.
3. **Two widths and 50 cubes per depth.** The paired tests are what make 50 cubes
   enough to separate these solvers; they do not make the result general.
4. A few short notebook runs overlapped the hybrid's timing run. Their effect is small
   next to a twelve-fold difference, but the times are not from an idle machine.

### Finding 8c: at width 1000 the database's gain grows, and so does its cost

Beam search against the hybrid at width 1000, paired on the same cubes (at depths 1-9
both solved every cube):

| Depth | Both solved | Hybrid only | Beam only | Neither | Beam ms/cube | Hybrid ms/cube |
|---|---|---|---|---|---|---|
| 10 | 50 | 0 | 0 | 0 | 68.0 | 77.3 |
| 11 | 48 | 2 | 0 | 0 | 96.9 | 139.0 |
| 12 | 47 | 2 | 0 | 1 | 106.5 | 322.4 |
| 13 | 44 | 5 | 0 | 1 | 123.3 | 447.8 |
| 14 | 40 | 5 | 0 | 5 | 158.5 | 1,268.7 |
| 15 | 32 | 7 | 0 | 11 | 203.5 | 2,476.6 |
| 16 | 34 | 9 | 0 | 7 | 202.4 | 2,155.6 |
| 17 | 17 | 11 | 0 | 22 | 281.0 | 4,828.9 |
| 18 | 22 | 14 | 0 | 14 | 269.1 | 3,775.6 |
| 19 | 18 | 12 | 0 | 20 | 299.7 | 4,949.8 |
| 20 | 16 | 14 | 0 | 20 | 298.8 | 5,616.9 |
| 50 | 3 | 10 | 0 | 37 | 339.8 | 7,346.8 |

- **The hybrid solved 91 cubes that beam search did not, and beam search solved none
  that the hybrid did not** (exact McNemar p = 8 x 10^-28). The width-100 result
  (38 to 0) was not a fluke of one setting; the gain more than doubles at the wider
  width.
- **Again no solution got longer:** on the 821 cubes both solved, the hybrid's was
  shorter on 23 and longer on none (mean -0.13 moves).
- **On fully scrambled cubes (depth 50) the hybrid solves 13 of 50 (26%)**, against 3
  for beam search at the same width and 0 for IDA\*. At depth 20: 60%, 32% and 6%.
- **Wherever IDA\* could verify it, the width-1000 hybrid was optimal on every cube**:
  693 of 693, worst case zero extra moves -- against 99% and a worst case of 19 extra
  moves for beam search at the same width. The same survivorship caveat applies: this
  only covers cubes IDA\* finished. The hybrid's deep solutions are not all short; its
  median length is 22 at depth 20 and 25 at depth 50, both above God's number.
- **The cost is 12.7 times beam search's** (33.5 s against 2.6 s, summing per-cube
  means over the 21 depths).
- **Side observation:** at depths 4-9 the hybrid is *faster* than beam search at width
  1000 (depth 6: 5.9 ms against 26.3 ms). The likely reason is that pruning under a
  tight bound leaves fewer surviving children than the beam's width, so fewer network
  evaluations are needed; untested.

### Finding 8d: at the higher budget too, a wider beam does better

Width 10,000 was **not in the original protocol**. It was added after Finding 8c left
the equal-time question open at the higher budget, because beam search at width
10,000 costs about as much per cube as the width-1000 hybrid. Same network, same
1,050 cubes, same 30-move budget.

Beam search at width 10,000 against the hybrid at width 1000, paired on the same
cubes (at depths 1-11 both solved every cube):

| Depth | Both solved | Beam only | Hybrid only | Neither | Beam ms/cube | Hybrid ms/cube |
|---|---|---|---|---|---|---|
| 12 | 48 | 0 | 1 | 1 | 1,021.9 | 322.4 |
| 13 | 49 | 1 | 0 | 0 | 1,008.4 | 447.8 |
| 14 | 45 | 3 | 0 | 2 | 1,372.3 | 1,268.7 |
| 15 | 38 | 6 | 1 | 5 | 1,892.5 | 2,476.6 |
| 16 | 41 | 2 | 2 | 5 | 1,808.0 | 2,155.6 |
| 17 | 23 | 10 | 5 | 12 | 2,807.3 | 4,828.9 |
| 18 | 33 | 5 | 3 | 9 | 2,487.2 | 3,775.6 |
| 19 | 28 | 14 | 2 | 6 | 2,689.7 | 4,949.8 |
| 20 | 25 | 8 | 5 | 12 | 3,065.2 | 5,616.9 |
| 50 | 11 | 15 | 2 | 22 | 3,919.9 | 7,346.8 |

- **The wider beam solved 64 cubes the hybrid did not; the hybrid solved 21 the wider
  beam did not** (exact McNemar p = 3.3 x 10^-6) -- in less total time (25.1 s against
  33.5 s, summing per-cube means over the 21 depths).
- **The gap is largest on fully scrambled cubes:** 26 of 50 (52%) against 13 (26%).
- **Solutions were slightly shorter, not longer:** on the 891 cubes both solved, the
  wider beam's solution was shorter on 41 and longer on 14 (0.15 moves shorter on
  average). Where IDA\* could check, it matched the optimum on 100% of 692 cubes after
  rounding (mean 0.02 extra moves, worst case 10).
- **The time profile differs with depth.** On easy cubes the hybrid is far cheaper
  (depth 8: 17.5 ms against 419 ms), since pruning keeps its beam small; from depth 15
  onward the wider beam is both cheaper and more successful.

**The reason given for running this experiment was wrong.** Finding 8c argued the
answer might flip because widening from 100 to 1000 added 21.8 points for the hybrid
but only 12.8 for beam search. The next tenfold widening added 22.2 points for beam
search (63.6% to 85.8%, depths 11-20). Beam search's gains had not slowed; its step
from 100 to 1000 was simply the smaller of its two steps. Two points were not a trend
-- worth a line in the report, since the experiment was chosen on that argument.

### Cost against reach (final)

Pooled over depths 11-20 (500 cubes per solver). This range was chosen after seeing
the data, as where the search solvers are not all at 100%.

| Solver | Cubes solved | Mean time per cube |
|---|---|---|
| Greedy | 19.0% | 0.4 ms |
| Beam search, width 100 | 50.8% | 28.9 ms |
| Beam search, width 1000 | 63.6% | 204 ms |
| Beam search, width 10,000 | **85.8%** | 1,899 ms |
| Hybrid, width 100 | 58.0% | 394 ms |
| Hybrid, width 1000 | 79.8% | 2,598 ms |
| IDA\*, 200k-node limit | 38.8% | 6,961 ms |

- **At both budgets tested, beam search reaches further for the same or less time:**
  width 1000 over hybrid width 100 at roughly 0.2-0.4 s per cube, and width 10,000 over
  hybrid width 1000 at roughly 2-3 s.
- **IDA\* with the corner database is out-reached by every search configuration from
  beam width 100 upward**, at a small fraction of its time -- within the limits already
  noted for its 200,000-node budget.

**Taken together, the answer to the proposal's main question:**

1. **The pattern database does improve the learned solver at a fixed search width:**
   significantly more cubes solved (38 to 0 at width 100, 91 to 0 at width 1000), and
   never a longer solution.
2. **But for the same time, a wider beam does better, at both budgets tested** (40 to
   10 and 64 to 21, both significant, each in less total time and with slightly shorter
   solutions). In this implementation, compute is better spent on search width than on
   the corner pattern database.
3. **This is a conclusion about this hybrid, not about pattern databases in general.**
   The hybrid's cost comes mostly from restarting its search under a longer bound after
   each failed pass, and the database covers corners only. At width 1000 the hybrid took
   only about a third more total time than the wider beam (33.5 s against 25.1 s), so a
   cheaper hybrid or a stronger database is exactly what could change the answer -- the
   natural future work.

## Scope decisions against the proposal

Checked against `AI proposal - revised.md`, two promised items are not being
built as written. The report needs to say so explicitly, with the reason, rather
than leave them silently missing.

### 2x2x2 network validation (proposal Section 2, Evaluation #5): replaced

**Promised:** check the trained network and search pipeline exhaustively
against the exact 2x2x2 optimal-move table.

**Done instead:** the 2x2x2 table validated the cube simulator (Step 3), and
solver correctness is checked against IDA*'s exact optimal solutions on the
actual 3x3x3 cubes being evaluated (Steps 7-8).

**Why:** this is the stronger check. It tests the network that is actually
being reported, on the puzzle and the very cubes being reported, instead of a
separately trained network on a smaller puzzle. It also covers the second half
of Evaluation #5, comparison with optimal solutions on a sample of 3x3x3 cubes.

### Correlated scrambles and memoization (research question, Section 4, Evaluation #3): narrowed

**Measured:** whether shared-prefix scramble generation creates repeated states,
by depth; how many target-network evaluations a within-iteration cache would
skip; and how many distinct training states that costs.

**Why within one iteration:** ADI's targets depend on the network's current
weights, which change every iteration, so a cache carried across iterations
returns stale targets. Within a single iteration the weights are fixed, so
skipping a duplicate evaluation is exact, and its saving can be measured
without training at all.

**Not measured:** whether less varied batches make the network learn worse --
the proposal's "reduce the training compute needed to reach a given solve
quality". That needs paired training runs. On this hardware each run takes
about two hours, and a single seed per setting would not separate the effect
from ordinary run-to-run variation. Left as future work.

---

## Cache study: does shared-prefix scrambling make memoization worthwhile?

**Built:** a `prefix_sharing` option on `adi.generate_scrambles` -- the
probability, at each step, that a scramble abandons its own history and
continues from another scramble's current position -- and
`scripts/cache_study.py`. The option is off by default, and a test checks that
at 0 the generator produces exactly the scrambles it did before, so the data the
network trained on is unchanged.

**What is measured** (see the scope decision above for why): a cache that is
exact because it lives inside one training iteration, where the network's
weights are fixed. A child position appearing several times in a batch then
needs evaluating only once, and skipping the repeats cannot change any target,
so the saving can be measured without training.

Settings as in training: 256 scrambles x depth 12 = 3,072 positions and 55,296
children per batch, averaged over 20 batches.

| Prefix sharing | Distinct children | Evaluations saved | Distinct training positions | Dedupe ms |
|---|---|---|---|---|
| 0.00 (independent, as trained) | 64.1% | 35.9% | 78.9% | 39.3 |
| 0.25 | 62.6% | 37.4% | 77.5% | 39.6 |
| 0.50 | 61.7% | 38.3% | 76.5% | 37.6 |
| 0.75 | 61.2% | 38.8% | 76.1% | 33.4 |
| 0.90 | 61.1% | 38.9% | 75.9% | 30.7 |

Distinct positions at each depth, as a share of the 256 scrambles:

| Prefix sharing | d1 | d2 | d3 | d4 | d5 | d6 | d8 | d12 |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 7% | 61% | 89% | 97% | 99% | 100% | 100% | 100% |
| 0.90 | 7% | 60% | 87% | 95% | 96% | 97% | 97% | 97% |

### Finding C1: a cache is worthwhile without correlated scrambles, and correlation adds almost nothing

With ordinary independent scrambles, **35.9% of child evaluations are already
duplicates**. Sharing prefixes at 90% raises that by only 3.0 points, to 38.9%,
and costs 3.0 points of distinct training positions (78.9% to 75.9%).

The proposal's hypothesis, as stated, is therefore not supported: correlated
generation is not what makes memoization useful, because the opportunity is
already there without it. This is a negative result on the mechanism and a
positive one on caching.

### Finding C2: where the duplicates come from

Measured on one batch of independent scrambles:

| Deduplicating... | Distinct children |
|---|---|
| the whole batch | 64.3% |
| only children of positions at depth 6-12 | 81.7% |
| only within each individual scramble | 79.4% |

| Positions compared, within one scramble | Children in common (of 18) |
|---|---|
| one move apart | 2.00 |
| two moves apart | 2.25 |

Two mechanisms account for the duplicates:

1. **Few positions exist near the solved cube.** There are only 18 positions
   one move from solved, so 256 scrambles yield just 7% distinct positions at
   depth 1 and 61% at depth 2. Scrambles collide there, and so do their
   children.
2. **ADI keeps every intermediate position**, so neighbouring positions in one
   scramble are a single move apart and share children. If the step between
   them was a turn of some face, then turning that same face again from the
   later position lands exactly where one of the other turns of that face lands
   from the earlier position. For a step of `R`: `R` then `R` is `R2`, and `R`
   then `R2` is `R'`, so those two children coincide (the third, `R` then `R'`,
   returns to the earlier position itself, which is not one of its own
   children). That predicts exactly 2 shared children, and 2.00 is what was
   measured. A position lying between two others is also a child of both.
   Deduplicating inside each scramble alone removes 20.6 of the 35.7 points.

**Why sharing prefixes cannot add much:** two scrambles that fork from the same
position then each draw their own random move from 18, so they coincide at the
next step only about one time in 18, and diverge from then on. Even at 90%
sharing, positions deeper than 5 moves remain about 97% distinct.

### Finding C3: the cache roughly pays for itself on CPU alone

One batch, on CPU while training was sharing the machine:

| Cost | ms |
|---|---|
| Encoding all 55,296 children | 109.4 |
| Finding the duplicates | 26.8 |
| Encoding avoided by skipping them (35.7%) | 39.1 |

A net saving of about 12 ms per iteration on encoding alone. The GPU
forward-pass saving comes on top of that but could not be measured while
training occupied the GPU.

**Side finding for the compute-efficiency discussion:** encoding children on the
CPU takes about 110 ms of a roughly 240 ms training iteration -- close to half.
On this hardware the bottleneck of ADI training is CPU-side input encoding, not
the GPU. Both figures were measured under contention and should be rechecked
once training has finished.

### What this answers, and what it does not

**Answers** the memoization half of the research question: an exact cache
within each iteration would skip about a third of target evaluations with
independent scrambling, and correlated generation is unnecessary for it.

**Does not answer** whether less varied batches change learning quality.
Deduplication cannot (it is exact); prefix sharing might, since it reduces
distinct training positions. That needs paired training runs and is future
work. The deduplication was measured, not built into the training loop, so the
network being evaluated was trained exactly as before.

**Figures for the report** (generated by the *Cache study* section of
`notebooks/01_results.ipynb`, which prints the underlying tables first):

- `figures/cache_prefix_sharing.png` -- evaluations a cache would skip and
  distinct training positions, against the sharing rate, on a full 0-100% axis
  so that the flatness is shown at its true size.
- `figures/cache_distinct_by_depth.png` -- distinct positions by depth,
  independent scrambles versus 90% sharing, showing repeats concentrated at
  depths 1-3 under both.

---

## Status

| Step | State |
|---|---|
| 1. 3x3x3 engine | Done, validated |
| 2. 2x2x2 engine | Done, validated |
| 3. 2x2x2 exhaustive BFS + ground-truth table | Done, matches published results |
| 4. Corner pattern database (3x3x3) | Done, complete and validated |
| 5. Encoding, network, ADI training loop | Done; trained to 60k iterations (Finding T1) |
| 6a. Greedy baseline, measured against ground truth | Done on every 10k snapshot, 20k-60k |
| 6b. Beam search baseline | Done at widths 100 and 1000 (Finding 6b-ii) |
| 7. Hybrid search and PDB-only IDA* | Done; IDA* evaluated in full (Finding 7d) |
| 8. Evaluation | Done (Findings 8a-8d): the database helps at equal width; a wider beam wins at equal time |
| Write-up | Yours; figures in `figures/`, tables in `notebooks/02_comparison.ipynb` |
| Cache study (proposal's memoization question) | Done; GPU saving to measure after training |

Test suite: 191 tests.
