# Research Proposal: A Hybrid Learned-and-Exact Heuristic Approach to Solving the 3×3 Rubik's Cube

## Project Goal

The goal of this project is to build a solver for the 3×3 Rubik's Cube that combines a curriculum-trained neural network with an exact, precomputed pattern-database heuristic inside a search algorithm. Rather than relying on the network alone to pick moves, the project studies how a learned value/policy estimate and an exact admissible heuristic can be combined during search, and how the training data itself can be generated more efficiently through correlated scrambles and a shared memoization cache. The project is explicitly framed as an extension of Autodidactic Iteration (ADI), the method introduced in McAleer et al. (2018), *"Solving the Rubik's Cube Without Human Knowledge,"* rather than a reproduction of it.

## Motivation

The Rubik's Cube is a well-studied benchmark in AI because it has an enormous state space (~4.3×10¹⁹ configurations) but a single, clearly defined goal state, and a known upper bound on solution length (God's Number: 20 moves in the half-turn metric). This combination makes it a useful testbed for search, planning, and learning-based decision making.

McAleer et al. (2018) showed that a neural network trained purely through self-play — using Autodidactic Iteration to bootstrap value and policy targets outward from the solved state, without human data or an external solver — can be combined with Monte Carlo Tree Search (MCTS) to solve 100% of randomly scrambled cubes, matching or beating a human-knowledge-based solver (Kociemba) in over half of cases. This result already answers, at large scale, the basic question of whether curriculum-style training from easy to hard states lets a network learn to solve the cube.

This project does not attempt to re-answer that question at the same scale — doing so required roughly 8 billion cube visits and 44 hours across three GPUs. Instead, it asks a set of questions the original paper leaves open:

1. **Can an exact, classical heuristic (a pattern database) meaningfully improve or accelerate a learned solver, and how should the two be combined?** DeepCube's search relies on the network alone; it never incorporates an admissible heuristic of the kind used by classical optimal solvers (e.g., Korf's IDA* with pattern databases).
2. **Does correlated (shared-prefix) scramble generation make caching/memoization useful at depths where independently generated scrambles never overlap?** DeepCube generates every training scramble independently from the solved state, which means a memoization cache built during training would see almost no repeat states beyond shallow depths.
3. **How much of DeepCube's performance can be recovered at a small fraction of its training compute**, by using a smaller network, a compact state representation, and the efficiency techniques above? This matters because most students and small research groups do not have access to multi-GPU training budgets, and understanding the accuracy-versus-compute tradeoff is itself a useful, honest contribution.

## Related Work

This project builds directly on:

- **McAleer, Agostinelli, Shmakov, and Baldi (2018), "Solving the Rubik's Cube Without Human Knowledge"** (arXiv:1805.07470). Introduces Autodidactic Iteration (ADI): a joint value/policy network trained by generating scrambles outward from the solved state and bootstrapping value targets via a depth-1 breadth-first search using the network's own current estimates, weighted by 1/depth to stabilize training. Combines the trained network with MCTS at test time (policy narrows branching, value estimates leaf quality) to solve 100% of fully scrambled cubes.
- **Korf (1997), "Finding Optimal Solutions to Rubik's Cube Using Pattern Databases."** Introduces pattern databases: exact, precomputed distance-to-solved tables for a reduced sub-problem (e.g., the 8 corner cubelets, ~88 million states), used as an admissible heuristic in IDA* search. Guarantees optimal solutions but is extremely slow on fully scrambled cubes (reported to take several days per cube).
- **Kociemba's two-phase algorithm.** A fast, general-purpose solver based on cube group theory. Always finds a solution quickly but not necessarily an optimal one.

This project's contribution is to combine ideas from the first two lines of work — a learned heuristic and an exact heuristic — which neither paper does, and to study data-generation efficiency (correlated scrambles, caching) that neither paper addresses.

## Research Question

**To what extent does combining a curriculum-trained value/policy network with an exact pattern-database heuristic improve solving efficiency and solution quality over either approach alone, and can correlated scramble generation with a shared memoization cache reduce the training compute needed to reach a given solve quality, compared to independent scramble generation?**

## Proposed Method

### 1. State representation and network

Cube states will be represented using the reduced cubie-based encoding (20 corner/edge cubelets × 24 possible positions each), following McAleer et al., rather than the full 54-sticker encoding, to keep the input compact. The network will output a value estimate and a move-probability distribution over the 12 possible moves, trained with a combined value and policy loss.

### 2. Pipeline validation on 2×2×2 (preliminary step, not the main deliverable)

Before committing compute to the 3×3 cube, the full training and evaluation pipeline will be validated on the 2×2×2 cube, whose state space (~3.6 million states) is small enough to fully enumerate via breadth-first search from the solved state. This produces an exact, complete optimal-move table, against which the trained network's predictions and the search pipeline's outputs can be checked exhaustively. This step exists purely to catch bugs in encoding, training, and evaluation code before scaling up — it is not a research result in itself.

### 3. Curriculum training with weighted sampling

Following ADI, training scrambles will be generated outward from the solved state up to a chosen maximum depth (scoped to what is computationally feasible, likely well below the full 20-move bound). Rather than strict sequential depth staging, samples of mixed depth will be drawn each iteration with loss weighted by 1/depth, as McAleer et al. found necessary to prevent divergence. Value and policy targets will be generated via the same bootstrapped depth-1 search used in ADI.

### 4. Correlated scramble generation and memoization cache

As an explicit alternative to independent scramble generation, training scrambles will also be generated as a shared tree — new scrambles are produced by extending previously generated scrambles rather than always starting fresh from the solved state — so that intermediate states are deliberately revisited. A memoization cache (state → best known value/move) will be maintained across training, persisting between iterations. Cache hit rate will be measured as a function of scramble depth, for both the independent and correlated generation strategies, to test whether correlated generation makes caching useful at greater depths than independent generation allows.

### 5. Pattern database construction and hybrid search

A corner-cubelet pattern database (~88 million states) will be built via exhaustive breadth-first search from the solved state, storing the exact optimal distance-to-solved for every corner configuration. This gives an admissible heuristic independent of the learned network. Several search configurations will be implemented and compared:

- **Network-only, greedy** (baseline, expected to perform worst, following McAleer et al.'s own "Greedy" ablation).
- **Network-only with MCTS** (reproducing DeepCube's approach at reduced scale).
- **Pattern-database-only with IDA*** (reproducing Korf's classical approach at reduced scale).
- **Hybrid**: pattern database used for admissible pruning/lower-bounding within the search, network value/policy used to guide move ordering and prioritize which branches to expand first.

## Evaluation

The project will evaluate along several axes:

1. **Solve quality**: solve rate and solution length (compared to the 20-move bound where feasible) across each search configuration, at a fixed maximum search budget (time or nodes expanded).
2. **Search efficiency**: nodes expanded per solve, for network-only, PDB-only, and hybrid search, to test whether the hybrid genuinely reduces search effort rather than merely combining two costs.
3. **Cache/memoization utility**: hit rate versus scramble depth, comparing independent versus correlated scramble generation, and the resulting effect on training wall-clock time to reach a given accuracy.
4. **Compute-efficiency comparison**: solve quality achieved as a function of training compute used, positioned explicitly against the reported compute budget in McAleer et al. (2018), to characterize the accuracy-versus-compute tradeoff at small scale.
5. **Correctness validation**: exhaustive comparison against the exact optimal-move table on 2×2×2, and comparison against Korf's known depth-15 results on a sample of 3×3 states, following the same evaluation McAleer et al. used.

## Expected Outcome

We expect the pattern-database-augmented hybrid search to expand fewer nodes and/or find shorter solutions than the network-only search at equivalent compute, since it has access to exact lower-bound information the learned value function does not. We expect correlated scramble generation to substantially increase cache hit rates at moderate depths compared to independent generation, though we expect this benefit to shrink as depth increases, reflecting the combinatorial growth of the state space. We do not expect to match DeepCube's full-scale solve rate or move-count results given the difference in available compute; instead, the project aims to characterize how much of that performance is recoverable at a small fraction of the original training cost, and to identify which specific components (exact heuristic, correlated data generation, caching) contribute most to that recovery. This project extends the combination of neural networks, search, and curriculum learning explored in McAleer et al. (2018) by incorporating classical exact heuristics and studying training-data efficiency, two directions the original work does not address.
