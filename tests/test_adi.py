import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rubiks import adi
from rubiks.cube import ALL_MOVES, apply_move, apply_sequence, is_solved, solved_state
from rubiks.network import CubeNet


def inverse_move(move):
    if move.endswith("'"):
        return move[0]
    if move.endswith("2"):
        return move
    return move + "'"


def test_move_permutations_match_the_engine():
    rng = np.random.default_rng(0)
    state = apply_sequence(
        solved_state(), [ALL_MOVES[rng.integers(18)] for _ in range(10)]
    )
    for index, move in enumerate(ALL_MOVES):
        assert np.array_equal(
            state[adi.MOVE_PERMUTATIONS[index]], apply_move(state, move)
        )


def test_is_solved_batch_agrees_with_engine():
    rng = np.random.default_rng(1)
    states = [solved_state()]
    for length in range(1, 8):
        states.append(
            apply_sequence(
                solved_state(), [ALL_MOVES[rng.integers(18)] for _ in range(length)]
            )
        )
    batch = np.stack(states)
    expected = np.array([is_solved(s) for s in states])
    assert np.array_equal(adi.is_solved_batch(batch), expected)


def test_generate_scrambles_shapes_and_depths():
    rng = np.random.default_rng(2)
    states, depths = adi.generate_scrambles(sequences=5, max_depth=4, rng=rng)
    assert states.shape == (20, 54)
    assert depths.shape == (20,)
    assert sorted(depths.tolist()) == sorted([1] * 5 + [2] * 5 + [3] * 5 + [4] * 5)


def test_generate_scrambles_states_are_reachable_and_mostly_unsolved():
    rng = np.random.default_rng(3)
    states, depths = adi.generate_scrambles(sequences=50, max_depth=6, rng=rng)
    # A depth-1 state is exactly one move from solved, so never solved itself.
    assert not adi.is_solved_batch(states[depths == 1]).any()


def test_children_of_matches_apply_move():
    rng = np.random.default_rng(4)
    states, _ = adi.generate_scrambles(sequences=3, max_depth=3, rng=rng)
    children = adi.children_of(states)
    assert children.shape == (9, 18, 54)
    for i, state in enumerate(states):
        for m, move in enumerate(ALL_MOVES):
            assert np.array_equal(children[i, m], apply_move(state, move))


def test_targets_for_depth_one_states_point_at_the_solving_move():
    """The one grounded signal in ADI: a state one move from solved must be
    labelled with that move, and valued at the goal reward."""
    device = torch.device("cpu")
    model = CubeNet(trunk=(64,), head=32)

    states = []
    solving_moves = []
    for move in ALL_MOVES:
        states.append(apply_move(solved_state(), move))
        solving_moves.append(ALL_MOVES.index(inverse_move(move)))

    target_values, target_policies = adi.compute_targets(
        model, np.stack(states), device
    )

    assert target_policies.tolist() == solving_moves
    assert torch.allclose(
        target_values, torch.full_like(target_values, adi.SOLVED_REWARD)
    )


def test_solved_children_are_treated_as_terminal():
    """A solved child must contribute its reward alone, not an estimated value.

    The value head is pinned to -50, so every non-terminal child scores
    -51. The solved child scores +1 only if its estimate is discarded; were
    the estimate used instead the target would come out at -49.
    """
    device = torch.device("cpu")
    model = CubeNet(trunk=(64,), head=32)
    with torch.no_grad():
        model.value_head[-1].weight.zero_()
        model.value_head[-1].bias.fill_(-50.0)

    state = apply_move(solved_state(), "R")[None, :]
    target_values, target_policies = adi.compute_targets(model, state, device)

    assert pytest.approx(target_values.item(), abs=1e-4) == adi.SOLVED_REWARD
    assert ALL_MOVES[target_policies.item()] == "R'"


def test_train_step_runs_and_reports_finite_losses():
    device = torch.device("cpu")
    model = CubeNet(trunk=(128,), head=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(5)
    states, depths = adi.generate_scrambles(sequences=8, max_depth=5, rng=rng)

    stats = adi.train_step(model, optimizer, states, depths, device)

    for key in ("loss", "value_loss", "policy_loss", "target_accuracy"):
        assert np.isfinite(stats[key])
    assert 0.0 <= stats["target_accuracy"] <= 1.0


def test_training_reduces_loss_on_a_fixed_shallow_batch():
    """Sanity check that the loop learns at all, on states it sees repeatedly."""
    device = torch.device("cpu")
    torch.manual_seed(0)
    model = CubeNet(trunk=(256,), head=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(6)
    states, depths = adi.generate_scrambles(sequences=16, max_depth=2, rng=rng)

    first = adi.train_step(model, optimizer, states, depths, device)["loss"]
    for _ in range(30):
        stats = adi.train_step(model, optimizer, states, depths, device)
    assert stats["loss"] < first
