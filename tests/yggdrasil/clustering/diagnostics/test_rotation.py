import numpy as np
import pytest

from yggdrasil.clustering.diagnostics.rotation import rotation_cost


def _block_indicator(n_per_block: int, n_blocks: int) -> np.ndarray:
    """Return an (n_per_block * n_blocks, n_blocks) one-hot indicator."""
    return np.repeat(np.eye(n_blocks, dtype=np.float64), n_per_block, axis=0)


def test_rotation_cost_returns_zero_for_one_dimensional_input():
    X = np.zeros((5, 1))

    assert rotation_cost(X) == 0.0


def test_rotation_cost_low_on_clean_indicator_basis():
    indicator = _block_indicator(n_per_block=4, n_blocks=3)

    cost = rotation_cost(indicator, random_state=0)

    assert 0.0 <= cost < 0.05


def test_rotation_cost_in_unit_interval():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))

    cost = rotation_cost(X, random_state=0)

    assert 0.0 <= cost <= 1.0


def test_rotation_cost_increases_with_wrong_k():
    indicator = _block_indicator(n_per_block=4, n_blocks=3)
    rng = np.random.default_rng(0)

    matched = rotation_cost(indicator, random_state=0)
    over = rotation_cost(
        np.hstack([indicator, rng.normal(0.0, 0.5, size=(indicator.shape[0], 1))]),
        random_state=0,
    )

    assert over > matched


def test_rotation_cost_deterministic_under_fixed_random_state():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))

    a = rotation_cost(X, random_state=42)
    b = rotation_cost(X, random_state=42)

    assert a == pytest.approx(b)


def test_rotation_cost_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2-D"):
        rotation_cost(np.zeros(5))
