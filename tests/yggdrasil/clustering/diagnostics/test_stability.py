import numpy as np
import pytest

from yggdrasil.clustering.diagnostics.stability import (
    label_stability,
    pairwise_ari_matrix,
)


def test_pairwise_ari_matrix_is_identity_on_identical_labelings():
    a = np.array([0, 0, 1, 1])

    M = pairwise_ari_matrix([a, a, a])

    assert M.shape == (3, 3)
    assert np.allclose(M, np.ones((3, 3)))


def test_pairwise_ari_matrix_invariant_under_label_permutation():
    a = np.array([0, 0, 1, 1])
    b = np.array([1, 1, 0, 0])

    M = pairwise_ari_matrix([a, b])

    assert M[0, 1] == pytest.approx(1.0)


def test_pairwise_ari_matrix_uses_nmi_when_requested():
    a = np.array([0, 0, 1, 1])
    b = np.array([0, 1, 0, 1])

    M = pairwise_ari_matrix([a, b], metric="nmi")

    assert 0.0 <= M[0, 1] <= 1.0


def test_label_stability_one_for_perfect_agreement():
    a = np.array([0, 0, 1, 1])

    assert label_stability([a, a]) == pytest.approx(1.0)


def test_label_stability_low_for_random_labelings():
    rng = np.random.default_rng(0)
    labelings = [rng.integers(0, 4, size=200) for _ in range(4)]

    assert label_stability(labelings) < 0.1


def test_label_stability_returns_nan_for_too_few_labelings():
    a = np.array([0, 0, 1, 1])

    assert np.isnan(label_stability([a]))


def test_label_stability_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        label_stability([np.array([0, 1]), np.array([0, 1, 0])])


def test_label_stability_rejects_unknown_metric():
    a = np.array([0, 0, 1, 1])

    with pytest.raises(ValueError, match="metric"):
        label_stability([a, a], metric="rand")
